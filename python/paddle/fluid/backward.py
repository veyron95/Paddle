#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from .proto import framework_pb2

from paddle.fluid import framework as framework
from paddle.fluid import program_guard
from . import core
import collections
import copy
import six
import logging
from .. import compat as cpt
from . import unique_name
from . import log_helper
import paddle.fluid
from .data_feeder import check_type
import warnings
__all__ = [
    'append_backward',
    'gradients',
]

_logger = log_helper.get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class ProgramStats(object):
    def __init__(self, block, ops):
        self.block = block
        self.ops = ops
        self.op_deps = {}  # op-> in_ops, out_ops
        self.var_op_deps = {}  # var as input op, var as output op

    def get_input_nodes(self):
        input_names = []
        for name in self.var_op_deps:
            if len(self.var_op_deps[name]["var_as_output_ops"]) == 0 and \
                    len(self.var_op_deps[name]["var_as_input_ops"]) > 0:
                if self.block.var(name).persistable:
                    continue
                input_names.append(name)
        for op in self.ops:
            if op.desc.type() == "read":
                input_names.extend(op.desc.output_arg_names())
        return input_names

    def get_reserved_vars(self):
        var_name = []
        for op in self.ops:
            if op.desc.type() == "seed":
                var_name.extend(op.desc.output_arg_names())
        return var_name

    def get_out_of_subgraph_vars(self, begin_op_idx, end_op_idx):
        var_name = []
        for i in range(begin_op_idx, end_op_idx, 1):
            for name in self.ops[i].desc.output_arg_names():
                if name in self.var_op_deps:
                    for idx in self.var_op_deps[name]["var_as_input_ops"]:
                        if idx >= end_op_idx:
                            var_name.append(name)
            for name in self.ops[i].desc.input_arg_names():
                if name in self.var_op_deps:
                    for idx in self.var_op_deps[name]["var_as_output_ops"]:
                        if idx < begin_op_idx:
                            var_name.append(name)
        return var_name

    def is_subgraph(self, var_group1, var_group2):
        # should traverse from var_group1 to var_group2
        # max op idx in var_group2
        # min op idx in var_group1
        min_op_idx = len(self.ops)
        max_op_idx = -1
        for name in var_group1:
            if name not in self.var_op_deps:
                return False, min_op_idx, max_op_idx
        for name in var_group2:
            if name not in self.var_op_deps:
                return False, min_op_idx, max_op_idx
        for name in var_group1:
            op_idx = self.var_op_deps[name]["var_as_input_ops"]
            for idx in op_idx:
                min_op_idx = min(min_op_idx, idx)
        for name in var_group2:
            op_idx = self.var_op_deps[name]["var_as_output_ops"]
            for idx in op_idx:
                max_op_idx = max(max_op_idx, idx)
        if min_op_idx >= max_op_idx:
            return False, min_op_idx, max_op_idx

        return True, min_op_idx, max_op_idx

    def _update_segment_start(self, min_idx, pre_segment_end_idx):
        """
        persist vars of amp-related cast should be included in recompute segment
        """

        def is_amp_cast(op):
            return op.desc.type() == 'cast' and self.block.var(
                op.desc.input_arg_names()[0]).persistable

        idx_ = min_idx - 1
        updated_min_idx = min_idx
        while idx_ > pre_segment_end_idx:
            if is_amp_cast(self.ops[idx_]):
                _logger.info("found amp-cast op: {}, : {}".format(self.ops[
                    idx_].desc.type(), self.ops[idx_].desc.input_arg_names()[
                        0]))
                updated_min_idx = idx_
                idx_ -= 1
            else:
                break

        return updated_min_idx

    def build_stats(self):
        for i, op in enumerate(self.ops):
            self.op_deps[i] = {"in_ops": [], "out_ops": []}
            for j, name in enumerate(op.desc.input_arg_names()):
                if name in self.var_op_deps:
                    self.op_deps[i]["in_ops"].extend(self.var_op_deps[name][
                        "var_as_output_ops"])
            for j, name in enumerate(op.desc.input_arg_names()):
                if name in self.var_op_deps:
                    self.var_op_deps[name]["var_as_input_ops"].extend([i])
                else:
                    self.var_op_deps[name] = {}
                    self.var_op_deps[name]["var_as_input_ops"] = [i]
                    self.var_op_deps[name]["var_as_output_ops"] = []

            for j, name in enumerate(op.desc.output_arg_names()):
                if name in self.var_op_deps:
                    self.var_op_deps[name]["var_as_output_ops"].extend([i])
                else:
                    self.var_op_deps[name] = {}
                    self.var_op_deps[name]["var_as_input_ops"] = []
                    self.var_op_deps[name]["var_as_output_ops"] = [i]

            for op_idx in self.op_deps[i]["in_ops"]:
                self.op_deps[op_idx]["out_ops"].extend([i])

    def sort_checkpoints(self, checkpoints_name):
        sorted_checkpoints = []
        for name in checkpoints_name:
            if name not in self.var_op_deps:
                _logger.info(
                    "Recompute Optimizer: deleted %s from checkpoints, because it is not used in paddle program."
                    % name)
            elif self.var_op_deps[name]["var_as_output_ops"] == []:
                # input nodes
                sorted_checkpoints.append((name, -1))
            else:
                sorted_checkpoints.append(
                    (name, max(self.var_op_deps[name]["var_as_output_ops"])))
        sorted_checkpoints = sorted(sorted_checkpoints, key=lambda x: x[1])
        return [x[0] for x in sorted_checkpoints]

    def modify_forward_desc_for_recompute(self):
        op_types = [op.desc.type() for op in self.ops]
        if "dropout" not in op_types:
            return

        op_idx = 0
        while op_idx < len(self.ops):
            op = self.ops[op_idx]
            if op.desc.type() != "dropout":
                op_idx += 1
                continue
            # already insert seed op before dropout
            if op.input('Seed') is not None and len(op.input('Seed')) == 1:
                op_idx += 1
                continue
            # add a seed op so that the two dropout op can generate same output
            op_unique_name = unique_name.generate("seed")
            var_unique_name = unique_name.generate_with_ignorable_key(".".join(
                [op_unique_name, 'tmp']))
            added_var = self.block.create_var(
                name=var_unique_name,
                dtype='int32',
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=False)
            seed = 0 if op.attr("fix_seed") is False else int(op.attr("seed"))

            op_device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName(
            )
            op_device = ""
            if op.desc.has_attr(op_device_attr_name):
                op_device = op.desc.attr(op_device_attr_name)

            # Setting the force_cpu of seed to true will make the output of seed in cpu memory, 
            # reduce the synchronous copy from GPU to CPU in dropout, and reduce the communication hang
            added_op = self.block._insert_op(
                index=op.idx,
                type='seed',
                inputs={},
                outputs={'Out': [added_var]},
                attrs={
                    'seed': seed,
                    'op_device': op_device,
                    'force_cpu': True
                })
            self.ops.insert(op_idx, added_op)
            # modify dropout op desc so that it accept a seed var as input
            op.desc.set_input("Seed", [var_unique_name])
            op.desc.remove_attr("fix_seed")
            op.desc.remove_attr("seed")
            self.block._sync_with_cpp()
            op_idx += 2


def _pretty_op_desc_(op_desc, prefix):
    out_s = "%s\tname:[%s]\n%s    \tinputs:[%s]\n%s    \toutputs:[%s]" % \
            (prefix + "_op", str(op_desc.type()), prefix + "_input", " ".join(op_desc.input_arg_names()),
             prefix + "_output", " ".join(op_desc.output_arg_names()))
    return out_s


def _add_needed_descs_to_block(descs, block, main_block, in_memory_vars):
    if len(descs) == 0:
        return []
    result_descs = []
    op_role_attr_name = \
        core.op_proto_and_checker_maker.kOpRoleAttrName()
    backward = core.op_proto_and_checker_maker.OpRole.Backward
    for desc in descs:
        if isinstance(desc, framework.Operator):
            desc = desc.desc
        if isinstance(desc, tuple):
            desc = desc[0]
        is_needed = False
        for name in desc.output_arg_names():
            if main_block.has_var(name) and main_block.var(name).persistable:
                continue
            if name not in in_memory_vars:
                is_needed = True
        if is_needed:
            new_op_desc = block.desc.append_op()
            new_op_desc.copy_from(desc)
            new_op_desc._set_attr(op_role_attr_name, backward)
            if desc.has_attr('op_device'):
                new_op_desc._set_attr('op_device', desc.attr('op_device'))
            result_descs.append(new_op_desc)
    return result_descs


def _add_descs_to_block(descs, block):
    if len(descs) == 0:
        return []
    result_descs = []
    op_role_attr_name = \
        core.op_proto_and_checker_maker.kOpRoleAttrName()
    backward = core.op_proto_and_checker_maker.OpRole.Backward
    for desc in descs:
        if isinstance(desc, framework.Operator):
            desc = desc.desc
        if isinstance(desc, tuple):
            desc = desc[0]
        new_op_desc = block.desc.append_op()
        new_op_desc.copy_from(desc)
        new_op_desc._set_attr(op_role_attr_name, backward)
        if desc.has_attr('op_device'):
            new_op_desc._set_attr('op_device', desc.attr('op_device'))
        result_descs.append(new_op_desc)
    return result_descs


def _find_loss_op_(loss):
    for op in reversed(loss.block.ops):
        assert isinstance(op, framework.Operator)
        if len(op.output_arg_names) == 1 and op.output_arg_names[
                0] == loss.name:
            loss.op = op
            break
    if loss.op is None:
        raise ValueError("loss.op is None. Should not happend")


def _rename_arg_(op_descs, old_name, new_name, begin_idx=None, end_idx=None):
    """
    Traverse all ops in op_descs[begin_idx : end_idx],
    if any op has inputs/outputs named "old_name", rename it as 'new_name'
    """
    if begin_idx is None:
        begin_idx = 0
    if end_idx is None:
        end_idx = len(op_descs)
    if isinstance(op_descs, (list, tuple)):
        for i in range(begin_idx, end_idx):
            op_desc = op_descs[i]
            if isinstance(op_desc, tuple):
                op_desc = op_desc[0]
            op_desc._rename_input(old_name, new_name)
            op_desc._rename_output(old_name, new_name)
    if isinstance(op_descs, collections.OrderedDict):
        for key, value in op_descs.items():
            if isinstance(value, (list, tuple)):
                for op_desc in value:
                    op_desc._rename_input(old_name, new_name)
                    op_desc._rename_output(old_name, new_name)


def _create_op_desc_(op_type, inputs, outputs, attrs):
    """
    Create a C++ OpDesc object with specified inputs, outputs and attributes.
    """
    op_desc = core.OpDesc()
    op_desc.set_type(op_type)
    for para, args in six.iteritems(inputs):
        op_desc.set_input(
            para,
            list(
                map(lambda arg: arg.decode() if isinstance(arg, six.binary_type) else arg,
                    args)))
    for para, args in six.iteritems(outputs):
        op_desc.set_output(
            para,
            list(
                map(lambda arg: arg.decode() if isinstance(arg, six.binary_type) else arg,
                    args)))

    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
    op_device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()

    if op_role_attr_name not in attrs:
        attrs[
            op_role_attr_name] = core.op_proto_and_checker_maker.OpRole.Backward
    if op_device_attr_name not in attrs:
        attrs[op_device_attr_name] = ""
    for name, val in six.iteritems(attrs):
        if isinstance(val, framework.Block):
            op_desc.set_block_attr(name, val.desc)
        else:
            op_desc._set_attr(name, val)
    return op_desc


def _create_loss_op_desc_(loss):
    op_desc = _create_op_desc_(
        "fill_constant", {}, {"Out": [_append_grad_suffix_(loss.name)]}, {
            "shape": [1],
            "value": 1.0,
            "dtype": loss.dtype,
            "force_cpu": False,
            core.op_proto_and_checker_maker.kOpRoleAttrName():
            int(core.op_proto_and_checker_maker.OpRole.Backward) |
            int(core.op_proto_and_checker_maker.OpRole.Loss),
            core.op_proto_and_checker_maker.kOpDeviceAttrName():
            loss.op.attr(core.op_proto_and_checker_maker.kOpDeviceAttrName())
        })
    return op_desc


def _infer_var_data_type_shape_(grad_var_name, block):
    """
    Infer the data type and shape of given grad variable
    """
    grad_var = block.desc.find_var(cpt.to_bytes(grad_var_name))
    fwd_name = _strip_grad_suffix_(grad_var_name) # 去除 @GRAD 尾缀
    if block.desc.has_var_recursive(cpt.to_bytes(fwd_name)):
        fwd_var = block.desc.find_var_recursive(cpt.to_bytes(fwd_name))
        grad_var.set_dtype(fwd_var.dtype()) # grad_var 与 fwd_var 保持一致的 dtype 和 shape
        grad_var.set_shape(fwd_var.shape())
    else:
        # TODO(jiabin): Maybe we should not to this to cause some unexpected error on dtype
        warnings.warn(
            "Set grad var: {} dtype to default FP32, since we can't find its related forward var".
            format(grad_var_name))
        grad_var.set_dtype(core.VarDesc.VarType.FP32)


def _all_in_set_(cands, s):
    """
    Test if all elements of 'cands' are in set 's'
    """
    if len(cands) == 0:
        return False
    for c in cands:
        if not c in s:
            return False
    return True


def _some_in_set_(cands, s):
    """
    Test if some elements of 'cands' are in set 's'
    """
    if len(cands) == 0:
        return False
    literal_set = cpt.to_text(s)
    literal_cands = cpt.to_text(cands)
    for c in literal_cands:
        if c in literal_set:
            return True
    return False


def _strip_grad_suffix_(name):
    """
    Strip the grad suffix from the given variable name
    e.g. x@GRAD ==> x
         y@GRAD@RENAME@1 ==> y
    """
    name = cpt.to_text(name)
    pos = name.find(core.grad_var_suffix())
    new_name = name[:pos] if pos != -1 else name
    new_pos = name.rfind('grad/')
    return new_name[new_pos + 5:] if new_pos != -1 else new_name


def _append_grad_suffix_(name):
    """
    Append grad suffix to the given variable name
    e.g. x ==> x@GRAD
    """
    return cpt.to_text(name) + core.grad_var_suffix()


def _accumulate_gradients_by_sum_op_(var_name,
                                     renamed_vars,
                                     pending_sum_ops,
                                     op_idx,
                                     op_device=""):
    """
    Use sum op to accumulate_gradients, the gradients are stored in renamed_vars.
    """
    if op_idx not in pending_sum_ops.keys():
        pending_sum_ops[op_idx] = []
    pending_sum_ops[op_idx].append(
        _create_op_desc_("sum", {"X": renamed_vars[var_name]}, {
            "Out": [var_name]
        }, {"use_mkldnn": False,
            "op_device": op_device}))
    renamed_vars[var_name] = [var_name]


def _accumulate_gradients_by_add_ops_(var_name,
                                      renamed_vars,
                                      pending_sum_ops,
                                      op_idx,
                                      op_device=""):
    """
    Use several inplace add op to accumulate_gradients, the gradients are stored in renamed_vars.
    """
    if op_idx not in pending_sum_ops.keys():
        pending_sum_ops[op_idx] = []
    out_name = renamed_vars[var_name][0]
    for i in range(1, len(renamed_vars[var_name])):
        x_name = out_name
        y_name = renamed_vars[var_name][i]
        if i != len(renamed_vars[var_name]) - 1:
            out_name = var_name + '@ADD@' + str(i)
        else:
            out_name = var_name
        pending_sum_ops[op_idx].append(
            _create_op_desc_("grad_add", {"X": [x_name],
                                          "Y": [y_name]}, {"Out": [out_name]},
                             {"use_mkldnn": False,
                              "op_device": op_device}))
    renamed_vars[var_name] = [var_name]

# op_descs 是反向图的 op_descs
# 函数名的直观理解，就是将 重复的输出 加起来 (实际上，也把输入也加了起来)
def _addup_repetitive_outputs_(op_descs, block_idx):
    """
    In backward part, an variable may be the output of more than one ops.
    And one op may yield its multiple outputs to the same variable.
    In these cases, the variable should be the accumulation of all the outputs.
    `sum_op`s are added to implement the accumulate.
    """
    _MAX_ADD_NUM_ = framework._global_flags()['FLAGS_max_inplace_grad_add']
    #pending_sum_ops = []
    # 1. 定义好几个数据结构 用于存储
    pending_sum_ops = collections.OrderedDict()     # 存储 后继 聚合算子 字典
    var_rename_count = collections.defaultdict(int) # var 更名次数
    renamed_vars = collections.defaultdict(list)    # key：var_name,  value：rename 后的 new_name
    renamed_var_start_idx = collections.defaultdict(list) # key：var_name,  val: start_idx ---- idx 记录的是反向图中 op 的序号(第一次 rename 的 op idx)
    var_device = collections.defaultdict(str)       # 记录 var 对应的 device
    for idx, op_desc in enumerate(op_descs):  # 2. 遍历 反向图
        op_device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        op_device = ""
        if op_desc.has_attr(op_device_attr_name):
            op_device = op_desc.attr(op_device_attr_name) # 获取 device
        for var_name in op_desc.input_arg_names():  # 3. 遍历 反向 op 的输入
            if "@GRAD" not in var_name:
                continue
            if len(renamed_vars[var_name]) > 1: # 4. 检查 var_name 对应的 grad_var 有多少个，即多少个 grad_var 的梯度需要聚合，进行聚合操作
                                                # （实际上是在当前反向 op 的输入之前，先接了 一个 sum op）
                if len(renamed_vars[var_name]) > _MAX_ADD_NUM_:
                    _accumulate_gradients_by_sum_op_(var_name, renamed_vars,
                                                     pending_sum_ops, idx,
                                                     var_device[var_name])
                else:
                    _accumulate_gradients_by_add_ops_(var_name, renamed_vars,
                                                      pending_sum_ops, idx,
                                                      var_device[var_name])

        # param_name 是 反向 op 的输出
        for param_idx, param_name in enumerate(op_desc.output_names()): # 5. 遍历反向 op 的输出
            arg_names = op_desc.output(param_name)
            for arg_idx, var_name in enumerate(arg_names):
                if "@GRAD" not in var_name: # 5.1 非@GRAD 忽略
                    continue 
                # if "@RENAME@" in var_name:
                #    continue
                if var_name == core.empty_var_name( # 5.2 empty_var ，inplace 忽略
                ) or var_name in op_desc.input_arg_names():
                    # empty variable or inplace op
                    continue
                if len(renamed_vars[var_name]) == 0: # 6. 判断当前 遇到的 var_name 是否记录过，如果没有，加入 renamed_vars 记录起来，并且记录下反向图中的 op 序号
                    # it's the first time we get the variable
                    renamed_vars[var_name] = [var_name] # 记录下来
                    renamed_var_start_idx[var_name] = idx # idx 记录的是反向图中 op 的序号
                else: # 7. 判断当前 遇到的 var_name 有记录过，并且只是记录了一次
                    if len(renamed_vars[var_name]) == 1:
                        new_name = var_name + "@RENAME@block" + str(block_idx) + "@" + \
                            str(var_rename_count[var_name])                    # 7.1 创建 new_name 
                        var_rename_count[var_name] += 1  # 更名次数+1            # 7.2  在 var_rename_count 中记录 var_name 更名次数
                        # rename original var_name
                        renamed_vars[var_name][0] = new_name # 修改原始的名字     # 7.3 renamed_vars 里面更新 名字
                        # before change: _rename_arg_(op_descs, var_name,
                        #                             new_name, 0, idx)
                        # rename arg from idx of the first appearance
                        # in backward, not always from 0
                        
                        # 在 op_descs 里面遍历 op, 修改对应的 var_name
                        _rename_arg_(op_descs, var_name, new_name,              # 7.4 在 op_descs 里面修改对应的 var_name （需要遍历操作）
                                     renamed_var_start_idx[var_name], idx)
                        _rename_arg_(pending_sum_ops, var_name, new_name)       # 7.5 在 后接的 sum op 里面修改对应的 var_name

                        for p in op_desc.output_names()[:param_idx]:    # 8. 遍历当前反向 op 的  前 param_idx 个输出
                            p_arg_names = op_desc.output(p)
                            if var_name in p_arg_names:                 # 8.1 如果当前反向 op 的 输出var_name 是有被 rename 的话，需要更新
                                op_desc.set_output(p, [
                                    new_name if x == var_name else x
                                    for x in p_arg_names
                                ])

                        arg_names = [
                            new_name if x == var_name else x
                            for x in arg_names[:arg_idx]
                        ] + arg_names[arg_idx:]

                    # 9. 如果记录次数不止是一次，也需要更名
                    new_name = var_name + "@RENAME@block" + str(block_idx) + "@" + \
                        str(var_rename_count[var_name])
                    var_rename_count[var_name] += 1 # 9.1 更名次数 +1
                    arg_names[arg_idx] = new_name   # 9.2 arg_names 对应位置更名
                    op_desc.set_output(param_name, arg_names) # 9.3 op_desc 里面更名
                    renamed_vars[var_name].append(new_name)   # 9.4 renamed_vars 里记录下来，新的命名
                    # 9.5 record the latest device
                    var_device[var_name] = op_device
    # 到这，已经遍历完反向图
    for var_name, inputs in six.iteritems(renamed_vars): # 10. 遍历所记录的 renamed_vars 字典, [var_name]:[new_name1, new_name2,...]
        if len(renamed_vars[var_name]) > 1: 
            if len(renamed_vars[var_name]) > _MAX_ADD_NUM_:
                _accumulate_gradients_by_sum_op_(             # 11. 将 op 输出的 那些 需要做梯度聚合的 var_name 进行聚合，即对应的 [new_name1, new_name2,...] 做聚合
                    var_name, renamed_vars, pending_sum_ops,  # 实际是在后面加上 sum op 
                    len(op_descs), var_device[var_name])
            else:
                _accumulate_gradients_by_add_ops_(
                    var_name, renamed_vars, pending_sum_ops,
                    len(op_descs), var_device[var_name])

    # sum_op descs are sorted according to their insert position
    # 12. 处理 sum op descs 在整个反向图中的位置
    for key, value in collections.OrderedDict(
            reversed(list(pending_sum_ops.items()))).items():

        # NOTE(zhiqiu): Since reversed, the idx of op_descs to be inserted will remains correct.
        # For example, [0, 1, 2], and we want to insert 'a' at idx 1, 'b' at idx 2, and the expected result is [0, 1, 'a', 2, 'b'].
        # If reversed, we first insert 'b' at idx 2, it becomes [0, 1, 2, 'b'], and then insert 'a' at idx 1, it becomes [0, 1, 'a', 2, 'b'].
        # If not reverse, we first insert 'a' at idx 1, it becomes [0, 1, 'a', 2], and then insert 'b' at idx 2, it becomes [0, 1, 'a', 'b', 2].
        idx = key
        for i, op in enumerate(value):
            op_descs.insert(idx + i, op)

    return op_descs

# 移除不需要计算梯度的分支，即对反向图进行剪枝操作
def _remove_no_grad_branch_(op_descs, no_grad_set):
    """
    Remove unnecessary grad ops
    A grad op can be removed in two cases:
        1. all outputs of the grad op are in 'no_grad_set'
        2. all grad inputs of the grad op are in 'no_grad_set'
    """
    # 1. 定义了一个 func 用于区分当前 op 是否可以移除
    def _op_can_be_removed_(op_desc, no_grad_set):
        out_arg_names = op_desc.output_arg_names()
        if len(out_arg_names) == 0 or _all_in_set_(out_arg_names, no_grad_set):
            return True # 1.1 输出个数0 或者 所有输出都在 no_grad_set 里面， 可移除
        if _all_in_set_([
                name for name in op_desc.input_arg_names()
                if name.find(core.grad_var_suffix()) != -1
        ], no_grad_set): # 1.2 所有那些 如  var@GRAD  的输入 都在 no_grad_set 里面， 可移除
            no_grad_set.update(out_arg_names) # 1.3 更新一下 no_grad_set 集合
            return True
        return False

    # 2. Remove ops whose outputs are all in no_grad_dict
    op_descs = [
        op_desc for op_desc in op_descs
        if not _op_can_be_removed_(op_desc, no_grad_set)
    ]
    # 3. Insert fill_zeros_like_op, 因为有的反向 op 的输入 var@grad 虽然在 no_grad_set 里
    # （那是因为对应 var 的 stop_gradient=True），但是 var@grad 可以是其他反向 op 的输入（比如当前op的输入），所以这时候既然不对 var 算梯度，那就直接 fill zeros 即可
    to_insert = [] # 存放待插入的 fill_zeros_like op desc
    for idx, op_desc in enumerate(op_descs): # 3.1 遍历所有 反向 op
        for arg in op_desc.input_arg_names():  # 3.2 拿到当前反向 op 的所有输入
            # 3.3 如果 arg is a gradient var name and arg should not have gradient
            if core.grad_var_suffix() in arg and arg in no_grad_set:
                x_in = _strip_grad_suffix_(arg) # 3.3.1 移除 @GRAD 尾缀
                # the reason should be: arg can be input of another grad op
                # and the op is a not-to-remove op
                to_insert.append((_create_op_desc_(
                    "fill_zeros_like", {"X": [x_in]}, {"Out": [arg]}, {}), idx))
    # 4. 在反向图中插入 fill_zeros_like op
    # p[1] ----- idx
    # p[0] ----- _create_op_desc_
    list([op_descs.insert(p[1], p[0]) for p in reversed(to_insert)])

    return op_descs # 5. 返回（剪枝 + 追加 fill_zeros_like op ）后的反向图 op_descs

# 反向图 op_descs、前向图ops_path、反向图所有op记录的 grad_names input
def _find_not_need_ops(grad_op_descs, forward_ops, input_grad_names_set):
    """
    Pruning Program with Structural Analysis Method of Computational Graph.
    The nodes of the computational graph composed of backward OPS should be
    interconnected. If there are unconnected sub-graphs in the computational graph,
    these sub-graphs should be cut off.   # 1. 明确定义：反向计算图中不链接的子图应该被裁剪掉

    Args:
        grad_op_descs(list[core.OpDesc]): The candidate backward OpDescs.
        forward_ops(list[Operator]): The forward ops.
        input_grad_names_set(set): this set is used to store the gradients' name
            which is generated by backward ops, and input_grad_names_set can help
            to prune the unnecessary backward ops.

    Return:
        (set[core.OpDesc]): A set of OpDescs which should be pruned.
    """
    # 2. 新建一个内部 Var 类
    class Var(object):
        def __init__(self, var_name):
            self.var_name = var_name
            self.gen_op = None
            self.pendding_ops = []

        def set_gen_op(self, gen_op):
            assert isinstance(gen_op, Op)
            assert self.gen_op is None
            self.gen_op = gen_op

        def add_pending_op(self, op):
            assert isinstance(op, Op)
            self.pendding_ops.append(op)
     # 3. 新建一个内部 Op 类
    class Op(object):
        def __init__(self, op_desc):
            self.op_desc = op_desc
            self.inputs = []
            self.outputs = []

        def insert_input(self, var):
            assert isinstance(var, Var)
            self.inputs.append(var)

        def insert_output(self, var):
            assert isinstance(var, Var)
            self.outputs.append(var)

    var_versions = dict()
    # 4. 新建一个内部 _create_node 类
    def _create_node(name):
        if name not in var_versions.keys():
            var_versions[name] = [Var(name)]
        else:
            var_versions[name].append(Var(name))
        return var_versions[name][-1]
    # 5. 新建一个内部 _create_or_get_last_version_node 类
    def _create_or_get_last_version_node(name):
        if name not in var_versions.keys():
            var_versions[name] = [Var(name)]
        return var_versions[name][-1]
    # 6. 新建一个内部 _create_op_node 类
    def _create_op_node(op_desc):
        op_node = Op(op_desc)
        for input in op_desc.input_arg_names():
            var = _create_or_get_last_version_node(name=input)
            var.add_pending_op(op_node)
            op_node.insert_input(var)
        for output in op_desc.output_arg_names():
            var = _create_node(name=output)
            var.set_gen_op(op_node)
            op_node.insert_output(var)
        return op_node

    # 7. Record the forward vars， 这里有疑问，为何将 grad_names 放到 forward_vars_set 里面？？
    forward_vars_set = set() if input_grad_names_set is None else set(
        input_grad_names_set)
    for op in forward_ops: # 8. 遍历前向的 op_path, 将所有输入和输出都放入 forward_vars_set， why ?
        forward_vars_set.update(op.desc.input_arg_names())
        forward_vars_set.update(op.desc.output_arg_names())

    # 9. Record the vars which are created during backward and is not generated by op.
    backward_vars_set = set() # 记录那些在反向过程会创建的 var, 但是这些 var 又没有被 op 生成
    # special_op_nodes is the candidate sub-graph head node.
    special_op_nodes = set() # 10. 子图头节点
    for op_desc in grad_op_descs: # 11. 遍历反向图的 op_desc
        input_set = set(op_desc.input_arg_names()) # 11.1 拿到当前反向 op 的输入集合
        # The new_vars are created during backward and is not generated by op.
        new_vars = input_set - forward_vars_set - backward_vars_set
        backward_vars_set.update(op_desc.output_arg_names())

        op_node = _create_op_node(op_desc)
        if len(new_vars) == len(input_set):
            special_op_nodes.add(op_node)

    # 12.开始准备 no_need_op_descs 的工作
    not_need_op_descs = []
    # Start traversing all candidate sub-graph headers to check whether
    # they are connected to backward computational graphs, and if they are
    # not, list them in not_need_op_descs
    for special_op_node in special_op_nodes:
        op_list = [special_op_node]
        ready_vars = set(special_op_node.inputs)
        remove_ops = True
        candidate_ops = [special_op_node]
        while len(candidate_ops) > 0:
            op_node = candidate_ops.pop(0)
            if _all_in_set_(op_node.inputs, ready_vars):
                for out_var in op_node.outputs:
                    candidate_ops.extend(out_var.pendding_ops)
                    op_list.extend(out_var.pendding_ops)
                ready_vars.update(op_node.outputs)
            else:
                remove_ops = False
                break
        if remove_ops:
            not_need_op_descs.extend([node.op_desc for node in op_list])
    not_need_op_descs_set = set(not_need_op_descs)
    grad_op_descs_set = set(grad_op_descs)
    # If a backward computational graph is simply one sub-graph header, the
    # not_need_op_descs will be whole graph, this IF clause avoids it.
    if grad_op_descs_set == not_need_op_descs_set: # 13. 特殊情况不裁剪
        return set()
    return not_need_op_descs_set


def serialize_op_decs(op_desc):
    protostr = op_desc.serialize_to_string()
    proto = framework_pb2.OpDesc.FromString(six.binary_type(protostr))
    return proto.__str__()


def _append_backward_ops_with_checkpoints_(
        block, ops, target_block, no_grad_dict, grad_to_var, checkpoints):
    """
    Create grad ops with forward ops, and insert them into given block

    Args:
        block(Block): the block where forward ops are
        ops(Op): the forward operators whose forward recomputation backward ops need to be added
        target_block(Block): the block which is going to hold new generated grad ops
        no_grad_dict(dict):
            key(int) block index
            val(str): corresponding forward variable name
        checkpoints: variables that a user defined as checkpoint for forward recomputation

    Algorithms:
        0) deal with forward recomputing program descs
        1) find ops between checkpoints, i.e. recompute_segments
        2) go through all forward ops and induct all variables that will be hold in memory
            a. variables that are used across segments will be held in memory
            b. output of dropout op will be held in memory
            c. input variables will be held in memory
        3) go through each recompute_segments, add backward ops with forward recomputation
            a. add ops in current recompute_segment as forward recomputation ops
            b. rename all non-checkpoint variables in recomputation ops
            c. add backward ops of current recomputation ops
            d. add sum op for repetitive_outputs
        4) remove no grad branch as it is in _remove_no_grad_branch_
        5) Note1: all appended ops' OpRole are Backward
        6) Note2: all variables with new name should be returned so that _append_backward_vars_ can be called
        7) Note3: current forward recomputation backpropagation does not handle programs with subblock
    """

    checkpoints_name = [x.name for x in checkpoints]
    checkpoints_name = list(set(checkpoints_name))
    local_block = block.program._create_block()
    buffer_block = block.program._create_block()
    # 0) deal with forward recomputing program descs
    program_stat = ProgramStats(block, ops)
    program_stat.modify_forward_desc_for_recompute()
    program_stat.build_stats()

    # 1) find ops between checkpoints, i.e. recompute_segments
    checkpoints_name = program_stat.sort_checkpoints(checkpoints_name)
    segments = []

    if len(checkpoints_name) == 1:
        # only one checkpoint
        max_op_idx = -1
        var_group = [checkpoints_name[0]]
        for name in var_group:
            if name not in program_stat.var_op_deps:
                break
            op_idx = program_stat.var_op_deps[name]["var_as_output_ops"]
            # only count the last generate op
            for idx in op_idx:
                max_op_idx = max(max_op_idx, idx)
        if max_op_idx > 0:
            segments.append([0, max_op_idx + 1])
    else:
        start_idx = 0
        pre_segment_end_idx = -1
        while True:
            if start_idx >= len(checkpoints_name) - 1:
                break
            # min_idx: checkpoint_1' s input op
            # max_idx: checkpoint_2' s output op
            flag, min_idx, max_idx = program_stat.is_subgraph(
                [checkpoints_name[start_idx]],
                [checkpoints_name[start_idx + 1]])
            if flag:
                # max_idx + 1 since the exact and used segment end idx is max_idx
                min_idx = program_stat._update_segment_start(
                    min_idx, pre_segment_end_idx)
                segments.append([min_idx, max_idx + 1])
            else:
                _logger.info("Could not recompute op range [{}] - [{}] ".format(
                    min_idx, max_idx + 1))

            start_idx += 1

    if segments != [] and segments[0][0] != 0:
        recompute_segments = [[0, segments[0][0]]] + segments
    else:
        recompute_segments = segments

    for i, (idx1, idx2) in enumerate(recompute_segments):
        _logger.info("recompute segment[{}]".format(i))
        _logger.info("segment start op: [{}]: [{}]".format(ops[idx1].desc.type(
        ), ops[idx1].desc.input_arg_names()))
        _logger.info("segment end op: [{}]: [{}]".format(ops[
            idx2 - 1].desc.type(), ops[idx2 - 1].desc.input_arg_names()))
        _logger.info("recompute segment[{}]".format(i))
        _logger.info("segment start op: [{}]: [{}]".format(ops[idx1].desc.type(
        ), ops[idx1].desc.input_arg_names()))
        _logger.info("segment end op: [{}]: [{}]".format(ops[
            idx2 - 1].desc.type(), ops[idx2 - 1].desc.input_arg_names()))

    # 2) go through all forward ops and induct all variables that will be hold in memory
    vars_should_be_hold = []
    # a. variables that are used across segments will be held in memory
    for segment in recompute_segments:
        vars_should_be_hold.extend(
            program_stat.get_out_of_subgraph_vars(segment[0], segment[1]))

    cross_vars = set(vars_should_be_hold) - set(checkpoints_name)
    _logger.info("found [{}] vars which cross recompute segment: [{}], better checkpoints might be set to reduce those vars".format( \
    len(cross_vars), cross_vars))

    # b. output of seed op should be kept in memory
    vars_should_be_hold.extend(program_stat.get_reserved_vars())
    # c. input variables are checkpoints
    vars_should_be_hold.extend(program_stat.get_input_nodes())
    vars_should_be_hold = list(set(vars_should_be_hold))

    # 3) go through each recompute_segments, add backward ops with forward recomputation
    grad_op_descs = []
    var_name_dict = {}

    vars_in_memory = vars_should_be_hold + checkpoints_name

    max_calculated_op_position = len(ops)
    device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
    if recompute_segments == []:
        gap_ops = ops[0:max_calculated_op_position]
        for op in reversed(gap_ops):
            if op.has_attr("sub_block"):
                raise Exception("Recompute don't support ops with sub_block"
                                "invoke op: %s" %
                                _pretty_op_desc_(op.desc, "with_sub_block"))
            grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
                op.desc, cpt.to_text(no_grad_dict[block.idx]), [])
            # Set device for grad_op according to forward Op
            if op.desc.has_attr(device_attr_name):
                op_device = op.desc.attr(device_attr_name)
                for op_desc in grad_op_desc:
                    op_desc._set_attr(device_attr_name, op_device)
            added_descs = _add_descs_to_block(grad_op_desc, local_block)
            grad_op_descs.extend(added_descs)
            grad_to_var.update(op_grad_to_var)

    for i, segment in enumerate(recompute_segments[::-1]):
        gap_ops = ops[segment[1]:max_calculated_op_position]
        max_calculated_op_position = segment[0]
        for op in reversed(gap_ops):
            if op.has_attr("sub_block"):
                raise Exception("Recompute don't support ops with sub_block"
                                "invoke op: %s" %
                                _pretty_op_desc_(op.desc, "with_sub_block"))
            grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
                op.desc, cpt.to_text(no_grad_dict[block.idx]), [])
            # Set device for grad_op according to forward Op
            if op.desc.has_attr(device_attr_name):
                op_device = op.desc.attr(device_attr_name)
                for op_desc in grad_op_desc:
                    op_desc._set_attr(device_attr_name, op_device)
            added_descs = _add_descs_to_block(grad_op_desc, local_block)
            grad_op_descs.extend(added_descs)
            grad_to_var.update(op_grad_to_var)

        ff_ops = ops[segment[0]:segment[1]]
        var_suffix = ".subprog_%d" % i

        for op in ff_ops:
            if op.has_attr("sub_block"):
                raise Exception("Recompute don't support ops with sub_block"
                                "invoke op: %s" %
                                _pretty_op_desc_(op.desc, "with_sub_block"))
            input_and_output_names = []
            input_and_output_names.extend(op.desc.input_arg_names())
            input_and_output_names.extend(op.desc.output_arg_names())
            for name in input_and_output_names:
                if block.var(name).persistable or name in checkpoints_name:
                    continue
                if name in vars_should_be_hold:
                    continue
                if name not in var_name_dict:
                    var_name_dict[name] = name + var_suffix

                    # we should create the rename var in subprog, otherwise its VarType will be BOOL
                    ref_var = block.program.global_block().var(name)
                    block.create_var(
                        name=var_name_dict[name],
                        shape=ref_var.shape,
                        dtype=ref_var.dtype,
                        type=ref_var.type,
                        persistable=ref_var.persistable,
                        stop_gradient=ref_var.stop_gradient)

        # 3.a. add ops in current recompute_segment as forward recomputation ops
        buffer_descs = _add_needed_descs_to_block(ff_ops, buffer_block, block,
                                                  vars_in_memory)
        added_descs = _add_descs_to_block(ff_ops, local_block)

        # 3.b. rename all non-checkpoint variables in recomputation ops
        for key in var_name_dict:
            _rename_arg_(buffer_descs, key, var_name_dict[key])

        # added_descs should be in grad_op_descs because it is backward op desc
        grad_op_descs.extend(buffer_descs)

        # 3.c. add backward ops for all ops in current segment 
        for op_desc in reversed(added_descs):
            grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
                op_desc, cpt.to_text(no_grad_dict[block.idx]), [])

            # Set device for grad_op according to forward Op
            if op_desc.has_attr(device_attr_name):
                op_device = op_desc.attr(device_attr_name)
                for g_op_desc in grad_op_desc:
                    g_op_desc._set_attr(device_attr_name, op_device)

            for key in var_name_dict:
                _rename_arg_(grad_op_desc, key, var_name_dict[key])
            grad_op_descs.extend(grad_op_desc)
            grad_to_var.update(op_grad_to_var)

    # 3.d. add sum op for repetitive_outputs
    grad_op_descs = _addup_repetitive_outputs_(grad_op_descs, block.idx)
    # 4) remove no grad branch as it is in _remove_no_grad_branch_
    grad_op_descs = _remove_no_grad_branch_(grad_op_descs,
                                            no_grad_dict[block.idx])
    added_descs = _add_descs_to_block(grad_op_descs, target_block)
    return program_stat, checkpoints_name, vars_should_be_hold, recompute_segments


def _get_sub_block_path(sub_block,
                        sub_block_op_desc,
                        no_grad_set,
                        op_path_dict,
                        sub_block_target_names=None):
    """
    Get output vars in subblock which will be assigned to parent block.
    It is used to find the grad path in subblock.

    Args:
        sub_block(Block): The sub-block in which to get op path.
        sub_block_op_desc: The op desc of the sub-block op such as 'while', 'conditional_block' and 'recurrent'.
        no_grad_set(set): The set of no grad var name. no_grad_set will be changed.
        op_path_dict(dict): op_path_dict will be changed.
            key(int) block index
            val(list) the op path of block(index)
        sub_block_target_names(set): Target var names of sub-block.
    Return:
        The forward op path of sub-block corresponding to backward op.
    """

    assert sub_block_op_desc.has_attr(
        "sub_block") and sub_block.idx == sub_block_op_desc._block_attr_id(
            "sub_block")
    assert isinstance(sub_block_target_names, (set, type(None)))

    if sub_block_target_names is None:
        sub_block_target_names = sub_block_op_desc.output_arg_names

    # TODO(huihuangzheng): add support for recurrent op.
    if sub_block_op_desc.type in ["conditional_block", "while"]:
        # Step1: get the output vars in sub-block
        sub_outputs = [
            sub_block._var_recursive(var) for var in sub_block_target_names
        ]
        for var in sub_block_target_names:
            for op_desc in sub_block.ops:
                if var in op_desc.output_arg_names:
                    for name in op_desc.input_arg_names:
                        sub_outputs.append(sub_block._var_recursive(name))

        # Step2: find op path of sub-block
        is_while = sub_block_op_desc.type in ["while"]
        sub_block_op_path = _find_op_path_(sub_block, sub_outputs, [],
                                           no_grad_set, op_path_dict, is_while)
        return sub_block_op_path
    return sub_block.ops


def _is_grad_op_(op):
    op_maker = core.op_proto_and_checker_maker
    backward = core.op_proto_and_checker_maker.OpRole.Backward
    if op_maker.kOpRoleVarAttrName() in op.attr_names and \
            int(op.all_attrs()[op_maker.kOpRoleAttrName()]) == int(backward):
        return True
    return False


def _rename_grad_name_(name, grad_order):
    return 'grad/' * grad_order + name


def _append_backward_ops_(block, # 前向 ops 所在的 block 
                          ops,   # op_path 里的 op， 对应的 反向 op 都需要创建
                          target_block, # 同 block 
                          no_grad_dict, # 哪些 var 不需要计算 grad
                          grad_to_var,  # 空字典传入
                          callbacks=None,
                          input_grad_names_set=None, # 一个命名很奇葩的 set， 记录反向图中用作输入的 grad_var
                          op_path_dict=None, # 主要是记录 sub_block 的 op_path
                          distop_context=None):
    """
    Create all grad ops, and insert them into given block

    Args:
        block(Block): the block where forward ops are
        ops(Op): the forward operators whose backward ops need to be added
        target_block(Block): the block which is going to hold new generated grad ops
        no_grad_dict(dict):
            key(int)  block index
            val(set) a set of variable names. These variables have no gradient
        grad_to_var(dict)(output argument):
            key(str): grad variable name
            val(str): corresponding forward variable name
        callbacks(callable object): a callable object used to decorate new generated grad ops
        input_grad_names_set(set): this set is used to store the gradients' name which is
            generated by backward ops, and input_grad_names_set can help to prune the unnecessary
            backward ops.          ##  ————     help to prune the unnecessary backward ops                              
        op_path_dict(dict): op_path_dict will be changed.
            key(int) block index
            val(list) the op path of block(index)
    """
    if callbacks is not None:
        assert (isinstance(callbacks, (list, tuple)))
        for cb in callbacks:
            if not hasattr(cb, '__call__'):
                raise ValueError("'callback' must be a callable object.")

    # 1. grad_op_descs holds created grad_op, and will be appended to target_block
    grad_op_descs = []
    program = block.program
    
    rename_var_map = {} # key:旧名字 -> val:新名字

    # 2. 根据 op_path 的反序创建 grad_op
    # add grad_op_desc by reversed ops
    for op in reversed(ops):
        grad_sub_block_list = []
        # If the op has its own sub-block, deal with the sub-block first
        if op.has_attr("sub_block"):
            sub_block = program.block(op._block_attr_id("sub_block"))
            grad_sub_block = program._create_block()
            grad_sub_block._set_forward_block_idx(sub_block.idx)
            # see follwing comments for why set None here.
            pre_input_grad_names_set = copy.copy(input_grad_names_set)
            input_grad_names_set = None
            sub_block_path = op_path_dict[op._block_attr_id("sub_block")]
            _append_backward_ops_(sub_block, sub_block_path, grad_sub_block,
                                  no_grad_dict, grad_to_var, callbacks,
                                  input_grad_names_set, op_path_dict)
            input_grad_names_set = pre_input_grad_names_set

            program._rollback()
            grad_sub_block_list.append(grad_sub_block.desc)

        # 3. Getting op's corresponding grad_op
        grad_op_desc, op_grad_to_var = core.get_grad_op_desc( # 获得 grad_op_desc，即成功拿到反向 op desc; op_grad_to_var 是一个 map, 记录反向 op 与前向 op var 之间的对应关系 
            op.desc, cpt.to_text(no_grad_dict[block.idx]), grad_sub_block_list)
        if distop_context is not None:
            for op_desc in grad_op_desc:
                assert op_desc.id() not in distop_context.gradopidx2opidx
                distop_context.gradopidx2opidx[op_desc.id()] = op.desc.id()

        # 4. Set device for grad_op according to forward Op
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        if op.desc.has_attr(device_attr_name):
            op_device = op.desc.attr(device_attr_name)
            for op_desc in grad_op_desc: # 4.1 给反向 op 设置对应的 device 属性
                op_desc._set_attr(device_attr_name, op_device)

        # 5. 处理多次调用 fluid.gradients 情况下，可能存在的同名问题，下面的注释真的一言难尽。。。
        # Rename internal gradient variables in multiple backward
        # so that they have different names with previous backward.
        # For example:
        #  y = x * x, 
        # grad = fluid.gradients(fluid.gradients(y, x) + y * y, x)
        # In second-time backward, gradient variable names of partial
        # forward network (y * y) may be have same names with first-time
        # fluid.gradients(y, x).
        # So rename here before _addup_repetitive_outputs_.
        if program._appending_grad_times > 1: # 这个计数，用于记录是第几次调用 gradient，如果不是第一次则进入下面的处理
            for op_desc in grad_op_desc:
                if not _is_grad_op_(op): # 应该是一些特殊情况
                    for name in op_desc.input_arg_names():
                        if name in rename_var_map:
                            op_desc._rename_input(name, rename_var_map[name])
                for name in op_desc.output_arg_names():
                    if "@GRAD" not in name:
                        continue
                    if block.desc.find_var(name.encode("ascii")):
                        new_name = _rename_grad_name_(            # 执行 rename 操作
                            name, program._appending_grad_times)  # 'grad/' * grad_order + name
                        op_desc._rename_output(name, new_name)    # 设置 新 name
                        rename_var_map[name] = new_name # 旧名字 -> 新名字，记录 旧：新 名字的关系

                        if name in op_grad_to_var:  # 如果 旧名字 也在 op_grad_to_var 里面，需要做替换
                            op_grad_to_var[new_name] = op_grad_to_var[name]
                            op_grad_to_var.pop(name)

        # 6. input_grad_names_set 非空的情况处理
        # If input_grad_names_set is not None, extend grad_op_descs only when any input grad in outputs of previous grad ops.
        # But this strategy is not suited for while op for some control flow,
        # for example, for while op, the grads maybe generated in next loop.
        # 6.1 第一次调 gradient 接口，其实不会执行进第一个分支
        # TODO：不是完全能 get 到
        if input_grad_names_set is not None: # 6.2 非第一次调 gradient 接口
            is_append_grad = False
            for op_desc in grad_op_desc:  # 6.2.1 遍历反向 op 
                input_grad_names = [  # 6.2.2 拿到当前反向 op 的所有带有 @GRAD 尾缀的输入 
                    name for name in op_desc.input_arg_names()
                    if name.find(core.grad_var_suffix()) != -1  # 找 grad_var 尾缀 -> @GRAD
                ]
                # some code of gradient ops, like increment, are not very
                # standard, there is no @GRAD in these ops' inputs.
                if len(input_grad_names) == 0:
                    is_append_grad = True # 特殊情况，忽略
                    break
                # 6.2.3 如果 input_grad_names 有一些 在 input_grad_names_set 里面
                if _some_in_set_(input_grad_names, input_grad_names_set):
                    grad_op_descs.append(op_desc)
                    is_append_grad = True
                    for name in op_desc.output_arg_names(): # 又将反向 op 的输出，放入 input_grad_names_set 里面
                        input_grad_names_set.add(name)
            if is_append_grad:
                grad_to_var.update(op_grad_to_var)
        else: # 6.1 第一次调 gradient，直接进这个分支
            grad_op_descs.extend(grad_op_desc) # 处理完一个 反向op, 加入 grad_op_descs list 里
            grad_to_var.update(op_grad_to_var) # 更新总的 grad_to_var 字典
    
    # 以上，完成了整个反向图的构造

    # 7. 梯度聚合 相关
    # sum parameter's gradients' var given multiple var gradient
    # TODO 绘图解释比较好
    grad_op_descs = _addup_repetitive_outputs_(grad_op_descs, block.idx)

    # if all outputs of the grad op are in no_grad_set, then just remove and fill zero
    # if all inputs of the grad op are in no_grad_set, just remove this op
    # 8. 剪枝操作
    grad_op_descs = _remove_no_grad_branch_(grad_op_descs,
                                            no_grad_dict[block.idx])

    # remove some backward ops
    # 9. 找到一些不需要的反向 op, (在剪枝操作的基础上，再根据 input_grad_names_set 的辅助来找一些不需要的反向 op)
    # 三个输入：反向图 op_descs、前向图 ops_path、反向图所有op记录的 grad_names input
    # 主要是处理反向图中的子图没有链接的问题 
    # TODO：不太理解这个背景，找嘉彬问一下
    not_need_ops = _find_not_need_ops(grad_op_descs, ops, input_grad_names_set) # input_grad_names_set 辅助找到一些不需要的反向 op
 
    # 10. 移除不需要的反向 op
    grad_op_descs = [
        op_desc for op_desc in grad_op_descs if op_desc not in not_need_ops
    ]

    # 11. 将反向 op 加入到 目标 block 中
    # append op_desc in grad_op_descs to target_block
    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
    backward = core.op_proto_and_checker_maker.OpRole.Backward
    for op_desc in grad_op_descs:
        new_op_desc = target_block.desc.append_op()
        new_op_desc.copy_from(op_desc) # 将创造的 反向 op_desc 加入
        new_op_desc._set_attr(op_role_attr_name, backward)
        grad_to_var["__current_op_desc__"] = new_op_desc
        if callbacks is not None:
            assert (isinstance(callbacks, (list, tuple)))
            for cb in callbacks:
                cb(block=target_block, context=grad_to_var)


def _is_grad_var_(var_name):
    return core.grad_var_suffix() in var_name


# Find the op who holds the sub_block as its "sub_block" attr
def _find_parent_op_(sub_block):
    sub_block_id = sub_block.idx

    if sub_block_id == 0:
        return None

    program = sub_block.program
    for block_id in six.moves.range(program.num_blocks):
        block_desc = program.block(block_id).desc
        for op_idx in six.moves.range(block_desc.op_size()):
            op = block_desc.op(op_idx)
            if op.has_attr("sub_block") and op._block_attr_id(
                    "sub_block") == sub_block_id:
                return op

    # NOTE(paddle-dev): When optimizer is added in conditional block,
    # sub_block may not be found.
    return None

# start_op_idx 其实是反向 op 的起点
# grad_info_map 空字典传入，运行完该方法后，作为输出（挺关键的一个输出）
# grad_to_var 主要是记录，grad var 对应的 forward var
# 有个问题：感觉每个函数并没有做到权责单一，函数名与实际做的操作不是完全统一
def _append_backward_vars_(block, start_op_idx, grad_to_var, grad_info_map):
    """
    Create new variables required by backward pass.

    Args:
        block(Block): the block where new variables will be created
        start_op_idx(int): Only variables required by ops in block.ops[start_op_idx : ] will be created
        grad_to_var(dict):
            key(str): grad variable name
            val(str): corresponding forward variable name
            In most cases, this dict is generated by _append_backward_ops_()
        grad_info_map(dict)(output argument):
            key(str): forward variable name
            val(tuple): a tuple of (str, Block), str is the corresponding grad name, Block is the block containing grad variable
    """
    ops_to_remove = []
    '''
    NOTE(paddle-dev): while_grad op may hold some inputs which are not found 
    in the parent/forward block, and they are also the outputs of while_grad 
    op. These kinds of inputs are the recursive outputs inside while_grad op. 
    They should be considered as "already created" when scanning the inner 
    ops of while_grad ops.  
    '''
    parent_op = _find_parent_op_(block) # 找到 block 的父节点 op
    parent_op_vars = []
    if parent_op is not None:
        input_args = parent_op.input_arg_names()
        output_args = parent_op.output_arg_names()
        for in_arg in input_args:
            if in_arg in output_args: # 应该是特殊 case，输入居然也在[输出]里面
                parent_op_vars.append(in_arg)

    # 遍历反向 op
    for op_idx in range(start_op_idx, block.desc.op_size()):
        op_desc = block.desc.op(op_idx)
        if op_desc.has_attr("sub_block"):
            sub_block = block.program.block(op_desc._block_attr_id("sub_block"))
            _append_backward_vars_(sub_block, 0, grad_to_var, grad_info_map) # 递归调用
        
        # 构造反向 op 的输入列表，并且都是 grad_var 的
        grad_var_ins = [
            var for var in op_desc.input_arg_names() if _is_grad_var_(var)
        ]
        # 构造反向 op 的输出列表，并且都是 grad_var 的
        grad_var_outs = [
            var for var in op_desc.output_arg_names() if _is_grad_var_(var)
        ]
        # 构造反向 op 的输入列表，不是 empty_var 即可
        inputs = [
            var for var in op_desc.input_arg_names()
            if var != core.empty_var_name()
        ]
        # 构造反向 op 的输出列表，不是 empty_var 即可
        outputs = [
            var for var in op_desc.output_arg_names()
            if var != core.empty_var_name()
        ]

        # If the outputs of grad op is empty, just remove it
        if not outputs:
            ops_to_remove.append(op_idx) # 如果某个反向 op 的输出是空，则当前这个反向 op 可以移除
            continue
        else: # outputs 不为空的分支
            '''
            If the output is not empty and there is any grad input, find 
            whether there is any existing input. If not, just remove it.
            '''
            if grad_var_ins:
                existing_grad_var_ins = [
                    var for var in grad_var_ins
                    if block.desc.has_var_recursive(cpt.to_bytes(var)) or var in # has_var_recursive 意思是递归地去找是否有这个 var 的存在 --- 命名有点怪
                    parent_op_vars
                ]
                if not existing_grad_var_ins: # 如果当前反向 op 的 grad_var_ins 找不到具体来自哪里，则当前反向 op 应该移除
                    '''
                    FIXME(paddle-dev, zengjinle): rnn_memory_helper_grad is used
                    in recurrent op. The input of this op does not even exist in 
                    the program! Therefore, any dependency analysis would not 
                    work to this op! If I do not add the following code, this op
                    would be pruned, and the calculation result would be wrong. 
                    Maybe we should re-design this op later...  
                    '''
                    if op_desc.type() not in ['rnn_memory_helper_grad']: # 非特殊情况，移除
                        ops_to_remove.append(op_idx)
                        continue
        
        # 这下面才是真正开始做 _append_backward_vars_ 的操作
        new_vars = set()
        # create new gradient variables （加个背景，这里是在 block.desc 里面创建 var）
        for grad_var_name in op_desc.output_arg_names(): # 遍历当前 反向 op 的所有输出 
            if block.desc.has_var_recursive(cpt.to_bytes(
                    grad_var_name)) or grad_var_name == core.empty_var_name():
                continue # 如果 block.desc 里面存在反向 op 的输出，或者 是一个 empty_var，则 跳过，看下一个 grad_var_name
            # 否则，在 block.desc 里面创建新 var，并且记录在 new_vars set 里面
            block.desc.var(cpt.to_bytes(grad_var_name))
            new_vars.add(grad_var_name)
            if grad_var_name not in grad_to_var: # 如果 grad_var_name 不在 grad_to_var， 跳过; 值得注意的是，grad_to_var 是之前已经生成好的了，里面记录 【反向var：前向var】的映射关系
                continue # 如果不在 grad_to_var 里面，证明这个  grad_var_name 不是我们的目标 反向var
            
            # 来到这，grad_var_name 满足几个条件: 
            # 1. has_var_recursive 找不到（在 block.desc.var 域内）；2. grad_var_name in grad_to_var（证明是我们的目标 grad_var）

            # 应该是这样： block.desc 里面 var 是独立的，和 op_desc.output_arg_names 无关，也就是说 op_desc 虽然有输入输出，但是这个 var 并不一定存在 block.desc 里面
            # grad_to_var[grad_var_name] 拿到前向var， grad_var_name 对应反向 var 
            grad_info_map[grad_to_var[grad_var_name]] = (grad_var_name, block)

        # infer_shape and infer_type
        op_desc.infer_var_type(block.desc)
        op_desc.infer_shape(block.desc)

        for arg in op_desc.output_arg_names(): # 遍历反向 op 的所有输出
            if arg in new_vars:
                _infer_var_data_type_shape_(arg, block) # 将 grad_var 的 dtype,shape 与 fwd_var 保持一致

    for op_idx in reversed(ops_to_remove): # 移除不需要的反向 op
        block.desc._remove_op(op_idx, op_idx + 1)


def _rename_grad_(block, start_op_idx, grad_to_var, target_grad_map):
    var_map = copy.copy(target_grad_map) # 可能为None(用户没提供)，可能不为None（用户提供了）
    # var_map ==> [out@GRAD : targets_gradient.name]
    for op_idx in range(start_op_idx, block.desc.op_size()): # 遍历反向图所有 op
        op_desc = block.desc.op(op_idx) # 拿到反向 op desc
        for name in op_desc.input_arg_names(): # 遍历反向 op 的所有输入
            if name in var_map: # 
                op_desc._rename_input(name, var_map[name]) # 修改反向op的 输入命名, 用 用户提供的那个替代

        for name in op_desc.output_arg_names(): # 遍历反向 op 的所有输出
            if "@GRAD" not in name:
                continue
            if block.desc.find_var(name.encode("ascii")):
                new_name = unique_name.generate(name) # 在后面加一个序号
                op_desc._rename_output(name, new_name) # 修改反向op的 输出命名
                var_map[name] = new_name # var_map[旧名] = 新名

    for g, ng in six.iteritems(var_map): # key: grad_name/grad_var, val: non_grad_name/var
        if g in grad_to_var:
            grad_to_var[ng] = grad_to_var[g]
            grad_to_var.pop(g)
            # 属于更新操作 grad_to_var[新名] = grad_to_var[旧名]


def _get_stop_gradients_(program):
    no_grad_dict = dict()
    assert isinstance(program, framework.Program)
    for block in program.blocks:
        assert isinstance(block, framework.Block)
        block_no_grad_set = set()
        for var in list(block.vars.values()):
            assert isinstance(var, framework.Variable)
            if var.stop_gradient:
                block_no_grad_set.add(_append_grad_suffix_(var.name))
        no_grad_dict[block.idx] = block_no_grad_set
    return no_grad_dict


def _get_son_parent_block_idx_dict(program, current_block_idx):

    son_parent_block_idx_dict = collections.OrderedDict()
    while current_block_idx >= 0:
        parent_block_idx = program.block(current_block_idx).parent_idx
        son_parent_block_idx_dict[current_block_idx] = parent_block_idx
        current_block_idx = parent_block_idx

    return son_parent_block_idx_dict


def _get_no_grad_set_name(no_grad_set):
    no_grad_set_name = set()
    if no_grad_set is not None:
        if isinstance(no_grad_set, (set, list, tuple)):
            for i, no_grad_var in enumerate(no_grad_set):
                if isinstance(no_grad_var, framework.Variable):
                    no_grad_set_name.add(no_grad_var.name)
                elif isinstance(no_grad_var, six.string_types):
                    no_grad_set_name.add(no_grad_var)
                else:
                    raise TypeError(
                        "The type of no_grad_set's member must be paddle.fluid.Variable or str, but received %s."
                        % (type(no_grad_var)))
        else:
            raise TypeError(
                "The type of no_grad_set should be set or list or tuple, but received {}".
                format(type(no_grad_set)))
    return no_grad_set_name


@framework.static_only
def append_backward(loss,
                    parameter_list=None,
                    no_grad_set=None,
                    callbacks=None,
                    checkpoints=None,
                    distop_context=None):
    """
    :api_attr: Static Graph

    This function appends backward part to main_program.

    A complete neural network training is made up of forward and backward
    propagation. However, when we configure a network, we only need to
    specify its forward part. This function uses the chain rule to automatically
    generate the backward part according to the forward part.

    In most cases, users do not need to invoke this function manually.
    It will be automatically invoked by the optimizer's `minimize` function.

    Parameters:
        loss(Tensor): The loss Tensor of the network.
        parameter_list(list[Tensor|str]|tuple[Tensor|str], optional): List/Tuple of Parameters or Parameter.names
                                           that need to be updated by optimizers.
                                           If it is None, all parameters
                                           will be updated.
                                           Default: None.
        no_grad_set(set[Tensor|str], optional): Set of Tensors or Tensor.names in the :ref:`api_guide_Block_en` 0 whose gradients
                               should be ignored. All Tensors with
                               `stop_gradient=True` from all blocks will
                               be automatically added into this set.
                               If this parameter is not None, the Tensors or Tensor.names in this set will be added to the default set.
                               Default: None.
        callbacks(list[callable object]|tuple[callable object], optional): List/Tuple of callback functions.
                                               The callbacks are used for
                                               doing some custom jobs during
                                               backward part building. All
                                               callable objects in it will
                                               be invoked once each time a
                                               new gradient operator is added
                                               into the program. The callable
                                               object must have two input
                                               parameters: ``block`` and ``context`` .
                                               The ``block`` is the :ref:`api_guide_Block_en` which
                                               the new gradient operator will
                                               be added to. The ``context`` is a
                                               map, whose keys are gradient
                                               Tensor names and values are
                                               corresponding original :ref:`api_guide_tensor_en` .
                                               In addition to this, the ``context``
                                               has another special key-value pair:
                                               the key is string ``__current_op_desc__``
                                               and the value is the op_desc of the
                                               gradient operator who has just
                                               triggered the callable object.
                                               Default: None.

    Returns:
        list of tuple ( :ref:`api_guide_tensor_en` , :ref:`api_guide_tensor_en` ): Pairs of parameter and its corresponding gradients.
        The key is the parameter and the value is gradient Tensor.

    Raises:
        AssertionError: If ``loss`` is not an instance of Tensor.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            paddle.enable_static()

            x = paddle.static.data(name='x', shape=[None, 13], dtype='int64')
            y = paddle.static.data(name='y', shape=[None, 1], dtype='float32')
            x_emb = paddle.static.nn.embedding(x, size=[100, 256])
            y_predict = paddle.static.nn.fc(x=x_emb, size=1, activation=None, name='my_fc')
            loss = F.square_error_cost(input=y_predict, label=y)
            avg_loss = paddle.mean(loss)

            # Get all weights in main_program, not include bias.
            all_weights = [param for param in paddle.static.default_main_program().block(0).all_parameters() if 'w_' in param.name]
            all_weights_name = [w.name for w in all_weights]

            # return all param_grads needed to be updated if parameter_list set default None.
            p_g_list1 = paddle.static.append_backward(loss=avg_loss)
            # output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD), (my_fc.b_0, my_fc.b_0@GRAD)]

            # return the param_grads corresponding to parameter_list that can be list of param (Tensor).
            p_g_list2 = paddle.static.append_backward(loss=avg_loss, parameter_list=all_weights)
            # output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD)]

            # parameter_list can be list of param.name (str).
            p_g_list3 = paddle.static.append_backward(loss=avg_loss, parameter_list=all_weights_name)
            # output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD)]

            # no_grad_set can be set of Tensors that means grad will be cut off from these Tensors.
            p_g_list4 = paddle.static.append_backward(loss=avg_loss, no_grad_set=set([x_emb]))
            # output: [(my_fc.w_0, my_fc.w_0@GRAD), (my_fc.b_0, my_fc.b_0@GRAD)]

            # no_grad_set can be set of Tensor.name when the Tensor is created inside layers and can't be specified explicitly.
            p_g_list5 = paddle.static.append_backward(loss=avg_loss, no_grad_set=set(['my_fc.b_0']))
            # output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD)]

            # return [] because all param_grads are filtered by no_grad_set.
            p_g_list6 = paddle.static.append_backward(loss=avg_loss, parameter_list=all_weights, no_grad_set=set(all_weights))

    """
    check_type(loss, 'loss', framework.Variable,
               'paddle.static.append_backward')

    if loss.op is None:
        # the loss is from a cloned program. Find loss op manually.
        _find_loss_op_(loss)

    loss.op._set_attr(core.op_proto_and_checker_maker.kOpRoleAttrName(),
                      int(core.op_proto_and_checker_maker.OpRole.Forward) |
                      int(core.op_proto_and_checker_maker.OpRole.Loss))

    if callbacks is not None:
        check_type(callbacks, 'callbacks', (list, tuple),
                   'paddle.static.append_backward')

    program = loss.block.program
    root_block = program.block(0)
    current_block_idx = program.current_block_idx
    current_block = program.block(current_block_idx)

    is_in_control_flow = current_block_idx != 0

    # Double grad is not supported in sub-block (control flow)
    if not is_in_control_flow:
        # _appending_grad_times used for double grad
        program._appending_grad_times += 1

    if no_grad_set is None:
        no_grad_set = set()
    else:
        no_grad_set = _get_no_grad_set_name(copy.copy(no_grad_set))
    no_grad_dict = _get_stop_gradients_(program)
    # no_grad_set only contains vars in block 0
    # Todo(liym27): support vars in sub block
    no_grad_dict[0].update(list(map(_append_grad_suffix_, no_grad_set)))

    # Currently it is only to support the optimizer.minimize
    # in a switch branch, which can append_backward in a sub_block.
    # Note: while_loop is in control flow, but it makes no sense to call optimizer in while.
    # Todo: report error when it is in while_loop
    if is_in_control_flow:
        # create grad block if in switch control flow.
        target_grad_block = program._create_block(
            parent_idx=current_block.parent_idx)
        target_grad_block._set_forward_block_idx(current_block_idx)
        # after _create_block, program.current_block changes
    else:
        target_grad_block = root_block

    son_parent_block_idx_dict = _get_son_parent_block_idx_dict(
        program, current_block_idx)

    block_fwd_op_num_dict = {}  # block_id: fwd_op_num
    for idx in son_parent_block_idx_dict:
        block_fwd_op_num_dict[idx] = program.block(idx).desc.op_size()

    grad_to_var = dict()

    op_desc = _create_loss_op_desc_(loss)
    target_grad_block.desc.append_op().copy_from(op_desc)

    for block_idx in son_parent_block_idx_dict:
        block = program.block(block_idx)

        block_no_grad_set = set(
            map(_strip_grad_suffix_, no_grad_dict[block_idx]))

        op_path_dict = dict()
        op_path = _find_op_path_(block, [loss], [], block_no_grad_set,
                                 op_path_dict)

        no_grad_vars = _find_no_grad_vars(block, op_path, [loss],
                                          block_no_grad_set)

        block_no_grad_set.update(no_grad_vars)
        no_grad_dict[block_idx].update(
            list(map(_append_grad_suffix_, block_no_grad_set)))

        input_grad_names_set = None
        # For double backward, input_grad_names is used for filtering
        # some non-used gradients op(s).

        # TODO(liym27): need a better design.
        # not support double grad in control flow sub-block now.
        if not is_in_control_flow:
            if program._appending_grad_times > 1:
                input_grad_names_set = set([_append_grad_suffix_(loss.name)])

        # TODO: support _append_backward_ops_with_checkpoints_ in
        #  sub-block (control flow)
        is_recompute = False
        if checkpoints != None and \
                isinstance(checkpoints, list) and \
                len(checkpoints) > 0:
            is_recompute = True
            program_stat, checkpoint_names, \
                vars_should_be_hold, \
                recompute_segments = \
                _append_backward_ops_with_checkpoints_(
                    root_block,
                    op_path,
                    root_block,
                    no_grad_dict,
                    grad_to_var,
                    checkpoints)
        else:
            _append_backward_ops_(
                block,  # the block where forward ops are in
                op_path,
                target_grad_block,
                no_grad_dict,
                grad_to_var,
                callbacks,
                input_grad_names_set=input_grad_names_set,
                op_path_dict=op_path_dict,
                distop_context=distop_context, )

    grad_info_map = dict()

    # if in control flow, target_grad_block is a created new block which only contains grad ops,
    # so fwd_op_num is set to 0.
    fwd_op_num = block_fwd_op_num_dict[
        current_block_idx] if not is_in_control_flow else 0

    # Because append_backward may be called multiple times,
    # we need rename the internal gradient variables so that they have
    # different names.
    _rename_grad_(target_grad_block, fwd_op_num, grad_to_var, {})

    _append_backward_vars_(target_grad_block, fwd_op_num, grad_to_var,
                           grad_info_map)

    program.current_block_idx = current_block_idx
    program._sync_with_cpp()

    if parameter_list is not None:
        check_type(parameter_list, 'parameter_list', (list, tuple, set),
                   'fluid.backward.append_backward')
        parameters = []
        for i, param in enumerate(parameter_list):
            check_type(param, 'parameter_list[%s]' % i, (framework.Variable,
                                                         six.string_types),
                       'fluid.backward.append_backward')
            if isinstance(param, framework.Variable):
                parameters.append(param.name)
            elif isinstance(param, six.string_types):
                parameters.append(param)
    else:
        params = program.global_block().all_parameters()
        parameters = [param.name for param in params if param.trainable]

    params_and_grads = []
    op_role_var_attr_name = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
    for param in parameters:
        if cpt.to_text(param) not in grad_info_map:
            continue
        grad_info = grad_info_map[param]
        grad_block = grad_info[1]
        if not grad_block.has_var(grad_info[0]):
            raise ValueError("grad block[{0}] did not have grad var {1}".format(
                grad_info[1], grad_info[0]))
        # Get the param var from the global block
        param_var = program.global_block().var(param)
        grad_var = grad_block.var(grad_info[0])
        if not is_in_control_flow:
            if loss.block.has_var(grad_info[0]):
                params_and_grads.append((param_var, grad_var))
            else:
                params_and_grads.append((param_var, None))
        else:
            params_and_grads.append((param_var, grad_var))

    for p, g in params_and_grads:
        if g is None:
            continue
        ops = grad_block.ops if is_in_control_flow else program.global_block(
        ).ops
        for op in reversed(ops):
            assert isinstance(op, framework.Operator)
            if g.name in op.output_arg_names:
                g.op = op
                break

        if g.op is None:
            raise ValueError("Unexpected branch")
        attr_val = [p.name, g.name]
        if g.op.has_attr(op_role_var_attr_name):
            attr_val.extend(g.op.attr(op_role_var_attr_name))
        g.op._set_attr(op_role_var_attr_name, attr_val)

    if is_recompute:
        return params_and_grads, checkpoint_names
    else:
        return params_and_grads


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, collections.Sequence) else [x]


def _is_ancestor_block(ancestor_block, block):
    prog = block.program
    ancestor_idx = ancestor_block.idx
    parent_idx = block.parent_idx

    while parent_idx != -1:
        if parent_idx == ancestor_idx:
            return True
        parent_idx = prog.block(parent_idx).parent_idx

    return False


def _get_output_names(cur_block, targets):
    """
    In `cur_block`, get output names those linked to targets.
    NOTE:
    1. `targets` can be in `cur_block`;
    Usually, `targets` is in `cur_block`. However, considering control flow,
    2. `targets` may be in sub-block but `cur_block` is an ancestor of `targets[0].block`;
    3. `targets` may be in the block which is ancestor of `cur_block`.
    """

    block = targets[0].block if targets else cur_block
    current_output_names = set([out.name for out in targets])

    # 1. If `targets` in cur_block or the ancestral block of `cur_block`
    if block.idx == cur_block.idx or _is_ancestor_block(block, cur_block):
        return current_output_names

    # 2. If `cur_block` is an ancestor of `targets[0].block`, run while loop
    prog = cur_block.program
    while block.idx != cur_block.idx:
        assert block.parent_idx != -1
        parent_block = prog.block(block.parent_idx)

        parent_block_output_names = set()
        for op in reversed(block.ops):
            if _some_in_set_(op.desc.output_arg_names(), current_output_names):
                for name in op.desc.input_arg_names():
                    current_output_names.add(name)
                    if not block.desc.find_var(cpt.to_bytes(name)) \
                            and parent_block.desc.find_var(cpt.to_bytes(name)):
                        parent_block_output_names.add(name)

        block = parent_block
        current_output_names = parent_block_output_names

    return current_output_names


def _find_no_grad_vars(block, op_path, targets, no_grad_set):
    """
    Find the vars which is not used in the program, and
    those vars belong to no_grad_var.
    """
    output_names = _get_output_names(block, targets)
    no_grad_var = []
    for i, op in reversed(list(enumerate(op_path))): # 反序 遍历 op_path
        # If the op has sub_block, it is too complicated to find the correct no_grad_var.
        # sub_block 情况不处理
        # 只处理没有 sub_block 的情况
        if not op.has_attr("sub_block"):
            for out_var in op.desc.output_arg_names(): # 遍历当前 op 的所有输出
                if out_var not in output_names and out_var not in op.desc.input_arg_names() and not block.vars[out_var].stop_gradient: # 后边 stop_gradient 似乎有点奇怪
                    no_grad_var.append(out_var)
        for name in op.desc.input_arg_names(): # 遍历当前 op 的所有输入, name
            if name not in no_grad_set:        # 如果 name 不在 no_grad_set
                output_names.add(name)         # 则将其 name 加入 output_names 里面， 遍历的下一个 op 需要用到
    return set(no_grad_var)


def _find_op_path_(block,
                   targets,
                   inputs,
                   no_grad_set,
                   op_path_dict=None,
                   is_while=False):
    """
    It is used to find the grad path in `block`. # 前后矛盾的声明？？？

    Args:
        block(Block): The block in which to get op path.
        targets(list[Variable]): The target variables.
        inputs(list[Variable]): The input variables.
        no_grad_set(set): The set of no grad var name. no_grad_set will be changed.
        op_path_dict(dict): op_path_dict will be changed. op_path_dict will be changed.
            key(int) block index
            val(list) the op path of block(index)
        is_while(bool): Whether or not `block` is while block
    Return:
        The forward op path of block corresponding to backward op.
    """

    input_names = set([inp.name for inp in inputs])
    output_names = _get_output_names(block, targets)
    if op_path_dict is None:
        op_path_dict = dict() # 前向还是反向？？，应该是前向

    relevant_op_flags = [True] * len(block.ops) # 给 block 里所有 op 打上 flag, True

    # All the inputs of the block are used if inputs is empty,
    if inputs: # 用户提供的 inputs 不为空
        for i, op in enumerate(block.ops): # 前向 遍历当前 block 里面的所有 op
            if _some_in_set_(
                    op.desc.input_arg_names(), #  op.desc.input_arg_names() 当前op所有输入。   input_names 是用户提供的一个输入
                    input_names) and core.has_non_empty_grad_op_maker(op.type): # 并且当前 op 的反向 op 不是 empty_grad_op_maker
                for name in op.desc.output_arg_names(): # 遍历op的输出names 
                    if name not in no_grad_set: # 过滤一下
                        input_names.add(name) # 为何将 op.desc.output_arg_names() 的 name 放入到 input_names 里面？？？
                                              # 因为当前 op 可能有后继 op，即 当前 op 的输出 是 后继 op 的输入，统一都放在 input_names 中

            else:# block 里，有些 op 的输入并不在用户提供的 inputs 里，所以这些 op 是没用使用到的，flags 记为 false
                relevant_op_flags[i] = False # 未使用的 op，flags 改为 False, 即表明这些 op 不需要创建反向 op

    # 将 block.ops 的列表反转
    for i, op in reversed(list(enumerate(block.ops))): # 反序 遍历 当前 block 里面的所有 op
        if op.has_attr("sub_block"):
            sub_block_id = op._block_attr_id("sub_block")
            sub_block = block.program.block(sub_block_id)
            sub_block_target_names = output_names & set(op.output_arg_names)
            sub_block_path = _get_sub_block_path(sub_block, op,               # sub_block_path
                                                 set(), op_path_dict,
                                                 sub_block_target_names)
            op_path_dict[sub_block_id] = sub_block_path

        if _some_in_set_(
                op.desc.output_arg_names(),
                output_names) and core.has_non_empty_grad_op_maker(op.type):
            for name in op.desc.input_arg_names(): # 遍历当前 op 的输入 name
                if name not in no_grad_set: # 过滤一下
                    output_names.add(name)  # 为何将 op.desc.input_arg_names() 的 name 放入 output_names 里面？
                                            # 因为现在是反向遍历，当前 op 的输入，是上一个 op 的输出
        else: # block里，有些op的输出并不在用户提供的outputs里，所以这些op是没用使用到的，flags记为false
            relevant_op_flags[i] = False

    if is_while:
        # If block is while block, dealing with op specifically again.
        # TODO(liym27): Consider special types of ops.
        for i, op in reversed(list(enumerate(block.ops))):
            if relevant_op_flags[i] == False \
                    and _some_in_set_(op.desc.output_arg_names(), output_names):
                relevant_op_flags[i] = True

    op_path = [
        block.ops[i] for i in range(len(block.ops)) if relevant_op_flags[i] # relevant_op_flags 记录相关的前向 op
    ]

    if inputs:
        for op in op_path:
            for name in op.desc.input_arg_names():
                if name not in input_names and block.vars[name].stop_gradient:#  op_path 里，所有op的输入 name, 如果不在 input_names 并且 stop_gradient 的话，加入 no_grad_set
                    no_grad_set.add(name)  

    return op_path


def calc_gradient(targets, inputs, target_gradients=None, no_grad_set=None):
    """
    Backpropagate the gradients of targets to inputs.

    Args:
        targets(Tensor|list[Tensor]|tuple[Tensor]): The target Tensors
        inputs(Tensor|list[Tensor]|tuple[Tensor]): The input Tensors
        target_gradients (Tensor|list[Tensor]|tuple[Tensor], optional): The gradient Tensors
            of targets which has the same shape with targets, If None, ones will
            be created for them.
        no_grad_set(set[Tensor|str], optional): Set of Tensors or Tensor.names in the :ref:`api_guide_Block_en` 0 whose gradients
                               should be ignored. All Tensors with
                               `stop_gradient=True` from all blocks will
                               be automatically added into this set.
                               If this parameter is not None, the Tensors or Tensor.names in this set will be added to the default set.
                               Default: None.

    Return:
        (list[Tensor]): A list of gradients for inputs
        If an input does not affect targets, the corresponding gradient Tensor
        will be None
    """
    targets = _as_list(targets)
    inputs = _as_list(inputs)
    target_gradients = _as_list(target_gradients)
    # 1. 拿到 block
    block = targets[0].block
    # 2. 拿到 prog
    prog = block.program
    # increase appending gradients times
    # 3. 记录调用 gradient 接口的次数，记录是第几次调用 gradient， 主要是用于区分是否是第二次或以上次数，用于决定是否要处理 命名问题
    prog._appending_grad_times += 1
    block_idx = block.idx
    # 4. 预处理 target_gradients
    if not target_gradients:
        target_gradients = [None] * len(targets)
    # 5. 判断 target_graients 合法性
    if len(targets) != len(target_gradients):
        raise ValueError(
            "Should have the same number of target_gradients as targets")
    # 6. 处理 no_grad_set
    if no_grad_set is None:
        no_grad_set = set()
    else:
        no_grad_set = _get_no_grad_set_name(copy.copy(no_grad_set)) # 6.1 做了些预处理
    # 7. 从整个 prog 里找到 stop_gradients 
    # no_grad_dict[block.idx] = block_no_grad_set（var is stop_gradient and add suffix @GRAD）
    no_grad_dict = _get_stop_gradients_(prog)
    # 8. 更新 no_grad_dict
    no_grad_dict[0].update(list(map(_append_grad_suffix_, no_grad_set)))
    # 9. 获得当前 block 里 op 的个数, fwd op 个数
    fwd_op_num = block.desc.op_size()

    # 10. 一个命名特别怪的东西。 这个 input，应该指的是 反向图中的 输入， grad input
    input_grad_names_set = set()

    # 11. 这个命名也很怪，key value 分别是啥呢？
    target_grad_map = {}
    # key: _append_grad_suffix_(target.name) ::: out->out@GRAD， 的确是 target@GRAD
    # val: 用户输入的 target_gradients 的 name 

    # 12. 预处理 target_gradinets，总的来说，就是用户没提供，则 fill_constant 1.0，否则使用用户提供的
    for i, grad in enumerate(target_gradients):
        target = targets[i]  # 拿一个 y / out 出来
        if grad is None:     # 为 None, fill_constant 1.0
            grad_name = _append_grad_suffix_(target.name) # out -> out@GRAD
            target_shape = target.name + '_shape' # out_shape
            block.desc.append_op().copy_from(
                _create_op_desc_("shape", {'Input': [target.name]},
                                 {"Out": [target_shape]}, {}))
            input_grad_names_set.add(target_shape)
            op_desc = _create_op_desc_("fill_constant",
                                       {"ShapeTensor": [target_shape]},
                                       {"Out": [grad_name]}, {
                                           "shape": target.shape,
                                           "value": 1.0,
                                           "dtype": target.dtype,
                                       })

            block.desc.append_op().copy_from(op_desc) # block.desc 里面增加这个 op_desc
            input_grad_names_set.add(grad_name) #  input_grad_names_set 什么鬼命名？input 是只对反向图来说，是输入, 这个放着 out@GRAD
        else: # 非 None 情况
            # 先预处理一些东西
            if target.block.idx != block_idx or target.block.program != prog:
                raise ValueError("all targets must be in the same block")
            if target.shape != grad.shape:
                raise ValueError(
                    "The shapes of target and grad are different: %s %s" % (
                        target.name, grad.name))
            # _append_grad_suffix_(target.name) --- out->out@GRAD
            target_grad_map[_append_grad_suffix_(target.name)] = grad.name  # grad.name ==> target_gradients 的名字
            # target_grad_map[out@GRAD] = grad.name(target_gradients.name), 
            
            input_grad_names_set.add(grad.name) 

            # input_grad_names_set
            # this set is used to store the gradients' name which is generated by backward ops, and input_grad_names_set can help to prune the unnecessary
            # backward ops. —— 这个描述并不完全准确

    # 13. 这段说明其实很难理解
    # 是不是说，用于处理二阶微分的情形？？
    # For double backward, input_grad_names is used for filter
    # some non-used gradients op.  
    # 未使用反向 op ?
    # 当第一次调用 gradient, input_grad_names_set = None
    # 如果不是第一次调用 gradient, 保留 input_grad_names_set， 传入到 _append_backward_ops_ 里面
    if prog._appending_grad_times == 1: 
        input_grad_names_set = None

    for input in inputs:
        if input.block.program != prog:
            raise "input must be in the same program as targets"
    # 14. 创建一个 block 级别的 no_grad_set
    block_no_grad_set = set(map(_strip_grad_suffix_, no_grad_dict[0])) # 这里又去掉 @GRAD 尾缀

    # 15. 创建一个 dict: op_path_dict, 应该是前向执行的 path, op_path_dict 主要是记录 sub_block 的 op path
    op_path_dict = dict()
    # key(int) block index, 
    # val(list) the op path of block(index)

    # 16. _find_op_path_ 应该就是在 block 里，在给定的 targets inputs之间，以及已知 block_no_grad_set 的基础上，
    #     获得 op_path 以及 op_path_dict
    # return：op_path ==== The forward op path of block corresponding to backward op.
    # TODO：这个点绘图解释
    op_path = _find_op_path_(block, targets, inputs, block_no_grad_set,
                             op_path_dict)

    # find no grad var by op_path
    # 17. 再反序遍历一边 op_path，找到不需要计算 grad 的 vars。
    #     注意，上一步的 block_no_grad_set，记录了 no_grad_set
    # TODO：绘图比较好解释
    no_grad_vars = _find_no_grad_vars(block, op_path, targets,
                                      block_no_grad_set)
    # 18. 更新
    block_no_grad_set.update(no_grad_vars)

    # 19. 更新 
    no_grad_dict[0].update(list(map(_append_grad_suffix_, block_no_grad_set))) # 又加上 @GRAD 尾缀
    
    # 这命名很奇葩，直接给一下 key val 岂不更好
    # key(str): grad variable name
    # val(str): corresponding forward variable name
    grad_to_var = dict() # 记录反向图中所有的 【grad_var：fwd_var】映射关系

    grad_info_map = dict()

    # 20. Create all grad ops, and insert them into given block
    # 比较关键的操作， 创建所有反向op，并加入到给定的 block 里面
    # TODO: 绘图比较好解释
    # TODO: 需要去理解里面提到的 sub_block 的概念
    _append_backward_ops_(
        block,
        op_path, # 前向执行的 op_path
        block, 
        no_grad_dict, # 记录着哪些 var 是不需要计算 grad 的
        grad_to_var,  # 空字典传入
        input_grad_names_set=input_grad_names_set, # 一个命名特别奇葩的东西，其实是用于记录 反向图 中 作为输入的一些 grad_var
        op_path_dict=op_path_dict)  # op_path_dict 主要是记录 sub_block 的 op path

    # Because calc_gradient may be called multiple times,
    # we need rename the internal gradient variables so that they have
    # different names.———— TODO 搞清楚 这个 rename 和 _append_backward_ops 里面的 改名 有什么区别？？
    # 在同一个 block 里面，可能执行多次 gradient 接口
    # 
    # 21. 对输入，输出都做了处理
    # fwd_op_num 实际上是作为 start_op_idx 
    _rename_grad_(block, fwd_op_num, grad_to_var, target_grad_map) # target_grad，如果用户没传入，则为空map；如果用户传入，则不为空，target_grad_map[out@GRAD]= 传入的 target_gradients.name 

    # 22. 增加反向 var ?
    # grad_info_map(dict)(output argument):
    #       key(str): forward variable name
    #       val(tuple): a tuple of (str, Block), str is the corresponding grad name, Block is the block containing grad variable
    _append_backward_vars_(block, fwd_op_num, grad_to_var, grad_info_map)
    prog._sync_with_cpp()

    # 23. 返回目标 grad_vars
    grad_vars = []
    for input_var in inputs:
        if input_var.name not in grad_info_map:
            grad_vars.append(None)
        else:
            grad_info = grad_info_map[input_var.name]
            grad_block = grad_info[1]
            grad_var = grad_block.var(grad_info[0])
            grad_vars.append(grad_var)

    if len(grad_vars) == 1:
        return grad_vars[0]
    else:
        return grad_vars


@framework.static_only
def gradients(targets, inputs, target_gradients=None, no_grad_set=None):
    """
    :api_attr: Static Graph

    Backpropagate the gradients of targets to inputs.

    Args:
        targets (Tensor|list[Tensor]|tuple[Tensor]): The target Tensors.
        inputs (Tensor|list[Tensor]|tuple[Tensor]): The input Tensors.
        target_gradients (Tensor|list[Tensor]|tuple[Tensor], optional): The gradient Tensor
            of targets which has the same shape with targets, If None, ones will
            be created for them.
        no_grad_set (set[Tensor|str], optional): Set of Tensors or Tensor.names in the :ref:`api_guide_Block_en` 0 whose gradients
            should be ignored. All Tensors with ``stop_gradient=True`` from all blocks will
            be automatically added into this set. If this parameter is not None, the Tensors or Tensor.names
            in this set will be added to the default set. Default: None.

    Return:
        (list[Tensor]): A list of gradients for inputs
        If an input does not affect targets, the corresponding gradient Tensor
        will be None.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            paddle.enable_static()

            x = paddle.static.data(name='x', shape=[None, 2, 8, 8], dtype='float32')
            x.stop_gradient=False
            y = paddle.static.nn.conv2d(x, 4, 1, bias_attr=False)
            y = F.relu(y)
            z = paddle.static.gradients([y], x)
            print(z) # [var x@GRAD : fluid.VarType.LOD_TENSOR.shape(-1L, 2L, 8L, 8L).astype(VarType.FP32)]
    """
    check_type(targets, 'targets', (framework.Variable, list, tuple),
               'paddle.static.gradients')
    check_type(inputs, 'inputs', (framework.Variable, list, tuple),
               'paddle.static.gradients')
    check_type(target_gradients, 'target_gradients', (
        framework.Variable, list, tuple, type(None)), 'paddle.static.gradients')

    outs = calc_gradient(targets, inputs, target_gradients, no_grad_set)
    return _as_list(outs)


@framework.static_only
def gradients_with_optimizer(program, optimizer, inputs=None, outputs=None):
    """
    :api_attr: Static Graph

    Backpropagate the gradients of the program and apply the gradients with the given optimizer.

    Args:
        program (Program): The input program.
        optimizer (Optimizer): The optimizer to apply the gradients.
        inputs (Tensor|list[Tensor]|tuple[Tensor], optional): The input Tensors.
            If None, the inputs will be created from the input variables in the given program. Default:None.
        outputs (Tensor|list[Tensor]|tuple[Tensor], optional): The output Tensors.
            If None, the outputs will be created from the output variables in the given program. Default: None.

    Return:
        tuple: tuple (optimize_ops, params_grads), A list of operators appended
            by gradients_with_optimizer and a list of (param, grad) variable pairs, param is
            ``Parameter``, grad is the gradient value corresponding to the parameter.
            The returned tuple can be passed to ``fetch_list`` in ``Executor.run()`` to
            indicate program pruning. If so, the program will be pruned by ``feed`` and
            ``fetch_list`` before run, see details in ``Executor``.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.static as static

            paddle.enable_static()

            img = static.data(name='image', shape=[None, 784])
            pred = static.nn.fc(x=img, size=10, activation='relu')
            loss = paddle.mean(pred)
            opt_ops, pram_grads = paddle.fluid.backward.gradients_with_optimizer(static.default_main_program(), opt)
            print(opt_ops)

    """
    check_type(program, 'program', paddle.fluid.Program,
               'paddle.static.gradients_with_optimizer')
    check_type(optimizer, 'optimizer', paddle.optimizer.Optimizer,
               'paddle.static.gradients_with_optimizer')

    if inputs is None or outputs is None:
        in_set = set()
        out_set = set()
        for block in program.blocks:
            for op in block.ops:
                for name in op.input_arg_names:
                    in_set.add(block.vars[name])
                for name in op.output_arg_names:
                    out_set.add(block.vars[name])
        if inputs is None:
            inputs = list(in_set.difference(out_set))
        if outputs is None:
            outputs = list(out_set.difference(in_set))

    grads = gradients(outputs, inputs)

    with program_guard(program, None):
        pram_grads = [(pram, grad) for pram, grad in zip(inputs, grads)
                      if isinstance(pram, paddle.fluid.framework.Parameter) and
                      grad is not None]

        optimize_ops = optimizer.apply_gradients(pram_grads)

    return optimize_ops, pram_grads
