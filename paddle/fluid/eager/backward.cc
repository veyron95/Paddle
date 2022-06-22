// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/eager/backward.h"

#include <deque>

#include "glog/logging.h"
#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/utils/hook_utils.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/grad_tensor_holder.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"

namespace egr {

/*
 * GeneralGrad is Helpper class to implement custom grad operation between
 * outputs and inputs.
 *
 * **/
class GeneralGrad {
 public:
  static GeneralGrad& Instance() { return *general_grad_; }

  // Get inputs's / no_grad_vars's GradNodes and InputMeta Info
  void GetTargetNodesInfo(
      const std::vector<paddle::experimental::Tensor>& inputs,
      bool is_no_grad_vars) {
    std::string msg = is_no_grad_vars ? "no_grad_vars" : "inputs";
    VLOG(6) << "Running in GetTargetNodesInfo.";
    if (!inputs.empty()) {
      VLOG(6) << msg << " are not empty.";
      size_t num_inputs = inputs.size();
      for (size_t i = 0; i < num_inputs; i++) {
        AutogradMeta* auto_grad_meta =
            EagerUtils::unsafe_autograd_meta(inputs[i]);
        auto* target_node = auto_grad_meta->GetMutableGradNode().get();

        if (orig_to_copied_node_mapping_.count(target_node)) {
          target_node = orig_to_copied_node_mapping_[target_node].get();
        } else {
          VLOG(6) << "Unable to find target node in "
                     "orig_to_copied_node_mapping_, likely indicating an "
                     "unused input";
        }

        PADDLE_ENFORCE_NOT_NULL(target_node,
                                paddle::platform::errors::Fatal(
                                    "There is no grad op for %s:[%d] or it's"
                                    "stop_gradient=True.",
                                    msg,
                                    i));
        if (is_no_grad_vars) {
          (no_grad_var_nodes_inputmeta_map_)[target_node] = auto_grad_meta;
        } else {  // normal input
          (input_target_nodes_inputmeta_map_)[target_node] = auto_grad_meta;
        }
      }
    }
  }

  // Fetch grad for tensor in target_node passing by, not endding_nodes
  void RegisterFetchGradHookForTensor(
      const std::vector<paddle::experimental::Tensor>& inputs) {
    VLOG(6) << "Running in RegisterFetchGradHookForTensor.";
    if (!inputs.empty()) {
      size_t num_inputs = inputs.size();
      for (size_t i = 0; i < num_inputs; i++) {
        AutogradMeta* auto_grad_meta =
            EagerUtils::unsafe_autograd_meta(inputs[i]);
        auto* target_node = auto_grad_meta->GetMutableGradNode().get();

        if (orig_to_copied_node_mapping_.count(target_node)) {
          target_node = orig_to_copied_node_mapping_[target_node].get();
          if (copied_node_to_endding_node_mapping_.count(target_node)) {
            VLOG(6) << "No need to call FetchGradForTensor for endding_nodes";
            continue;
          }
        } else {
          VLOG(6) << "Unable to find target node in "
                     "orig_to_copied_node_mapping_, likely indicating an "
                     "unused input";
        }

        PADDLE_ENFORCE_NOT_NULL(
            target_node,
            paddle::platform::errors::Fatal(
                "There is no grad op for inputs:[%d] or it's"
                "stop_gradient=True.",
                i));

        if (!IsEnddingNodes(target_node)) {
          // Fetch grad for tensor in target_node passing by.
          auto fetched_grad =
              egr_utils_api::FetchGradForTensor(inputs[i], target_node);
          results_map_[target_node] = fetched_grad;
        }
      }
    }
  }

  // Purify potential_startup_nodes_, remove nodes those are the same as
  // input_target_nodes
  void PurifyPotentialStartUpNodes() {
    VLOG(6) << "Running in PurifyPotentialStartUpNodes";
    if (input_target_nodes_inputmeta_map_.empty()) return;
    std::unordered_set<GradNodeBase*> potential_startup_nodes_to_be_erased;
    for (auto startup_op : potential_startup_nodes_) {
      auto iter = input_target_nodes_inputmeta_map_.find(startup_op);
      if (iter != input_target_nodes_inputmeta_map_.end()) {
        potential_startup_nodes_to_be_erased.emplace(iter->first);
      }
    }
    if (!potential_startup_nodes_to_be_erased.empty()) {
      for (auto nodes : potential_startup_nodes_to_be_erased) {
        potential_startup_nodes_.erase(nodes);
      }
    }
  }

  // Remove some nodes those doesn't need to be
  // stored in potential_startup_nodes_
  void UpdateGraphInfo() {
    std::unordered_set<GradNodeBase*> startup_ops;
    VLOG(6) << "Running in UpdateGraphInfo";
    std::deque<GradNodeBase*> queue;
    for (auto& target_nodes_inputmeta_pair :
         input_target_nodes_inputmeta_map_) {
      queue.push_back(target_nodes_inputmeta_pair.first);
      needed_nodes_.emplace(target_nodes_inputmeta_pair.first);
    }
    std::unordered_set<GradNodeBase*> visited;
    std::unordered_set<GradNodeBase*> input_target_nodes_passing_by;
    while (!queue.empty()) {
      auto* target_node = queue.front();
      queue.pop_front();
      if (visited.count(target_node)) {
        continue;
      }
      visited.insert(target_node);
      if (!(depending_nodes_)[target_node].empty()) {
        auto precedding_nodes = (depending_nodes_)[target_node];
        for (auto pre_nodes : precedding_nodes) {
          queue.push_back(pre_nodes);
          needed_nodes_.emplace(pre_nodes);
          if (IsInputTargetNodes(pre_nodes)) {
            input_target_nodes_passing_by.emplace(pre_nodes);
          }
        }
      } else {  // startup_ops have no precedding nodes
        VLOG(6) << "Emplace startup_ops";
        startup_ops.emplace(target_node);
        needed_nodes_.emplace(target_node);
      }
    }

    for (auto& target_nodes_inputmeta_pair :
         input_target_nodes_inputmeta_map_) {
      if (!input_target_nodes_passing_by.count(
              target_nodes_inputmeta_pair.first)) {
        endding_nodes_.emplace(target_nodes_inputmeta_pair.first);
      }
    }

    // Purify potential_startup_nodes_ again, remove some
    // potential startup_nodes that unreach to input target nodes
    if (!startup_ops.empty()) {
      std::unordered_set<GradNodeBase*> potential_startup_nodes_to_be_erased;
      for (auto node : potential_startup_nodes_) {
        if (startup_ops.count(node) == 0) {
          VLOG(6) << "Set up potential_startup_nodes_to_be_erased";
          potential_startup_nodes_to_be_erased.emplace(node);
        }
      }
      if (!potential_startup_nodes_to_be_erased.empty()) {
        for (auto node : potential_startup_nodes_to_be_erased) {
          VLOG(6) << "Erase nodes in potential_startup_nodes_to_be_erased";
          potential_startup_nodes_.erase(node);
        }
      }
    }
  }

  // Get Graph Info Betweent input target GradNode and outputs，
  // record depending_nodes_, potential_startup_nodes_
  void GetGraphInfoBetweenTargets(const std::deque<GradNodeBase*>& init_queue) {
    VLOG(6) << "Runing In GetGraphInfoBetweenTargets";

    // Calculate in_degree for each node
    std::unordered_map<GradNodeBase*, int> node_in_degree_map;

    // Copy nodes
    std::deque<GradNodeBase*> queue = init_queue;
    std::unordered_set<GradNodeBase*> visited;

    // Visit each node exactly once in any order
    while (!queue.empty()) {
      GradNodeBase* node = queue.front();
      queue.pop_front();

      if (visited.count(node)) {
        continue;
      }
      visited.insert(node);

      // Find and append next nodes
      const paddle::small_vector<std::vector<GradSlotMeta>,
                                 kSlotSmallVectorSize>& metas =
          node->OutputMeta();
      for (const auto& meta_list : metas) {
        for (const GradSlotMeta& meta : meta_list) {
          const auto& edge = meta.GetEdge();
          GradNodeBase* next_node = edge.GetMutableGradNode().get();

          // Next node could be nullptr if it is leaf tensor with no
          // AccumulationNode attached
          // Or it could also originated from dispensable inputs
          if (!next_node) continue;

          // Update in_degree
          if (!node_in_degree_map.count(next_node)) {
            node_in_degree_map[next_node] = 0;
          }
          node_in_degree_map[next_node]++;

          // Record depending relationship
          (depending_nodes_)[next_node].emplace(node);
          queue.push_back(next_node);
        }
      }
    }
  }

  void ModifyReadyQueue(std::deque<GradNodeBase*>* queue) {
    std::deque<GradNodeBase*> tmp_queue;
    for (auto nodes : potential_startup_nodes_) {
      tmp_queue.push_back(nodes);
    }
    tmp_queue.swap(*queue);
  }

  // Set result for input target grad_var when potential_startup_nodes_ is empty
  void SetResultForInputTargetVar(
      const std::unordered_map<GradNodeBase*,
                               std::unique_ptr<GradTensorHolder>>&
          node_input_buffers_dict) {
    if (potential_startup_nodes_.size() == 0) {
      for (auto input_target_node : *GetInputTargetNodesInputMetaMap()) {
        // out rank_info of forward op
        auto rank_info = input_target_node.second->OutRankInfo();
        auto iter = node_input_buffers_dict.find(input_target_node.first);
        if (iter != node_input_buffers_dict.end()) {
          auto& target_result =
              (iter->second)->Buffers()[rank_info.first][rank_info.second];
          // save the target result
          results_map_[input_target_node.first] =
              std::make_shared<paddle::experimental::Tensor>(target_result);
        }
      }
    }
  }

  void SetResultForEnddingNodes(
      paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                           kSlotSmallVectorSize> grad_output,
      GradNodeBase* node) {
    if (IsEnddingNodes(node)) {
      VLOG(6) << "Set result for endding_nodes_ with grad_output_tensors";
      results_map_[node] =
          std::make_shared<paddle::experimental::Tensor>(grad_output[0][0]);
    }
  }

  std::vector<paddle::experimental::Tensor> GetResults(
      const std::vector<paddle::experimental::Tensor>& inputs,
      bool allow_unused,
      bool create_graph) {
    VLOG(6) << "Running in GetResults";
    if (inputs.empty()) return {};

    std::vector<paddle::experimental::Tensor> results;
    results.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
      auto& input = inputs[i];
      AutogradMeta* auto_grad_meta = EagerUtils::unsafe_autograd_meta(input);

      auto* target_node = auto_grad_meta->GetMutableGradNode().get();
      if (orig_to_copied_node_mapping_.count(target_node)) {
        target_node = orig_to_copied_node_mapping_[target_node].get();
        if (copied_node_to_endding_node_mapping_.count(target_node)) {
          target_node = copied_node_to_endding_node_mapping_[target_node];
        }
      } else {
        VLOG(6) << "Unable to find target node in "
                   "orig_to_copied_node_mapping_, likely indicating an unused "
                   "input";
      }
      auto iter = results_map_.find(target_node);
      if (iter != results_map_.end()) {
        // set StopGradient = !create_graph
        AutogradMeta* tensor_auto_grad_meta =
            EagerUtils::autograd_meta(iter->second.get());
        tensor_auto_grad_meta->SetStopGradient(!create_graph);
        results.emplace_back(*(iter->second.get()));
      } else {
        PADDLE_ENFORCE_EQ(allow_unused,
                          true,
                          paddle::platform::errors::InvalidArgument(
                              "The %d-th input does not appear in the backward "
                              "graph. Please check the input tensor or set "
                              "allow_unused=True to get None result.",
                              i));
        results.emplace_back();
      }
    }
    Clear();
    return results;
  }

  void SetNodeToAccumulationNode(GradNodeBase* node) {
    if (dynamic_cast<egr::GradNodeAccumulation*>(node)) return;
    if (!(depending_nodes_)[node].empty()) {
      auto precedding_nodes = (depending_nodes_)[node];
      for (auto pre_nodes : precedding_nodes) {
        paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
            pre_nodes_edges = pre_nodes->MutableOutputMeta();
        for (size_t i = 0; i < pre_nodes_edges.size(); i++) {
          for (size_t j = 0; j < pre_nodes_edges[i].size(); j++) {
            auto edge_ = pre_nodes_edges[i][j].GetEdge();
            if (edge_.GetGradNode() == node) {
              auto autograd_meta = egr::AutogradMeta(edge_);
              Edge& pre_node_edge = pre_nodes_edges[i][j].GetMutableEdge();

              if (grad_node_trans_mapping_.count(node)) {
                pre_node_edge.SetGradNode(grad_node_trans_mapping_[node]);
              } else {
                std::shared_ptr<GradNodeBase> shared_grad_node_accumulation =
                    std::make_shared<egr::GradNodeAccumulation>(&autograd_meta);
                pre_node_edge.SetGradNode(shared_grad_node_accumulation);
                grad_node_trans_mapping_[node] = shared_grad_node_accumulation;
              }

              auto* grad_node = pre_node_edge.GetGradNode();
              needed_nodes_.emplace(grad_node);
              endding_nodes_.emplace(grad_node);
              copied_node_to_endding_node_mapping_[node] = grad_node;
              input_target_nodes_inputmeta_map_[grad_node] =
                  input_target_nodes_inputmeta_map_[node];

              VLOG(6)
                  << node->name() << " (addr:" << node
                  << ") has been transformed to GradNodeAccumulation (addr: "
                  << grad_node << ")";

              // Copy Hook func
              if (node->GradientHooksRegistered()) {
                VLOG(6) << "Copy hook func from node: " << node->name()
                        << " (addr: " << node
                        << ") to GradNodeAccumulation (addr: " << grad_node
                        << ")";
                grad_node->SetGradientHookFuntions(
                    node->GetGradientHookFuntions());
              }
            }
          }
        }
      }
    }
  }

  void GenSubGraph(std::deque<GradNodeBase*>* queue) {
    std::deque<GradNodeBase*> queue_ = *queue;
    std::unordered_set<GradNodeBase*> visited;

    while (!queue_.empty()) {
      GradNodeBase* node = queue_.front();
      queue_.pop_front();

      if (visited.count(node)) {
        continue;
      }
      visited.insert(node);

      if (IsInputTargetNodes(node)) {
        if (IsEnddingNodes(node)) {
          SetNodeToAccumulationNode(node);
          continue;
        }
      }

      paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
          meta = node->MutableOutputMeta();
      for (size_t i = 0; i < meta.size(); i++) {
        for (size_t j = 0; j < meta[i].size(); j++) {
          Edge& edge = meta[i][j].GetMutableEdge();
          std::shared_ptr<GradNodeBase> next_node = edge.GetMutableGradNode();
          if (!next_node) continue;

          if (IsInputTargetNodes(node)) {
            if (meta.size() != 1 && !IsNeededNodes(next_node.get()) &&
                IsNeededNodes(node) && !IsEnddingNodes(node)) {
              VLOG(6) << "Clear meta[" << i << "][" << j
                      << "] for node: " << node->name() << " (addr: " << node
                      << ")";
              meta[i][j].Clear();
            }
          }
          // Update BFS queue
          queue_.push_back(next_node.get());
        }
      }
    }
  }

  void PreparedForGeneralGrad(
      const std::vector<paddle::experimental::Tensor>& inputs,
      const std::vector<paddle::experimental::Tensor>& no_grad_vars,
      std::deque<GradNodeBase*>* queue,
      const std::unordered_map<GradNodeBase*,
                               std::unique_ptr<GradTensorHolder>>&
          node_input_buffers_dict) {
    // Get no_grad_vars's GradNodes and InputMeta Info
    GetTargetNodesInfo(no_grad_vars, true /* is_no_grad_vars */);
    // Get inputs's GradNodes and InputMeta Info
    GetTargetNodesInfo(inputs, false /* is_no_grad_vars */);
    // Purify potentialstartup_ops, remove those nodes that are the same as
    // input_target_nodes
    PurifyPotentialStartUpNodes();
    // Get Graph Info Betweent input target gradnode and outputs
    // Record the depending_nodes_ and potential_startup_nodes_
    GetGraphInfoBetweenTargets(*queue);
    // Update Graph Info, remove some nodes in
    // potential_startup_nodes_
    UpdateGraphInfo();
    // Reset queue. Queue is empty only when
    // 1.input equals to output. 2.input can not reach to output.
    ModifyReadyQueue(queue);
    // Set result for input target grad_var when queue is empty
    if (queue->empty()) {
      SetResultForInputTargetVar(node_input_buffers_dict);
    } else {
      GenSubGraph(queue);
      RegisterFetchGradHookForTensor(inputs);
    }
  }

  bool IsNeededNodes(GradNodeBase* node) { return needed_nodes_.count(node); }

  bool IsEnddingNodes(GradNodeBase* node) { return endding_nodes_.count(node); }

  bool IsInputTargetNodes(GradNodeBase* node) {
    auto iter = input_target_nodes_inputmeta_map_.find(node);
    if (iter != input_target_nodes_inputmeta_map_.end()) {
      return true;
    }
    return false;
  }

  std::unordered_map<GradNodeBase*, AutogradMeta*>*
  GetNoGradVarNodesInputMetaMap() {
    return &no_grad_var_nodes_inputmeta_map_;
  }

  std::unordered_map<GradNodeBase*, AutogradMeta*>*
  GetInputTargetNodesInputMetaMap() {
    return &input_target_nodes_inputmeta_map_;
  }

  std::unordered_set<GradNodeBase*>* GetPotentialStartupNodes() {
    return &potential_startup_nodes_;
  }

  void Clear() {
    no_grad_var_nodes_inputmeta_map_.clear();
    input_target_nodes_inputmeta_map_.clear();
    potential_startup_nodes_.clear();
    depending_nodes_.clear();
    results_map_.clear();
    copied_grad_nodes_.clear();
    orig_to_copied_node_mapping_.clear();
    copied_node_to_endding_node_mapping_.clear();
    needed_nodes_.clear();
    grad_node_trans_mapping_.clear();
    endding_nodes_.clear();
  }

  GradNodeBase* CopyGradNode(const std::shared_ptr<GradNodeBase>& orig_node) {
    if (orig_to_copied_node_mapping_.count(orig_node.get())) {
      return orig_to_copied_node_mapping_[orig_node.get()].get();
    }
    std::shared_ptr<GradNodeBase> copied_node = orig_node->Copy();

    // Save node and update mapping
    orig_to_copied_node_mapping_[orig_node.get()] = copied_node;
    copied_grad_nodes_.push_back(copied_node);

    return copied_node.get();
  }

  void ReconstructBackwardGraph(
      const std::deque<GradNodeBase*>& orig_init_queue) {
    std::deque<GradNodeBase*> queue = orig_init_queue;
    std::unordered_set<GradNodeBase*> visited;

    // BFS and recursively copy the grad nodes
    while (!queue.empty()) {
      GradNodeBase* orig_node = queue.front();
      queue.pop_front();
      if (visited.count(orig_node)) {
        continue;
      }
      visited.insert(orig_node);

      PADDLE_ENFORCE(
          orig_to_copied_node_mapping_.count(orig_node),
          paddle::platform::errors::Fatal(
              "Cannot reconstruct backward graph,"
              "unable to find copied target for certain grad node."));
      GradNodeBase* copied_node = orig_to_copied_node_mapping_[orig_node].get();

      const paddle::small_vector<std::vector<GradSlotMeta>,
                                 kSlotSmallVectorSize>& orig_meta =
          orig_node->OutputMeta();
      paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
          copied_edges = copied_node->MutableOutputMeta();
      for (size_t i = 0; i < orig_meta.size(); i++) {
        for (size_t j = 0; j < orig_meta[i].size(); j++) {
          const Edge& orig_edge = orig_meta[i][j].GetEdge();
          Edge& copied_edge = copied_edges[i][j].GetMutableEdge();

          std::shared_ptr<GradNodeBase> orig_next_node =
              orig_edge.GetMutableGradNode();
          if (!orig_next_node) continue;

          // Copy Next Node
          std::shared_ptr<GradNodeBase> copied_next_node;
          if (orig_to_copied_node_mapping_.count(orig_next_node.get())) {
            copied_next_node =
                orig_to_copied_node_mapping_[orig_next_node.get()];

          } else {
            copied_next_node = orig_next_node->Copy();
            orig_to_copied_node_mapping_[orig_next_node.get()] =
                copied_next_node;
            copied_grad_nodes_.push_back(copied_next_node);
          }

          // Update Edge's Grad Node
          copied_edge.SetGradNode(copied_next_node);

          // Update BFS queue
          queue.push_back(orig_next_node.get());
        }
      }
    }
  }

 private:
  GeneralGrad() = default;
  static GeneralGrad* general_grad_;
  // no_grad_vars's GradNode and GradNode's InputMeta.
  std::unordered_map<GradNodeBase*, AutogradMeta* /* InputMeta */>
      no_grad_var_nodes_inputmeta_map_;
  // inputs's GradNode and GradNode's InputMeta.
  std::unordered_map<GradNodeBase*, AutogradMeta* /* InputMeta */>
      input_target_nodes_inputmeta_map_;
  // Record all the potential startup_nodes, will be changed.
  std::unordered_set<GradNodeBase*> potential_startup_nodes_;
  std::unordered_map<GradNodeBase* /* next node */,
                     std::unordered_set<GradNodeBase*> /* pre nodes */>
      depending_nodes_;
  std::unordered_map<GradNodeBase*,
                     std::shared_ptr<paddle::experimental::Tensor>>
      results_map_;

  std::vector<std::shared_ptr<GradNodeBase>> copied_grad_nodes_;
  std::unordered_map<GradNodeBase*, std::shared_ptr<GradNodeBase>>
      orig_to_copied_node_mapping_;
  std::unordered_map<GradNodeBase*, GradNodeBase*>
      copied_node_to_endding_node_mapping_;
  std::unordered_set<GradNodeBase*> needed_nodes_;
  // Record which grad_node has been transformed to AccumulationNode
  std::unordered_map<GradNodeBase*, std::shared_ptr<GradNodeBase>>
      grad_node_trans_mapping_;
  std::unordered_set<GradNodeBase*> endding_nodes_;

  DISABLE_COPY_AND_ASSIGN(GeneralGrad);
};

std::unordered_map<GradNodeBase*, int> getInDegreeMap(
    const std::deque<GradNodeBase*>& init_queue) {
  // Calculate in_degree for each node
  // We can completely remove this pass, if in_degree were set during forward
  // pass
  std::unordered_map<GradNodeBase*, int> node_in_degree_map;

  // Copy nodes
  std::deque<GradNodeBase*> queue = init_queue;
  std::unordered_set<GradNodeBase*> visited;

  // Visit each node exactly once in any order
  while (!queue.empty()) {
    GradNodeBase* node = queue.front();
    queue.pop_front();

    if (visited.count(node)) {
      continue;
    }
    visited.insert(node);

    PADDLE_ENFORCE_NOT_NULL(
        node,
        paddle::platform::errors::Fatal(
            "We got null node when we traverse the backward graph, and this "
            "should not happened please check your code and contact us."));
    // Find and append next nodes
    const paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
        metas = node->OutputMeta();
    for (const auto& meta_list : metas) {
      for (const GradSlotMeta& meta : meta_list) {
        const auto& edge = meta.GetEdge();
        GradNodeBase* next_node = edge.GetMutableGradNode().get();
        // Next node could be nullptr if it is leaf tensor with no
        // AccumulationNode attached
        // Or it could also originated from dispensable inputs
        if (!next_node) continue;

        // Update in_degree
        if (!node_in_degree_map.count(next_node))
          node_in_degree_map[next_node] = 0;
        node_in_degree_map[next_node]++;
        queue.push_back(next_node);
      }
    }
  }

  return node_in_degree_map;
}

// Enforce GradNode has TensorWrappers as Input
void EnforceGradNodeHasInput(GradNodeBase* node) {
  VLOG(6) << "Running in EnforceGradNodeHasInput";
  PADDLE_ENFORCE_NE(
      node->IsTensorWrappersCleared(),
      true,
      paddle::platform::errors::Fatal(
          "The TensorWrappers of %s do not exist. This may be because:\n"
          "You calculate backward twice for the same subgraph without "
          "setting retain_graph=True. Please set retain_graph=True in the "
          "first backward/grad call.\n",
          node->name()));
}

void DuplicateCheck(const std::vector<paddle::experimental::Tensor>& inputs,
                    bool is_input) {
  std::unordered_set<AutogradMeta*> visisted_ins;
  std::string msg = is_input ? "inputs" : "outputs";
  for (auto in : inputs) {
    AutogradMeta* auto_grad_meta = EagerUtils::unsafe_autograd_meta(in);
    PADDLE_ENFORCE_EQ(
        visisted_ins.count(auto_grad_meta),
        0,
        paddle::platform::errors::AlreadyExists(
            "%s contain duplicate tensor %s, please check %s carefully.",
            msg,
            in.name(),
            msg));
    visisted_ins.insert(auto_grad_meta);
  }
}

GeneralGrad* GeneralGrad::general_grad_ = new GeneralGrad();

std::vector<paddle::experimental::Tensor> RunBackward(
    const std::vector<paddle::experimental::Tensor>& tensors,  // output
    const std::vector<paddle::experimental::Tensor>& grad_tensors,
    bool retain_graph,
    bool create_graph = false,
    const std::vector<paddle::experimental::Tensor>& inputs = {},
    bool allow_unused = false,
    const std::vector<paddle::experimental::Tensor>& no_grad_vars = {}) {
  VLOG(3) << "Start Backward";

  // *Gradient Hook should happen at node-level
  // *Inplace version check should perform at node-level
  // *Cross-batch accumulation happens at forward pass

  // GeneralGrad
  bool is_general_grad = !inputs.empty();
  if (is_general_grad) GeneralGrad::Instance().Clear();

  /* --- Initialization --- */
  // 1. Init queue with starting nodes
  // 2. Prepare initial input buffers
  std::deque<GradNodeBase*> queue;
  std::deque<GradNodeBase*> orig_queue;
  std::unordered_map<GradNodeBase*, std::unique_ptr<GradTensorHolder>>
      node_input_buffers_dict;
  for (size_t i = 0; i < tensors.size(); i++) {
    const paddle::experimental::Tensor& tensor = tensors[i];

    AutogradMeta* auto_grad_meta = EagerUtils::nullable_autograd_meta(tensor);
    if (auto_grad_meta == nullptr) {
      VLOG(3) << "Skip auto grad since there is no grad op for var or loss is "
                 "stop_gradient=True: "
              << tensor.name();
      continue;
    }
    // Get grad input info from target tensors
    auto input_info = auto_grad_meta->OutRankInfo();

    VLOG(2) << "Out Rank of Tensor is slot: " << input_info.first
            << ", rank: " << input_info.second;
    // Get target GradNodeBase from target tensors
    auto shared_grad_node = auto_grad_meta->GetMutableGradNode();

    if (shared_grad_node == nullptr || shared_grad_node.get() == nullptr ||
        auto_grad_meta->StopGradient()) {
      VLOG(3) << "Skip auto grad since there is no grad op for var or loss is "
                 "stop_gradient=True: "
              << tensor.name();
      continue;
    }

    // TODO(zhanlve): Copy and Modify GradNode if is_general_grad
    GradNodeBase* grad_node = shared_grad_node.get();
    if (is_general_grad) {
      // Save orig grad node
      orig_queue.push_back(grad_node);

      // Replace grad_node with copied grad_node
      grad_node = GeneralGrad::Instance().CopyGradNode(shared_grad_node);

      // Record potential startup grad node
      GeneralGrad::Instance().GetPotentialStartupNodes()->insert(grad_node);
    }

    // Prepare GradTensorHolder
    if (!node_input_buffers_dict.count(grad_node)) {
      VLOG(6) << "Create Value for grad input tensor " << i
              << " of grad node: " << grad_node->name();
      node_input_buffers_dict[grad_node] =
          std::make_unique<GradTensorHolder>(grad_node->InputMeta());
    }
    bool copy_from_grad_t =
        grad_tensors.size() > 0 && grad_tensors[i].initialized();
    if (copy_from_grad_t) {
      PADDLE_ENFORCE(
          grad_tensors.size() == tensors.size(),
          paddle::platform::errors::Fatal(
              "Detected size mismatch between tensors and grad_tensors"
              "grad_tensors should either have "
              "size = 0 or same size as tensors."));
      // Feed given tensor if it's provided
      VLOG(6) << "Fill grad input tensor " << i << "with give grad tensor";

      // Deep copy
      node_input_buffers_dict[grad_node]->CopyValueFromTensor(
          input_info.first, input_info.second, grad_tensors[i]);
    } else {
      VLOG(6) << "Fill grad input tensor " << i << " with 1.0";
      // Initialize tensor with 1.0
      // Forward Tensor "tensor" is passed to indicate tensortype, datatype and
      // dims
      // GradTensorHolder will initialize another tensor with same tensortype,
      // datatype and dims but filled with 1.0
      node_input_buffers_dict[grad_node]->CopyValueFromTensor(
          input_info.first, input_info.second, tensor, /*fill_one=*/true);
    }

    // Prepare queue, potential startup_nodes
    queue.push_back(grad_node);
  }

  if (is_general_grad) {
    // Copy Backward Graph
    GeneralGrad::Instance().ReconstructBackwardGraph(orig_queue);
  }

  if (is_general_grad) {
    // Prepare several vital preprocess for GeneralGrad
    GeneralGrad::Instance().PreparedForGeneralGrad(
        inputs, no_grad_vars, &queue, node_input_buffers_dict);
  }

  VLOG(6) << "Update In degree Map for backward";
  // 3. Compute in_degree for each node
  std::unordered_map<GradNodeBase*, int> node_in_degree_map =
      getInDegreeMap(queue);

  VLOG(6) << "Startup_ops's size is :" << queue.size();

  /* --- Topological Visit --- */
  // 1. Pop queue
  // 2. Run node
  //    |- Check and capture target result
  //    |- node(grads)
  //    |- Prepare for next node
  // 3. Update queue
  VLOG(3) << "Run Backward";
  while (!queue.empty()) {
    GradNodeBase* node = queue.front();
    VLOG(6) << "GradNode " << node->name() << " is ready.";

    paddle::platform::RecordEvent node_record_event(
        std::string((*node).name()),
        paddle::platform::TracerEventType::Operator,
        1);

    if (queue.size() > 1 && node_in_degree_map[node] != 0) {
      queue.pop_front();
      continue;
    }
    queue.pop_front();

    // Run node: This is where Hook happens
    auto node_input_buffer_iter = node_input_buffers_dict.find(node);
    PADDLE_ENFORCE_NE(
        node_input_buffer_iter,
        node_input_buffers_dict.end(),
        paddle::platform::errors::Fatal(
            "Unable to find next node in the GradTensorHolder \n"
            "Trying to run Node without configuring its GradTensorHolder."));

    std::unique_ptr<GradTensorHolder> node_input_buffer =
        std::move(node_input_buffer_iter->second);

    // no_grad_vars
    if (!no_grad_vars.empty() && is_general_grad) {
      auto iter =
          GeneralGrad::Instance().GetNoGradVarNodesInputMetaMap()->find(node);
      if (iter !=
          GeneralGrad::Instance().GetNoGradVarNodesInputMetaMap()->end()) {
        VLOG(6) << "Change the input buffer[slot][rank] by Zeros";
        auto rank_info = (iter->second)->OutRankInfo();
        node_input_buffer->SetBufferSlotRankZeros(rank_info.first,
                                                  rank_info.second);
      }
    }

    // Check input
    EnforceGradNodeHasInput(node);

    VLOG(6) << "Run Backward Kernel with GradTensorHolder.";
    // Run Pre Backward Node and get outputs
    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         kSlotSmallVectorSize>
        grad_output_tensors = (*node)(
            node_input_buffer->Buffers(), create_graph, is_general_grad);

    if (!inputs.empty() && is_general_grad) {
      GeneralGrad::Instance().SetResultForEnddingNodes(grad_output_tensors,
                                                       node);
    }

    // retain_grad or not
    if (!retain_graph) {
      VLOG(6)
          << "retain_graph is false, need to clear the TensorWrapper of nodes.";
      node->ClearTensorWrappers();
    }

    // TODO(jiabin): Should we erase it or find a more efficient way.
    node_input_buffers_dict.erase(node_input_buffer_iter);

    // Prepare GradTensorHolder for next node
    const paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
        metas = node->OutputMeta();
    PADDLE_ENFORCE(metas.size() == grad_output_tensors.size() || metas.empty(),
                   paddle::platform::errors::Fatal(
                       "Number of edges should be either empty ( for leaf node "
                       ") or the same as number of output grad tensors, but we "
                       "got edges size is: %d, grad_output size is: %d",
                       metas.size(),
                       grad_output_tensors.size()));

    for (size_t i = 0; i < metas.size(); i++) {
      for (size_t j = 0; j < metas[i].size(); j++) {
        const Edge& edge = metas[i][j].GetEdge();
        if (!edge.IsInitialized()) {
          continue;
        }
        auto edge_rank = edge.GetEdgeRankInfo();
        // Since we make edge has as same rank as bwd outputs, we indexing them
        // with the same rank(i, j)
        auto next_node_shared = edge.GetMutableGradNode();
        VLOG(3) << "Found pending node: " << next_node_shared->name();
        // Next node could be nullptr if it is leaf tensor with no
        // AccumulationNode attached
        // Or it could also originated from dispensable inputs
        if (!next_node_shared || !next_node_shared.get() ||
            grad_output_tensors[i].empty()) {
          continue;
        }

        PADDLE_ENFORCE_LT(
            j,
            grad_output_tensors[i].size(),
            paddle::platform::errors::Fatal(
                "Rank of grad_output_tensors should be less than "
                "grad_output_tensors[i].size(), which is: %d. This error may "
                "indicate autoprune or autograd api error. ",
                grad_output_tensors.size()));
        paddle::experimental::Tensor& grad_output_tensor =
            grad_output_tensors[i][j];

        if ((!grad_output_tensor.defined() ||
             !grad_output_tensor.initialized())) {
          VLOG(6) << "We get grad_output_tensor with slot: " << i
                  << ", rank: " << j << " as uninitialized or undefined tensor";
        }

        VLOG(6) << "Get Edge and grad_output_tensor with slot: " << i
                << ", rank: " << j
                << " 's name is: " << grad_output_tensor.name();

        auto* next_node = next_node_shared.get();
        if (!node_input_buffers_dict.count(next_node)) {
          const auto& input_meta = next_node->InputMeta();
          auto grad_tensor_holder =
              std::make_unique<GradTensorHolder>(input_meta);
          VLOG(6) << "Construct GradTensorHolder for grad node: "
                  << next_node->name();
          node_input_buffers_dict[next_node] = std::move(grad_tensor_holder);
        }

        VLOG(6) << "Sum grad inputs for edge slot: " << edge_rank.first
                << ", rank: " << edge_rank.second;

        node_input_buffers_dict[next_node]->add(edge_rank.first,
                                                edge_rank.second,
                                                grad_output_tensor,
                                                create_graph);

        // Update queue
        node_in_degree_map[next_node]--;

        PADDLE_ENFORCE(
            node_in_degree_map[next_node] >= 0,
            paddle::platform::errors::Fatal(
                "Detected in-degree value smaller than zero. For Node: %s"
                "Node's in-degree cannot be negative.",
                next_node->name()));

        if (node_in_degree_map[next_node] == 0) {
          if (dynamic_cast<egr::GradNodeAccumulation*>(next_node)) {
            queue.push_front(std::move(next_node));
          } else {
            queue.push_back(std::move(next_node));
          }
        }
      }
    }
  }

  if (!is_general_grad) return {};
  return GeneralGrad::Instance().GetResults(inputs, allow_unused, create_graph);
}

void Backward(
    const std::vector<paddle::experimental::Tensor>& tensors,  // outputs
    const std::vector<paddle::experimental::Tensor>& grad_tensors,
    bool retain_graph) {
  VLOG(3) << "Run in Backward";
  paddle::platform::RecordEvent backward_record_event(
      "backward", paddle::platform::TracerEventType::Operator, 1);
  RunBackward(tensors, grad_tensors, retain_graph);
  phi::autotune::AutoTuneStatus::Instance().Update();
}

std::vector<paddle::experimental::Tensor> Grad(
    const std::vector<paddle::experimental::Tensor>& tensors,  // outputs
    const std::vector<paddle::experimental::Tensor>& inputs,
    const std::vector<paddle::experimental::Tensor>& grad_tensors,
    bool retain_graph,
    bool create_graph,
    bool only_inputs,
    bool allow_unused,
    const std::vector<paddle::experimental::Tensor>& no_grad_vars) {
  VLOG(3) << "Run in Grad";

  DuplicateCheck(inputs, true /* is_input */);
  DuplicateCheck(tensors, false /* is_input */);

  return RunBackward(tensors,
                     grad_tensors,
                     retain_graph,
                     create_graph,
                     inputs,
                     allow_unused,
                     no_grad_vars);
}
}  // namespace egr
