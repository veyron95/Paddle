/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_reducescatter_op.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class CReduceScatterOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");

    int rid = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
    int nranks = comm->nranks();

    auto out_dims = in->dims();
    PADDLE_ENFORCE_EQ(out_dims[0] % nranks,
                      0,
                      platform::errors::InvalidArgument(
                          "The input tensor X's "
                          "dim[0] (%d) should be divisible by nranks(%d)",
                          out_dims[0],
                          nranks));
    out_dims[0] = out_dims[0] / nranks;
    out->mutable_data<T>(out_dims, place);

    int64_t recv_numel = in->numel() / nranks;
    const T* send_buff = in->data<T>();
    T* recv_buff = out->data<T>();
    int dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(in->dtype()));

    gpuStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      // should ExecutionContext for calc stream.
      stream = ctx.cuda_device_context().stream();
    } else {
      stream = comm->stream();
    }

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::ncclReduceScatter(send_buff,
                                             recv_buff,
                                             recv_numel,
                                             static_cast<ncclDataType_t>(dtype),
                                             ncclSum,
                                             comm->comm(),
                                             stream));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

PD_REGISTER_STRUCT_KERNEL(c_reducescatter,
                          GPU,
                          ALL_LAYOUT,
                          ops::CReduceScatterOpCUDAKernel,
                          float,
                          double,
#if NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000
                          plat::bfloat16,
#endif
                          int,
                          int64_t,
                          plat::float16) {
}
