/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/complex_functors.h"

#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"

#include "paddle/pten/kernels/hybird/eigen/common.h"
// #include "paddle/pten/kernels/common/eigen/common.h"

// #include "paddle/pten/kernels/common/math/reduce_function.h"
#include "paddle/pten/kernels/hybird/eigen/reduce.h"
#include "paddle/pten/kernels/hybird/math/cast_func.h"

#include "paddle/pten/core/tensor_status.h"

#include "paddle/pten/kernels/cpu/utils.h"
#include "paddle/pten/kernels/cuda/utils.h"

namespace pten {
namespace math {

template <typename InType, typename OutType>
struct CastDataTypeFunctor {
  HOSTDEVICE inline OutType operator()(InType in) const {
    return static_cast<OutType>(in);
  }
};

template <typename DeviceContext, typename InType>
struct CastDataType {
  CastDataType(const DeviceContext& dev_ctx,
               const DenseTensor& in,
               DenseTensor* out)
      : dev_ctx_(dev_ctx), in_(in), out_(out) {}
  const DeviceContext& dev_ctx_;
  const DenseTensor& in_;
  DenseTensor* out_;

  template <typename OutType>
  void apply() {
    auto* in_begin = in_.data<InType>();
    auto* in_end = in_begin + in_.numel();
    auto* out_begin = out_->mutable_data<OutType>();

    if (paddle::platform::is_cpu_place(in_.place())) {
      paddle::platform::Transform<paddle::platform::CPUDeviceContext> trans;
      trans(paddle::platform::CPUDeviceContext(),
            in_begin,
            in_end,
            out_begin,
            CastDataTypeFunctor<InType, OutType>());

#if defined(__NVCC__) || defined(__HIPCC__)
    } else if (paddle::platform::is_gpu_place(in_.place())) {
      paddle::platform::Transform<paddle::platform::CUDADeviceContext> trans;
      trans(dev_ctx_,
            in_begin,
            in_end,
            out_begin,
            CastDataTypeFunctor<InType, OutType>());
      dev_ctx_.Wait();
#endif
    } else {
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Place type is not supported when casting data type."));
    }
  }
};

struct SumGradFunctor {
  template <typename DeviceContext,
            typename X,
            typename Y,
            typename DX,
            typename DY,
            typename Dim>
  void operator()(const DeviceContext& place,
                  X* x,
                  Y* y,
                  DX* dx,
                  DY* dy,
                  const Dim& dim,
                  int size) {
    dx->device(place) = dy->broadcast(dim);
  }
};

template <typename DeviceContext, typename T, size_t D, typename Functor>
void ReduceGradFunctor(const DeviceContext& context,
                       const DenseTensor& input0,  // X
                       const DenseTensor& input1,  // Out
                       const DenseTensor& input2,  // GradOut
                       const std::vector<int64_t>& dims,
                       DenseTensor* output /*GradX*/) {
  auto x = pten::EigenTensor<T, D>::From(input0);
  auto x_grad = pten::EigenTensor<T, D>::From(*output);
  auto x_rank = static_cast<int>(x.dimensions().size());
  auto x_dims = input0.dims();
  auto reduced_dims_v = paddle::framework::vectorize(x_dims);
  std::vector<int64_t> dims_ref = dims;
  Eigen::array<int, D> broadcast_dim;
  for (size_t i = 0; i < D; ++i) broadcast_dim[i] = 1;

  int broad_cats_times = 1;
  for (size_t i = 0; i < dims_ref.size(); ++i) {
    if (dims_ref[i] < 0) {
      dims_ref[i] = x_rank + dims_ref[i];
    }
    reduced_dims_v[dims_ref[i]] = 1;
    broadcast_dim[dims_ref[i]] = x_dims[dims_ref[i]];
    broad_cats_times *= x_dims[dims_ref[i]];
  }
  auto reduced_dims = paddle::framework::make_ddim(reduced_dims_v);
  auto x_reduce = pten::EigenTensor<T, D>::From(input1, reduced_dims);
  auto x_reduce_grad = pten::EigenTensor<T, D>::From(input2, reduced_dims);

  auto& place = *context.eigen_device();

  Functor functor;
  functor(place,
          &x,
          &x_reduce,
          &x_grad,
          &x_reduce_grad,
          broadcast_dim,
          broad_cats_times);
}

inline void GetOriginDimFromShuffled(const DDim& src_dim,
                                     const std::vector<int64_t>& dims,
                                     std::vector<int64_t>* origin_dim) {
  DDim shuffled_dims(src_dim);
  size_t n = src_dim.size();
  std::vector<int64_t> perm_axis(n);
  pten::eigen::GetShuffledDim(src_dim, &shuffled_dims, dims, &perm_axis);
  for (size_t i = 0; i < n; ++i) {
    (*origin_dim)[perm_axis[i]] = i;
  }
}

template <typename DeviceContext, typename T, typename Functor>
void HandleLargeDimGrad(const DeviceContext& context,
                        const DenseTensor& x,
                        const DenseTensor& out,
                        const DenseTensor& dout,
                        const std::vector<int64_t>& dims,
                        DenseTensor* dx) {
  const int64_t unreduced = out.numel();
  const int64_t reduced = x.numel() / unreduced;
  DDim out_dim(out.dims());
  DDim x_dim(x.dims());

  // transpose and reshape X
  DenseTensor shuffled_x(
      pten::make_intrusive<paddle::experimental::SharedStorage>(x.place()),
      x.meta());
  pten::eigen::GetShuffledInput<DeviceContext, T>(
      context, x, &shuffled_x, dims);
  DDim shuffled_dim = shuffled_x.dims();
  shuffled_x.Resize({unreduced, reduced});

  // reshape dX {unreduced, reduced}
  dx->Resize({unreduced, reduced});
  ReduceGradFunctor<DeviceContext, T, 2, Functor>(
      context, shuffled_x, out, dout, {1}, dx);

  // transpose dX
  std::vector<int64_t> origin_axis(x_dim.size());
  GetOriginDimFromShuffled(x_dim, dims, &origin_axis);

  DenseTensor dx_tmp(
      pten::make_intrusive<paddle::experimental::SharedStorage>(dx->place()),
      dx->meta());
  pten::Copy(context, *dx, false, &dx_tmp);

  dx_tmp.Resize(shuffled_dim);
  dx->Resize(x_dim);

  TransposeNormal<DeviceContext, T> trans;
  trans(context, dx_tmp, dx, origin_axis);
}

template <typename DeviceContext>
inline void TransDataType(
    const DeviceContext& dev_ctx,
    const paddle::framework::OpKernelType& kernel_type_for_var,
    const paddle::framework::OpKernelType& expected_kernel_type,
    const DenseTensor& in,
    DenseTensor* out) {
  out->Resize(in.dims());
  auto src_type = kernel_type_for_var.data_type_;
  auto dst_type = expected_kernel_type.data_type_;

  switch (src_type) {
    case paddle::framework::proto::VarType::FP16:
      paddle::framework::VisitDataType(
          dst_type,
          CastDataType<DeviceContext, paddle::platform::float16>(
              dev_ctx, in, out));
      break;
    case paddle::framework::proto::VarType::BF16:
      paddle::framework::VisitDataType(
          dst_type,
          CastDataType<DeviceContext, paddle::platform::bfloat16>(
              dev_ctx, in, out));
      break;
    case paddle::framework::proto::VarType::FP32:
      paddle::framework::VisitDataType(
          dst_type, CastDataType<DeviceContext, float>(dev_ctx, in, out));
      break;
    case paddle::framework::proto::VarType::FP64:
      paddle::framework::VisitDataType(
          dst_type, CastDataType<DeviceContext, double>(dev_ctx, in, out));
      break;
    case paddle::framework::proto::VarType::INT32:
      paddle::framework::VisitDataType(
          dst_type, CastDataType<DeviceContext, int>(dev_ctx, in, out));
      break;
    case paddle::framework::proto::VarType::INT64:
      paddle::framework::VisitDataType(
          dst_type, CastDataType<DeviceContext, int64_t>(dev_ctx, in, out));
      break;
    case paddle::framework::proto::VarType::BOOL:
      paddle::framework::VisitDataType(
          dst_type, CastDataType<DeviceContext, bool>(dev_ctx, in, out));
      break;
    case paddle::framework::proto::VarType::INT16:
      paddle::framework::VisitDataType(
          dst_type, CastDataType<DeviceContext, int16_t>(dev_ctx, in, out));
      break;
    case paddle::framework::proto::VarType::UINT8:
      paddle::framework::VisitDataType(
          dst_type, CastDataType<DeviceContext, uint8_t>(dev_ctx, in, out));
      break;
    default:
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Data type (%s) is not supported when casting data type.",
          paddle::framework::DataTypeToString(src_type)));
  }
}

template <typename DeviceContext,
          typename T,
          typename Functor,
          bool kNoNeedBufferX = false,
          bool kNoNeedBufferY = false>
struct ReduceGradKernel {
  void ComputeFromInput(const DeviceContext& dev_ctx,
                        const DenseTensor& input0,  // X
                        const DenseTensor& input1,  // Out
                        const DenseTensor& input2,  // GradOut
                        bool reduce_all,
                        const std::vector<int64_t>& dims,
                        DenseTensor* GradX) const {
    GradX->mutable_data<T>();

    // The dims has full dim, set the reduce_all is True
    const auto& input_dim_size = input0.dims().size();
    std::set<int> dims_set(dims.begin(), dims.end());
    bool full_dim = true;
    for (auto i = 0; i < input_dim_size; i++) {
      if (dims_set.find(i) == dims_set.end()) {
        full_dim = false;
        break;
      }
    }
    reduce_all = (reduce_all || full_dim);
    if (reduce_all) {
      auto x = pten::EigenVector<T>::Flatten(input0);
      auto x_reduce = pten::EigenVector<T>::Flatten(input1);
      auto x_reduce_grad = pten::EigenVector<T>::Flatten(input2);
      auto x_grad = pten::EigenVector<T>::Flatten(*GradX);
      auto& place = *dev_ctx.eigen_device();
      auto broadcast_dim =
          Eigen::array<int, 1>({{static_cast<int>(input0.numel())}});
      Functor functor;
      functor(place,
              &x,
              &x_reduce,
              &x_grad,
              &x_reduce_grad,
              broadcast_dim,
              broadcast_dim[0]);
    } else {
      int rank = input0.dims().size();
      switch (rank) {
        case 1:
          ReduceGradFunctor<DeviceContext, T, 1, Functor>(
              dev_ctx, input0, input1, input2, dims, GradX);
          break;
        case 2:
          ReduceGradFunctor<DeviceContext, T, 2, Functor>(
              dev_ctx, input0, input1, input2, dims, GradX);
          break;
        case 3:
          ReduceGradFunctor<DeviceContext, T, 3, Functor>(
              dev_ctx, input0, input1, input2, dims, GradX);
          break;
        case 4:
          ReduceGradFunctor<DeviceContext, T, 4, Functor>(
              dev_ctx, input0, input1, input2, dims, GradX);
          break;
        case 5:
          ReduceGradFunctor<DeviceContext, T, 5, Functor>(
              dev_ctx, input0, input1, input2, dims, GradX);
          break;
        case 6:
          ReduceGradFunctor<DeviceContext, T, 6, Functor>(
              dev_ctx, input0, input1, input2, dims, GradX);
          break;
        default:
          HandleLargeDimGrad<DeviceContext, T, Functor>(
              dev_ctx, input0, input1, input2, dims, GradX);
          break;
      }
    }
  }

  void Compute(const DeviceContext& dev_ctx,
               const DenseTensor& X,
               const DenseTensor& Out,
               const DenseTensor& GradOut,
               bool reduce_all,
               const std::vector<int64_t>& dims,
               int in_dtype,
               DenseTensor* GradX) {
    if (in_dtype >= 0) {
      auto in_kernel_type = paddle::framework::OpKernelType(
          TransToProtoVarType(GradOut.dtype()), dev_ctx.GetPlace());
      auto out_kernel_type = paddle::framework::OpKernelType(
          static_cast<paddle::framework::proto::VarType::Type>(in_dtype),
          dev_ctx.GetPlace());

      DenseTensorMeta tmp_meta(TransToPtenDataType(out_kernel_type.data_type_),
                               GradOut.dims(),
                               GradOut.layout());
      DenseTensor tmp_tensor(
          pten::make_intrusive<paddle::experimental::SharedStorage>(
              GradOut.place()),
          tmp_meta);

      TransDataType(
          dev_ctx, in_kernel_type, out_kernel_type, GradOut, &tmp_tensor);

      if (kNoNeedBufferX && kNoNeedBufferY) {
        ComputeFromInput(
            dev_ctx, *GradX, tmp_tensor, tmp_tensor, reduce_all, dims, GradX);

      } else if (kNoNeedBufferX) {
        // NOTE: EigenTensor::From() uses tensor->data()
        // if op has NoNeedBufferVarsInferer, the corresponding kNoNeedBufferX
        // or
        // kNoNeedBufferY should set true
        // and use fake var that has same dims.
        ComputeFromInput(
            dev_ctx, *GradX, Out, tmp_tensor, reduce_all, dims, GradX);

      } else if (kNoNeedBufferY) {
        // NOTE(dengkaipeng): Out is unnecessary in some reduce kernel and
        // not be set as Input in grad Maker, use Out_grad to replace here
        ComputeFromInput(
            dev_ctx, X, tmp_tensor, tmp_tensor, reduce_all, dims, GradX);

      } else {
        ComputeFromInput(dev_ctx, X, Out, tmp_tensor, reduce_all, dims, GradX);
      }

    } else {
      if (kNoNeedBufferX && kNoNeedBufferY) {
        ComputeFromInput(
            dev_ctx, *GradX, GradOut, GradOut, reduce_all, dims, GradX);

      } else if (kNoNeedBufferX) {
        ComputeFromInput(
            dev_ctx, *GradX, Out, GradOut, reduce_all, dims, GradX);

      } else if (kNoNeedBufferY) {
        ComputeFromInput(dev_ctx, X, GradOut, GradOut, reduce_all, dims, GradX);

      } else {
        ComputeFromInput(dev_ctx, X, Out, GradOut, reduce_all, dims, GradX);
      }
    }
  }
};

template <typename DeviceContext,
          typename T,
          typename Functor,
          bool kNoNeedBufferX = false>
struct ReduceSumGradKernel {
  // use for loop to speed up Eigen broadcast. 4 timer faster then broadcast
  void ComputeFromInput(const DenseTensor& input0,
                        const DenseTensor& input2,
                        const std::vector<int64_t>& dims,
                        DenseTensor* GradX) const {
    GradX->mutable_data<T>();
    const auto* input2_d = input2.data<T>();
    auto* output_d = GradX->mutable_data<T>();

    // handle reduce_all
    if (input2.dims().size() == 1 && input2.dims()[0] == 1) {
      for (int64_t i = 0; i < paddle::framework::product(input0.dims()); ++i) {
        output_d[i] = input2_d[0];
      }
      return;
    }

    // handle reduce by one dimension
    int reduce_dim_index = dims[0];
    if (reduce_dim_index < 0) {
      reduce_dim_index += input0.dims().size();
    }

    const auto& input_dim = input0.dims();
    int64_t before_dim = 1;
    for (int i = 0; i < reduce_dim_index; ++i) {
      before_dim *= input_dim[i];
    }
    int64_t reduce_dim = input_dim[reduce_dim_index];
    int64_t after_dim = 1;
    for (int i = reduce_dim_index + 1; i < input_dim.size(); ++i) {
      after_dim *= input_dim[i];
    }
    for (int64_t i = 0; i < before_dim; ++i) {
      for (int64_t j = 0; j < reduce_dim; ++j) {
        for (int64_t k = 0; k < after_dim; ++k) {
          output_d[i * reduce_dim * after_dim + j * after_dim + k] =
              input2_d[i * after_dim + k];
        }
      }
    }
  }

  void Compute(const DeviceContext& dev_ctx,
               const DenseTensor& X,
               const DenseTensor& Out,
               const DenseTensor& GradOut,
               bool reduce_all,
               const std::vector<int64_t>& dims,
               int in_dtype,
               DenseTensor* GradX) {
    if (dev_ctx.GetPlace().type() == typeid(paddle::platform::CPUPlace) &&
        dims.size() == 1) {
      if (in_dtype >= 0) {
        auto in_kernel_type = paddle::framework::OpKernelType(
            TransToProtoVarType(GradOut.dtype()), dev_ctx.GetPlace());
        auto out_kernel_type = paddle::framework::OpKernelType(
            static_cast<paddle::framework::proto::VarType::Type>(in_dtype),
            dev_ctx.GetPlace());

        DenseTensorMeta tmp_meta(
            TransToPtenDataType(out_kernel_type.data_type_),
            GradOut.dims(),
            GradOut.layout());
        DenseTensor tmp_tensor(
            pten::make_intrusive<paddle::experimental::SharedStorage>(
                GradOut.place()),
            tmp_meta);

        TransDataType(
            dev_ctx, in_kernel_type, out_kernel_type, GradOut, &tmp_tensor);
        ComputeFromInput(X, tmp_tensor, dims, GradX);
      } else {
        ComputeFromInput(X, GradOut, dims, GradX);
      }
      return;
    }
    // default use Eigen broadcast
    ReduceGradKernel<DeviceContext, T, Functor, kNoNeedBufferX> kernel;
    kernel.Compute(dev_ctx, X, Out, GradOut, reduce_all, dims, in_dtype, GradX);
  }
};

}  // namespace math
}  // namespace pten
