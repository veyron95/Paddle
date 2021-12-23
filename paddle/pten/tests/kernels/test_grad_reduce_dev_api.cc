/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// #include <gtest/gtest.h>
// #include <memory>

// // #include "paddle/pten/include/math.h"
// #include "paddle/pten/kernels/cpu/grad_reduce.h"
// // paddle/pten/kernels/cpu/grad_reduce.h

// #include "paddle/pten/api/lib/utils/allocator.h"
// #include "paddle/pten/core/dense_tensor.h"
// #include "paddle/pten/core/kernel_registry.h"

// namespace pten {
// namespace tests {

// namespace framework = paddle::framework;
// using DDim = paddle::framework::DDim;

// TEST(API, grad_reduce_sum_cpu) {
//   // 1. create tensor

//   const auto alloc =
//   std::make_shared<paddle::experimental::DefaultAllocator>(
//       paddle::platform::CPUPlace());

//   // ======= dense_x
//   pten::DenseTensor dense_x(alloc,
//                             pten::DenseTensorMeta(pten::DataType::FLOAT32,
//                                                   framework::make_ddim({3,
//                                                   3}),
//                                                   pten::DataLayout::NCHW));
//   auto* dense_x_data = dense_x.mutable_data<float>();

//   // ======= dense_out
//   pten::DenseTensor dense_out(alloc,
//                             pten::DenseTensorMeta(pten::DataType::FLOAT32,
//                                                   framework::make_ddim({1}),
//                                                   pten::DataLayout::NCHW));
//   auto* dense_out_data = dense_out.mutable_data<float>();

//   // ======== dense_grad_out
//   pten::DenseTensor dense_grad_out(alloc,
//                             pten::DenseTensorMeta(pten::DataType::FLOAT32,
//                                                   framework::make_ddim({1}),
//                                                   pten::DataLayout::NCHW));
//   auto* dense_grad_out_data = dense_grad_out.mutable_data<float>();

//   // ======== dense_grad_x
//   pten::DenseTensor dense_grad_x(alloc,
//                             pten::DenseTensorMeta(pten::DataType::FLOAT32,
//                                                   framework::make_ddim({1}),
//                                                   pten::DataLayout::NCHW));
//   // auto* dense_grad_x_data = dense_grad_x.mutable_data<float>();

//   dense_out_data[0] = 9.0;
//   dense_grad_out_data[0] = 4.0;
//   for (size_t i = 0; i < 9; ++i) {
//     dense_x_data[i] = 1.0;
//   }

//   std::vector<float> grad_x_result(9, 4.0);

//   // paddle::experimental::Tensor
//   x(std::make_shared<pten::DenseTensor>(dense_x));
//   // paddle::experimental::Tensor
//   out(std::make_shared<pten::DenseTensor>(dense_out));
//   // paddle::experimental::Tensor
//   grad_out(std::make_shared<pten::DenseTensor>(dense_grad_out));

//   paddle::platform::DeviceContextPool& pool =
//       paddle::platform::DeviceContextPool::Instance();
//   auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());

//   // const CPUContext& dev_ctx,
//   //                  const DenseTensor& X,
//   //                  const DenseTensor& Out,
//   //                  const DenseTensor& GradOut,
//   //                  bool reduce_all,
//   //                  // const std::vector<int>& dims,
//   //                  int in_dtype,
//   //                  DenseTensor* GradX) {
//   // 2. test API
//   pten::GradReduceSum<float>(
//     *(static_cast<paddle::platform::CPUDeviceContext*>(dev_ctx)),
//      dense_x, dense_out, dense_grad_out, true, -1, &dense_grad_x);

//   // 3. check result
//   // ASSERT_EQ(grad_x.shape().size(), 2);
//   // ASSERT_EQ(grad_x.shape()[0], 3);
//   // ASSERT_EQ(grad_x.shape()[1], 3);
//   // ASSERT_EQ(grad_x.numel(), 9);
//   // ASSERT_EQ(grad_x.type(), pten::DataType::kFLOAT32);
//   // ASSERT_EQ(grad_x.layout(), pten::DataLayout::kNCHW);
//   // ASSERT_EQ(grad_x.initialized(), true);

//   // auto dense_grad_x =
//   std::dynamic_pointer_cast<pten::DenseTensor>(grad_x.impl());
//   // for (size_t i = 0; i < 9; i++) {
//   //   ASSERT_NEAR(grad_x_result[i], dense_grad_x->data<float>()[i], 1e-6f);
//   // }
// }

// TEST(DEV_API, sum) {
//   // 1. create tensor
//   const auto alloc =
//   std::make_shared<paddle::experimental::DefaultAllocator>(
//       paddle::platform::CPUPlace());
//   pten::DenseTensor dense_x(alloc,
//                             pten::DenseTensorMeta(pten::DataType::FLOAT32,
//                                                   framework::make_ddim({3,
//                                                   4}),
//                                                   pten::DataLayout::NCHW));
//   auto* dense_x_data = dense_x.mutable_data<float>();

//   float sum = 0.0;
//   for (size_t i = 0; i < 12; ++i) {
//     dense_x_data[i] = i * 1.0;
//     sum += i * 1.0;
//   }
//   paddle::platform::DeviceContextPool& pool =
//       paddle::platform::DeviceContextPool::Instance();
//   auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());

//   std::vector<int64_t> axis = {0, 1};
//   // 2. test API
//   auto out = pten::Sum<float>(
//       *(static_cast<paddle::platform::CPUDeviceContext*>(dev_ctx)),
//       dense_x,
//       axis,
//       pten::DataType::FLOAT32,
//       false);

//   // 3. check result
//   ASSERT_EQ(out.dims().size(), 1);
//   ASSERT_EQ(out.numel(), 1);
//   ASSERT_EQ(out.meta().dtype, pten::DataType::FLOAT32);
//   ASSERT_EQ(out.meta().layout, pten::DataLayout::NCHW);

//   auto expect_result = sum;
//   auto actual_result = out.data<float>()[0];
//   ASSERT_NEAR(expect_result, actual_result, 1e-6f);
// }

// }  // namespace tests
// }  // namespace pten
