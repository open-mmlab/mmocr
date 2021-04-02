// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor> RROIAlign_forward_cuda(const at::Tensor& input,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width);

at::Tensor RROIAlign_backward_cuda(const at::Tensor& grad,
                                 const at::Tensor& rois,
                                 const at::Tensor& con_idx_x,
                                 const at::Tensor& con_idx_y,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width);

at::Tensor compute_flow_cuda(const at::Tensor& boxes,
                             const int height,
                             const int width);
