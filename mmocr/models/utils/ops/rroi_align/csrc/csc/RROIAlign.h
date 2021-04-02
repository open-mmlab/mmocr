// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
std::tuple<at::Tensor, at::Tensor, at::Tensor> RROIAlign_forward(const at::Tensor& input,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return RROIAlign_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
  //return RROIAlign_forward_cpu(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
}

at::Tensor RROIAlign_backward(const at::Tensor& grad,
                                 const at::Tensor& rois,
                                 const at::Tensor& con_idx_x,
                                 const at::Tensor& con_idx_y,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return RROIAlign_backward_cuda(grad, rois, con_idx_x, con_idx_y, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
