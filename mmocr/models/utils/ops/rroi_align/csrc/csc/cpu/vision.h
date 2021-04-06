// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/extension.h>


at::Tensor ROIAlign_forward_cpu(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width,
                                const int sampling_ratio);
