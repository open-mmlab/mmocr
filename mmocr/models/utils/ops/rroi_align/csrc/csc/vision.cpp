// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "RROIAlign.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("rroi_align_forward", &RROIAlign_forward, "RROIAlign_forward");
  m.def("rroi_align_backward", &RROIAlign_backward, "RROIAlign_backward");

}
