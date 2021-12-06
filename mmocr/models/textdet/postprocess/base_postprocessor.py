# Copyright (c) OpenMMLab. All rights reserved.


class BasePostprocessor:

    def __init__(self, text_repr_type='poly'):
        assert text_repr_type in ['poly', 'quad'
                                  ], f'Invalid text repr type {text_repr_type}'

        self.text_repr_type = text_repr_type

    def is_valid_instance(self, area, confidence, area_thresh,
                          confidence_thresh):

        return bool(area >= area_thresh and confidence > confidence_thresh)
