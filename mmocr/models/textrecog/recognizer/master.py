# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import DETECTORS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@DETECTORS.register_module()
class MASTER(EncodeDecodeRecognizer):

    def __init__(self,
                 preprocessor=None,
                 backbone=None,
                 encoder=None,
                 decoder=None,
                 loss=None,
                 label_convertor=None,
                 train_cfg=None,
                 test_cfg=None,
                 max_seq_len=40,
                 pretrained=None,
                 init_cfg=None):
        super(MASTER, self).__init__(
            preprocessor,
            backbone,
            encoder,
            decoder,
            loss,
            label_convertor,
            train_cfg,
            test_cfg,
            max_seq_len,
            pretrained,
            init_cfg=init_cfg)
