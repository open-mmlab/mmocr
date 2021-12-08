import warnings

from mmcv.runner import BaseModule

from mmocr.models.builder import FUSION, build_loss


@FUSION.register_module()
class BaseFusion(BaseModule):
    """Base Fusion for multimodal fusion."""

    def __init__(self,
                 loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # Loss
        assert loss is not None
        self.loss = build_loss(loss)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    def forward_train(self, major_modality, extra_modality, targets_dict,
                      img_metas):
        raise NotImplementedError

    def forward_test(self, major_modality, extra_modality, targets_dict,
                     img_metas):
        raise NotImplementedError

    def forward(self,
                major_modality,
                extra_modality,
                targets_dict=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(major_modality, extra_modality,
                                      targets_dict, img_metas)

        return self.forward_test(major_modality, extra_modality, img_metas)
