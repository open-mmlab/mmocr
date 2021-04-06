import torch.nn as nn

from mmocr.models.builder import DECODERS


@DECODERS.register_module()
class BaseDecoder(nn.Module):
    """Base decoder class for text recognition."""

    def __init__(self, **kwargs):
        super().__init__()

    def init_weights(self):
        pass

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def forward(self,
                feat,
                out_enc,
                targets_dict=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets_dict, img_metas)

        return self.forward_test(feat, out_enc, img_metas)
