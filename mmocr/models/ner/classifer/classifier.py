import torch

from mmdet.models.builder import DETECTORS, build_loss
from mmocr.models.builder import build_decoder, build_encoder
from mmocr.models.textrecog.recognizer.base import BaseRecognizer


@DETECTORS.register_module()
class NerClassifier(BaseRecognizer):
    """Base class for encode-decode recognizer."""

    def __init__(self,
                 encoder=None,
                 decoder=None,
                 loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)
        self.loss = build_loss(loss)

    def extract_feat(self, imgs):
        """Extract features from images."""
        return

    def forward_train(self, imgs, img_metas, **kwargs):
        device = next(self.encoder.parameters()).device
        x = self.encoder(img_metas)
        logits, x = self.decoder(x)
        labels = []
        attention_masks = []
        for i in range(len(img_metas)):
            label = torch.tensor(img_metas[i]['labels']).to(device)
            attention_mask = torch.tensor(
                img_metas[i]['attention_mask']).to(device)
            labels.append(label)
            attention_masks.append(attention_mask)
        labels = torch.stack(labels, 0)
        attention_masks = torch.stack(attention_masks, 0)
        loss = self.loss(logits, labels, attention_masks)
        return {'loss_cls': loss}

    def forward_test(self, imgs, img_metas, **kwargs):
        x = self.encoder(img_metas)
        logits, x = self.decoder(x)
        return x

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]])
        """
        pass

    def simple_test(self, img, img_metas, **kwargs):
        pass
