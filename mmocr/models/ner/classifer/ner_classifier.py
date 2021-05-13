from mmdet.models.builder import DETECTORS, build_loss
from mmocr.models.builder import build_convertor, build_decoder, build_encoder
from mmocr.models.textrecog.recognizer.base import BaseRecognizer


@DETECTORS.register_module()
class NerClassifier(BaseRecognizer):
    """Base class for NER classifier."""

    def __init__(self,
                 encoder=None,
                 decoder=None,
                 loss=None,
                 label_convertor=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.label_convertor = build_convertor(label_convertor)
        self.encoder = build_encoder(encoder)
        decoder.update(num_labels=self.label_convertor.num_labels)
        self.decoder = build_decoder(decoder)
        loss.update(num_labels=self.label_convertor.num_labels)
        self.loss = build_loss(loss)

    def extract_feat(self, imgs):
        """Extract features from images."""
        return

    def forward_train(self, imgs, img_metas, **kwargs):
        encode_out = self.encoder(img_metas)
        logits, _ = self.decoder(encode_out)
        loss = self.loss(logits, img_metas)
        return loss

    def forward_test(self, imgs, img_metas, **kwargs):
        encode_out = self.encoder(img_metas)
        _, preds = self.decoder(encode_out)
        pred_entities = self.label_convertor.convert_pred2entities(preds)
        return pred_entities

    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def simple_test(self, img, img_metas, **kwargs):
        pass
