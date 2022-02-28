import torch.nn as nn

from mmocr.models.builder import DETECTORS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@DETECTORS.register_module()
class MASTER(EncodeDecodeRecognizer):
    # need to inherit BaseRecognizer or EncodeDecodeRecognizer in mmocr
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
                 pretrained=None):
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
                                        pretrained)

    def init_weights(self, pretrained=None):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_train(self, img, img_metas):
        """
        Args:
            img (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                contains: 'img_shape', 'filename', and may also contain
                'ori_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        feat = self.extract_feat(img)
        feat = feat[-1]

        gt_labels = [img_meta['text'] for img_meta in img_metas]

        targets_dict = self.label_convertor.str2tensor(gt_labels)

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)

        out_dec = self.decoder(
            feat, out_enc, targets_dict, img_metas, train_mode=True)

        loss_inputs = (
            out_dec,
            targets_dict,
            img_metas,
        )
        losses = self.loss(*loss_inputs)

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        """Test function with test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        feat = self.extract_feat(img)
        feat = feat[-1]

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)

        out_dec = self.decoder(
            feat, out_enc, None, img_metas, train_mode=False)

        label_indexes, label_scores = self.label_convertor.tensor2idx(
            out_dec, img_metas)
        label_strings = self.label_convertor.idx2str(label_indexes)

        # flatten batch results
        results = []
        for string, score in zip(label_strings, label_scores):
            results.append(dict(text=string, score=score))

        return results