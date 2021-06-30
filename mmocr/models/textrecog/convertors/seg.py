import cv2
import numpy as np
import torch

import mmocr.utils as utils
from mmocr.models.builder import CONVERTORS
from .base import BaseConvertor


@CONVERTORS.register_module()
class SegConvertor(BaseConvertor):
    """Convert between text, index and tensor for segmentation based pipeline.

    Args:
        dict_type (str): Type of dict, should be either 'DICT36' or 'DICT90'.
        dict_file (None|str): Character dict file path. If not none, the
            file is of higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, the list
        is of higher priority than dict_type, but lower than dict_file.
        with_unknown (bool): If True, add `UKN` token to class.
        lower (bool): If True, convert original string to lower case.
    """

    def __init__(self,
                 dict_type='DICT36',
                 dict_file=None,
                 dict_list=None,
                 with_unknown=True,
                 lower=False,
                 **kwargs):
        super().__init__(dict_type, dict_file, dict_list)
        assert isinstance(with_unknown, bool)
        assert isinstance(lower, bool)

        self.with_unknown = with_unknown
        self.lower = lower
        self.update_dict()

    def update_dict(self):
        # background
        self.idx2char.insert(0, '<BG>')

        # unknown
        self.unknown_idx = None
        if self.with_unknown:
            self.idx2char.append('<UKN>')
            self.unknown_idx = len(self.idx2char) - 1

        # update char2idx
        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx

    def tensor2str(self, output, img_metas=None):
        """Convert model output tensor to string labels.
        Args:
            output (tensor): Model outputs with size: N * C * H * W
            img_metas (list[dict]): Each dict contains one image info.
        Returns:
            texts (list[str]): Decoded text labels.
            scores (list[list[float]]): Decoded chars scores.
        """
        # assert utils.is_type_list(img_metas, dict)
        # assert len(img_metas) == output.size(0)
        outputs = []
        texts, scores = [], []
        for b in range(output.size(0)):
            seg_pred = output[b].detach()
            seg_res = torch.argmax(
                seg_pred, dim=0).cpu().numpy().astype(np.int32)
            seg_thr = np.where(seg_res == 0, 0, 255).astype(np.uint8)
            _, labels, stats, centroids = cv2.connectedComponentsWithStats(
                seg_thr)
            component_num = stats.shape[0]
            all_res = []
            # print('stats',stats)
            for i in range(component_num):
                temp_loc = (labels == i)
                # print('labels',labels.shape)
                temp_value = seg_res[temp_loc]
                print(temp_value[0],temp_value.shape)
                temp_center = centroids[i]

                temp_max_num = 0
                temp_max_cls = -1
                temp_total_num = 0
                for c in range(len(self.idx2char)):
                    c_num = np.sum(temp_value == c)
                    # if c_num != 0:
                    #     print(c_num)
                    temp_total_num += c_num
                    if c_num > temp_max_num:
                        temp_max_num = c_num
                        temp_max_cls = c

                if temp_max_cls == 0:
                    continue
                temp_max_score = 1.0 * temp_max_num / temp_total_num
                # print(temp_max_cls)
                all_res.append(
                    [temp_max_cls, temp_center, temp_max_num, temp_max_score])
                # all_res.append(
                #     [temp_max_cls, temp_max_score])

            all_res = sorted(all_res, key=lambda s: s[1][0])
            chars, char_scores = [], []
            for res in all_res:
                temp_area = res[2]
                if temp_area < 20:
                    continue
                temp_char_index = res[0]
                # if temp_char_index >= len(self.idx2char):
                #     temp_char = ''
                # elif temp_char_index <= 0:
                #     temp_char = ''
                # elif temp_char_index == self.unknown_idx:
                #     temp_char = ''
                # else:
                #     temp_char = self.idx2char[temp_char_index]
                # chars.append(temp_char)
                # char_scores.append(res[3])
                texts.append(temp_char_index)
                scores.append(res[3])
            outputs.append(texts)
            outputs.append(scores)
            # text = ''.join(chars)

            # texts.append(text)
            # scores.append(char_scores)

        # return texts, scores
        return outputs
