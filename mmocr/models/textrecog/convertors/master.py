# Copyright (c) OpenMMLab. All rights reserved.
import torch

import mmocr.utils as utils
from mmocr.models.builder import CONVERTORS
from .attn import AttnConvertor


@CONVERTORS.register_module()
class MasterConvertor(AttnConvertor):
    """Convert between text, index and tensor for encoder-decoder based
    pipeline.

    Args:
        dict_type (str): Type of dict, should be one of {'DICT36', 'DICT90'}.
        dict_file (None|str): Character dict file path. If not none,
            higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, higher
            priority than dict_type, but lower than dict_file.
        with_unknown (bool): If True, add `UKN` token to class.
        max_seq_len (int): Maximum sequence length of label.
        lower (bool): If True, convert original string to lower case.
        start_end_same (bool): Whether use the same index for
            start and end token or not. Default: True.
    """

    def __init__(self,
                 dict_type='DICT90',
                 dict_file=None,
                 dict_list=None,
                 with_unknown=True,
                 max_seq_len=40,
                 lower=False,
                 start_end_same=True,
                 **kwargs):
        super().__init__(dict_type, dict_file, dict_list)
        assert isinstance(with_unknown, bool)
        assert isinstance(max_seq_len, int)
        assert isinstance(lower, bool)

        self.with_unknown = with_unknown
        self.max_seq_len = max_seq_len
        self.lower = lower
        self.start_end_same = start_end_same

        self.update_dict()

    def update_dict(self):
        start_token = '<SOS>'
        end_token = '<EOS>'
        unknown_token = '<UKN>'
        padding_token = '<PAD>'

        # unknown
        self.unknown_idx = None
        if self.with_unknown:
            self.idx2char.append(unknown_token)
            self.unknown_idx = len(self.idx2char) - 1

        # SOS/EOS
        self.idx2char.append(start_token)
        self.start_idx = len(self.idx2char) - 1
        if not self.start_end_same:
            self.idx2char.append(end_token)
        self.end_idx = len(self.idx2char) - 1

        # padding
        self.idx2char.append(padding_token)
        self.padding_idx = len(self.idx2char) - 1

        # update char2idx
        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx