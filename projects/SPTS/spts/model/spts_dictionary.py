# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmocr.models.common.dictionary import Dictionary
from mmocr.registry import TASK_UTILS


@TASK_UTILS.register_module()
class SPTSDictionary(Dictionary):
    """The class generates a dictionary for recognition. It pre-defines four
    special tokens: ``start_token``, ``end_token``, ``pad_token``, and
    ``unknown_token``, which will be sequentially placed at the end of the
    dictionary when their corresponding flags are True.

    Args:
        dict_file (str): The path of Character dict file which a single
            character must occupies a line.
        num_bins (int): Number of bins dividing the image, which is used to
            shift the character indexes. Defaults to 1000.
        with_start (bool): The flag to control whether to include the start
            token. Defaults to False.
        with_end (bool): The flag to control whether to include the end token.
            Defaults to False.
        with_seq end (bool): The flag to control whether to include the
            sequence end token. Defaults to False.
        same_start_end (bool): The flag to control whether the start token and
            end token are the same. It only works when both ``with_start`` and
            ``with_end`` are True. Defaults to False.
        with_padding (bool):The padding token may represent more than a
            padding. It can also represent tokens like the blank token in CTC
            or the background token in SegOCR. Defaults to False.
        with_unknown (bool): The flag to control whether to include the
            unknown token. Defaults to False.
        start_token (str): The start token as a string. Defaults to '<BOS>'.
        end_token (str): The end token as a string. Defaults to '<EOS>'.
        seq_end_token (str): The sequence end token as a string. Defaults to
            '<SEQEOS>'.
        start_end_token (str): The start/end token as a string. if start and
            end is the same. Defaults to '<BOS/EOS>'.
        padding_token (str): The padding token as a string.
            Defaults to '<PAD>'.
        unknown_token (str, optional): The unknown token as a string. If it's
            set to None and ``with_unknown`` is True, the unknown token will be
            skipped when converting string to index. Defaults to '<UKN>'.
    """

    def __init__(
        self,
        dict_file: str,
        num_bins: int = 1000,
        with_start: bool = False,
        with_end: bool = False,
        with_seq_end: bool = False,
        same_start_end: bool = False,
        with_padding: bool = False,
        with_unknown: bool = False,
        start_token: str = '<BOS>',
        end_token: str = '<EOS>',
        seq_end_token: str = '<SEQEOS>',
        start_end_token: str = '<BOS/EOS>',
        padding_token: str = '<PAD>',
        unknown_token: str = '<UKN>',
    ) -> None:
        self.with_seq_end = with_seq_end
        self.seq_end_token = seq_end_token

        super().__init__(
            dict_file=dict_file,
            with_start=with_start,
            with_end=with_end,
            same_start_end=same_start_end,
            with_padding=with_padding,
            with_unknown=with_unknown,
            start_token=start_token,
            end_token=end_token,
            start_end_token=start_end_token,
            padding_token=padding_token,
            unknown_token=unknown_token)

        self.num_bins = num_bins
        self._shift_idx()

    @property
    def num_classes(self) -> int:
        """int: Number of output classes. Special tokens are counted.
        """
        return len(self._dict) + self.num_bins

    def _shift_idx(self):
        idx_terms = [
            'start_idx', 'end_idx', 'unknown_idx', 'seq_end_idx', 'padding_idx'
        ]
        for term in idx_terms:
            value = getattr(self, term)
            if value:
                setattr(self, term, value + self.num_bins)
        for char in self._dict:
            self._char2idx[char] += self.num_bins

    def _update_dict(self):
        """Update the dict with tokens according to parameters."""
        # BOS/EOS
        self.start_idx = None
        self.end_idx = None
        # unknown
        self.unknown_idx = None
        # TODO: Check if this line in Dictionary is correct and
        # work as expected
        # if self.with_unknown and self.unknown_token is not None:
        if self.with_unknown:
            self._dict.append(self.unknown_token)
            self.unknown_idx = len(self._dict) - 1

        if self.with_start and self.with_end and self.same_start_end:
            self._dict.append(self.start_end_token)
            self.start_idx = len(self._dict) - 1
            self.end_idx = self.start_idx
            if self.with_seq_end:
                self._dict.append(self.seq_end_token)
                self.seq_end_idx = len(self.dict) - 1
        else:
            if self.with_end:
                self._dict.append(self.end_token)
                self.end_idx = len(self._dict) - 1
            if self.with_seq_end:
                self._dict.append(self.seq_end_token)
                self.seq_end_idx = len(self.dict) - 1
            if self.with_start:
                self._dict.append(self.start_token)
                self.start_idx = len(self._dict) - 1

        # padding
        self.padding_idx = None
        if self.with_padding:
            self._dict.append(self.padding_token)
            self.padding_idx = len(self._dict) - 1

        # update char2idx
        self._char2idx = {}
        for idx, char in enumerate(self._dict):
            self._char2idx[char] = idx

    def idx2str(self, index: Sequence[int]) -> str:
        """Convert a list of index to string.

        Args:
            index (list[int]): The list of indexes to convert to string.

        Return:
            str: The converted string.
        """
        assert isinstance(index, (list, tuple))
        string = ''
        for i in index:
            assert i < self.num_classes, f'Index: {i} out of range! Index ' \
                                        f'must be less than {self.num_classes}'
            # TODO: find its difference from ignore_chars
            # in TextRecogPostprocessor
            shifted_i = i - self.num_bins
            if self._dict[shifted_i] is not None:
                string += self._dict[shifted_i]
        return string
