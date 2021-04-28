from mmocr.models.builder import CONVERTORS


@CONVERTORS.register_module()
class BIOConvertor:
    """Convert between text, index and tensor for text recognize pipeline.

    Args:
        dict_type (str): Type of dict, should be either 'DICT36' or 'DICT90'.
        dict_file (None|str): Character dict file path. If not none,
            the dict_file is of higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, the list
            is of higher priority than dict_type, but lower than dict_file.
    """
    start_idx = end_idx = padding_idx = 0
    unknown_idx = None
    lower = False


    def __init__(self, dict_type='DICT90', dict_file=None, dict_list=None):
        assert dict_type in ('DICT36', 'DICT90')
        assert dict_file is None or isinstance(dict_file, str)
        self.idx2char = []

    def num_classes(self):
        """Number of output classes."""
        return len(self.idx2char)

    def str2idx(self, strings):
        """Convert strings to indexes.

        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
        """
        assert isinstance(strings, list)

        indexes = []
        for string in strings:
            if self.lower:
                string = string.lower()
            index = []
            for char in string:
                char_idx = self.char2idx.get(char, self.unknown_idx)
                if char_idx is None:
                    raise Exception(f'Chararcter: {char} not in dict,'
                                    f' please check gt_label and use'
                                    f' custom dict file,'
                                    f' or set "with_unknown=True"')
                index.append(char_idx)
            indexes.append(index)

        return indexes

    def str2tensor(self, strings):
        """Convert text-string to input tensor.

        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            tensors (list[torch.Tensor]): [torch.Tensor([1,2,3,3,4]),
                torch.Tensor([5,4,6,3,7])].
        """
        raise NotImplementedError


