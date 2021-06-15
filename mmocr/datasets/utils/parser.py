import json

from mmocr.datasets.builder import PARSERS


@PARSERS.register_module()
class LineStrParser:
    """Parse string of one line in annotation file to dict format.

    Args:
        keys (list[str]): Keys in result dict.
        keys_idx (list[int]): Value index in sub-string list
            for each key above.
        separator (str): Separator to separate string to list of sub-string.
    """

    def __init__(self,
                 keys=['filename', 'text'],
                 keys_idx=[0, 1],
                 separator=' '):
        assert isinstance(keys, list)
        assert isinstance(keys_idx, list)
        assert isinstance(separator, str)
        assert len(keys) > 0
        assert len(keys) == len(keys_idx)
        self.keys = keys
        self.keys_idx = keys_idx
        self.separator = separator

    def get_item(self, data_ret, index):
        map_index = index % len(data_ret)
        line_str = data_ret[map_index]
        line_str = line_str.split(self.separator)
        if len(line_str) <= max(self.keys_idx):
            raise Exception(
                f'key index: {max(self.keys_idx)} out of range: {line_str}')

        line_info = {}
        for i, key in enumerate(self.keys):
            line_info[key] = line_str[self.keys_idx[i]]
        return line_info


@PARSERS.register_module()
class LineJsonParser:
    """Parse json-string of one line in annotation file to dict format.

    Args:
        keys (list[str]): Keys in both json-string and result dict.
    """

    def __init__(self, keys=[], **kwargs):
        assert isinstance(keys, list)
        assert len(keys) > 0
        self.keys = keys

    def get_item(self, data_ret, index):
        map_index = index % len(data_ret)
        line_json_obj = json.loads(data_ret[map_index])
        line_info = {}
        for key in self.keys:
            if key not in line_json_obj:
                raise Exception(f'key {key} not in line json {line_json_obj}')
            line_info[key] = line_json_obj[key]

        return line_info
