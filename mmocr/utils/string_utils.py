# Copyright (c) OpenMMLab. All rights reserved.
class StringStripper:
    """Removing the leading and/or the trailing characters based on the string
    argument passed.

    Args:
        strip (bool): Whether remove characters from both left and right of
            the string. Default: True.
        strip_pos (str): Which position for removing, can be one of
            ('both', 'left', 'right'), Default: 'both'.
        strip_str (str|None): A string specifying the set of characters
            to be removed from the left and right part of the string.
            If None, all leading and trailing whitespaces
            are removed from the string. Default: None.
    """

    def __init__(self, strip=True, strip_pos='both', strip_str=None):
        assert isinstance(strip, bool)
        assert strip_pos in ('both', 'left', 'right')
        assert strip_str is None or isinstance(strip_str, str)

        self.strip = strip
        self.strip_pos = strip_pos
        self.strip_str = strip_str

    def __call__(self, in_str):

        if not self.strip:
            return in_str

        if self.strip_pos == 'left':
            return in_str.lstrip(self.strip_str)
        elif self.strip_pos == 'right':
            return in_str.rstrip(self.strip_str)
        else:
            return in_str.strip(self.strip_str)
