from typing import List, Tuple, Union


def find_other_label_name_of_biolabel(classes: Union[List[str], Tuple[str]]):
    """Find the original name of BIO label `O`

    Args:
        classes (List[str]): The list or tuple of class_names.
    """
    valid_other_label_names = ('other', 'Other', 'OTHER')
    for c in classes:
        if c in valid_other_label_names:
            return c
    return None
