import os

import mmcv


def drop_orientation(img_file):
    """Check if the image has orientation information. If yes, ignore it by
    converting the image format to png, and return new filename, otherwise
    return the original filename.

    Args:
        img_file(str): The image path

    Returns:
        The converted image filename with proper postfix
    """
    assert isinstance(img_file, str)
    assert img_file

    # read imgs with ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')
    # read imgs with orientations as dataloader does when training and testing
    img_color = mmcv.imread(img_file, 'color')
    # make sure imgs have no orientation info, or annotation gt is wrong.
    if img.shape[:2] == img_color.shape[:2]:
        return img_file

    target_file = os.path.splitext(img_file)[0] + '.png'
    # read img with ignoring orientation information
    img = mmcv.imread(img_file, 'unchanged')
    mmcv.imwrite(img, target_file)
    os.remove(img_file)
    print(f'{img_file} has orientation info. Ignore it by converting to png')
    return target_file


def is_not_png(img_file):
    """Check img_file is not png image.

    Args:
        img_file(str): The input image file name

    Returns:
        The bool flag indicating whether it is not png
    """
    assert isinstance(img_file, str)
    assert img_file

    suffix = os.path.splitext(img_file)[1]

    return suffix not in ['.PNG', '.png']
