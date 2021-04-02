import os
import os.path as osp

import mmcv


def check_ignore_orientation(img_file):
    """Check if the image has orientation information.

    If yes, ignore it by converting the image format to png, otherwise return
    the original filename.

    Args:
        img_file(str): The image path

    Returns:
        The converted image filename with proper postfix
    """
    assert isinstance(img_file, str)
    assert img_file

    # read imgs with ignoring oritations
    img = mmcv.imread(img_file, 'unchanged')
    # read imgs with oritations as dataloader does when training and testing
    img_color = mmcv.imread(img_file, 'color')
    # make sure imgs have no oritation info, or annotation gt is wrong.
    if img.shape[:2] == img_color.shape[:2]:
        return img_file
    else:
        target_file = osp.splitext(img_file)[0] + '.png'
        # read img with ignoring orientation information
        img = mmcv.imread(img_file, 'unchanged')
        mmcv.imwrite(img, target_file)
        os.remove(img_file)
        print(
            f'{img_file} has orientation info. Ingore it by converting to png')
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

    suffix = osp.splitext(img_file)[1]

    return (suffix not in ['.PNG', '.png'])


def convert_annotations(image_infos, out_json_name):
    """Convert the annotion into coco style.

    Args:
        image_infos(list): The list of image information dicts
        out_json_name(str): The output json filename

    Returns:
        out_json(dict): The coco style dict
    """
    assert isinstance(image_infos, list)
    assert isinstance(out_json_name, str)
    assert out_json_name

    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    for image_info in image_infos:
        image_info['id'] = img_id
        anno_infos = image_info.pop('anno_info')
        out_json['images'].append(image_info)
        for anno_info in anno_infos:
            anno_info['image_id'] = img_id
            anno_info['id'] = ann_id
            out_json['annotations'].append(anno_info)
            ann_id += 1
            # if image_info['file_name'].find('png'):
            #    img = mmcv.imread('data/ctw1500/imgs/'+
            #        image_info['file_name'], 'color')
            #    show_img_boundary(img, anno_info['segmentation'] )
        img_id += 1
        # print(img_id)
    cat = dict(id=1, name='text')
    out_json['categories'].append(cat)

    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')
    mmcv.dump(out_json, out_json_name)

    return out_json
