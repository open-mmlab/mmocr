# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

from mmocr.utils import dump_ocr_data


def convert_annotations(root_path, split):
    """Convert original annotations to mmocr format.

    The annotation format is as the following:
        word_1.png, "flying"
        word_2.png, "today"
        word_3.png, "means"
    See the format of converted annotation in mmocr.utils.dump_ocr_data.

    Args:
        root_path (str): The root path of the dataset
        split (str): The split of dataset. Namely: training or test
    """
    assert isinstance(root_path, str)
    assert isinstance(split, str)

    img_info = []
    with open(
            osp.join(root_path, 'annotations',
                     f'Challenge2_{split}_Task3_GT.txt'),
            encoding='"utf-8-sig') as f:
        annos = f.readlines()
    for anno in annos:
        seg = ' ' if split == 'Test1015' else ', "'
        # text may contain comma ','
        dst_img_name, word = anno.split(seg)
        word = word.replace('"\n', '')

        img_info.append({
            'file_name': osp.basename(dst_img_name),
            'anno_info': [{
                'text': word
            }]
        })

    return img_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and test set of IC13')
    parser.add_argument('root_path', help='Root dir path of IC13')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path

    for split in ['Train', 'Test', 'Test1015']:
        img_info = convert_annotations(root_path, split)
        dump_ocr_data(img_info,
                      osp.join(root_path, f'{split.lower()}_label.json'),
                      'textrecog')
        print(f'{split} split converted.')


if __name__ == '__main__':
    main()
