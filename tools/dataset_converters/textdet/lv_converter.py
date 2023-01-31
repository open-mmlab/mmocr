# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv

import mmengine
from mmocr.utils import dump_ocr_data


def collect_files(data_dir):
    """Collect all images and their corresponding groundtruth files.

    Args:
        data_dir (str): The directory to dataset

    Returns:
        files (list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(data_dir, str)
    assert data_dir

    ann_list, imgs_list = [], []
    for video_dir in os.listdir(data_dir):
        for frame_dir in os.listdir(osp.join(data_dir, video_dir)):
            crt_dir = osp.join(data_dir, video_dir, frame_dir)
            if not osp.isdir(crt_dir):
                continue
            for crt_file in os.listdir(crt_dir):
                if crt_file.endswith('xml'):
                    ann_path = osp.join(crt_dir, crt_file)
                    img_path = osp.join(crt_dir,
                                        crt_file.replace('xml', 'png'))
                    if os.path.exists(img_path):
                        ann_list.append(ann_path)
                        imgs_list.append(img_path)
                    else:
                        continue

    files = list(zip(imgs_list, ann_list))
    assert len(files), f'No images found in {data_dir}'
    print(f'Loaded {len(files)} images from {data_dir}')

    return files


def collect_annotations(files, nproc=1):
    """Collect the annotation information.

    Args:
        files (list): The list of tuples (image_file, groundtruth_file)
        nproc (int): The number of process to collect annotations

    Returns:
        images (list): The list of image information dicts
    """
    assert isinstance(files, list)
    assert isinstance(nproc, int)

    if nproc > 1:
        images = mmengine.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmengine.track_progress(load_img_info, files)

    return images


def load_img_info(files):
    """Load the information of one image.

    Args:
        files (tuple): The tuple of (img_file, groundtruth_file)

    Returns:
        img_info (dict): The dict of the img and annotation information
    """
    assert isinstance(files, tuple)

    img_file, gt_file = files
    assert osp.basename(gt_file).split('.')[0] == osp.basename(img_file).split(
        '.')[0]
    # read imgs while ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')
    img_file = os.path.split(img_file)[-1]

    img_info = dict(
        file_name=img_file,
        height=img.shape[0],
        width=img.shape[1],
        segm_file=osp.join(osp.basename(gt_file)))

    if osp.splitext(gt_file)[1] == '.xml':
        img_info = load_xml_info(gt_file, img_info)
    else:
        raise NotImplementedError

    return img_info


def load_xml_info(gt_file, img_info):
    """Collect the annotation information.

    The annotation format is as the following:
    <annotation>
        <object>
            <name>hierarchy</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>657</xmin>
                <ymin>467</ymin>
                <xmax>839</xmax>
                <ymax>557</ymax>
            </bndbox>
        </object>
    </annotation>

    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    obj = ET.parse(gt_file)
    root = obj.getroot()
    anno_info = []
    for obj in root.iter('object'):
        x = max(0, int(obj.find('bndbox').find('xmin').text))
        y = max(0, int(obj.find('bndbox').find('ymin').text))
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)

        w, h = abs(xmax - x), abs(ymax - y)
        bbox = [x, y, w, h]
        segmentation = [x, y, x + w, y, x + w, y + h, x, y + h]

        anno = dict(
            iscrowd=0,
            category_id=1,
            bbox=bbox,
            area=w * h,
            segmentation=[segmentation])
        anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training, val and test set of Lecture Video DB ')
    parser.add_argument('root_path', help='Root dir path of Lecture Video DB')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path

    for split in ['train', 'val', 'test']:
        print(f'Processing {split} set...')
        with mmengine.Timer(
                print_tmpl='It takes {}s to convert LV annotation'):
            files = collect_files(osp.join(root_path, 'imgs', split))
            image_infos = collect_annotations(files, nproc=args.nproc)
            dump_ocr_data(image_infos,
                          osp.join(root_path, 'instances_' + split + '.json'),
                          'textdet')


if __name__ == '__main__':
    main()
