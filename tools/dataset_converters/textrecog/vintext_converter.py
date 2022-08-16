# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmcv

from mmocr.utils import crop_img, dump_ocr_data


def collect_files(img_dir, gt_dir):
    """Collect all images and their corresponding groundtruth files.

    Args:
        img_dir (str): The image directory
        gt_dir (str): The groundtruth directory

    Returns:
        files (list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir

    ann_list, imgs_list = [], []
    for img_file in os.listdir(img_dir):
        ann_file = 'gt_' + str(int(img_file[2:6])) + '.txt'
        ann_list.append(osp.join(gt_dir, ann_file))
        imgs_list.append(osp.join(img_dir, img_file))

    files = list(zip(imgs_list, ann_list))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

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
        images = mmcv.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)

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
    assert int(osp.basename(gt_file)[3:-4]) == int(
        osp.basename(img_file)[2:-4])
    # read imgs while ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')

    img_info = dict(
        file_name=osp.join(osp.basename(img_file)),
        height=img.shape[0],
        width=img.shape[1],
        segm_file=osp.join(osp.basename(gt_file)))

    if osp.splitext(gt_file)[1] == '.txt':
        img_info = load_txt_info(gt_file, img_info)
    else:
        raise NotImplementedError

    return img_info


def load_txt_info(gt_file, img_info):
    """Collect the annotation information.

    The annotation format is as the following:
    x1,y1,x2,y2,x3,y3,x4,y4,text
    118,15,147,15,148,46,118,46,LƯỢNG
    149,9,165,9,165,43,150,43,TỐT
    167,9,180,9,179,43,167,42,ĐỂ
    181,12,193,12,193,43,181,43,CÓ
    195,13,215,14,215,46,196,46,VIỆC
    217,13,237,14,239,47,217,46,LÀM,

    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    with open(gt_file, encoding='utf-8') as f:
        anno_info = []
        for line in f:
            line = line.strip('\n')
            ann = line.split(',')
            bbox = ann[0:8]
            word = line[len(','.join(bbox)) + 1:]
            bbox = [int(coord) for coord in bbox]
            # Ignore hard samples
            if word == '###':
                continue
            assert len(bbox) == 8
            anno = dict(bbox=bbox, word=word)
            anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def generate_ann(root_path, split, image_infos, preserve_vertical):
    """Generate cropped annotations and label txt file.

    Args:
        root_path (str): The root path of the dataset
        split (str): The split of dataset. Namely: training or test
        image_infos (list[dict]): A list of dicts of the img and
            annotation information
        preserve_vertical (bool): Whether to preserve vertical texts
    """
    dst_image_root = osp.join(root_path, 'crops', split)
    ignore_image_root = osp.join(root_path, 'ignores', split)
    if split == 'training':
        dst_label_file = osp.join(root_path, 'train_label.json')
    elif split == 'test':
        dst_label_file = osp.join(root_path, 'test_label.json')
    elif split == 'unseen_test':
        dst_label_file = osp.join(root_path, 'unseen_test_label.json')
    os.makedirs(dst_image_root, exist_ok=True)

    img_info = []
    for image_info in image_infos:
        index = 1
        src_img_path = osp.join(root_path, 'imgs', split,
                                image_info['file_name'])
        image = mmcv.imread(src_img_path)
        src_img_root = image_info['file_name'].split('.')[0]

        for anno in image_info['anno_info']:
            word = anno['word']
            dst_img = crop_img(image, anno['bbox'], 0, 0)
            h, w, _ = dst_img.shape

            dst_img_name = f'{src_img_root}_{index}.png'
            index += 1
            # Skip invalid annotations
            if min(dst_img.shape) == 0:
                continue
            # Skip vertical texts
            if not preserve_vertical and h / w > 2 and split == 'training':
                dst_img_path = osp.join(ignore_image_root, dst_img_name)
                mmcv.imwrite(dst_img, dst_img_path)
                continue

            dst_img_path = osp.join(dst_image_root, dst_img_name)
            mmcv.imwrite(dst_img, dst_img_path)
            img_info.append({
                'file_name': dst_img_name,
                'anno_info': [{
                    'text': word
                }]
            })
    dump_ocr_data(img_info, dst_label_file, 'textrecog')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and test set of VinText ')
    parser.add_argument('root_path', help='Root dir path of VinText')
    parser.add_argument(
        '--preserve-vertical',
        help='Preserve samples containing vertical texts',
        action='store_true')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of processes')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path
    for split in ['training', 'test', 'unseen_test']:
        print(f'Processing {split} set...')
        with mmcv.Timer(
                print_tmpl='It takes {}s to convert VinText annotation'):
            files = collect_files(
                osp.join(root_path, 'imgs', split),
                osp.join(root_path, 'annotations'))
            image_infos = collect_annotations(files, nproc=args.nproc)
            generate_ann(root_path, split, image_infos, args.preserve_vertical)


if __name__ == '__main__':
    main()
