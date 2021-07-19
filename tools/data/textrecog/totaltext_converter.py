import argparse
import glob
import os
import os.path as osp
import re

import mmcv
import numpy as np
import scipy.io as scio
import yaml
from shapely.geometry import Polygon

from mmocr.datasets.pipelines.crop import crop_img
from mmocr.utils.fileio import list_to_file


def collect_files(img_dir, gt_dir, split):
    """Collect all images and their corresponding groundtruth files.

    Args:
        img_dir(str): The image directory
        gt_dir(str): The groundtruth directory
        split(str): The split of dataset. Namely: training or test
    Returns:
        files(list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir

    # note that we handle png and jpg only. Pls convert others such as gif to
    # jpg or png offline
    suffixes = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']
    # suffixes = ['.png']

    imgs_list = []
    for suffix in suffixes:
        imgs_list.extend(glob.glob(osp.join(img_dir, '*' + suffix)))

    imgs_list = sorted(imgs_list)
    ann_list = sorted(
        [osp.join(gt_dir, gt_file) for gt_file in os.listdir(gt_dir)])

    files = [(img_file, gt_file)
             for (img_file, gt_file) in zip(imgs_list, ann_list)]
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, nproc=1):
    """Collect the annotation information.

    Args:
        files(list): The list of tuples (image_file, groundtruth_file)
        nproc(int): The number of process to collect annotations
    Returns:
        images(list): The list of image information dicts
    """
    assert isinstance(files, list)
    assert isinstance(nproc, int)

    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)

    return images


def get_contours_mat(gt_path):
    """Get the contours and words for each ground_truth mat file.

    Args:
        gt_path(str): The relative path of the ground_truth mat file
    Returns:
        contours(list[lists]): A list of lists of contours
        for the text instances
        words(list[list]): A list of lists of words (string)
        for the text instances
    """
    assert isinstance(gt_path, str)

    contours = []
    words = []
    data = scio.loadmat(gt_path)
    data_polygt = data['polygt']

    for i, lines in enumerate(data_polygt):
        X = np.array(lines[1])
        Y = np.array(lines[3])

        point_num = len(X[0])
        word = lines[4]
        if len(word) == 0:
            word = '???'
        else:
            word = word[0]

        if word == '#':
            word = '###'
            continue

        words.append(word)

        arr = np.concatenate([X, Y]).T
        contour = []
        for i in range(point_num):
            contour.append(arr[i][0])
            contour.append(arr[i][1])
        contours.append(np.asarray(contour))

    return contours, words


def load_mat_info(img_info, gt_file):
    """Load the information of one ground truth in .mat format.

    Args:
        img_info(dict): The dict of only the image information
        gt_file(str): The relative path of the ground_truth mat
        file for one image
    Returns:
        img_info(dict): The dict of the img and annotation information
    """
    assert isinstance(img_info, dict)
    assert isinstance(gt_file, str)

    contours, words = get_contours_mat(gt_file)
    anno_info = []
    for contour, word in zip(contours, words):
        if contour.shape[0] == 2:
            continue
        coordinates = np.array(contour).reshape(-1, 2)
        polygon = Polygon(coordinates)

        # convert to COCO style XYWH format
        min_x, min_y, max_x, max_y = polygon.bounds
        bbox = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
        anno = dict(word=word, bbox=bbox)
        anno_info.append(anno)

    img_info.update(anno_info=anno_info)
    return img_info


def process_line(line, contours, words):
    """Get the contours and words by processing each line in the gt file.

    Args:
        line(str): The line in gt file containing annotation info
        contours(list[lists]): A list of lists of contours
        for the text instances
        words(list[list]): A list of lists of words (string)
        for the text instances
    Returns:
        contours(list[lists]): A list of lists of contours
        for the text instances
        words(list[list]): A list of lists of words (string)
        for the text instances
    """

    line = '{' + line.replace('[[', '[').replace(']]', ']') + '}'
    ann_dict = re.sub('([0-9]) +([0-9])', r'\1,\2', line)
    ann_dict = re.sub('([0-9]) +([ 0-9])', r'\1,\2', ann_dict)
    ann_dict = re.sub('([0-9]) -([0-9])', r'\1,-\2', ann_dict)
    ann_dict = ann_dict.replace("[u',']", "[u'#']")
    ann_dict = yaml.load(ann_dict)

    X = np.array([ann_dict['x']])
    Y = np.array([ann_dict['y']])

    if len(ann_dict['transcriptions']) == 0:
        word = '???'
    else:
        word = ann_dict['transcriptions'][0]
        if len(ann_dict['transcriptions']) > 1:
            for ann_word in ann_dict['transcriptions'][1:]:
                word += ',' + ann_word
        word = str(eval(word))
    words.append(word)

    point_num = len(X[0])

    arr = np.concatenate([X, Y]).T
    contour = []
    for i in range(point_num):
        contour.append(arr[i][0])
        contour.append(arr[i][1])
    contours.append(np.asarray(contour))

    return contours, words


def get_contours_txt(gt_path):
    """Get the contours and words for each ground_truth txt file.

    Args:
        gt_path(str): The relative path of the ground_truth mat file
    Returns:
        contours(list[lists]): A list of lists of contours
        for the text instances
        words(list[list]): A list of lists of words (string)
        for the text instances
    """
    assert isinstance(gt_path, str)

    contours = []
    words = []

    with open(gt_path, 'r') as f:
        tmp_line = ''
        for idx, line in enumerate(f):
            line = line.strip()
            if idx == 0:
                tmp_line = line
                continue
            if not line.startswith('x:'):
                tmp_line += ' ' + line
                continue
            else:
                complete_line = tmp_line
                tmp_line = line
            contours, words = process_line(complete_line, contours, words)

        if tmp_line != '':
            contours, words = process_line(tmp_line, contours, words)

        for word in words:

            if word == '#':
                word = '###'
                continue

    return contours, words


def load_txt_info(gt_file, img_info):
    """Load the information of one ground truth in .txt format.

    Args:
        img_info(dict): The dict of only the image information
        gt_file(str): The relative path of the ground_truth mat
        file for one image
    Returns:
        img_info(dict): The dict of the img and annotation information
    """

    contours, words = get_contours_txt(gt_file)
    anno_info = []
    for contour, word in zip(contours, words):
        if contour.shape[0] == 2:
            continue
        coordinates = np.array(contour).reshape(-1, 2)
        polygon = Polygon(coordinates)

        # convert to COCO style XYWH format
        min_x, min_y, max_x, max_y = polygon.bounds
        bbox = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
        anno = dict(word=word, bbox=bbox)
        anno_info.append(anno)

    img_info.update(anno_info=anno_info)
    return img_info


def generate_ann(root_path, split, image_infos):
    """Generate cropped annotations and label txt file.

    Args:
        root_path(str): The relative path of the totaltext file
        split(str): The split of dataset. Namely: training or test
        image_infos(list[dict]): A list of dicts of the img and
        annotation information
    """

    dst_image_root = osp.join(root_path, 'dst_imgs', split)
    if split == 'training':
        dst_label_file = osp.join(root_path, 'train_label.txt')
    elif split == 'test':
        dst_label_file = osp.join(root_path, 'test_label.txt')
    os.makedirs(dst_image_root, exist_ok=True)

    lines = []
    for image_info in image_infos:
        index = 1
        src_img_path = osp.join(root_path, 'imgs', image_info['file_name'])
        image = mmcv.imread(src_img_path)
        src_img_root = osp.splitext(image_info['file_name'])[0].split('/')[1]

        for anno in image_info['anno_info']:
            word = anno['word']
            dst_img = crop_img(image, anno['bbox'])
            dst_img_name = f'{src_img_root}_{index}.png'
            index += 1
            dst_img_path = osp.join(dst_image_root, dst_img_name)
            mmcv.imwrite(dst_img, dst_img_path)
            lines.append(f'{osp.basename(dst_image_root)}/{dst_img_name} '
                         f'{word}')
    list_to_file(dst_label_file, lines)


def load_img_info(files):
    """Load the information of one image.

    Args:
        files(tuple): The tuple of (img_file, groundtruth_file)
    Returns:
        img_info(dict): The dict of the img and annotation information
    """
    assert isinstance(files, tuple)

    img_file, gt_file = files
    # read imgs with ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')

    split_name = osp.basename(osp.dirname(img_file))
    img_info = dict(
        # remove img_prefix for filename
        file_name=osp.join(split_name, osp.basename(img_file)),
        height=img.shape[0],
        width=img.shape[1],
        # anno_info=anno_info,
        segm_file=osp.join(split_name, osp.basename(gt_file)))

    if osp.splitext(gt_file)[1] == '.mat':
        img_info = load_mat_info(img_info, gt_file)
    elif osp.splitext(gt_file)[1] == '.txt':
        img_info = load_txt_info(gt_file, img_info)
    else:
        raise NotImplementedError

    return img_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert totaltext annotations to COCO format')
    parser.add_argument('root_path', help='totaltext root path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--split-list',
        nargs='+',
        help='a list of splits. e.g., "--split_list training test"')

    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path
    out_dir = args.out_dir if args.out_dir else root_path
    mmcv.mkdir_or_exist(out_dir)

    img_dir = osp.join(root_path, 'imgs')
    gt_dir = osp.join(root_path, 'annotations')

    set_name = {}
    for split in args.split_list:
        set_name.update({split: 'instances_' + split + '.json'})
        assert osp.exists(osp.join(img_dir, split))

    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(
                print_tmpl='It takes {}s to convert totaltext annotation'):
            files = collect_files(
                osp.join(img_dir, split), osp.join(gt_dir, split), split)
            image_infos = collect_annotations(files, nproc=args.nproc)
            generate_ann(root_path, split, image_infos)


if __name__ == '__main__':
    main()
