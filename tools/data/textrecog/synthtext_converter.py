import argparse
import os
from functools import partial

import mmcv
import numpy as np
from scipy.io import loadmat


def parse_args():
    parser = argparse.ArgumentParser(
        description='Crop images in Synthtext-style dataset in '
        'prepration for MMOCR\'s use')
    parser.add_argument(
        'anno_path', help='Path to gold annotation data (gt.mat)')
    parser.add_argument('img_path', help='Path to images')
    parser.add_argument('out_dir', help='Path of output images and labels')
    parser.add_argument(
        '--n_proc',
        default=1,
        type=int,
        help='Number of processes to run with')
    args = parser.parse_args()
    return args


def load_gt_datum(datum):
    img_path, txt, wordBB, charBB = datum
    words = []
    word_bboxes = []
    char_bboxes = []

    # when there's only one word in txt
    # scipy will load it as a string
    if type(txt) is str:
        words = txt.split()
    else:
        for line in txt:
            words += line.split()

    # From (2, 4, num_boxes) to (num_boxes, 4, 2)
    if len(wordBB.shape) == 2:
        wordBB = wordBB[:, :, np.newaxis]
    cur_wordBB = wordBB.transpose(2, 1, 0)
    for box in cur_wordBB:
        word_bboxes.append(
            [max(round(coord), 0) for pt in box for coord in pt])

    # Validate word bboxes.
    if len(words) != len(word_bboxes):
        return

    # From (2, 4, num_boxes) to (num_boxes, 4, 2)
    cur_charBB = charBB.transpose(2, 1, 0)
    for box in cur_charBB:
        char_bboxes.append(
            [max(round(coord), 0) for pt in box for coord in pt])

    char_bbox_idx = 0
    char_bbox_grps = []

    for word in words:
        temp_bbox = char_bboxes[char_bbox_idx:char_bbox_idx + len(word)]
        char_bbox_idx += len(word)
        char_bbox_grps.append(temp_bbox)

    # Validate char bboxes.
    # If the length of the last char bbox is correct, then
    # all the previous bboxes are also valid
    if len(char_bbox_grps[len(words) - 1]) != len(words[-1]):
        return

    return img_path, words, word_bboxes, char_bbox_grps


def load_gt_data(filename, n_proc):
    mat_data = loadmat(filename, simplify_cells=True)
    imnames = mat_data['imnames']
    txt = mat_data['txt']
    wordBB = mat_data['wordBB']
    charBB = mat_data['charBB']
    return mmcv.track_parallel_progress(
        load_gt_datum, list(zip(imnames, txt, wordBB, charBB)), nproc=n_proc)


def process(data, img_path_prefix, out_dir):
    if data is None:
        return
    # Dirty hack for multi-processing
    img_path, words, word_bboxes, char_bbox_grps = data
    img_dir, img_name = os.path.split(img_path)
    img_name = os.path.splitext(img_name)[0]
    input_img = mmcv.imread(os.path.join(img_path_prefix, img_path))

    output_sub_dir = os.path.join(out_dir, img_dir)
    if not os.path.exists(output_sub_dir):
        try:
            os.makedirs(output_sub_dir)
        except FileExistsError:
            pass  # occurs when multi-proessing

    for i, word in enumerate(words):
        output_image_patch_name = f'{img_name}_{i}.png'
        output_label_name = f'{img_name}_{i}.txt'
        output_image_patch_path = os.path.join(output_sub_dir,
                                               output_image_patch_name)
        output_label_path = os.path.join(output_sub_dir, output_label_name)
        if os.path.exists(output_image_patch_path) and os.path.exists(
                output_label_path):
            continue

        word_bbox = word_bboxes[i]
        min_x, max_x = min(word_bbox[::2]), max(word_bbox[::2])
        min_y, max_y = min(word_bbox[1::2]), max(word_bbox[1::2])
        cropped_img = input_img[min_y:max_y, min_x:max_x]
        if cropped_img.shape[0] <= 0 or cropped_img.shape[1] <= 0:
            continue

        char_bbox_grp = np.array(char_bbox_grps[i])
        char_bbox_grp[:, ::2] -= min_x
        char_bbox_grp[:, 1::2] -= min_y

        mmcv.imwrite(cropped_img, output_image_patch_path)
        with open(output_label_path, 'w') as output_label_file:
            output_label_file.write(word + '\n')
            for cbox in char_bbox_grp:
                output_label_file.write('%d %d %d %d %d %d %d %d\n' %
                                        tuple(cbox.tolist()))


def main():
    args = parse_args()
    print('Loading annoataion data...')
    data = load_gt_data(args.anno_path, args.n_proc)
    process_with_outdir = partial(
        process, img_path_prefix=args.img_path, out_dir=args.out_dir)
    print('Creating cropped images and gold labels...')
    mmcv.track_parallel_progress(process_with_outdir, data, nproc=args.n_proc)
    print('Done')


if __name__ == '__main__':
    main()
