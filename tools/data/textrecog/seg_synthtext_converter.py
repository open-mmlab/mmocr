import argparse
import codecs
import json
import os.path as osp

import cv2


def read_json(fpath):
    with codecs.open(fpath, 'r', 'utf-8') as f:
        obj = json.load(f)
    return obj


def parse_old_label(img_prefix, in_path):
    imgid2imgname = {}
    imgid2anno = {}
    idx = 0
    with open(in_path, 'r') as fr:
        for line in fr:
            line = line.strip().split()
            img_full_path = osp.join(img_prefix, line[0])
            if not osp.exists(img_full_path):
                continue
            img = cv2.imread(img_full_path)
            h, w = img.shape[:2]
            img_info = {}
            img_info['file_name'] = line[0]
            img_info['height'] = h
            img_info['width'] = w
            imgid2imgname[idx] = img_info
            imgid2anno[idx] = []
            for i in range(len(line[1:]) // 8):
                seg = [int(x) for x in line[(1 + i * 8):(1 + (i + 1) * 8)]]
                points_x = seg[0:2:8]
                points_y = seg[1:2:9]
                box = [
                    min(points_x),
                    min(points_y),
                    max(points_x),
                    max(points_y)
                ]
                new_anno = {}
                new_anno['iscrowd'] = 0
                new_anno['category_id'] = 1
                new_anno['bbox'] = box
                new_anno['segmentation'] = [seg]
                imgid2anno[idx].append(new_anno)
            idx += 1

    return imgid2imgname, imgid2anno


def gen_line_dict_file(out_path, imgid2imgname, imgid2anno):
    # import pdb; pdb.set_trace()
    with codecs.open(out_path, 'w', 'utf-8') as fw:
        for key, value in imgid2imgname.items():
            if key in imgid2anno:
                anno = imgid2anno[key]
                line_dict = {}
                line_dict['file_name'] = value['file_name']
                line_dict['height'] = value['height']
                line_dict['width'] = value['width']
                line_dict['annotations'] = anno
                line_dict_str = json.dumps(line_dict)
                fw.write(line_dict_str + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img-prefix',
        help='image prefix, to generate full image path with "image_name"')
    parser.add_argument(
        '--in-path',
        help='mapping file of image_name and ann_file,'
        ' "image_name ann_file" in each line')
    parser.add_argument(
        '--out-path', help='output txt path with line-json format')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    imgid2imgname, imgid2anno = parse_old_label(args.img_prefix, args.in_path)
    gen_line_dict_file(args.out_path, imgid2imgname, imgid2anno)
    print('finish')


if __name__ == '__main__':
    main()
