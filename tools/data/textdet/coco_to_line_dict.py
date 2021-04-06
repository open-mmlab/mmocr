import argparse
import codecs
import json


def read_json(fpath):
    with codecs.open(fpath, 'r', 'utf-8') as f:
        obj = json.load(f)
    return obj


def parse_coco_json(in_path):
    json_obj = read_json(in_path)
    image_infos = json_obj['images']
    annotations = json_obj['annotations']
    imgid2imgname = {}
    img_ids = []
    for image_info in image_infos:
        imgid2imgname[image_info['id']] = image_info
        img_ids.append(image_info['id'])
    imgid2anno = {}
    for img_id in img_ids:
        imgid2anno[img_id] = []
    for anno in annotations:
        img_id = anno['image_id']
        new_anno = {}
        new_anno['iscrowd'] = anno['iscrowd']
        new_anno['category_id'] = anno['category_id']
        new_anno['bbox'] = anno['bbox']
        new_anno['segmentation'] = anno['segmentation']
        if img_id in imgid2anno.keys():
            imgid2anno[img_id].append(new_anno)

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
    parser.add_argument('--in-path', help='input json path with coco format')
    parser.add_argument(
        '--out-path', help='output txt path with line-json format')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    imgid2imgname, imgid2anno = parse_coco_json(args.in_path)
    gen_line_dict_file(args.out_path, imgid2imgname, imgid2anno)
    print('finish')


if __name__ == '__main__':
    main()
