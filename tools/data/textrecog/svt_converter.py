import argparse
import os
import os.path as osp
import xml.etree.ElementTree as ET

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate testset of svt by cropping box image.')
    parser.add_argument(
        'root_path',
        help='Root dir path of svt, where test.xml in,'
        'for example, "data/mixture/svt/svt1/"')
    parser.add_argument(
        '--resize',
        action='store_true',
        help='Whether resize cropped image to certain size.')
    parser.add_argument('--height', default=32, help='Resize height.')
    parser.add_argument('--width', default=100, help='Resize width.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path

    # inputs
    src_label_file = osp.join(root_path, 'test.xml')
    if not osp.exists(src_label_file):
        raise Exception(
            f'{src_label_file} not exists, please check and try again.')
    src_image_root = root_path

    # outputs
    dst_label_file = osp.join(root_path, 'test_label.txt')
    dst_image_root = osp.join(root_path, 'image')
    os.makedirs(dst_image_root, exist_ok=True)

    tree = ET.parse(src_label_file)
    root = tree.getroot()

    index = 1
    with open(dst_label_file, 'w', encoding='utf-8') as fw:
        total_img_num = len(root)
        for i, image in enumerate(root):
            src_img_name = image[0].text
            print(f'[{i}/{total_img_num}] Process image: {src_img_name}')
            src_img_path = osp.join(src_image_root, src_img_name)
            src_img = cv2.imread(src_img_path)
            lex = image[2].text.lower()
            for box in image[4]:
                height = int(box.attrib['height'])
                width = int(box.attrib['width'])
                lefttop_x = int(box.attrib['x'])
                lefttop_y = int(box.attrib['y'])
                text_label = box[0].text.lower()
                try:
                    dst_img = src_img[lefttop_y:(lefttop_y + height),
                                      lefttop_x:(lefttop_x + width)]
                    if args.resize:
                        dst_img = cv2.resize(dst_img,
                                             (args.width, args.height))
                    dst_img_name = 'img_' + f'{index:04}' + '.jpg'
                    index += 1
                    dst_img_path = osp.join(dst_image_root, dst_img_name)
                    cv2.imwrite(dst_img_path, dst_img)
                    fw.write(
                        osp.basename(dst_image_root) + '/' + dst_img_name +
                        ' ' + text_label + ' ' + lex + '\n')
                except Exception as e:
                    print(e)
                    continue
    print(f'Finish to generate svt testset, '
          f'with label file {dst_label_file}')


if __name__ == '__main__':
    main()
