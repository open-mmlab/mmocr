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
        i = 1
        for image_node in root.findall('image'):
            image_name = image_node.find('imageName').text
            print(f'[{i}/{total_img_num}] Process image: {image_name}')
            i += 1
            lexicon = image_node.find('lex').text.lower()
            lexicon_list = lexicon.split(',')
            lex_size = len(lexicon_list)
            src_img = cv2.imread(osp.join(src_image_root, image_name))
            for rectangle in image_node.find('taggedRectangles'):
                x = int(rectangle.get('x'))
                y = int(rectangle.get('y'))
                w = int(rectangle.get('width'))
                h = int(rectangle.get('height'))
                rb, re = max(0, y), max(0, y + h)
                cb, ce = max(0, x), max(0, x + w)
                dst_img = src_img[rb:re, cb:ce]
                text_label = rectangle.find('tag').text.lower()
                if args.resize:
                    dst_img = cv2.resize(dst_img, (args.width, args.height))
                dst_img_name = f'img_{index:04}' + '.jpg'
                index += 1
                dst_img_path = osp.join(dst_image_root, dst_img_name)
                cv2.imwrite(dst_img_path, dst_img)
                fw.write(f'{osp.basename(dst_image_root)}/{dst_img_name} '
                         f'{text_label} {lex_size} {lexicon}\n')

    print(f'Finish to generate svt testset, '
          f'with label file {dst_label_file}')


if __name__ == '__main__':
    main()
