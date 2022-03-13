# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import xml.etree.ElementTree as ET
import zipfile
from xml.etree.ElementTree import ParseError


def extract(root_path):
    idx = 0
    for language in ['English', 'Korean', 'Mixed']:
        for camera in ['Digital_Camera', 'Mobile_Phone']:
            crt_path = osp.join(root_path, 'KAIST', language, camera)
            zips = os.listdir(crt_path)
            for zip in zips:
                extracted_path = osp.join(root_path, 'tmp', zip)
                extract_zipfile(osp.join(crt_path, zip), extracted_path)
                for file in os.listdir(extracted_path):
                    if file.endswith('xml'):
                        src_ann = os.path.join(extracted_path, file)
                        # Filtering broken annotations
                        try:
                            ET.parse(src_ann)
                        except ParseError:
                            continue
                        src_img = None
                        im_names = [
                            file.replace('xml', suffix)
                            for suffix in ['jpg', 'JPG']
                        ]
                        for im in im_names:
                            im_path = osp.join(extracted_path, im)
                            if osp.exists(im_path):
                                src_img = im_path
                        if src_img:
                            shutil.move(
                                src_ann,
                                osp.join(root_path, 'annotations',
                                         str(idx).zfill(5) + '.xml'))
                            shutil.move(
                                src_img,
                                osp.join(root_path, 'imgs',
                                         str(idx).zfill(5) + '.jpg'))
                            idx += 1


def extract_zipfile(zip_path, dst_dir, delete=True):

    files = zipfile.ZipFile(zip_path)
    for file in files.namelist():
        files.extract(file, dst_dir)
    if delete:
        os.remove(zip_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Extract KAIST zips')
    parser.add_argument('root_path', help='Root path of KAIST')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path
    assert osp.exists(root_path)
    extract(root_path)
    shutil.rmtree(osp.join(args.root_path, 'tmp'))
    shutil.rmtree(osp.join(args.root_path, 'KAIST'))


if __name__ == '__main__':
    main()
