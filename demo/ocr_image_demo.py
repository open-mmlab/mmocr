# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv
from mmdet.apis import init_detector

from mmocr.apis.inference import model_inference
from mmocr.core.visualize import det_recog_show_result
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.utils.box_util import stitch_boxes_into_lines


def det_and_recog_inference(args, det_model, recog_model):
    image_path = args.img
    end2end_res = {'filename': image_path}
    end2end_res['result'] = []

    image = mmcv.imread(image_path)
    det_result = model_inference(det_model, image)
    bboxes = det_result['boundary_result']

    box_imgs = []
    for bbox in bboxes:
        box_res = {}
        box_res['box'] = [round(x) for x in bbox[:-1]]
        box_res['box_score'] = float(bbox[-1])
        box = bbox[:8]
        if len(bbox) > 9:
            min_x = min(bbox[0:-1:2])
            min_y = min(bbox[1:-1:2])
            max_x = max(bbox[0:-1:2])
            max_y = max(bbox[1:-1:2])
            box = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
        box_img = crop_img(image, box)
        if args.batch_mode:
            box_imgs.append(box_img)
        else:
            recog_result = model_inference(recog_model, box_img)
            text = recog_result['text']
            text_score = recog_result['score']
            if isinstance(text_score, list):
                text_score = sum(text_score) / max(1, len(text))
            box_res['text'] = text
            box_res['text_score'] = text_score

        end2end_res['result'].append(box_res)

    if args.batch_mode:
        batch_size = args.batch_size
        for chunk_idx in range(len(box_imgs) // batch_size + 1):
            start_idx = chunk_idx * batch_size
            end_idx = (chunk_idx + 1) * batch_size
            chunk_box_imgs = box_imgs[start_idx:end_idx]
            if len(chunk_box_imgs) == 0:
                continue
            recog_results = model_inference(
                recog_model, chunk_box_imgs, batch_mode=True)
            for i, recog_result in enumerate(recog_results):
                text = recog_result['text']
                text_score = recog_result['score']
                if isinstance(text_score, list):
                    text_score = sum(text_score) / max(1, len(text))
                end2end_res['result'][start_idx + i]['text'] = text
                end2end_res['result'][start_idx + i]['text_score'] = text_score

    return end2end_res


def main():
    parser = ArgumentParser()
    parser.add_argument('img', type=str, help='Input Image file.')
    parser.add_argument(
        'out_file', type=str, help='Output file name of the visualized image.')
    parser.add_argument(
        '--det-config',
        type=str,
        default='./configs/textdet/psenet/'
        'psenet_r50_fpnf_600e_icdar2015.py',
        help='Text detection config file.')
    parser.add_argument(
        '--det-ckpt',
        type=str,
        default='https://download.openmmlab.com/'
        'mmocr/textdet/psenet/'
        'psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth',
        help='Text detection checkpint file (local or url).')
    parser.add_argument(
        '--recog-config',
        type=str,
        default='./configs/textrecog/sar/'
        'sar_r31_parallel_decoder_academic.py',
        help='Text recognition config file.')
    parser.add_argument(
        '--recog-ckpt',
        type=str,
        default='https://download.openmmlab.com/'
        'mmocr/textrecog/sar/'
        'sar_r31_parallel_decoder_academic-dba3a4a3.pth',
        help='Text recognition checkpint file (local or url).')
    parser.add_argument(
        '--batch-mode',
        action='store_true',
        help='Whether use batch mode for text recognition.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for text recognition inference '
        'if batch_mode is True above.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--imshow',
        action='store_true',
        help='Whether show image with OpenCV.')
    parser.add_argument(
        '--ocr-in-lines',
        action='store_true',
        help='Whether group ocr results in lines.')
    args = parser.parse_args()

    if args.device == 'cpu':
        args.device = None
    # build detect model
    detect_model = init_detector(
        args.det_config, args.det_ckpt, device=args.device)
    if hasattr(detect_model, 'module'):
        detect_model = detect_model.module
    if detect_model.cfg.data.test['type'] == 'ConcatDataset':
        detect_model.cfg.data.test.pipeline = \
            detect_model.cfg.data.test['datasets'][0].pipeline

    # build recog model
    recog_model = init_detector(
        args.recog_config, args.recog_ckpt, device=args.device)
    if hasattr(recog_model, 'module'):
        recog_model = recog_model.module
    if recog_model.cfg.data.test['type'] == 'ConcatDataset':
        recog_model.cfg.data.test.pipeline = \
            recog_model.cfg.data.test['datasets'][0].pipeline

    det_recog_result = det_and_recog_inference(args, detect_model, recog_model)
    print(f'result: {det_recog_result}')
    mmcv.dump(
        det_recog_result,
        args.out_file + '.json',
        ensure_ascii=False,
        indent=4)

    if args.ocr_in_lines:
        res = det_recog_result['result']
        res = stitch_boxes_into_lines(res, 10, 0.5)
        det_recog_result['result'] = res
        mmcv.dump(
            det_recog_result,
            args.out_file + '.line.json',
            ensure_ascii=False,
            indent=4)

    img = det_recog_show_result(args.img, det_recog_result)
    mmcv.imwrite(img, args.out_file)
    if args.imshow:
        mmcv.imshow(img, 'predicted results')


if __name__ == '__main__':
    main()
