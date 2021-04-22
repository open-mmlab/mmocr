from argparse import ArgumentParser

import mmcv

from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.core.det_recog_visualize import det_recog_show_result, write_json
from mmocr.datasets.pipelines.crop import crop_img


def det_and_recog_inference(args, det_model, recog_model):
    image_path = args.img
    end2end_res = {'filename': image_path}
    end2end_res['result'] = []

    image = mmcv.imread(image_path)
    det_result = model_inference(det_model, image)
    bboxes = det_result['boundary_result']

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

        recog_result = model_inference(recog_model, box_img)

        text = recog_result['text']
        text_score = recog_result['score']
        if isinstance(text_score, list):
            text_score = sum(text_score) / max(1, len(text))
        box_res['text'] = text
        box_res['text_score'] = text_score
        end2end_res['result'].append(box_res)

    return end2end_res


def main():
    parser = ArgumentParser()
    parser.add_argument('img', type=str, help='Input Image file.')
    parser.add_argument(
        'out_file', type=str, help='Output file name of the visualized image.')
    parser.add_argument(
        '--detect-config',
        type=str,
        default='./configs/textdet/psenet/'
        'psenet_r50_fpnf_600e_icdar2015.py',
        help='Text detection config file.')
    parser.add_argument(
        '--detect-ckpt',
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
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--imshow',
        action='store_true',
        help='Whether show image with OpenCV.')
    args = parser.parse_args()

    # build detect model
    detect_model = init_detector(
        args.detect_config, args.detect_ckpt, device=args.device)
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
    write_json(det_recog_result, args.out_file + '.json')

    img = det_recog_show_result(args.img, det_recog_result)
    mmcv.imwrite(img, args.out_file)
    if args.imshow:
        mmcv.imshow(img, 'predicted results')


if __name__ == '__main__':
    main()
