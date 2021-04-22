from argparse import ArgumentParser

import mmcv

from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.core.end2end_visualize import end2end_show_result, write_json
from mmocr.core.prepare_model import prepare_det_model, prepare_recog_model
from mmocr.datasets.pipelines.crop import crop_img


def end2end_inference(args, det_model, recog_model):
    image_path = args.img
    end2end_res = {'filename': image_path}
    end2end_res['result'] = []

    image = mmcv.imread(image_path)
    det_result = model_inference(det_model, image)
    bboxes = det_result['boundary_result']

    for bbox in bboxes:
        box_res = {}
        box_res['box'] = [round(x) for x in bbox[:-1]]
        box_res['box_score'] = float('{:.4f}'.format(bbox[-1]))
        box = bbox[:8]
        if len(bbox) > 9:
            min_x = min(bbox[0:-1:2])
            min_y = min(bbox[1:-1:2])
            max_x = max(bbox[0:-1:2])
            max_y = max(bbox[1:-1:2])
            box = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
        box_img = crop_img(image, box)

        if args.recog_alg == 'crnn':
            box_img = mmcv.bgr2gray(box_img, keepdim=True)

        recog_result = model_inference(recog_model, box_img)

        text = recog_result['text']
        text_score = recog_result['score']
        if isinstance(text_score, list):
            text_score = sum(text_score) / max(1, len(text))
        box_res['text'] = text
        box_res['text_score'] = float('{:.4f}'.format(text_score))
        end2end_res['result'].append(box_res)

    return end2end_res


def main():
    parser = ArgumentParser()
    parser.add_argument('img', type=str, help='Input Image file.')
    parser.add_argument(
        'out_file', type=str, help='Output file name of the visualized image.')
    parser.add_argument(
        '--detect-alg',
        type=str,
        default='psenet',
        choices=['psenet', 'panet', 'dbnet', 'textsnake', 'maskrcnn'],
        help='Type of text detection algorithm.')
    parser.add_argument(
        '--recog-alg',
        type=str,
        default='crnn',
        choices=['sar', 'crnn', 'seg', 'nrtr', 'robust_scanner'],
        help='Type of text recognition algorithm.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--imshow',
        action='store_true',
        help='Whether show image with OpenCV.')
    args = parser.parse_args()

    # build detect model
    detect_config, detect_checkpoint = prepare_det_model(
        model_type=args.detect_alg)
    detect_model = init_detector(
        detect_config, detect_checkpoint, device=args.device)
    if hasattr(detect_model, 'module'):
        detect_model = detect_model.module
    if detect_model.cfg.data.test['type'] == 'ConcatDataset':
        detect_model.cfg.data.test.pipeline = \
            detect_model.cfg.data.test['datasets'][0].pipeline

    # build recog model
    recog_config, recog_checkpoint = prepare_recog_model(
        model_type=args.recog_alg)
    recog_model = init_detector(
        recog_config, recog_checkpoint, device=args.device)
    if hasattr(recog_model, 'module'):
        recog_model = recog_model.module
    if recog_model.cfg.data.test['type'] == 'ConcatDataset':
        recog_model.cfg.data.test.pipeline = \
            recog_model.cfg.data.test['datasets'][0].pipeline

    end2end_result = end2end_inference(args, detect_model, recog_model)
    print(f'result: {end2end_result}')
    write_json(end2end_result, args.out_file + '.json')

    img = end2end_show_result(args.img, end2end_result)
    mmcv.imwrite(img, args.out_file)
    if args.imshow:
        mmcv.imshow(img, 'predicted results')


if __name__ == '__main__':
    main()
