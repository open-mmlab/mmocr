from argparse import ArgumentParser

import mmcv

from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.core.end2end_visualize import end2end_show_result, write_json
from mmocr.core.prepare_model import prepare_det_model, prepare_recog_model
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.models import build_detector  # noqa: F401


def end2end_inference(image_path, det_model, recog_model):
    end2end_res = {'filename': image_path}
    end2end_res['result'] = []

    image = mmcv.imread(image_path)
    det_result = model_inference(det_model, image)
    bboxes = det_result['boundary_result']

    for bbox in bboxes:
        box_res = {}
        box_res['box'] = [round(x) for x in bbox[:8]]
        box_res['box_score'] = float('{:.4f}'.format(bbox[8]))
        box_img = crop_img(image, bbox[:8])

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
        'save_path', type=str, help='Path to save visualized image.')
    parser.add_argument(
        '--detect-model',
        type=str,
        default='psenet',
        choices=['psenet', 'panet', 'dbnet', 'textsnake', 'maskrcnn'],
        help='Text detect model type.')
    parser.add_argument(
        '--recog-model',
        type=str,
        default='sar',
        choices=['sar', 'crnn', 'seg', 'nrtr', 'robustscanner'],
        help='Text recognize model type.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--imshow',
        action='store_true',
        help='Whether show image with OpenCV.')
    args = parser.parse_args()

    # build detect model
    detect_config, detect_checkpoint = prepare_det_model(
        model_type=args.detect_model)
    detect_model = init_detector(
        detect_config, detect_checkpoint, device=args.device)
    if hasattr(detect_model, 'module'):
        detect_model = detect_model.module
    if detect_model.cfg.data.test['type'] == 'ConcatDataset':
        detect_model.cfg.data.test.pipeline = \
            detect_model.cfg.data.test['datasets'][0].pipeline

    # build recog model
    recog_config, recog_checkpoint = prepare_recog_model(
        model_type=args.recog_model)
    recog_model = init_detector(
        recog_config, recog_checkpoint, device=args.device)
    if hasattr(recog_model, 'module'):
        recog_model = recog_model.module
    if recog_model.cfg.data.test['type'] == 'ConcatDataset':
        recog_model.cfg.data.test.pipeline = \
            recog_model.cfg.data.test['datasets'][0].pipeline

    end2end_result = end2end_inference(args.img, detect_model, recog_model)
    print(f'result: {end2end_result}')
    write_json(end2end_result, args.save_path + '.json')

    img = end2end_show_result(args.img, end2end_result)
    mmcv.imwrite(img, args.save_path)
    if args.imshow:
        mmcv.imshow(img, 'predicted results')


if __name__ == '__main__':
    main()
