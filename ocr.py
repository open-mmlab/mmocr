from argparse import ArgumentParser, Namespace

import mmcv
from mmdet.apis import init_detector

from mmocr.apis.inference import model_inference
from mmocr.core.visualize import det_recog_show_result
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.utils.box_util import stitch_boxes_into_lines

import os

textdet_models = {
    'DB_r18': {
        'config': 'dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
        'ckpt': 'dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth'
    },
    'DB_r50': {
        'config': 'dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py',
        'ckpt': 'dbnet/dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20210325-91cef9af.pth'
    },
    'DRRG': {
        'config': 'drrg/drrg_r50_fpn_unet_1200e_ctw1500.py',
        'ckpt': 'drrg/drrg_r50_fpn_unet_1200e_ctw1500-1abf4f67.pth'
    },
    'FCE_ICDAR15': {
        'config': 'fcenet/fcenet_r50_fpn_1500e_icdar2015.py',
        'ckpt': 'fcenet/fcenet_r50_fpn_1500e_icdar2015-d435c061.pth'
    },
    'FCE_CTW_DCNv2': {
        'config': 'fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500.py',
        'ckpt': 'fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500-05d740bb.pth'
    },
    'MaskRCNN_CTW': {
        'config': 'maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py',
        'ckpt': 'maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.pth'
    },
    'MaskRCNN_ICDAR15': {
        'config': 'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015.py',
        'ckpt': 'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth'
    },
    'MaskRCNN_ICDAR17': {
        'config': 'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017.py',
        'ckpt': 'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth'
    },
    'PANet_CTW': {
        'config': 'panet/panet_r18_fpem_ffm_600e_ctw1500.py',
        'ckpt': 'panet/panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.pth'
    },
    'PANet_ICDAR15': {
        'config': 'panet/panet_r18_fpem_ffm_600e_icdar2015.py',
        'ckpt': 'panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth'
    },
    'PS_CTW': {
        'config': 'psenet/psenet_r50_fpnf_600e_ctw1500.py',
        'ckpt': 'psenet/psenet_r50_fpnf_600e_ctw1500_20210401-216fed50.pth'
    },
    'PS_ICDAR15': {
        'config': 'psenet/psenet_r50_fpnf_600e_icdar2015.py',
        'ckpt': 'psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth'
    },
    'TextSnake': {
        'config': 'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500.py',
        'ckpt': 'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth'
    }
}

textrecog_models = {
    'CRNN': {
        'config': 'crnn/crnn_academic_dataset.py',
        'ckpt': 'crnn/crnn_academic-a723a1c5.pth'
    },
    'SAR': {
        'config': 'sar/sar_r31_parallel_decoder_academic.py',
        'ckpt': 'sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth'
    },
    'NRTR_1/16-1/8': {
        'config': 'nrtr/nrtr_r31_1by16_1by8_academic.py',
        'ckpt': 'nrtr/nrtr_r31_academic_20210406-954db95e.pth'
    },
    'NRTR_1/8-1/4': {
        'config': 'nrtr/nrtr_r31_1by8_1by4_academic.py',
        'ckpt': 'nrtr/nrtr_r31_1by8_1by4_academic_20210406-ce16e7cc.pth'
    },
    'RobustScanner': {
        'config': 'robustscanner/robustscanner_r31_academic.py',
        'ckpt': 'robustscanner/robustscanner_r31_academic-5f05874f.pth'
    },
    'SEG': {
        'config': 'seg/seg_r31_1by16_fpnocr_academic.py',
        'ckpt': 'seg/seg_r31_1by16_fpnocr_academic-72235b11.pth'
    },
    'CRNN_TPS': {
        'config': 'tps/crnn_tps_academic_dataset.py',
        'ckpt': 'tps/crnn_tps_academic_dataset_20210510-d221a905.pth'
    }
}


def det_and_recog_inference(args, det_model, recog_model):
    image = args.img
    if isinstance(image, str):
        end2end_res = {'filename': image}
    else:
        end2end_res = {}
    end2end_res['result'] = []

    image = mmcv.imread(image)
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
    args = parse_args()
    ocr = MMOCR(**vars(args))
    ocr.readtext(**vars(args))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', type=str, help='Input Image file.')
    parser.add_argument('--out_img', type=str,
                        help='Output file name of the visualized image.')
    parser.add_argument('--textdet', type=str,
                        default='DB_r18', help='Text detection algorithm')
    parser.add_argument('--textrecog', type=str,
                        default='CRNN', help='Text recognition algorithm')
    parser.add_argument('--batch-mode', action='store_false',
                        help='Whether use batch mode for text recognition.')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for text recognition inference if batch_mode is True above.')
    parser.add_argument('--device', default='cuda:0',
                        help='Device used for inference.')
    parser.add_argument('--export-json', action='store_false',
                        help='Whether export the ocr results in a json file.')
    parser.add_argument('--details', action='store_True',
                        help='Whether include the text boxes coordinates and confidence values')
    parser.add_argument('--imshow', action='store_false',
                        help='Whether show image with OpenCV.')
    parser.add_argument('--ocr-in-lines', action='store_false',
                        help='Whether group ocr results in lines.')
    args = parser.parse_args()
    return args


class MMOCR:
    def __init__(self, textdet='DB_r18', textrecog='CRNN', device='cuda:0', **kwargs):
        self.td = textdet
        self.tr = textrecog
        if device == 'cpu':
            self.device = 0
        else:
            self.device = device

        if self.td not in textdet_models:
            raise ValueError(
                self.td, 'is not a supported text detection algorthm')
        elif self.tr not in textrecog_models:
            raise ValueError(
                self.tr, 'is not a supported text recognition algorithm')

        dir_path = os.path.dirname(os.path.realpath(__file__))
        # build detect model
        det_config = dir_path + '/configs/textdet/' + textdet_models[self.td]["config"]
        det_ckpt = 'https://download.openmmlab.com/mmocr/textdet/' + \
            textdet_models[self.td]["ckpt"]

        self.detect_model = init_detector(
            det_config, det_ckpt, device=self.device)

        # build recog model
        recog_config = dir_path + '/configs/textrecog/' + textrecog_models[self.tr]["config"]
        recog_ckpt = 'https://download.openmmlab.com/mmocr/textrecog/' + \
            textrecog_models[self.tr]["ckpt"]

        self.recog_model = init_detector(
            recog_config, recog_ckpt, device=self.device)

        # Attribute check
        for model in [self.recog_model, self.detect_model]:
            if hasattr(model, 'module'):
                model = model.module
            if model.cfg.data.test['type'] == 'ConcatDataset':
                model.cfg.data.test.pipeline = \
                    model.cfg.data.test['datasets'][0].pipeline

    def readtext(self, img, out_img=None, details=False, export_json=False, batch_mode=False, batch_size=4, imshow=False, ocr_in_lines=False, **kwargs):
        args = locals()
        [args.pop(x, None) for x in ['kwargs', 'self']]
        args = Namespace(**args)
        det_recog_result = det_and_recog_inference(
            args, self.detect_model, self.recog_model)
        if args.export_json:
            mmcv.dump(
                det_recog_result,
                out_img + '.json',
                ensure_ascii=False,
                indent=4)
        if args.ocr_in_lines:
            res = det_recog_result['result']
            res = stitch_boxes_into_lines(res, 10, 0.5)
            det_recog_result['result'] = res
            mmcv.dump(
                det_recog_result,
                args.out_img + '.line.json',
                ensure_ascii=False,
                indent=4)
        if args.out_img:
            img = det_recog_show_result(img, det_recog_result)
            mmcv.imwrite(img, out_img)
            if args.imshow:
                mmcv.imshow(img, 'predicted results')
        if not args.details:
            det_recog_result = [x['text'] for x in det_recog_result['result']]
        return det_recog_result


if __name__ == "__main__":
    main()