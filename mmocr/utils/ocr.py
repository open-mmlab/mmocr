from argparse import ArgumentParser, Namespace
from pathlib import Path

import mmcv
import numpy as np
from mmdet.apis import init_detector

from mmocr.apis.inference import model_inference
from mmocr.core.visualize import det_recog_show_result
from mmocr.datasets.pipelines.crop import crop_img

textdet_models = {
    'DB_r18': {
        'config': 'dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
        'ckpt':
        'dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth'
    },
    'DB_r50': {
        'config':
        'dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py',
        'ckpt':
        'dbnet/dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20210325-91cef9af.pth'
    },
    'DRRG': {
        'config': 'drrg/drrg_r50_fpn_unet_1200e_ctw1500.py',
        'ckpt': 'drrg/drrg_r50_fpn_unet_1200e_ctw1500-1abf4f67.pth'
    },
    'FCE_IC15': {
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
    'MaskRCNN_IC15': {
        'config': 'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015.py',
        'ckpt':
        'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth'
    },
    'MaskRCNN_IC17': {
        'config': 'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017.py',
        'ckpt':
        'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth'
    },
    'PANet_CTW': {
        'config': 'panet/panet_r18_fpem_ffm_600e_ctw1500.py',
        'ckpt':
        'panet/panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.pth'
    },
    'PANet_IC15': {
        'config': 'panet/panet_r18_fpem_ffm_600e_icdar2015.py',
        'ckpt':
        'panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth'
    },
    'PS_CTW': {
        'config': 'psenet/psenet_r50_fpnf_600e_ctw1500.py',
        'ckpt': 'psenet/psenet_r50_fpnf_600e_ctw1500_20210401-216fed50.pth'
    },
    'PS_IC15': {
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
        'config': 'robust_scanner/robustscanner_r31_academic.py',
        'ckpt': 'robust_scanner/robustscanner_r31_academic-5f05874f.pth'
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


# Post processing function for end2end ocr
def det_recog_pp(args, result):
    final_results = []
    for arr, out_img, export, det_recog_result in zip(args.arrays,
                                                      args.out_img,
                                                      args.export, result):
        if out_img or args.imshow:
            res_img = det_recog_show_result(arr, det_recog_result)
            if out_img:
                mmcv.imwrite(res_img, out_img)
            if args.imshow:
                mmcv.imshow(res_img, 'predicted results')
        if not args.details:
            simple_res = {}
            simple_res['filename'] = det_recog_result['filename']
            simple_res['text'] = [
                x['text'] for x in det_recog_result['result']
            ]
            final_result = simple_res
        else:
            final_result = det_recog_result
        if export:
            mmcv.dump(final_result, export, ensure_ascii=False, indent=4)
        if args.print_result:
            print(final_result, end='\n\n')
        final_results.append(final_result)
    return final_results


# Post processing function for separate det/recog inference
def single_pp(args, result, model):
    for arr, out_img, export, res in zip(args.arrays, args.out_img,
                                         args.export, result):
        if export:
            mmcv.dump(res, export, ensure_ascii=False, indent=4)
        if out_img or args.imshow:
            model.show_result(arr, res, out_file=out_img, show=args.imshow)
        if args.print_result:
            print(res, end='\n\n')
    return result


# End2end ocr inference pipeline
def det_and_recog_inference(args, det_model, recog_model):
    end2end_res = []
    # Find bounding boxes in the images (text detection)
    det_result = single_inference(det_model, args.arrays, args.batch_mode,
                                  args.det_batch_size)
    bboxes_list = [res['boundary_result'] for res in det_result]

    # For each bounding box, the image is cropped and sent to the recognition
    # model either one by one or all together depending on the batch_mode
    for filename, arr, bboxes in zip(args.filenames, args.arrays, bboxes_list):
        img_e2e_res = {}
        img_e2e_res['filename'] = filename
        img_e2e_res['result'] = []
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
            box_img = crop_img(arr, box)
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
            img_e2e_res['result'].append(box_res)

        if args.batch_mode:
            recog_results = single_inference(recog_model, box_imgs, True,
                                             args.recog_batch_size)
            for i, recog_result in enumerate(recog_results):
                text = recog_result['text']
                text_score = recog_result['score']
                if isinstance(text_score, (list, tuple)):
                    text_score = sum(text_score) / max(1, len(text))
                img_e2e_res['result'][i]['text'] = text
                img_e2e_res['result'][i]['text_score'] = text_score

        end2end_res.append(img_e2e_res)
    return end2end_res


# Separate det/recog inference pipeline
def single_inference(model, arrays, batch_mode, batch_size):
    result = []
    if batch_mode:
        if batch_size == 0:
            result = model_inference(model, arrays, batch_mode=True)
        else:
            n = batch_size
            arr_chunks = [arrays[i:i + n] for i in range(0, len(arrays), n)]
            for chunk in arr_chunks:
                result.extend(model_inference(model, chunk, batch_mode=True))
    else:
        for arr in arrays:
            result.append(model_inference(model, arr, batch_mode=False))
    return result


# Arguments pre-processing function
def args_processing(args):
    # Check if the input is a list/tuple that
    # contains only np arrays or strings
    if isinstance(args.img, (list, tuple)):
        img_list = args.img
        if not all([isinstance(x, (np.ndarray, str)) for x in args.img]):
            raise AssertionError('Images must be strings or numpy arrays')

    # Create a list of the images
    if isinstance(args.img, str):
        img_path = Path(args.img)
        if img_path.is_dir():
            img_list = [str(x) for x in img_path.glob('*')]
        else:
            img_list = [str(img_path)]
    elif isinstance(args.img, np.ndarray):
        img_list = [args.img]

    # Read all image(s) in advance to reduce wasted time
    # re-reading the images for vizualisation output
    args.arrays = [mmcv.imread(x) for x in img_list]

    # Create a list of filenames (used for output imgages and result files)
    if isinstance(img_list[0], str):
        args.filenames = [str(Path(x).stem) for x in img_list]
    else:
        args.filenames = [str(x) for x in range(len(img_list))]

    # If given an output argument, create a list of output image filenames
    num_res = len(img_list)
    if args.out_img:
        out_img_path = Path(args.out_img)
        if out_img_path.is_dir():
            args.out_img = [
                str(out_img_path / f'out_{x}.png') for x in args.filenames
            ]
        else:
            args.out_img = [str(args.out_img)]
            if args.batch_mode:
                raise AssertionError(
                    'Output of multiple images inference must be a directory')
    else:
        args.out_img = [None] * num_res

    # If given an export argument, create a list of
    # result filenames for each image
    if args.export:
        export_path = Path(args.export)
        args.export = [
            str(export_path / f'out_{x}.{args.export_format}')
            for x in args.filenames
        ]
    else:
        args.export = [None] * num_res

    return args


# Create an inference pipeline with parsed arguments
def main():
    args = parse_args()
    ocr = MMOCR(**vars(args))
    ocr.readtext(**vars(args))


# Parse CLI arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', type=str, help='Input image file or folder path.')
    parser.add_argument(
        '--out-img',
        type=str,
        default='',
        help='Output file/folder name for visualization')
    parser.add_argument(
        '--det',
        type=str,
        default='PANet_IC15',
        help='Text detection algorithm')
    parser.add_argument(
        '--det-config',
        type=str,
        default='',
        help='Path to the custom config of the selected det model')
    parser.add_argument(
        '--recog', type=str, default='SEG', help='Text recognition algorithm')
    parser.add_argument(
        '--recog-config',
        type=str,
        default='',
        help='Path to the custom config of the selected recog model')
    parser.add_argument(
        '--batch-mode',
        action='store_true',
        help='Whether use batch mode for inference')
    parser.add_argument(
        '--recog-batch-size',
        type=int,
        default=0,
        help='Batch size for text recognition')
    parser.add_argument(
        '--det-batch-size',
        type=int,
        default=0,
        help='Batch size for text detection')
    parser.add_argument(
        '--single-batch-size',
        type=int,
        default=0,
        help='Batch size for separate det/recog inference')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--export',
        type=str,
        default='',
        help='Folder where the results of each image are exported')
    parser.add_argument(
        '--export-format',
        type=str,
        default='json',
        help='Format of the exported result file(s)')
    parser.add_argument(
        '--details',
        action='store_true',
        help='Whether include the text boxes coordinates and confidence values'
    )
    parser.add_argument(
        '--imshow',
        action='store_true',
        help='Whether show image with OpenCV.')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Prints the recognised text')
    args = parser.parse_args()
    if args.det == "''":
        args.det = None
    if args.recog == "''":
        args.recog = None
    return args


class MMOCR:

    def __init__(self,
                 det='PANet_IC15',
                 det_config='',
                 recog='SEG',
                 recog_config='',
                 device='cuda:0',
                 **kwargs):
        self.td = det
        self.tr = recog
        if device == 'cpu':
            self.device = 0
        else:
            self.device = device

        # Check if the det/recog model choice is valid
        if self.td and self.td not in textdet_models:
            raise ValueError(self.td,
                             'is not a supported text detection algorthm')
        elif self.tr and self.tr not in textrecog_models:
            raise ValueError(self.tr,
                             'is not a supported text recognition algorithm')

        # By default, the config folder should be in the cwd
        dir_path = str(Path.cwd())

        if self.td:
            # Build detection model
            if not det_config:
                det_config = dir_path + '/configs/textdet/' + textdet_models[
                    self.td]['config']
            det_ckpt = 'https://download.openmmlab.com/mmocr/textdet/' + \
                textdet_models[self.td]['ckpt']

            self.detect_model = init_detector(
                det_config, det_ckpt, device=self.device)
        else:
            self.detect_model = None

        if self.tr:
            # Build recognition model
            if not recog_config:
                recog_config = dir_path + '/configs/textrecog/' + \
                    textrecog_models[self.tr]['config']
            recog_ckpt = 'https://download.openmmlab.com/mmocr/textrecog/' + \
                textrecog_models[self.tr]['ckpt']

            self.recog_model = init_detector(
                recog_config, recog_ckpt, device=self.device)
        else:
            self.recog_model = None

        # Attribute check
        for model in list(filter(None, [self.recog_model, self.detect_model])):
            if hasattr(model, 'module'):
                model = model.module
            if model.cfg.data.test['type'] == 'ConcatDataset':
                model.cfg.data.test.pipeline = \
                    model.cfg.data.test['datasets'][0].pipeline

    def readtext(self,
                 img,
                 out_img=None,
                 details=False,
                 export=None,
                 export_format='json',
                 batch_mode=False,
                 recog_batch_size=0,
                 det_batch_size=0,
                 single_batch_size=0,
                 imshow=False,
                 print_result=False,
                 **kwargs):
        args = locals()
        [args.pop(x, None) for x in ['kwargs', 'self']]
        args = Namespace(**args)

        # Input and output arguments processing
        args = args_processing(args)

        pp_result = None

        # Send args and models to the MMOCR model inference API
        # and call post-processing functions for the output
        if self.detect_model and self.recog_model:
            det_recog_result = det_and_recog_inference(args, self.detect_model,
                                                       self.recog_model)
            pp_result = det_recog_pp(args, det_recog_result)
        else:
            for model in list(
                    filter(None, [self.recog_model, self.detect_model])):
                result = single_inference(model, args.arrays, args.batch_mode,
                                          args.single_batch_size)
                pp_result = single_pp(args, result, model)

        return pp_result


if __name__ == '__main__':
    main()
