#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.image.misc import tensor2imgs
from mmcv.runner import load_checkpoint
from mmcv.utils.config import Config

from mmocr.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.core.visualize import det_recog_show_result
from mmocr.datasets.kie_dataset import KIEDataset
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.models import build_detector
from mmocr.utils.box_util import stitch_boxes_into_lines
from mmocr.utils.fileio import list_from_file
from mmocr.utils.model import revert_sync_batchnorm


# Parse CLI arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', type=str, help='Input image file or folder path.')
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='Output file/folder name for visualization')
    parser.add_argument(
        '--det',
        type=str,
        default='PANet_IC15',
        help='Pretrained text detection algorithm')
    parser.add_argument(
        '--det-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected det model. It '
        'overrides the settings in det')
    parser.add_argument(
        '--det-ckpt',
        type=str,
        default='',
        help='Path to the custom checkpoint file of the selected det model. '
        'It overrides the settings in det')
    parser.add_argument(
        '--recog',
        type=str,
        default='SEG',
        help='Pretrained text recognition algorithm')
    parser.add_argument(
        '--recog-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected recog model. It'
        'overrides the settings in recog')
    parser.add_argument(
        '--recog-ckpt',
        type=str,
        default='',
        help='Path to the custom checkpoint file of the selected recog model. '
        'It overrides the settings in recog')
    parser.add_argument(
        '--kie',
        type=str,
        default='',
        help='Pretrained key information extraction algorithm')
    parser.add_argument(
        '--kie-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected kie model. It'
        'overrides the settings in kie')
    parser.add_argument(
        '--kie-ckpt',
        type=str,
        default='',
        help='Path to the custom checkpoint file of the selected kie model. '
        'It overrides the settings in kie')
    parser.add_argument(
        '--config-dir',
        type=str,
        default=os.path.join(str(Path.cwd()), 'configs/'),
        help='Path to the config directory where all the config files '
        'are located. Defaults to "configs/"')
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
    parser.add_argument(
        '--merge', action='store_true', help='Merge neighboring boxes')
    parser.add_argument(
        '--merge-xdist',
        type=float,
        default=20,
        help='The maximum x-axis distance to merge boxes')
    args = parser.parse_args()
    if args.det == 'None':
        args.det = None
    if args.recog == 'None':
        args.recog = None
    # Warnings
    if args.merge and not (args.det and args.recog):
        warnings.warn(
            'Box merging will not work if the script is not'
            ' running in detection + recognition mode.', UserWarning)
    if not os.path.samefile(args.config_dir, os.path.join(str(
            Path.cwd()))) and (args.det_config != ''
                               or args.recog_config != ''):
        warnings.warn(
            'config_dir will be overridden by det-config or recog-config.',
            UserWarning)
    return args


class MMOCR:

    def __init__(self,
                 det='PANet_IC15',
                 det_config='',
                 det_ckpt='',
                 recog='SEG',
                 recog_config='',
                 recog_ckpt='',
                 kie='',
                 kie_config='',
                 kie_ckpt='',
                 config_dir=os.path.join(str(Path.cwd()), 'configs/'),
                 device='cuda:0',
                 **kwargs):

        textdet_models = {
            'DB_r18': {
                'config':
                'dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
                'ckpt':
                'dbnet/'
                'dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth'
            },
            'DB_r50': {
                'config':
                'dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py',
                'ckpt':
                'dbnet/'
                'dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.pth'
            },
            'DRRG': {
                'config':
                'drrg/drrg_r50_fpn_unet_1200e_ctw1500.py',
                'ckpt':
                'drrg/drrg_r50_fpn_unet_1200e_ctw1500_20211022-fb30b001.pth'
            },
            'FCE_IC15': {
                'config':
                'fcenet/fcenet_r50_fpn_1500e_icdar2015.py',
                'ckpt':
                'fcenet/fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth'
            },
            'FCE_CTW_DCNv2': {
                'config':
                'fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500.py',
                'ckpt':
                'fcenet/' +
                'fcenet_r50dcnv2_fpn_1500e_ctw1500_20211022-e326d7ec.pth'
            },
            'MaskRCNN_CTW': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.pth'
            },
            'MaskRCNN_IC15': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth'
            },
            'MaskRCNN_IC17': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth'
            },
            'PANet_CTW': {
                'config':
                'panet/panet_r18_fpem_ffm_600e_ctw1500.py',
                'ckpt':
                'panet/'
                'panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.pth'
            },
            'PANet_IC15': {
                'config':
                'panet/panet_r18_fpem_ffm_600e_icdar2015.py',
                'ckpt':
                'panet/'
                'panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth'
            },
            'PS_CTW': {
                'config': 'psenet/psenet_r50_fpnf_600e_ctw1500.py',
                'ckpt':
                'psenet/psenet_r50_fpnf_600e_ctw1500_20210401-216fed50.pth'
            },
            'PS_IC15': {
                'config':
                'psenet/psenet_r50_fpnf_600e_icdar2015.py',
                'ckpt':
                'psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth'
            },
            'TextSnake': {
                'config':
                'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500.py',
                'ckpt':
                'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth'
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
            'SAR_CN': {
                'config':
                'sar/sar_r31_parallel_decoder_chinese.py',
                'ckpt':
                'sar/sar_r31_parallel_decoder_chineseocr_20210507-b4be8214.pth'
            },
            'NRTR_1/16-1/8': {
                'config': 'nrtr/nrtr_r31_1by16_1by8_academic.py',
                'ckpt':
                'nrtr/nrtr_r31_1by16_1by8_academic_20211124-f60cebf4.pth'
            },
            'NRTR_1/8-1/4': {
                'config': 'nrtr/nrtr_r31_1by8_1by4_academic.py',
                'ckpt':
                'nrtr/nrtr_r31_1by8_1by4_academic_20211123-e1fdb322.pth'
            },
            'RobustScanner': {
                'config': 'robust_scanner/robustscanner_r31_academic.py',
                'ckpt': 'robustscanner/robustscanner_r31_academic-5f05874f.pth'
            },
            'SATRN': {
                'config': 'satrn/satrn_academic.py',
                'ckpt': 'satrn/satrn_academic_20211009-cb8b1580.pth'
            },
            'SATRN_sm': {
                'config': 'satrn/satrn_small.py',
                'ckpt': 'satrn/satrn_small_20211009-2cf13355.pth'
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

        kie_models = {
            'SDMGR': {
                'config': 'sdmgr/sdmgr_unet16_60e_wildreceipt.py',
                'ckpt':
                'sdmgr/sdmgr_unet16_60e_wildreceipt_20210520-7489e6de.pth'
            }
        }

        self.td = det
        self.tr = recog
        self.kie = kie
        self.device = device

        # Check if the det/recog model choice is valid
        if self.td and self.td not in textdet_models:
            raise ValueError(self.td,
                             'is not a supported text detection algorthm')
        elif self.tr and self.tr not in textrecog_models:
            raise ValueError(self.tr,
                             'is not a supported text recognition algorithm')
        elif self.kie:
            if self.kie not in kie_models:
                raise ValueError(
                    self.kie, 'is not a supported key information extraction'
                    ' algorithm')
            elif not (self.td and self.tr):
                raise NotImplementedError(
                    self.kie, 'has to run together'
                    ' with text detection and recognition algorithms.')

        self.detect_model = None
        if self.td:
            # Build detection model
            if not det_config:
                det_config = os.path.join(config_dir, 'textdet/',
                                          textdet_models[self.td]['config'])
            if not det_ckpt:
                det_ckpt = 'https://download.openmmlab.com/mmocr/textdet/' + \
                    textdet_models[self.td]['ckpt']

            self.detect_model = init_detector(
                det_config, det_ckpt, device=self.device)
            self.detect_model = revert_sync_batchnorm(self.detect_model)

        self.recog_model = None
        if self.tr:
            # Build recognition model
            if not recog_config:
                recog_config = os.path.join(
                    config_dir, 'textrecog/',
                    textrecog_models[self.tr]['config'])
            if not recog_ckpt:
                recog_ckpt = 'https://download.openmmlab.com/mmocr/' + \
                    'textrecog/' + textrecog_models[self.tr]['ckpt']

            self.recog_model = init_detector(
                recog_config, recog_ckpt, device=self.device)
            self.recog_model = revert_sync_batchnorm(self.recog_model)

        self.kie_model = None
        if self.kie:
            # Build key information extraction model
            if not kie_config:
                kie_config = os.path.join(config_dir, 'kie/',
                                          kie_models[self.kie]['config'])
            if not kie_ckpt:
                kie_ckpt = 'https://download.openmmlab.com/mmocr/' + \
                    'kie/' + kie_models[self.kie]['ckpt']

            kie_cfg = Config.fromfile(kie_config)
            self.kie_model = build_detector(
                kie_cfg.model, test_cfg=kie_cfg.get('test_cfg'))
            self.kie_model = revert_sync_batchnorm(self.kie_model)
            self.kie_model.cfg = kie_cfg
            load_checkpoint(self.kie_model, kie_ckpt, map_location=self.device)

        # Attribute check
        for model in list(filter(None, [self.recog_model, self.detect_model])):
            if hasattr(model, 'module'):
                model = model.module
            if model.cfg.data.test['type'] == 'ConcatDataset':
                model.cfg.data.test.pipeline = \
                    model.cfg.data.test['datasets'][0].pipeline

    def readtext(self,
                 img,
                 output=None,
                 details=False,
                 export=None,
                 export_format='json',
                 batch_mode=False,
                 recog_batch_size=0,
                 det_batch_size=0,
                 single_batch_size=0,
                 imshow=False,
                 print_result=False,
                 merge=False,
                 merge_xdist=20,
                 **kwargs):
        args = locals()
        [args.pop(x, None) for x in ['kwargs', 'self']]
        args = Namespace(**args)

        # Input and output arguments processing
        self._args_processing(args)
        self.args = args

        pp_result = None

        # Send args and models to the MMOCR model inference API
        # and call post-processing functions for the output
        if self.detect_model and self.recog_model:
            det_recog_result = self.det_recog_kie_inference(
                self.detect_model, self.recog_model, kie_model=self.kie_model)
            pp_result = self.det_recog_pp(det_recog_result)
        else:
            for model in list(
                    filter(None, [self.recog_model, self.detect_model])):
                result = self.single_inference(model, args.arrays,
                                               args.batch_mode,
                                               args.single_batch_size)
                pp_result = self.single_pp(result, model)

        return pp_result

    # Post processing function for end2end ocr
    def det_recog_pp(self, result):
        final_results = []
        args = self.args
        for arr, output, export, det_recog_result in zip(
                args.arrays, args.output, args.export, result):
            if output or args.imshow:
                if self.kie_model:
                    res_img = det_recog_show_result(arr, det_recog_result)
                else:
                    res_img = det_recog_show_result(
                        arr, det_recog_result, out_file=output)
                if args.imshow and not self.kie_model:
                    mmcv.imshow(res_img, 'inference results')
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
                mmcv.dump(final_result, export, indent=4)
            if args.print_result:
                print(final_result, end='\n\n')
            final_results.append(final_result)
        return final_results

    # Post processing function for separate det/recog inference
    def single_pp(self, result, model):
        for arr, output, export, res in zip(self.args.arrays, self.args.output,
                                            self.args.export, result):
            if export:
                mmcv.dump(res, export, indent=4)
            if output or self.args.imshow:
                res_img = model.show_result(arr, res, out_file=output)
                if self.args.imshow:
                    mmcv.imshow(res_img, 'inference results')
            if self.args.print_result:
                print(res, end='\n\n')
        return result

    def generate_kie_labels(self, result, boxes, class_list):
        idx_to_cls = {}
        if class_list is not None:
            for line in list_from_file(class_list):
                class_idx, class_label = line.strip().split()
                idx_to_cls[class_idx] = class_label

        max_value, max_idx = torch.max(result['nodes'].detach().cpu(), -1)
        node_pred_label = max_idx.numpy().tolist()
        node_pred_score = max_value.numpy().tolist()
        labels = []
        for i in range(len(boxes)):
            pred_label = str(node_pred_label[i])
            if pred_label in idx_to_cls:
                pred_label = idx_to_cls[pred_label]
            pred_score = node_pred_score[i]
            labels.append((pred_label, pred_score))
        return labels

    def visualize_kie_output(self,
                             model,
                             data,
                             result,
                             out_file=None,
                             show=False):
        """Visualizes KIE output."""
        img_tensor = data['img'].data
        img_meta = data['img_metas'].data
        gt_bboxes = data['gt_bboxes'].data.numpy().tolist()
        if img_tensor.dtype == torch.uint8:
            # The img tensor is the raw input not being normalized
            # (For SDMGR non-visual)
            img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        else:
            img = tensor2imgs(
                img_tensor.unsqueeze(0), **img_meta.get('img_norm_cfg', {}))[0]
        h, w, _ = img_meta.get('img_shape', img.shape)
        img_show = img[:h, :w, :]
        model.show_result(
            img_show, result, gt_bboxes, show=show, out_file=out_file)

    # End2end ocr inference pipeline
    def det_recog_kie_inference(self, det_model, recog_model, kie_model=None):
        end2end_res = []
        # Find bounding boxes in the images (text detection)
        det_result = self.single_inference(det_model, self.args.arrays,
                                           self.args.batch_mode,
                                           self.args.det_batch_size)
        bboxes_list = [res['boundary_result'] for res in det_result]

        if kie_model:
            kie_dataset = KIEDataset(
                dict_file=kie_model.cfg.data.test.dict_file)

        # For each bounding box, the image is cropped and
        # sent to the recognition model either one by one
        # or all together depending on the batch_mode
        for filename, arr, bboxes, out_file in zip(self.args.filenames,
                                                   self.args.arrays,
                                                   bboxes_list,
                                                   self.args.output):
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
                    box = [
                        min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
                    ]
                box_img = crop_img(arr, box)
                if self.args.batch_mode:
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

            if self.args.batch_mode:
                recog_results = self.single_inference(
                    recog_model, box_imgs, True, self.args.recog_batch_size)
                for i, recog_result in enumerate(recog_results):
                    text = recog_result['text']
                    text_score = recog_result['score']
                    if isinstance(text_score, (list, tuple)):
                        text_score = sum(text_score) / max(1, len(text))
                    img_e2e_res['result'][i]['text'] = text
                    img_e2e_res['result'][i]['text_score'] = text_score

            if self.args.merge:
                img_e2e_res['result'] = stitch_boxes_into_lines(
                    img_e2e_res['result'], self.args.merge_xdist, 0.5)

            if kie_model:
                annotations = copy.deepcopy(img_e2e_res['result'])
                # Customized for kie_dataset, which
                # assumes that boxes are represented by only 4 points
                for i, ann in enumerate(annotations):
                    min_x = min(ann['box'][::2])
                    min_y = min(ann['box'][1::2])
                    max_x = max(ann['box'][::2])
                    max_y = max(ann['box'][1::2])
                    annotations[i]['box'] = [
                        min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
                    ]
                ann_info = kie_dataset._parse_anno_info(annotations)
                ann_info['ori_bboxes'] = ann_info.get('ori_bboxes',
                                                      ann_info['bboxes'])
                ann_info['gt_bboxes'] = ann_info.get('gt_bboxes',
                                                     ann_info['bboxes'])
                kie_result, data = model_inference(
                    kie_model,
                    arr,
                    ann=ann_info,
                    return_data=True,
                    batch_mode=self.args.batch_mode)
                # visualize KIE results
                self.visualize_kie_output(
                    kie_model,
                    data,
                    kie_result,
                    out_file=out_file,
                    show=self.args.imshow)
                gt_bboxes = data['gt_bboxes'].data.numpy().tolist()
                labels = self.generate_kie_labels(kie_result, gt_bboxes,
                                                  kie_model.class_list)
                for i in range(len(gt_bboxes)):
                    img_e2e_res['result'][i]['label'] = labels[i][0]
                    img_e2e_res['result'][i]['label_score'] = labels[i][1]

            end2end_res.append(img_e2e_res)
        return end2end_res

    # Separate det/recog inference pipeline
    def single_inference(self, model, arrays, batch_mode, batch_size=0):
        result = []
        if batch_mode:
            if batch_size == 0:
                result = model_inference(model, arrays, batch_mode=True)
            else:
                n = batch_size
                arr_chunks = [
                    arrays[i:i + n] for i in range(0, len(arrays), n)
                ]
                for chunk in arr_chunks:
                    result.extend(
                        model_inference(model, chunk, batch_mode=True))
        else:
            for arr in arrays:
                result.append(model_inference(model, arr, batch_mode=False))
        return result

    # Arguments pre-processing function
    def _args_processing(self, args):
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
        # re-reading the images for visualization output
        args.arrays = [mmcv.imread(x) for x in img_list]

        # Create a list of filenames (used for output images and result files)
        if isinstance(img_list[0], str):
            args.filenames = [str(Path(x).stem) for x in img_list]
        else:
            args.filenames = [str(x) for x in range(len(img_list))]

        # If given an output argument, create a list of output image filenames
        num_res = len(img_list)
        if args.output:
            output_path = Path(args.output)
            if output_path.is_dir():
                args.output = [
                    str(output_path / f'out_{x}.png') for x in args.filenames
                ]
            else:
                args.output = [str(args.output)]
                if args.batch_mode:
                    raise AssertionError('Output of multiple images inference'
                                         ' must be a directory')
        else:
            args.output = [None] * num_res

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


if __name__ == '__main__':
    main()
