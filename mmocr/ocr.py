# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import mmcv
import mmengine
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint

from mmocr.data import KIEDataSample, TextDetDataSample, TextRecogDataSample
from mmocr.registry import MODELS
from mmocr.utils import register_all_modules
from mmocr.utils.img_utils import crop_img
from mmocr.visualization import (TextDetLocalVisualizer,
                                 TextRecogLocalVisualizer)
from mmocr.visualization.visualize import det_recog_show_result


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img_path', type=str, help='Input image file or folder path.')
    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='Output file/folder name for visualization')
    parser.add_argument(
        '--det',
        type=str,
        default=None,
        help='Pretrained text detection algorithm')
    parser.add_argument(
        '--det-config',
        type=str,
        default=None,
        help='Path to the custom config file of the selected det model. It '
        'overrides the settings in det')
    parser.add_argument(
        '--det-ckpt',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected det model. '
        'It overrides the settings in det')
    parser.add_argument(
        '--recog',
        type=str,
        default=None,
        help='Pretrained text recognition algorithm')
    parser.add_argument(
        '--recog-config',
        type=str,
        default=None,
        help='Path to the custom config file of the selected recog model. It'
        'overrides the settings in recog')
    parser.add_argument(
        '--recog-ckpt',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected recog model. '
        'It overrides the settings in recog')
    parser.add_argument(
        '--kie',
        type=str,
        default=None,
        help='Pretrained key information extraction algorithm')
    parser.add_argument(
        '--kie-config',
        type=str,
        default=None,
        help='Path to the custom config file of the selected kie model. It'
        'overrides the settings in kie')
    parser.add_argument(
        '--kie-ckpt',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected kie model. '
        'It overrides the settings in kie')
    parser.add_argument(
        '--config-dir',
        type=str,
        default=os.path.join(str(Path.cwd()), 'configs/'),
        help='Path to the config directory where all the config files '
        'are located. Defaults to "configs/"')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device used for inference.')
    parser.add_argument(
        '--imshow',
        action='store_true',
        help='Whether show image with OpenCV.')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Prints the recognised text')
    args = parser.parse_args()
    # Warnings
    if not os.path.samefile(args.config_dir, os.path.join(str(
            Path.cwd()))) and (args.det_config != ''
                               or args.recog_config != ''):
        warnings.warn(
            'config_dir will be overridden by det-config or recog-config.',
            UserWarning)
    return args


class MMOCR:
    """MMOCR API for text detection, recognition, KIE inference.

    Args:
        det (str): Name of the detection model. Default to 'FCE_IC15'.
        det_config (str): Path to the config file for the detection model.
            Default to None.
        det_ckpt (str): Path to the checkpoint file for the detection model.
            Default to None.
        recog (str): Name of the recognition model. Default to 'CRNN'.
        recog_config (str): Path to the config file for the recognition model.
            Default to None.
        recog_ckpt (str): Path to the checkpoint file for the recognition
            model. Default to None.
        kie (str): Name of the KIE model. Default to None.
        kie_config (str): Path to the config file for the KIE model. Default
            to None.
        kie_ckpt (str): Path to the checkpoint file for the KIE model.
            Default to None.
        config_dir (str): Path to the directory containing config files.
            Default to 'configs/'.
        device (torch.device): Device to use for inference. Default to 'cuda'.
    """

    def __init__(self,
                 det: str = None,
                 det_config: str = None,
                 det_ckpt: str = None,
                 recog: str = None,
                 recog_config: str = None,
                 recog_ckpt: str = None,
                 kie: str = None,
                 kie_config: str = None,
                 kie_ckpt: str = None,
                 config_dir: str = os.path.join(str(Path.cwd()), 'configs/'),
                 device: torch.device = 'cuda',
                 **kwargs) -> None:
        self.det = det
        self.det_config = det_config
        self.det_ckpt = det_ckpt
        self.recog = recog
        self.recog_config = recog_config
        self.recog_ckpt = recog_ckpt
        self.kie = kie
        self.kie_config = kie_config
        self.kie_ckpt = kie_ckpt
        self.config_dir = config_dir
        self.device = device

    def inference(self,
                  img_path: str,
                  out_dir: str = None,
                  imshow: bool = False,
                  print_result: bool = False,
                  **kwargs) -> None:
        """Inferences text detection, recognition, and KIE on an image or a
        folder of images.

        Args:
            img_path (str): Path to the image or folder of images.
            out_dir (str): Output file/folder name for visualization.
            imshow (bool): Whether show image with OpenCV.
            print_result (bool): Whether to print the results
        """
        # build models, pipelines and visualizers
        if self.det:
            det_model, det_config = self.init_model(self.det, self.det_config,
                                                    self.det_ckpt, 'textdet')
            det_pipeline = Compose(self.get_pipeline(det_config))

        if self.recog:
            recog_model, recog_config = self.init_model(
                self.recog, self.recog_config, self.recog_ckpt, 'textrecog')
            if self.det:
                # For end2end, use LoadImageFromNDArray instead as the input
                # of the recognition model is the output of the detection
                # model.
                recog_pipeline = Compose(
                    self.get_pipeline(recog_config, e2e=True))
            else:
                recog_pipeline = Compose(self.get_pipeline(recog_config))
        if self.kie:
            kie_model, kie_config = self.init_model(self.kie, self.kie_config,
                                                    self.kie_ckpt, 'kie')
            kie_pipeline = Compose(self.get_pipeline(kie_config))

        # load file names
        img_path_list = []
        if os.path.isfile(img_path):
            img_path_list.append(img_path)
        elif os.path.isdir(img_path):
            img_path_list = [
                os.path.join(img_path, img_name)
                for img_name in os.listdir(img_path)
            ]
        else:
            raise ValueError('img_path must be a file or directory')

        # inference
        for img_path in img_path_list:
            # det only
            if self.det and not self.recog:
                packed_results = det_pipeline(dict(img_path=img_path))
                det_results = self.det_inference(det_model, packed_results)
                if len(det_results) == 0:
                    print(f'No text detected in {img_path}')
                    continue
                self.postprocess(
                    img_path,
                    det_results=det_results,
                    visualizer=TextDetLocalVisualizer(),
                    out_dir=out_dir,
                    imshow=imshow,
                    print_result=print_result,
                    mode='det')
            # recog only
            elif self.recog and not self.det:
                packed_results = recog_pipeline(dict(img_path=img_path))
                recog_results = self.recog_inference(recog_model,
                                                     packed_results)
                if len(recog_results) == 0:
                    print(f'No text recognized in {img_path}')
                    continue
                self.postprocess(
                    img_path,
                    recog_results=recog_results,
                    visualizer=TextRecogLocalVisualizer(),
                    out_dir=out_dir,
                    imshow=imshow,
                    print_result=print_result,
                    mode='recog')
            # e2e
            elif self.det and self.recog and not self.kie:
                packed_results = det_pipeline(dict(img_path=img_path))
                det_results = self.det_inference(det_model, packed_results)
                det_imgs = self.parse_det_results(img_path, det_results)
                if len(det_imgs) == 0:
                    print(f'No text detected in {img_path}')
                    continue
                recog_results = []
                for det_img in det_imgs:
                    recog_results.append(recog_pipeline(det_img))
                recog_results = self.recog_inference(recog_model,
                                                     recog_results)
                self.postprocess(
                    img_path,
                    det_results=det_results,
                    recog_results=recog_results,
                    out_dir=out_dir,
                    imshow=imshow,
                    print_result=print_result,
                    mode='e2e')
            # kie
            elif self.kie:
                packed_results = kie_pipeline(img_path)
                kie_results = self.kie_inference(kie_model, packed_results)
                self.postprocess(img_path, kie_results, out_dir, imshow,
                                 print_result, 'kie')

    def det_inference(self, det_model: nn.ModuleList,
                      packed_results: dict) -> List[TextDetDataSample]:
        """Batch inference for detection models.

        Args:
            det_model (nn.ModuleList): A detection model.
            packed_results (dict): A dict containing packed results with keys
                of ``inputs`` and ``data_samples``.
        Returns:
            list[TextDetDataSample]: A list of N datasamples of prediction
            results. Results are stored in ``pred_instances``.
        """
        batch_inputs, batch_data_samples = det_model.data_preprocessor(
            [packed_results])
        return det_model.predict(batch_inputs, batch_data_samples)

    def recog_inference(
            self, recog_model: nn.ModuleList,
            packed_results: Union[List[dict],
                                  dict]) -> List[TextRecogDataSample]:
        """Batch inference for recognition models.

        Args:
            recog_model (nn.ModuleList): A recognition model.
            packed_results (List[dict] or dict): Dict containing packed results
                with keys of ``inputs`` and ``data_samples``.
        Returns:
            list[TextRecogDataSample]: A list of N datasamples of prediction
            results. Results are stored in ``pred_instances``.
        """
        if isinstance(packed_results, dict):
            packed_results = [packed_results]
        batch_inputs, batch_data_samples = recog_model.data_preprocessor(
            packed_results)
        return recog_model.predict(batch_inputs, batch_data_samples)

    def kie_inference(self, kie_model: nn.ModuleList,
                      packed_results: dict) -> List[TextDetDataSample]:
        """Batch inference for kie models.

        Args:
            kie_model (nn.ModuleList): A kie model.
            packed_results (dict): A dict containing packed results with keys
                of ``inputs`` and ``data_samples``.
        Returns:
            list[TextRecogDataSample]: A list of N datasamples of prediction
            results. Results are stored in ``pred_instances``.
        """
        batch_inputs, batch_data_samples = kie_model.data_preprocessor(
            [packed_results])
        return kie_model.predict(batch_inputs, batch_data_samples)

    def parse_det_results(self, img_path: str,
                          det_results: TextDetDataSample) -> List[Dict]:
        """Parse detection results including cropping.

        Args:
            img_path (str): Path to the image.
            det_results (TextDetDataSample): One predicted textdet datasamples,
                containing polygons and scores for current images.
        Returns:
            list(dict): A list of dictionaries with keys of ``img`` standing
                for cropped img based on polygon.
        """
        img = mmcv.imread(img_path)
        polygons = det_results[0].pred_instances.polygons
        parsed_results = []
        for poly in polygons:
            if len(poly) != 8:
                rect = cv2.minAreaRect(poly)
                vertices = cv2.boxPoints(rect)
                poly = vertices.flatten()
            croped_img = crop_img(img, poly.tolist())
            parsed_results.append(dict(img=croped_img))
        return parsed_results

    def load_image(self, img_path: str, pipeline: Compose) -> dict:
        """Load one image and preprocess it through test pipeline.

        Args:
            img_path (str): Path to the one image.
            pipeline (Compose): Pipelines to preprocess the image.

        Return:
            dict: A dictionary of preprocessed image and meta infomations
        """
        results = dict(img_path=img_path)
        results = pipeline(results)
        return results

    def postprocess(
            self,
            img_path: str,
            det_results: Optional[TextDetDataSample] = None,
            recog_results: Optional[Union[TextRecogDataSample,
                                          List[TextRecogDataSample]]] = None,
            kie_results: Optional[KIEDataSample] = None,
            visualizer: Optional[Union[TextRecogLocalVisualizer,
                                       TextDetLocalVisualizer]] = None,
            out_dir: str = None,
            imshow: bool = False,
            print_result: bool = False,
            mode: str = 'e2e'):
        """Postprocess the results.

        Args:
            img_path (str): Path to the image.
            det_results (TextDetDataSample, optional): Detection results.
                Defaults to None.
            recog_results (TextRecogDataSample or List[TextRecogDataSample],
                optional): Recognition results. Defaults to None.
            kie_results (KIEDataSample, optional): Kie results. Defaults to
                None.
            visualizer (Optional[Union[TextRecogLocalVisualizer,
                                        TextDetLocalVisualizer]]): A visualizer
            out_dir (str): Path to the output directory.
            imshow (bool): Whether to show the results.
            print_result (bool): Whether to print the results.
            mode (str): Mode for inference in [``det``, ``recog``, ``e2e``,
                ``kie``].
        """
        # Get original image
        img = mmcv.imread(img_path)
        img_name = os.path.basename(img_path)
        if out_dir:
            out_dir = os.path.join(out_dir, img_name)
        if mode == 'det':
            visualizer.add_datasample(
                img_name,
                img,
                pred_sample=det_results[0],
                show=imshow,
                out_file=out_dir)
        elif mode == 'recog':
            visualizer.add_datasample(
                img_name,
                img,
                pred_sample=recog_results[0],
                show=imshow,
                out_file=out_dir)
        elif mode == 'e2e':
            res_img = det_recog_show_result(
                img, det_results, recog_results, out_file=out_dir)
            if imshow:
                mmcv.imshow(res_img, 'inference results')

        if print_result:
            print('-' * 60)
            print('img path:', img_path)
            # Det only print
            if mode == 'det':
                polygons = det_results[0].pred_instances.polygons
                scores = det_results[0].pred_instances.scores
                for polygon, score in zip(polygons, scores):
                    print(f'polygon: {polygon}, score: {score}')
            elif mode == 'recog':
                print(f'text: {recog_results[0].pred_text.item}, '
                      f'score: {recog_results[0].pred_text.score}')
            elif mode == 'e2e':
                polygons = det_results[0].pred_instances.polygons
                det_scores = det_results[0].pred_instances.scores
                texts = []
                text_scores = []
                for i, polygon in enumerate(polygons):
                    texts.append(recog_results[i].pred_text.item)
                    text_scores.append(recog_results[i].pred_text.score)
                for polygon, det_score, text, text_score in zip(
                        polygons, det_scores, texts, text_scores):
                    print(f'polygon: {polygon}, det_score: {det_score}, '
                          f'text: {text}, text_score: {text_score}')

    def get_pipeline(self, cfg_dir: str, e2e: bool = False) -> dict:
        """Get pipeline config from config file.

        Args:
            cfg_dir (str): Path to the config file.
            e2e (bool): If true, use ``LoadImageFromNDArray`` instead of
                ``LoadImageFromFile``.
        Returns:
            dict: A dictionary of pipeline.
        """
        config = Config.fromfile(cfg_dir)
        test_pipeline = config.test_pipeline
        if e2e:
            # For end2end, use LoadImageFromNDArray instead as the input
            # of the recognition model is the output of the detection
            # model.
            test_pipeline[0] = dict(type='mmdet.LoadImageFromNDArray')
        # For inference, the key of ``instances`` is not used.
        test_pipeline[-1]['meta_keys'] = test_pipeline[-1]['meta_keys'][:-1]
        return test_pipeline

    def get_model_config(self, model_name: str) -> Dict:
        """Get the model configuration including model config and checkpoint
        url.

        Args:
            model_name (str): Name of the model.
        Returns:
            dict: Model configuration.
        """
        model_dict = {
            'Tesseract': {},
            # Detection models
            'DB_r18': {
                'config':
                'dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
                'ckpt':
                'textdet/'
                'dbnet/'
                'dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth'
            },
            'DB_r50': {
                'config':
                'dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py',
                'ckpt':
                'textdet/'
                'dbnet/'
                'dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.pth'
            },
            'DBPP_r50': {
                'config':
                'dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015.py',
                'ckpt':
                'textdet/'
                'dbnet/'
                'dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.pth'
            },
            'DRRG': {
                'config':
                'drrg/drrg_r50_fpn_unet_1200e_ctw1500.py',
                'ckpt':
                'textdet/'
                'drrg/drrg_r50_fpn_unet_1200e_ctw1500_20211022-fb30b001.pth'
            },
            'FCE_IC15': {
                'config':
                'fcenet/fcenet_r50_fpn_1500e_icdar2015.py',
                'ckpt':
                'textdet/'
                'fcenet/fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth'
            },
            'FCE_CTW_DCNv2': {
                'config':
                'fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500.py',
                'ckpt':
                'textdet/'
                'fcenet/'
                'fcenet_r50dcnv2_fpn_1500e_ctw1500_20211022-e326d7ec.pth'
            },
            'MaskRCNN_CTW': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py',
                'ckpt':
                'textdet/'
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.pth'
            },
            'MaskRCNN_IC15': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015.py',
                'ckpt':
                'textdet/'
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth'
            },
            'MaskRCNN_IC17': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017.py',
                'ckpt':
                'textdet/'
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth'
            },
            'PANet_CTW': {
                'config':
                'panet/panet_r18_fpem_ffm_600e_ctw1500.py',
                'ckpt':
                'textdet/'
                'panet/'
                'panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.pth'
            },
            'PANet_IC15': {
                'config':
                'panet/panet_r18_fpem_ffm_600e_icdar2015.py',
                'ckpt':
                'textdet/'
                'panet/'
                'panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth'
            },
            'PS_CTW': {
                'config':
                'psenet/psenet_r50_fpnf_600e_ctw1500.py',
                'ckpt':
                'textdet/'
                'psenet/psenet_r50_fpnf_600e_ctw1500_20210401-216fed50.pth'
            },
            'PS_IC15': {
                'config':
                'psenet/psenet_r50_fpnf_600e_icdar2015.py',
                'ckpt':
                'textdet/'
                'psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth'
            },
            'TextSnake': {
                'config':
                'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500.py',
                'ckpt':
                'textdet/'
                'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth'
            },
            # Recognition models
            'CRNN': {
                'config': 'crnn/crnn_academic_dataset.py',
                'ckpt': 'textrecog/crnn/crnn_academic-a723a1c5.pth'
            },
            'SAR': {
                'config':
                'sar/sar_r31_parallel_decoder_academic.py',
                'ckpt':
                'textrecog/sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth'
            },
            'SAR_CN': {
                'config':
                'sar/sar_r31_parallel_decoder_chinese.py',
                'ckpt':
                'textrecog/'
                'sar/sar_r31_parallel_decoder_chineseocr_20210507-b4be8214.pth'
            },
            'NRTR_1/16-1/8': {
                'config':
                'nrtr/nrtr_r31_1by16_1by8_academic.py',
                'ckpt':
                'textrecog/'
                'nrtr/nrtr_r31_1by16_1by8_academic_20211124-f60cebf4.pth'
            },
            'NRTR_1/8-1/4': {
                'config':
                'nrtr/nrtr_r31_1by8_1by4_academic.py',
                'ckpt':
                'textrecog/'
                'nrtr/nrtr_r31_1by8_1by4_academic_20211123-e1fdb322.pth'
            },
            'RobustScanner': {
                'config':
                'robust_scanner/robustscanner_r31_academic.py',
                'ckpt':
                'textrecog/'
                'robustscanner/robustscanner_r31_academic-5f05874f.pth'
            },
            'SATRN': {
                'config': 'satrn/satrn_academic.py',
                'ckpt': 'textrecog/satrn/satrn_academic_20211009-cb8b1580.pth'
            },
            'SATRN_sm': {
                'config': 'satrn/satrn_small.py',
                'ckpt': 'textrecog/satrn/satrn_small_20211009-2cf13355.pth'
            },
            'ABINet': {
                'config': 'abinet/abinet_academic.py',
                'ckpt': 'textrecog/abinet/abinet_academic-f718abf6.pth'
            },
            'SEG': {
                'config': 'seg/seg_r31_1by16_fpnocr_academic.py',
                'ckpt':
                'textrecog/seg/seg_r31_1by16_fpnocr_academic-72235b11.pth'
            },
            'CRNN_TPS': {
                'config':
                'tps/crnn_tps_academic_dataset.py',
                'ckpt':
                'textrecog/tps/crnn_tps_academic_dataset_20210510-d221a905.pth'
            },
            'MASTER': {
                'config': 'master/master_r31_12e_ST_MJ_SA.py',
                'ckpt': 'textrecog/master/master_r31_12e_ST_MJ_SA-787edd36.pth'
            },
            # KIE models
            'SDMGR': {
                'config':
                'sdmgr/sdmgr_unet16_60e_wildreceipt.py',
                'ckpt':
                'kie/sdmgr/sdmgr_unet16_60e_wildreceipt_20210520-7489e6de.pth'
            }
        }
        if model_name not in model_dict:
            raise ValueError(f'Model {model_name} is not supported.')
        else:
            return model_dict[model_name]

    def init_model(self,
                   model_name: str,
                   model_config: str = None,
                   model_ckpt: str = None,
                   model_type: str = 'textdet'):
        """Initialize model and load checkpoint.

        Args:
            model_name (str): Name of the model.
            model_config (str): Path to the config file. Defaults to None.
            model_ckpt (str): Path to the checkpoint file. Defaults to None.
            model_type (str): Type of the model, textrecog or textdet or kie.
                Defaults to 'textdet'.

        Returns:
            Tuole(nn.Module, dict): Model and model config

            - nn.Module: Initialized Model.
            - dict: Model config.
        """
        model_dict = self.get_model_config(model_name)
        # load model config
        if model_config is None:
            model_config = os.path.join(self.config_dir, model_type,
                                        model_dict['config'])
        # load checkpoint file
        if model_ckpt is None:
            model_ckpt = 'https://download.openmmlab.com/mmocr' + \
                f"/{model_dict['ckpt']}"

        model = self.init_detector(
            model_config, model_ckpt, device=self.device)
        return model, model_config

    def init_detector(self,
                      config: Union[str, mmengine.Config],
                      checkpoint: Optional[str] = None,
                      device: torch.device = 'cuda:0',
                      cfg_options: Optional[dict] = None):
        """Initialize a detector from config file.

        Args:
            config (str or mmengine.Config): Config file path or the config
                object.
            checkpoint (str, optional): Checkpoint path. If left as None, the
                model will not load any weights. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to
                'cuda:0'.
            cfg_options (dict, optional): Options to override some settings in
                the used config. Defaults to None.
        Returns:
            nn.Module: The constructed detector.
        """
        if isinstance(config, str):
            config = Config.fromfile(config)
        elif not isinstance(config, Config):
            raise TypeError('config must be a filename or Config object, '
                            f'but got {type(config)}')

        if cfg_options is not None:
            config.merge_from_dict(cfg_options)
        if config.model.get('pretrained'):
            config.model.pretrained = None
        model = MODELS.build(config.model)

        if checkpoint is not None:
            checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.cfg = config
        model.to(device)
        model.eval()
        return model


def main():
    args = parse_args()
    ocr = MMOCR(**vars(args))
    ocr.inference(**vars(args))


if __name__ == '__main__':
    register_all_modules(init_default_scope=True)
    main()
