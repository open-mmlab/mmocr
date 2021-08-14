import io
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from unittest import mock

import mmcv
import numpy as np
import pytest
import torch

from mmocr.apis import init_detector
from mmocr.datasets.kie_dataset import KIEDataset
from mmocr.utils.ocr import MMOCR


def test_ocr_init_errors():
    # Test assertions
    with pytest.raises(ValueError):
        _ = MMOCR(det='test')
    with pytest.raises(ValueError):
        _ = MMOCR(recog='test')
    with pytest.raises(ValueError):
        _ = MMOCR(kie='test')
    with pytest.raises(NotImplementedError):
        _ = MMOCR(det=None, recog=None, kie='SDMGR')
    with pytest.raises(NotImplementedError):
        _ = MMOCR(det='DB_r18', recog=None, kie='SDMGR')


cfg_default_prefix = os.path.join(str(Path.cwd()), 'configs/')


@pytest.mark.parametrize(
    'det, recog, kie, config_dir, gt_cfg, gt_ckpt',
    [('DB_r18', None, '', '',
      cfg_default_prefix + 'textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
      'https://download.openmmlab.com/mmocr/textdet/'
      'dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth'),
     (None, 'CRNN', '', '',
      cfg_default_prefix + 'textrecog/crnn/crnn_academic_dataset.py',
      'https://download.openmmlab.com/mmocr/textrecog/'
      'crnn/crnn_academic-a723a1c5.pth'),
     ('DB_r18', 'CRNN', 'SDMGR', '', [
         cfg_default_prefix +
         'textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
         cfg_default_prefix + 'textrecog/crnn/crnn_academic_dataset.py',
         cfg_default_prefix + 'kie/sdmgr/sdmgr_unet16_60e_wildreceipt.py'
     ], [
         'https://download.openmmlab.com/mmocr/textdet/'
         'dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth',
         'https://download.openmmlab.com/mmocr/textrecog/'
         'crnn/crnn_academic-a723a1c5.pth',
         'https://download.openmmlab.com/mmocr/kie/'
         'sdmgr/sdmgr_unet16_60e_wildreceipt_20210520-7489e6de.pth'
     ]),
     ('DB_r18', 'CRNN', 'SDMGR', 'test/', [
         'test/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
         'test/textrecog/crnn/crnn_academic_dataset.py',
         'test/kie/sdmgr/sdmgr_unet16_60e_wildreceipt.py'
     ], [
         'https://download.openmmlab.com/mmocr/textdet/'
         'dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth',
         'https://download.openmmlab.com/mmocr/textrecog/'
         'crnn/crnn_academic-a723a1c5.pth',
         'https://download.openmmlab.com/mmocr/kie/'
         'sdmgr/sdmgr_unet16_60e_wildreceipt_20210520-7489e6de.pth'
     ])],
)
@mock.patch('mmocr.utils.ocr.init_detector')
@mock.patch('mmocr.utils.ocr.build_detector')
@mock.patch('mmocr.utils.ocr.Config.fromfile')
@mock.patch('mmocr.utils.ocr.load_checkpoint')
def test_ocr_init(mock_loading, mock_config, mock_build_detector,
                  mock_init_detector, det, recog, kie, config_dir, gt_cfg,
                  gt_ckpt):

    def loadcheckpoint_assert(*args, **kwargs):
        assert args[1] == gt_ckpt[-1]

    mock_loading.side_effect = loadcheckpoint_assert
    with mock.patch('mmocr.utils.ocr.revert_sync_batchnorm'):
        if kie == '':
            if config_dir == '':
                _ = MMOCR(det=det, recog=recog)
            else:
                _ = MMOCR(det=det, recog=recog, config_dir=config_dir)
        else:
            if config_dir == '':
                _ = MMOCR(det=det, recog=recog, kie=kie)
            else:
                _ = MMOCR(det=det, recog=recog, kie=kie, config_dir=config_dir)
        if isinstance(gt_cfg, str):
            gt_cfg = [gt_cfg]
        if isinstance(gt_ckpt, str):
            gt_ckpt = [gt_ckpt]

        i_range = range(len(gt_cfg))
        if kie:
            i_range = i_range[:-1]
            mock_config.assert_called_with(gt_cfg[-1])
            mock_build_detector.assert_called_once()
            mock_loading.assert_called_once()
        calls = [
            mock.call(gt_cfg[i], gt_ckpt[i], device='cuda:0') for i in i_range
        ]
        mock_init_detector.assert_has_calls(calls)


@pytest.mark.parametrize(
    'det, det_config, det_ckpt, recog, recog_config, recog_ckpt,'
    'kie, kie_config, kie_ckpt, config_dir, gt_cfg, gt_ckpt',
    [('DB_r18', 'test.py', '', 'CRNN', 'test.py', '', 'SDMGR', 'test.py', '',
      'configs/', ['test.py', 'test.py', 'test.py'], [
          'https://download.openmmlab.com/mmocr/textdet/'
          'dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth',
          'https://download.openmmlab.com/mmocr/textrecog/'
          'crnn/crnn_academic-a723a1c5.pth',
          'https://download.openmmlab.com/mmocr/kie/'
          'sdmgr/sdmgr_unet16_60e_wildreceipt_20210520-7489e6de.pth'
      ]),
     ('DB_r18', '', 'test.ckpt', 'CRNN', '', 'test.ckpt', 'SDMGR', '',
      'test.ckpt', '', [
          'textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
          'textrecog/crnn/crnn_academic_dataset.py',
          'kie/sdmgr/sdmgr_unet16_60e_wildreceipt.py'
      ], ['test.ckpt', 'test.ckpt', 'test.ckpt']),
     ('DB_r18', 'test.py', 'test.ckpt', 'CRNN', 'test.py', 'test.ckpt',
      'SDMGR', 'test.py', 'test.ckpt', '', ['test.py', 'test.py', 'test.py'],
      ['test.ckpt', 'test.ckpt', 'test.ckpt'])])
@mock.patch('mmocr.utils.ocr.init_detector')
@mock.patch('mmocr.utils.ocr.build_detector')
@mock.patch('mmocr.utils.ocr.Config.fromfile')
@mock.patch('mmocr.utils.ocr.load_checkpoint')
def test_ocr_init_customize_config(mock_loading, mock_config,
                                   mock_build_detector, mock_init_detector,
                                   det, det_config, det_ckpt, recog,
                                   recog_config, recog_ckpt, kie, kie_config,
                                   kie_ckpt, config_dir, gt_cfg, gt_ckpt):

    def loadcheckpoint_assert(*args, **kwargs):
        assert args[1] == gt_ckpt[-1]

    mock_loading.side_effect = loadcheckpoint_assert
    with mock.patch('mmocr.utils.ocr.revert_sync_batchnorm'):
        _ = MMOCR(
            det=det,
            det_config=det_config,
            det_ckpt=det_ckpt,
            recog=recog,
            recog_config=recog_config,
            recog_ckpt=recog_ckpt,
            kie=kie,
            kie_config=kie_config,
            kie_ckpt=kie_ckpt,
            config_dir=config_dir)

        i_range = range(len(gt_cfg))
        if kie:
            i_range = i_range[:-1]
            mock_config.assert_called_with(gt_cfg[-1])
            mock_build_detector.assert_called_once()
            mock_loading.assert_called_once()
        calls = [
            mock.call(gt_cfg[i], gt_ckpt[i], device='cuda:0') for i in i_range
        ]
        mock_init_detector.assert_has_calls(calls)


@mock.patch('mmocr.utils.ocr.init_detector')
@mock.patch('mmocr.utils.ocr.build_detector')
@mock.patch('mmocr.utils.ocr.Config.fromfile')
@mock.patch('mmocr.utils.ocr.load_checkpoint')
@mock.patch('mmocr.utils.ocr.model_inference')
def test_single_inference(mock_model_inference, mock_loading, mock_config,
                          mock_build_detector, mock_init_detector):

    def dummy_inference(model, arr, batch_mode):
        return arr

    mock_model_inference.side_effect = dummy_inference
    mmocr = MMOCR()

    data = list(range(20))
    model = 'dummy'
    res = mmocr.single_inference(model, data, batch_mode=False)
    assert (data == res)
    mock_model_inference.reset_mock()

    res = mmocr.single_inference(model, data, batch_mode=True)
    assert (data == res)
    mock_model_inference.assert_called_once()
    mock_model_inference.reset_mock()

    res = mmocr.single_inference(model, data, batch_mode=True, batch_size=100)
    assert (data == res)
    mock_model_inference.assert_called_once()
    mock_model_inference.reset_mock()

    res = mmocr.single_inference(model, data, batch_mode=True, batch_size=3)
    assert (data == res)


@mock.patch('mmocr.utils.ocr.init_detector')
@mock.patch('mmocr.utils.ocr.load_checkpoint')
def MMOCR_testobj(mock_loading, mock_init_detector, **kwargs):
    # returns an MMOCR object bypassing the
    # checkpoint initialization step
    def init_detector_skip_ckpt(config, ckpt, device):
        return init_detector(config, device=device)

    def modify_kie_class(model, ckpt, map_location):
        model.class_list = 'tests/data/kie_toy_dataset/class_list.txt'

    mock_init_detector.side_effect = init_detector_skip_ckpt
    mock_loading.side_effect = modify_kie_class
    kwargs['det'] = kwargs.get('det', 'DB_r18')
    kwargs['recog'] = kwargs.get('recog', 'CRNN')
    kwargs['kie'] = kwargs.get('kie', 'SDMGR')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return MMOCR(**kwargs, device=device)


@mock.patch('mmocr.utils.ocr.KIEDataset')
def test_readtext(mock_kiedataset):
    # Fixing the weights of models to prevent them from
    # generating invalid results and triggering other assertion errors
    torch.manual_seed(4)
    random.seed(4)
    mmocr = MMOCR_testobj()
    mmocr_det = MMOCR_testobj(kie='', recog='')
    mmocr_recog = MMOCR_testobj(kie='', det='', recog='CRNN_TPS')
    mmocr_det_recog = MMOCR_testobj(kie='')

    def readtext(imgs, ocr_obj=mmocr, **kwargs):
        # filename can be different depends on how
        # the the image was loaded
        e2e_res = ocr_obj.readtext(imgs, **kwargs)
        for res in e2e_res:
            res.pop('filename')
        return e2e_res

    def kiedataset_with_test_dict(**kwargs):
        kwargs['dict_file'] = 'tests/data/kie_toy_dataset/dict.txt'
        return KIEDataset(**kwargs)

    mock_kiedataset.side_effect = kiedataset_with_test_dict

    # Single image
    toy_dir = 'tests/data/toy_dataset/imgs/test/'
    toy_img1_path = toy_dir + 'img_1.jpg'
    str_e2e_res = readtext(toy_img1_path)
    toy_img1 = mmcv.imread(toy_img1_path)
    np_e2e_res = readtext(toy_img1)
    assert str_e2e_res == np_e2e_res

    # Multiple images
    toy_img2_path = toy_dir + 'img_2.jpg'
    toy_img2 = mmcv.imread(toy_img2_path)
    toy_imgs = [toy_img1, toy_img2]
    toy_img_paths = [toy_img1_path, toy_img2_path]
    np_e2e_results = readtext(toy_imgs)
    str_e2e_results = readtext(toy_img_paths)
    str_tuple_e2e_results = readtext(tuple(toy_img_paths))
    assert np_e2e_results == str_e2e_results
    assert str_e2e_results == str_tuple_e2e_results

    # Batch mode test
    toy_imgs.append(toy_dir + 'img_3.jpg')
    e2e_res = readtext(toy_imgs)
    full_batch_e2e_res = readtext(toy_imgs, batch_mode=True)
    assert full_batch_e2e_res == e2e_res
    batch_e2e_res = readtext(
        toy_imgs, batch_mode=True, recog_batch_size=2, det_batch_size=2)
    assert batch_e2e_res == full_batch_e2e_res

    # Batch mode test with DBNet only
    full_batch_det_res = mmocr_det.readtext(toy_imgs, batch_mode=True)
    det_res = mmocr_det.readtext(toy_imgs)
    batch_det_res = mmocr_det.readtext(
        toy_imgs, batch_mode=True, single_batch_size=2)
    assert len(full_batch_det_res) == len(det_res)
    assert len(batch_det_res) == len(det_res)
    assert all([
        np.allclose(full_batch_det_res[i]['boundary_result'],
                    det_res[i]['boundary_result'])
        for i in range(len(full_batch_det_res))
    ])
    assert all([
        np.allclose(batch_det_res[i]['boundary_result'],
                    det_res[i]['boundary_result'])
        for i in range(len(batch_det_res))
    ])

    # Batch mode test with CRNN_TPS only (CRNN doesn't support batch inference)
    full_batch_recog_res = mmocr_recog.readtext(toy_imgs, batch_mode=True)
    recog_res = mmocr_recog.readtext(toy_imgs)
    batch_recog_res = mmocr_recog.readtext(
        toy_imgs, batch_mode=True, single_batch_size=2)
    assert full_batch_recog_res == recog_res
    assert batch_recog_res == recog_res

    # Test export
    with tempfile.TemporaryDirectory() as tmpdirname:
        mmocr.readtext(toy_imgs, export=tmpdirname)
        assert len(os.listdir(tmpdirname)) == len(toy_imgs)
    with tempfile.TemporaryDirectory() as tmpdirname:
        mmocr_det.readtext(toy_imgs, export=tmpdirname)
        assert len(os.listdir(tmpdirname)) == len(toy_imgs)
    with tempfile.TemporaryDirectory() as tmpdirname:
        mmocr_recog.readtext(toy_imgs, export=tmpdirname)
        assert len(os.listdir(tmpdirname)) == len(toy_imgs)

    # Test output
    # Single image
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_output = os.path.join(tmpdirname, '1.jpg')
        mmocr.readtext(toy_imgs[0], output=tmp_output)
        assert os.path.exists(tmp_output)
    # Multiple images
    with tempfile.TemporaryDirectory() as tmpdirname:
        mmocr.readtext(toy_imgs, output=tmpdirname)
        assert len(os.listdir(tmpdirname)) == len(toy_imgs)

    # Test imshow
    with mock.patch('mmocr.utils.ocr.mmcv.imshow') as mock_imshow:
        mmocr.readtext(toy_img1_path, imshow=True)
        mock_imshow.assert_called_once()
        mock_imshow.reset_mock()
        mmocr.readtext(toy_imgs, imshow=True)
        assert mock_imshow.call_count == len(toy_imgs)

    # Test print_result
    with io.StringIO() as capturedOutput:
        sys.stdout = capturedOutput
        res = mmocr.readtext(toy_imgs, print_result=True)
        assert json.loads('[%s]' % capturedOutput.getvalue().strip().replace(
            '\n\n', ',').replace("'", '"')) == res
        sys.stdout = sys.__stdout__
    with io.StringIO() as capturedOutput:
        sys.stdout = capturedOutput
        res = mmocr.readtext(toy_imgs, details=True, print_result=True)
        assert json.loads('[%s]' % capturedOutput.getvalue().strip().replace(
            '\n\n', ',').replace("'", '"')) == res
        sys.stdout = sys.__stdout__

    # Test merge
    with mock.patch('mmocr.utils.ocr.stitch_boxes_into_lines') as mock_merge:
        mmocr_det_recog.readtext(toy_imgs, merge=True)
        assert mock_merge.call_count == len(toy_imgs)
