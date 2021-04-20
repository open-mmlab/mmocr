import os
import os.path as osp
import shutil
import urllib


def prepare_ckpt(checkpoint_file, checkpoint_url):
    """Prepare checkpoint file."""
    if not osp.exists(checkpoint_file):
        print(f'Downloading {checkpoint_url} ...')
        local_filename, _ = urllib.request.urlretrieve(checkpoint_url)
        os.makedirs(osp.dirname(checkpoint_file), exist_ok=True)
        shutil.move(local_filename, checkpoint_file)
        print(f'Saved as {checkpoint_file}')
    else:
        print(f'Using existing checkpoint {checkpoint_file}')


def prepare_det_model(model_type='psenet'):
    """Prepare text detection config and checkpoint files."""
    config_mapping = {
        'psenet':
        './configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py',
        'panet':
        './configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py',
        'dbnet':
        './configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py',
        'textsnake':
        './configs/textdet/textsnake/textsnake_r50_fpn_unet_1200e_ctw1500.py',
        'maskrcnn':
        './configs/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015.py'
    }
    ckpt_mapping = {
        'psenet':
        './checkpoints/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth',
        'panet':
        './checkpoints/'
        'panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth',
        'dbnet':
        './checkpoints/'
        'dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-aa96e477.pth',
        'textsnake':
        './checkpoints/'
        'textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth',
        'maskrcnn':
        './checkpoints/'
        'mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth'
    }
    ckpt_url_mapping = {
        'psenet':
        'https://download.openmmlab.com/mmocr/textdet/psenet/'
        'psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth',
        'panet':
        'https://download.openmmlab.com/mmocr/textdet/panet/'
        'panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth',
        'dbnet':
        'https://download.openmmlab.com/mmocr/textdet/dbnet/'
        'dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-aa96e477.pth',
        'textsnake':
        'https://download.openmmlab.com/mmocr/textdet/textsnake/'
        'textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth',
        'maskrcnn':
        'https://download.openmmlab.com/mmocr/textdet/maskrcnn/'
        'mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth'
    }
    if model_type not in config_mapping:
        raise Exception(f'model type {model_type} not support')

    config_file = config_mapping[model_type]
    checkpoint_file = ckpt_mapping[model_type]
    checkpoint_url = ckpt_url_mapping[model_type]
    prepare_ckpt(checkpoint_file, checkpoint_url)

    return config_file, checkpoint_file


def prepare_recog_model(model_type='sar'):
    """Prepare text recognition config and checkpoint files."""
    config_mapping = {
        'sar': './configs/textrecog/sar/sar_r31_parallel_decoder_academic.py',
        'crnn': './configs/textrecog/crnn/crnn_academic_dataset.py',
        'robust_scanner':
        './configs/textrecog/robust_scanner/robustscanner_r31_academic.py',
        'seg': './configs/textrecog/seg/seg_r31_1by16_fpnocr_academic.py',
        'nrtr': './configs/textrecog/nrtr/nrtr_r31_1by16_1by8_academic.py'
    }
    ckpt_mapping = {
        'sar': './checkpoints/sar_r31_parallel_decoder_academic-dba3a4a3.pth',
        'crnn': './checkpoints/crnn_academic-a723a1c5.pth',
        'robust_scanner':
        './checkpoints/robustscanner_r31_academic-5f05874f.pth',
        'seg': './checkpoints/seg_r31_1by16_fpnocr_academic-72235b11.pth',
        'nrtr': './checkpoints/nrtr_r31_academic_20210406-954db95e.pth'
    }
    ckpt_url_mapping = {
        'sar':
        'https://download.openmmlab.com/mmocr/textrecog/sar/'
        'sar_r31_parallel_decoder_academic-dba3a4a3.pth',
        'crnn':
        'https://download.openmmlab.com/mmocr/textrecog/crnn/'
        'crnn_academic-a723a1c5.pth',
        'robust_scanner':
        'https://download.openmmlab.com/mmocr/textrecog/'
        'robustscanner/robustscanner_r31_academic-5f05874f.pth',
        'seg':
        'https://download.openmmlab.com/mmocr/textrecog/seg/'
        'seg_r31_1by16_fpnocr_academic-72235b11.pth',
        'nrtr':
        'https://download.openmmlab.com/mmocr/textrecog/nrtr/'
        'nrtr_r31_academic_20210406-954db95e.pth'
    }
    if model_type not in config_mapping:
        raise Exception(f'model type {model_type} not support')

    config_file = config_mapping[model_type]
    checkpoint_file = ckpt_mapping[model_type]
    checkpoint_url = ckpt_url_mapping[model_type]
    prepare_ckpt(checkpoint_file, checkpoint_url)

    return config_file, checkpoint_file
