import cv2
import torch
import torch.nn as nn
from collections import OrderedDict
import os
import numpy as np
from argparse import ArgumentParser

import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.apis import init_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

from mmocr.apis.inference import model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401
import mmocr.models.textrecog.backbones as K

config = './seg_r31_1by16_fpnocr_academic.py'
checkpoint = './seg_r31_1by16_fpnocr_academic-72235b11.pth'
if isinstance(config, str):
    config = mmcv.Config.fromfile(config)

elif not isinstance(config, mmcv.Config):
    raise TypeError('config must be a filename or Config object, '
                    f'but got {type(config)}')

config.model.pretrained = None
config.model.train_cfg = None
model = build_detector(config.model, test_cfg=config.get('test_cfg'))
state_dict = torch.load(
    checkpoint, map_location=lambda storage, loc: storage)
# model.CLASSES = state_dict['meta']['CLASSES']
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    new_state_dict.update(v)

del [new_state_dict['CLASSES']]
# # # load params
# print(new_state_dict.keys())
model.load_state_dict(new_state_dict)
model.eval()
# # convert pth-model to pt-model
# example = torch.rand(1, 1, 32, 512).cpu()
# traced_script_module = torch.jit.trace(model, example)
# traced_script_module.save("./src/crnn.pt")

model.to(device='cuda:0')

if config.data.test['type'] == 'ConcatDataset':
    config.data.test.pipeline = config.data.test['datasets'][
        0].pipeline

imgs = cv2.imread('G:/OCR/mmocr-main/demo/images/1.png')
imgs = np.uint8(imgs)
imgs = [imgs]
is_ndarray = isinstance(imgs[0], np.ndarray)
device = next(model.parameters()).device

config = config.copy()
# set loading pipeline type
config.data.test.pipeline[0].type = 'LoadImageFromNdarray'

config.data.test.pipeline = replace_ImageToTensor(config.data.test.pipeline)
test_pipeline = Compose(config.data.test.pipeline)

datas = []
for img in imgs:
    # prepare data
    if is_ndarray:
        # directly add img
        data = dict(img=img)
    else:
        # add information into dict
        data = dict(img_info=dict(filename=img), img_prefix=None)

    data = test_pipeline(data)
    datas.append(data)

data = collate(datas, samples_per_gpu=1)

# process img_metas

data['img_metas'] = data['img_metas'].data

data['img'] = data['img'].data
device = next(model.parameters()).device  # model device
data = scatter(data, [device])[0]
imm = cv2.resize(imgs[0], (192, 64))

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
new_img = imm / 255.

std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]

_std = np.array(std).reshape((1, 1, 3))
_mean = np.array(mean).reshape((1, 1, 3))

new_img = (new_img - _mean) / _std
# print(new_img[0][0])
# print(data['img'][0].permute(1, 2, 0)[0][0])
# forward the model
with torch.no_grad():
    # results = model(return_loss=False, rescale=True, **data)
    # results = model(data['img'])
    # print(data)
    traced_script_module = torch.jit.trace(model, data['img'])
    traced_script_module.save("./segocr.pt")
