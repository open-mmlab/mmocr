# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os

import mmcv
import torch
from ts.torch_handler.base_handler import BaseHandler

from mmocr.apis import init_detector, model_inference


class MMOCRHandler(BaseHandler):
    threshold = 0.5

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')

        self.model = init_detector(self.config_file, checkpoint, self.device)
        self.initialized = True

    def preprocess(self, data):
        images = []

        for row in data:
            image = row.get('data') or row.get('body')
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = mmcv.imfrombytes(image)
            images.append(image)

        return images

    def inference(self, data, *args, **kwargs):
        results = model_inference(self.model, data)
        return results

    def postprocess(self, data):
        # Format output following the example OCRHandler format
        output = []
        for image_index, image_result in enumerate(data):
            output.append([])
            if image_result.get('boundary_result', None):
                for bbox in image_result['boundary_result']:
                    output[image_index].append({
                        'bbox': [round(x) for x in bbox[:-1]],
                        'score':
                        float(bbox[-1])
                    })
            if image_result.get('text', None):
                output[image_index].append({
                    'text': image_result['text'],
                    'score': image_result['score']
                })
        return output
