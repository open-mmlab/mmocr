import numpy as np
import torchvision.transforms.functional as TF

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class ToTensorNER:

    def __init__(self):
        pass

    def __call__(self, results):
        results['input_ids'] = TF.to_tensor(
            np.array(results['input_ids'].copy()))
        results['attention_mask'] = TF.to_tensor(
            np.array(results['attention_mask'].copy()))
        results['token_type_ids'] = TF.to_tensor(
            np.array(results['token_type_ids'].copy()))
        return results
