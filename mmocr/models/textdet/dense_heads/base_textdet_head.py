# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

from mmcv.runner import BaseModule

from mmocr.registry import MODELS


@MODELS.register_module()
class BaseTextDetHead(BaseModule):
    """Base head for text detection, build the loss and postprocessor.

    Args:
        loss (dict): Config to build loss.
        postprocessor (dict): Config to build postprocessor.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 loss: Dict,
                 postprocessor: Dict,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(loss, dict)
        assert isinstance(postprocessor, dict)

        self.loss = MODELS.build(loss)
        self.postprocessor = MODELS.build(postprocessor)
