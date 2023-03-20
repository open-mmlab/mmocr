# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional, Sequence, Union

import mmcv
import mmengine.fileio as fileio
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmocr.registry import HOOKS
from mmocr.structures import TextDetDataSample, TextRecogDataSample


# TODO Files with the same name will be overwritten for multi datasets
@HOOKS.register_module()
class VisualizationHook(Hook):
    """Detection Visualization Hook. Used to visualize validation and testing
    process prediction results.

    Args:
        enable (bool): Whether to enable this hook. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. It's only useful for text detection. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Defaults to False.
        wait_time (float): The interval of show in seconds. Defaults
            to 0.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
    """

    def __init__(
        self,
        enable: bool = False,
        interval: int = 50,
        score_thr: float = 0.3,
        show: bool = False,
        draw_pred: bool = False,
        draw_gt: bool = False,
        wait_time: float = 0.,
        backend_args: Optional[dict] = None,
    ) -> None:
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.score_thr = score_thr
        self.show = show
        self.draw_pred = draw_pred
        self.draw_gt = draw_gt
        self.wait_time = wait_time
        self.backend_args = backend_args
        self.enable = enable

    # TODO after MultiDatasetWrapper, rewrites this function and try to merge
    # with after_val_iter and after_test_iter
    def after_val_iter(self, runner: Runner, batch_idx: int,
                       data_batch: Sequence[dict],
                       outputs: Sequence[Union[TextDetDataSample,
                                               TextRecogDataSample]]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (Sequence[dict]): Data from dataloader.
            outputs (Sequence[:obj:`TextDetDataSample` or
                :obj:`TextRecogDataSample`]): Outputs from model.
        """
        # TODO: data_batch does not include annotation information
        if self.enable is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        if total_curr_iter % self.interval == 0:
            for output in outputs:
                img_path = output.img_path
                img_bytes = fileio.get(
                    img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                self._visualizer.add_datasample(
                    osp.splitext(osp.basename(img_path))[0],
                    img,
                    data_sample=output,
                    draw_gt=self.draw_gt,
                    draw_pred=self.draw_pred,
                    show=self.show,
                    wait_time=self.wait_time,
                    pred_score_thr=self.score_thr,
                    step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int,
                        data_batch: Sequence[dict],
                        outputs: Sequence[Union[TextDetDataSample,
                                                TextRecogDataSample]]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (Sequence[dict]): Data from dataloader.
            outputs (Sequence[:obj:`TextDetDataSample` or
                :obj:`TextRecogDataSample`]): Outputs from model.
        """

        if self.enable is False:
            return

        for output in outputs:
            img_path = output.img_path
            img_bytes = fileio.get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            self._visualizer.add_datasample(
                osp.splitext(osp.basename(img_path))[0],
                img,
                data_sample=output,
                show=self.show,
                draw_gt=self.draw_gt,
                draw_pred=self.draw_pred,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                step=batch_idx)
