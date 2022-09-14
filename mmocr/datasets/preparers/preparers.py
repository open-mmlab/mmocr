# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from functools import partial
from typing import Dict, Tuple

import mmcv
import mmengine

from mmocr.datasets.preparers import BaseDatasetPreparer
from mmocr.registry import TASK_UTILS
from mmocr.utils import txt_loader


@TASK_UTILS.register_module()
class ICDAR2015Preparer(BaseDatasetPreparer):
    """ICDAR 2015 dataset preparer. Check out the dataset meta file for more
    details. [dataset_zoo/icdar2015.yml]

    Official Website: [https://rrc.cvc.uab.es/?ch=4&com=introduction]
    """

    file_mapper = [
        ('ch4_training_images', 'imgs/train'),
        ('ch4_training_localization_transcription_gt', 'annotations/train'),
        ('ch4_test_images', 'imgs/test'),
        ('Challenge4_Test_Task4_GT', 'annotations/test'),
        ('ch4_training_word_images_gt/gt.txt', 'annotations/train.txt'),
        ('ch4_training_word_images_gt', 'imgs/train'),
        ('ch4_test_word_images_gt', 'imgs/test'),
        ('Challenge4_Test_Task3_GT.txt', 'annotations/test.txt')
    ]

    def process_single_det(self, file: Tuple, split: str) -> Dict:
        """Process single image for detection task.

        Args:
            file (Tuple): A tuple (txt_file, img_file) of path to annotation
                file and image file.
            split (str): Split of the dataset.

        Returns:
            Dict: A dict containing the information of single image.
        """
        txt_file, img_file = file
        det_instances = list()
        for anno in txt_loader(txt_file, self.cfg.Data.Annotation.Separator,
                               self.cfg.Data.Annotation.Format,
                               self.cfg.Data.Annotation.Encoding):
            anno = list(anno.values())
            poly = list(map(float, anno[0:-1]))
            text = anno[-1]
            det_instance = self._pack_det_instance(
                poly=poly, ignore=text == self.ignore_flag)
            det_instances.append(det_instance)

        img = mmcv.imread(img_file)
        h, w = img.shape[:2]

        return {
            'instances': det_instances,
            'img_path': f'{split}/' + osp.basename(img_file),
            'height': h,
            'width': w
        }

    def _convert(self) -> None:
        """Convert the annotation to MMOCR format."""
        for split in self.splits:
            if self.task == 'det':
                files = [
                    (osp.join(self.data_root, f'annotations/{split}', 'gt_' +
                              osp.basename(file).replace('jpg', 'txt')), file)
                    for file in self._retrieve_img_list(split)
                ]
                process_func = partial(self.process_single_det, split=split)
                data_list = mmengine.track_parallel_progress(
                    process_func, files, nproc=self.nproc)
                self.data_list[split] = data_list
            elif self.task == 'rec':
                data_list = list()
                for anno in txt_loader(
                        osp.join(self.data_root, 'annotations',
                                 f'{split}.txt'),
                        format='img,text',
                        encoding=self.cfg.Data.Annotation.Encoding):
                    rec_instance = self._pack_rec_instance(
                        anno['text'].strip().replace('"', ''))
                    rec_instance['img_path'] = anno['img']
                    data_list.append(rec_instance)
                self.data_list[split] = data_list
            else:
                raise NotImplementedError
