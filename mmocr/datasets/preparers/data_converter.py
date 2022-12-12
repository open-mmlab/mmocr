# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import shutil
from abc import abstractmethod
from functools import partial
from typing import Dict, List, Optional, Tuple

import mmcv
from mmengine import mkdir_or_exist, track_parallel_progress

from mmocr.utils import bbox2poly, crop_img, poly2bbox
from .data_preparer import (DATA_CONVERTERS, DATA_DUMPERS, DATA_GATHER,
                            DATA_PARSERS)


class BaseDataConverter:
    """Base class for data processor.

    Args:
        splits (List): A list of splits to be processed.
        data_root (str): Path to the data root.
        gatherer (Dict): Config dict for gathering the dataset files.
        parser (Dict): Config dict for parsing the dataset files.
        dumper (Dict): Config dict for dumping the dataset files.
        nproc (int): Number of processes to process the data.
        task (str): Task of the dataset.
        dataset_name (str): Dataset name.
        delete (Optional[List]): A list of files to be deleted after
            conversion.
    """

    def __init__(
        self,
        splits: List,
        data_root: str,
        gatherer: Dict,
        parser: Dict,
        dumper: Dict,
        nproc: int,
        task: str,
        dataset_name: str,
        img_dir: str,
        delete: Optional[List] = None,
        config_path: str = 'configs/',
    ):
        assert isinstance(nproc, int) and nproc > 0, \
            'nproc must be a positive integer.'
        self.splits = splits
        self.data_root = data_root
        self.nproc = nproc
        self.task = task
        self.dataset_name = dataset_name
        self.delete = delete
        self.config_path = config_path
        if img_dir is None:
            self.img_dir = f'{task}_imgs'
        else:
            self.img_dir = img_dir
        parser.setdefault('nproc', default=nproc)
        self.parser = DATA_PARSERS.build(parser)
        dumper.setdefault('task', default=task)
        self.dumper = DATA_DUMPERS.build(dumper)
        gatherer.setdefault('splits', default=splits)
        gatherer.setdefault(
            'img_root', default=osp.join(self.data_root, self.img_dir))
        gatherer.setdefault(
            'ann_root', default=osp.join(self.data_root, 'annotations'))
        self.gatherer = DATA_GATHER.build(gatherer)

    def __call__(self):
        """Process the data.

        Returns:
            Dict: A dict that maps each split to the path of the annotation
                files.
        """
        gather_results = self.gatherer()
        assert len(gather_results) == len(self.splits), \
            'The number of splits does not match the number of gather results.'

        # Convert and dump annotations to MMOCR format
        for split, gather_result in zip(self.splits, gather_results):
            print(f'Parsing {split} split...')
            # Convert dataset annotations to MMOCR format
            samples = self.parser.parse_files(*gather_result, split=split)
            print(f'Packing {split} annotations...')
            func = partial(self.pack_instance, split=split)
            samples = track_parallel_progress(func, samples, nproc=self.nproc)
            samples = self.add_meta(samples)
            # Dump annotation files
            self.dumper.dump(samples, self.data_root, split)
        self.clean()

    @abstractmethod
    def pack_instance(self, sample: Tuple, split: str) -> Dict:
        """Pack the parsed annotation info to an MMOCR format instance.

        Args:
            sample (Tuple): A tuple of (img_file, ann_file).
               - img_path (str): Path to image file.
               - instances (Sequence[Dict]): A list of converted annos.
            split (str): The split of the instance.

        Returns:
            Dict: An MMOCR format instance.
        """

    @abstractmethod
    def add_meta(self, sample: List) -> Dict:
        """Add meta information to the sample.

        Args:
            sample (List): A list of samples of the dataset.

        Returns:
            Dict: A dict contains the meta information and samples.
        """

    def clean(self) -> None:
        for d in self.delete:
            delete_file = osp.join(self.data_root, d)
            if osp.exists(delete_file):
                shutil.rmtree(delete_file)


@DATA_CONVERTERS.register_module()
class TextDetDataConverter(BaseDataConverter):
    """Text detection data converter.

    Args:
        splits (List): A list of splits to be processed.
        data_root (str): Path to the data root.
        gatherer (Dict): Config dict for gathering the dataset files.
        parser (Dict): Config dict for parsing the dataset files.
        dumper (Dict): Config dict for dumping the dataset files.
        dataset_name (str): Name of the dataset.
        nproc (int): Number of processes to process the data.
        delete (Optional[List]): A list of files to be deleted after
            conversion. Defaults to ['annotations].
    """

    def __init__(self,
                 splits: List,
                 data_root: str,
                 gatherer: Dict,
                 parser: Dict,
                 dumper: Dict,
                 dataset_name: str,
                 nproc: int,
                 delete: List = ['annotations']) -> None:
        super().__init__(
            splits=splits,
            data_root=data_root,
            gatherer=gatherer,
            parser=parser,
            dumper=dumper,
            dataset_name=dataset_name,
            nproc=nproc,
            delete=delete,
            task='textdet',
            img_dir='textdet_imgs')

    def pack_instance(self,
                      sample: Tuple,
                      split: str,
                      bbox_label: int = 0) -> Dict:
        """Pack the parsed annotation info to an MMOCR format instance.

        Args:
            sample (Tuple): A tuple of (img_file, instances).
               - img_path (str): Path to the image file.
               - instances (Sequence[Dict]): A list of converted annos. Each
                    element should be a dict with the following keys:
                    - 'poly' or 'box'
                    - 'ignore'
                    - 'bbox_label' (optional)
            split (str): The split of the instance.

        Returns:
            Dict: An MMOCR format instance.
        """

        img_path, instances = sample

        img = mmcv.imread(img_path)
        h, w = img.shape[:2]

        packed_instances = list()
        for instance in instances:
            poly = instance.get('poly', None)
            box = instance.get('box', None)
            assert box or poly
            packed_sample = dict(
                polygon=poly if poly else list(
                    bbox2poly(box).astype('float64')),
                bbox=box if box else list(poly2bbox(poly).astype('float64')),
                bbox_label=bbox_label,
                ignore=instance['ignore'])
            packed_instances.append(packed_sample)

        packed_instances = dict(
            instances=packed_instances,
            img_path=img_path.replace(self.data_root + '/', ''),
            height=h,
            width=w)

        return packed_instances

    def add_meta(self, sample: List) -> Dict:
        meta = {
            'metainfo': {
                'dataset_type': 'TextDetDataset',
                'task_name': 'textdet',
                'category': [{
                    'id': 0,
                    'name': 'text'
                }]
            },
            'data_list': sample
        }
        return meta


@DATA_CONVERTERS.register_module()
class TextSpottingDataConverter(BaseDataConverter):
    """Text spotting data converter.

    Args:
        splits (List): A list of splits to be processed.
        data_root (str): Path to the data root.
        gatherer (Dict): Config dict for gathering the dataset files.
        parser (Dict): Config dict for parsing the dataset files.
        dumper (Dict): Config dict for dumping the dataset files.
        dataset_name (str): Name of the dataset.
        nproc (int): Number of processes to process the data.
        delete (Optional[List]): A list of files to be deleted after
            conversion. Defaults to ['annotations'].
    """

    def __init__(self,
                 splits: List,
                 data_root: str,
                 gatherer: Dict,
                 parser: Dict,
                 dumper: Dict,
                 dataset_name: str,
                 nproc: int,
                 delete: List = ['annotations']) -> None:
        # Textspotting task shares the same images with textdet task
        super().__init__(
            splits=splits,
            data_root=data_root,
            gatherer=gatherer,
            parser=parser,
            dumper=dumper,
            dataset_name=dataset_name,
            nproc=nproc,
            delete=delete,
            task='textspotting',
            img_dir='textdet_imgs')

    def pack_instance(self,
                      sample: Tuple,
                      split: str,
                      bbox_label: int = 0) -> Dict:
        """Pack the parsed annotation info to an MMOCR format instance.

        Args:
            sample (Tuple): A tuple of (img_file, ann_file).
               - img_path (str): Path to image file.
               - instances (Sequence[Dict]): A list of converted annos. Each
                    element should be a dict with the following keys:
                    - 'poly' or 'box'
                    - 'text'
                    - 'ignore'
                    - 'bbox_label' (optional)
            split (str): The split of the instance.

        Returns:
            Dict: An MMOCR format instance.
        """

        img_path, instances = sample

        img = mmcv.imread(img_path)
        h, w = img.shape[:2]

        packed_instances = list()
        for instance in instances:
            assert 'text' in instance, 'Text is not found in the instance.'
            poly = instance.get('poly', None)
            box = instance.get('box', None)
            assert box or poly
            packed_sample = dict(
                polygon=poly if poly else list(
                    bbox2poly(box).astype('float64')),
                bbox=box if box else list(poly2bbox(poly).astype('float64')),
                bbox_label=bbox_label,
                ignore=instance['ignore'],
                text=instance['text'])
            packed_instances.append(packed_sample)

        packed_instances = dict(
            instances=packed_instances,
            img_path=img_path.replace(self.data_root + '/', ''),
            height=h,
            width=w)

        return packed_instances

    def add_meta(self, sample: List) -> Dict:
        meta = {
            'metainfo': {
                'dataset_type': 'TextSpottingDataset',
                'task_name': 'textspotting',
                'category': [{
                    'id': 0,
                    'name': 'text'
                }]
            },
            'data_list': sample
        }
        return meta


@DATA_CONVERTERS.register_module()
class TextRecogDataConverter(BaseDataConverter):
    """Text recognition data converter.

    Args:
        splits (List): A list of splits to be processed.
        data_root (str): Path to the data root.
        gatherer (Dict): Config dict for gathering the dataset files.
        parser (Dict): Config dict for parsing the dataset annotations.
        dumper (Dict): Config dict for dumping the dataset files.
        dataset_name (str): Name of the dataset.
        nproc (int): Number of processes to process the data.
        delete (Optional[List]): A list of files to be deleted after
            conversion. Defaults to ['annotations].
    """

    def __init__(self,
                 splits: List,
                 data_root: str,
                 gatherer: Dict,
                 parser: Dict,
                 dumper: Dict,
                 dataset_name: str,
                 nproc: int,
                 delete: List = ['annotations']):
        super().__init__(
            splits=splits,
            data_root=data_root,
            gatherer=gatherer,
            parser=parser,
            dumper=dumper,
            dataset_name=dataset_name,
            nproc=nproc,
            task='textrecog',
            img_dir='textrecog_imgs',
            delete=delete)

    def pack_instance(self, sample: Tuple, split: str) -> Dict:
        """Pack the text info to a recognition instance.

        Args:
            samples (Tuple): A tuple of (img_name, text).
            split (str): The split of the instance.

        Returns:
            Dict: The packed instance.
        """

        img_name, text = sample
        packed_instance = dict(
            instances=[dict(text=text)],
            img_path=osp.join(self.img_dir, split, osp.basename(img_name)))

        return packed_instance

    def add_meta(self, sample: List) -> Dict:
        meta = {
            'metainfo': {
                'dataset_type': 'TextRecogDataset',
                'task_name': 'textrecog'
            },
            'data_list': sample
        }
        return meta


@DATA_CONVERTERS.register_module()
class TextRecogCropConverter(TextRecogDataConverter):
    """Text recognition crop converter. This converter will crop the text from
    the original image. The parser used for this Converter should be a TextDet
    parser.

    Args:
        splits (List): A list of splits to be processed.
        data_root (str): Path to the data root.
        gatherer (Dict): Config dict for gathering the dataset files.
        parser (Dict): Config dict for parsing the dataset annotations.
        dumper (Dict): Config dict for dumping the dataset files.
        dataset_name (str): Name of the dataset.
        nproc (int): Number of processes to process the data.
        long_edge_pad_ratio (float): The ratio of padding the long edge of the
            cropped image. Defaults to 0.1.
        short_edge_pad_ratio (float): The ratio of padding the short edge of
            the cropped image. Defaults to 0.05.
        delete (Optional[List]): A list of files to be deleted after
            conversion. Defaults to ['annotations].
    """

    def __init__(self,
                 splits: List,
                 data_root: str,
                 gatherer: Dict,
                 parser: Dict,
                 dumper: Dict,
                 dataset_name: str,
                 nproc: int,
                 long_edge_pad_ratio: float = 0.0,
                 short_edge_pad_ratio: float = 0.0,
                 delete: List = ['annotations']):
        super(TextRecogDataConverter, self).__init__(
            task='textrecog',
            splits=splits,
            data_root=data_root,
            gatherer=gatherer,
            parser=parser,
            dumper=dumper,
            dataset_name=dataset_name,
            nproc=nproc,
            img_dir='textdet_imgs',
            delete=delete)
        self.lepr = long_edge_pad_ratio
        self.sepr = short_edge_pad_ratio
        # Crop converter crops the images of textdet to patches
        self.cropped_img_dir = 'textrecog_imgs'
        self.crop_save_path = osp.join(self.data_root, self.cropped_img_dir)
        mkdir_or_exist(self.crop_save_path)
        for split in splits:
            mkdir_or_exist(osp.join(self.crop_save_path, split))

    def pack_instance(self, sample: Tuple, split: str) -> List:
        """Crop patches from image.

        Args:
            samples (Tuple): A tuple of (img_name, text).
            split (str): The split of the instance.

        Return:
            List: The list of cropped patches.
        """

        def get_box(instance: Dict) -> List:
            if 'box' in instance:
                return bbox2poly(instance['box']).tolist()
            if 'poly' in instance:
                return bbox2poly(poly2bbox(instance['poly'])).tolist()

        data_list = []
        img_path, instances = sample
        img = mmcv.imread(img_path)
        for i, instance in enumerate(instances):
            box, text = get_box(instance), instance['text']
            if instance['ignore']:
                continue
            patch = crop_img(img, box, self.lepr, self.sepr)
            if patch.shape[0] == 0 or patch.shape[1] == 0:
                continue
            patch_name = osp.splitext(
                osp.basename(img_path))[0] + f'_{i}' + osp.splitext(
                    osp.basename(img_path))[1]
            dst_path = osp.join(self.crop_save_path, split, patch_name)
            mmcv.imwrite(patch, dst_path)
            rec_instance = dict(
                instances=[dict(text=text)],
                img_path=osp.join(self.cropped_img_dir, split, patch_name))
            data_list.append(rec_instance)

        return data_list

    def add_meta(self, sample: List) -> Dict:
        # Since the TextRecogCropConverter packs all of the patches in a single
        # image into a list, we need to flatten the list.
        sample = [item for sublist in sample for item in sublist]
        return super().add_meta(sample)


@DATA_CONVERTERS.register_module()
class WildReceiptConverter(BaseDataConverter):
    """MMOCR only supports wildreceipt dataset for KIE task now. This converter
    converts the wildreceipt dataset from close set to open set.

    Args:
        splits (List): A list of splits to be processed.
        data_root (str): Path to the data root.
        gatherer (Dict): Config dict for gathering the dataset files.
        parser (Dict): Config dict for parsing the dataset annotations.
        dumper (Dict): Config dict for dumping the dataset files.
        nproc (int): Number of processes to process the data.
        delete (Optional[List]): A list of files to be deleted after
            conversion. Defaults to ['annotations].
        merge_bg_others (bool): If True, give the same label to "background"
                class and "others" class. Defaults to True.
        ignore_idx (int): Index for ``ignore`` class. Defaults to 0.
        others_idx (int): Index for ``others`` class. Defaults to 25.
    """

    def __init__(self,
                 splits: List,
                 data_root: str,
                 gatherer: Dict,
                 parser: Dict,
                 dumper: Dict,
                 dataset_name: str,
                 nproc: int,
                 delete: Optional[List] = None,
                 merge_bg_others: bool = False,
                 ignore_idx: int = 0,
                 others_idx: int = 25):
        self.ignore_idx = ignore_idx
        self.others_idx = others_idx
        self.merge_bg_others = merge_bg_others
        parser.update(dict(ignore=ignore_idx))
        super().__init__(
            splits=splits,
            data_root=data_root,
            gatherer=gatherer,
            parser=parser,
            dumper=dumper,
            dataset_name=dataset_name,
            nproc=nproc,
            task='kie',
            img_dir='kie_imgs',
            delete=delete)

    def add_meta(self, samples: List) -> List:
        """No meta info is required for the wildreceipt dataset."""
        return samples

    def pack_instance(self, sample: str, split: str):
        """Pack line-json str of close set to line-json str of open set.

        Args:
            sample (str): The string to be deserialized to
                the close set dictionary object.
            split (str): The split of the instance.
        """
        # Two labels at the same index of the following two lists
        # make up a key-value pair. For example, in wildreceipt,
        # closeset_key_inds[0] maps to "Store_name_key"
        # and closeset_value_inds[0] maps to "Store_addr_value".
        closeset_key_inds = list(range(2, self.others_idx, 2))
        closeset_value_inds = list(range(1, self.others_idx, 2))

        openset_node_label_mapping = {
            'bg': 0,
            'key': 1,
            'value': 2,
            'others': 3
        }
        if self.merge_bg_others:
            openset_node_label_mapping['others'] = openset_node_label_mapping[
                'bg']

        closeset_obj = json.loads(sample)
        openset_obj = {
            'file_name': closeset_obj['file_name'],
            'height': closeset_obj['height'],
            'width': closeset_obj['width'],
            'annotations': []
        }

        edge_idx = 1
        label_to_edge = {}
        for anno in closeset_obj['annotations']:
            label = anno['label']
            if label == self.ignore_idx:
                anno['label'] = openset_node_label_mapping['bg']
                anno['edge'] = edge_idx
                edge_idx += 1
            elif label == self.others_idx:
                anno['label'] = openset_node_label_mapping['others']
                anno['edge'] = edge_idx
                edge_idx += 1
            else:
                edge = label_to_edge.get(label, None)
                if edge is not None:
                    anno['edge'] = edge
                    if label in closeset_key_inds:
                        anno['label'] = openset_node_label_mapping['key']
                    elif label in closeset_value_inds:
                        anno['label'] = openset_node_label_mapping['value']
                else:
                    tmp_key = 'key'
                    if label in closeset_key_inds:
                        label_with_same_edge = closeset_value_inds[
                            closeset_key_inds.index(label)]
                    elif label in closeset_value_inds:
                        label_with_same_edge = closeset_key_inds[
                            closeset_value_inds.index(label)]
                        tmp_key = 'value'
                    edge_counterpart = label_to_edge.get(
                        label_with_same_edge, None)
                    if edge_counterpart is not None:
                        anno['edge'] = edge_counterpart
                    else:
                        anno['edge'] = edge_idx
                        edge_idx += 1
                    anno['label'] = openset_node_label_mapping[tmp_key]
                    label_to_edge[label] = anno['edge']

        openset_obj['annotations'] = closeset_obj['annotations']

        return json.dumps(openset_obj, ensure_ascii=False)
