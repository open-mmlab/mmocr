# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import re
import shutil
from abc import abstractmethod
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import mmcv
from mmengine import mkdir_or_exist, track_parallel_progress

from mmocr.utils import bbox2poly, crop_img, list_files, poly2bbox
from .data_preparer import DATA_CONVERTERS, DATA_DUMPERS, DATA_PARSERS


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
        config_path (str): Path to the configs. Defaults to 'configs/'.
    """

    def __init__(self,
                 splits: List,
                 data_root: str,
                 gatherer: Dict,
                 parser: Dict,
                 dumper: Dict,
                 nproc: int,
                 task: str,
                 dataset_name: str,
                 delete: Optional[List] = None,
                 config_path: str = 'configs/'):
        assert isinstance(nproc, int) and nproc > 0, \
            'nproc must be a positive integer.'
        self.splits = splits
        self.data_root = data_root
        self.nproc = nproc
        self.task = task
        self.dataset_name = dataset_name
        self.delete = delete
        self.config_path = config_path
        self.img_dir = f'{task}_imgs'
        parser.update(dict(nproc=nproc))
        dumper.update(dict(task=task, dataset_name=dataset_name))
        self.parser = DATA_PARSERS.build(parser)
        self.dumper = DATA_DUMPERS.build(dumper)
        gather_type = gatherer.pop('type')
        self.gatherer_args = gatherer
        if gather_type == 'pair_gather':
            self.gatherer = self.pair_gather
        elif gather_type == 'mono_gather':
            self.gatherer = self.mono_gather
        else:
            raise NotImplementedError

    def __call__(self):
        """Process the data."""
        # Convert and dump annotations to MMOCR format
        for split in self.splits:
            print(f'Parsing {split} split...')
            # Gather the info such as file names required by parser
            img_path = osp.join(self.data_root, self.img_dir, split)
            ann_path = osp.join(self.data_root, 'annotations')
            gatherer_args = dict(
                img_path=img_path, ann_path=ann_path, split=split)
            gatherer_args.update(self.gatherer_args)
            files = self.gatherer(**gatherer_args)
            # Convert dataset annotations to MMOCR format
            samples = self.parser.parse_files(files, split)
            print(f'Packing {split} annotations...')
            func = partial(self.pack_instance, split=split)
            samples = track_parallel_progress(func, samples, nproc=self.nproc)
            samples = self.add_meta(samples)
            # Dump annotation files
            self.dumper.dump(samples, self.data_root, split)
        self.generate_dataset_config()
        self.clean()

    def _generate_dataset_config_string(self) -> str:
        """Generate the dataset config string.

        Returns:
            str: The dataset config string.
        """
        return None

    def generate_dataset_config(self) -> None:
        """Generate dataset config file. Dataset config is a python file that
        contains the dataset information.

        Examples:
        Generated dataset config
        >>> ic15_rec_data_root = 'data/icdar2015/'
        >>> icdar2015_textrecog_train = dict(
        >>>     type='OCRDataset',
        >>>     data_root=ic15_rec_data_root,
        >>>     ann_file='textrecog_train.json',
        >>>     test_mode=False,
        >>>     pipeline=None)
        >>> icdar2015_textrecog_test = dict(
        >>>     type='OCRDataset',
        >>>     data_root=ic15_rec_data_root,
        >>>     ann_file='textrecog_test.json',
        >>>     test_mode=True,
        >>>     pipeline=None)

        Args:
            dataset_config (Dict): A dict contains the dataset config string of
            each split.
        """
        dataset_config = self._generate_dataset_config_string()
        if dataset_config is None:
            return
        cfg_path = osp.join(self.config_path, self.task, '_base_', 'datasets',
                            f'{self.dataset_name}.py')
        if osp.exists(cfg_path):
            while True:
                c = input(f'{cfg_path} already exists, overwrite? (Y/n) ') \
                    or 'Y'
                if c.lower() == 'y':
                    break
                if c.lower() == 'n':
                    return
        mkdir_or_exist(osp.dirname(cfg_path))
        with open(cfg_path, 'w') as f:
            f.write(
                f'{self.dataset_name}_{self.task}_data_root = \'{self.data_root}\'\n'  # noqa: E501
            )
        for split in self.splits:
            with open(cfg_path, 'a') as f:
                f.write([split])

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

    def mono_gather(self, ann_path: str, mapping: str, split: str,
                    **kwargs) -> str:
        """Gather the dataset file. Specifically for the case that only one
        annotation file is needed. For example,

            img_001.jpg \
            img_002.jpg ---> train.json
            img_003.jpg /

        Args:
            anno_path (str): Path to the annotations.
            mapping (str): Mapping rule of the annotation names. For example,
                "f'{split}.json'" will return 'train.json' when the split is
                'train'.
            split (str): The current split.

        Returns:
            str: Path to the annotation file.
        """

        return osp.join(ann_path, eval(mapping))

    def pair_gather(self, img_path: str, suffixes: List, rule: Sequence,
                    **kwargs) -> List[Tuple]:
        """Gather the dataset files. Specifically for the paired annotations.
        That is to say, each image has a corresponding annotation file. For
        example,

            img_1.jpg <---> gt_img_1.txt
            img_2.jpg <---> gt_img_2.txt
            img_3.jpg <---> gt_img_3.txt

        Args:
            img_path (str): Path to the images.
            suffixes (List[str]): File suffixes that used for searching.
            rule (Sequence): The rule for pairing the files. The
                    first element is the matching pattern for the file, and the
                    second element is the replacement pattern, which should
                    be a regular expression. For example, to map the image
                    name img_1.jpg to the annotation name gt_img_1.txt,
                    the rule is
                        [r'img_(\d+)\.([jJ][pP][gG])', r'gt_img_\1.txt'] # noqa: W605 E501

        Returns:
            List[Tuple]: A list of tuples (img_path, ann_path).
        """
        files = list()
        for file in list_files(img_path, suffixes):
            file2 = re.sub(rule[0], rule[1], osp.basename(file))
            file2 = file.replace(osp.basename(file), file2)
            file2 = file2.replace(self.img_dir, 'annotations')
            files.append((file, file2))

        return files

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
            task='textdet')

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

    def _generate_dataset_config_string(self) -> str:
        cfg = ''
        for split in self.splits:
            cfg = f'\n{self.dataset_name}_{self.task}_{split} = dict(\n'
            cfg += '    type=\'OCRDataset\',\n'
            cfg += '    data_root=' + f'{self.dataset_name}_{self.task}_data_root,\n'  # noqa: E501
            cfg += f'    ann_file=\'{self.task}_{split}.json\',\n'
            if split == 'train':
                cfg += '    filter_cfg=dict(filter_empty_gt=True, min_size=32),\n'  # noqa: E501
            elif split in ['test', 'val']:
                cfg += '    test_mode=True,\n'
            cfg += '    pipeline=None)\n'

        return cfg


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
        super().__init__(
            splits=splits,
            data_root=data_root,
            gatherer=gatherer,
            parser=parser,
            dumper=dumper,
            dataset_name=dataset_name,
            nproc=nproc,
            delete=delete,
            task='textspotting')
        # Textspotting task shares the same images with textdet task
        self.img_dir = 'textdet_imgs'

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
            instances=packed_instances, img_path=img_path, height=h, width=w)

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
        super().__init__(
            splits=splits,
            data_root=data_root,
            gatherer=gatherer,
            parser=parser,
            dumper=dumper,
            dataset_name=dataset_name,
            nproc=nproc,
            delete=delete)
        self.lepr = long_edge_pad_ratio
        self.sepr = short_edge_pad_ratio
        # Crop converter crops the images of textdet to patches
        self.img_dir = 'textdet_imgs'
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
