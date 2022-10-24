# Dataset Preparer

## One-click data preparation script

MMOCR provides a unified one-stop data preparation script `prepare_dataset.py`.

Only one line of command is needed to complete the data download, decompression, and format conversion.

```bash
python tools/dataset_converters/prepare_dataset.py [$DATASET_NAME] --task [$TASK] --nproc [$NPROC]
```

| ARGS         | Type | Description                                                                                                                               |
| ------------ | ---- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| dataset_name | str  | (required) dataset name.                                                                                                                  |
| --task       | str  | Convert the dataset to the format of a specified task supported by MMOCR. options are: 'textdet', 'textrecog', 'textspotting', and 'kie'. |
| --nproc      | int  | Number of processors to be used. Defaults to 4.                                                                                           |

For example, the following command shows how to use the script to prepare the ICDAR2015 dataset for text detection task.

```bash
python tools/dataset_converters/prepare_dataset.py icdar2015 --task textdet
```

Also, the script supports preparing multiple datasets at the same time. For example, the following command shows how to prepare the ICDAR2015 and TotalText datasets for text recognition task.

```bash
python tools/dataset_converters/prepare_dataset.py icdar2015 totaltext --task textrecog
```

The following table shows the supported datasets.

| Dataset Name | Text Detection | Text Recognition | Text Spotting | KIE |
| ------------ | -------------- | ---------------- | ------------- | --- |
| icdar2015    | ✓              | ✓                | ✓             |     |
| totaltext    | ✓              | ✓                | ✓             |     |
| wildreceipt  | ✓              | ✓                | ✓             | ✓   |

## Advanced Usage\[Coming Soon\]
