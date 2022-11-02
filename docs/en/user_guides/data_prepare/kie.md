# Key Information Extraction\[Deprecated\]

```{warning}
This page is deprecated and all these scripts will be eventually migrated into dataset preparer, a brand new module designed to ease these lengthy dataset preparation steps. [Check it out](./dataset_preparer.md)!
```

## Overview

The structure of the key information extraction dataset directory is organized as follows.

```text
└── wildreceipt
  ├── class_list.txt
  ├── dict.txt
  ├── image_files
  ├── openset_train.txt
  ├── openset_test.txt
  ├── test.txt
  └── train.txt
```

## Preparation Steps

### WildReceipt

- Just download and extract [wildreceipt.tar](https://download.openmmlab.com/mmocr/data/wildreceipt.tar).

### WildReceiptOpenset

- Step0: have [WildReceipt](#WildReceipt) prepared.
- Step1: Convert annotation files to OpenSet format:

```bash
# You may find more available arguments by running
# python tools/data/kie/closeset_to_openset.py -h
python tools/data/kie/closeset_to_openset.py data/wildreceipt/train.txt data/wildreceipt/openset_train.txt
python tools/data/kie/closeset_to_openset.py data/wildreceipt/test.txt data/wildreceipt/openset_test.txt
```

```{note}
You can learn more about the key differences between CloseSet and OpenSet annotations in our [tutorial](../tutorials/kie_closeset_openset.md).
```
