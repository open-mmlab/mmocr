# 关键信息提取

## 概览

关键信息提取任务的数据集，文件目录应按如下配置：

```text
└── wildreceipt
  ├── class_list.txt
  ├── dict.txt
  ├── image_files
  ├── test.txt
  └── train.txt
```

## 准备步骤

### WildReceipt

- 下载并解压 [wildreceipt.tar](https://download.openmmlab.com/mmocr/data/wildreceipt.tar)

### WildReceiptOpenset

- 准备好 [WildReceipt](#WildReceipt)。
- 转换 WildReceipt 成 OpenSet 格式:
```bash
# 你可以运行以下命令以获取更多可用参数：
# python tools/data/kie/closeset_to_openset.py -h
python tools/data/kie/closeset_to_openset.py data/wildreceipt/train.txt data/wildreceipt/openset_train.txt
python tools/data/kie/closeset_to_openset.py data/wildreceipt/test.txt data/wildreceipt/openset_test.txt
```

```{note}
[这篇教程](../tutorials/kie_closeset_openset.md)里讲述了更多 CloseSet 和 OpenSet 数据格式之间的区别。
```
