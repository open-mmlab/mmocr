# 命名实体识别（专名识别）

## 概览

命名实体识别任务的数据集，文件目录应按如下配置：

```text
└── cluener2020
  ├── cluener_predict.json
  ├── dev.json
  ├── README.md
  ├── test.json
  ├── train.json
  └── vocab.txt

```

## 准备步骤

### CLUENER2020

- 下载并解压 [cluener_public.zip](https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip) 至 `cluener2020/`。

- 下载 [vocab.txt](https://download.openmmlab.com/mmocr/data/cluener_public/vocab.txt) 然后将 `vocab.txt` 移动到 `cluener2020/` 文件夹下
