# Named Entity Recognition

## Overview

The structure of the named entity recognition dataset directory is organized as follows.

```text
└── cluener2020
  ├── cluener_predict.json
  ├── dev.json
  ├── README.md
  ├── test.json
  ├── train.json
  └── vocab.txt
```

## Preparation Steps

### CLUENER2020

- Download and extract [cluener_public.zip](https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip) to `cluener2020/`
- Download [vocab.txt](https://download.openmmlab.com/mmocr/data/cluener_public/vocab.txt) and move `vocab.txt` to `cluener2020/`
