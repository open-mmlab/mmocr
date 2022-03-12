#!/usr/bin/env bash

# gather models
sed -e '$a\\n' -s ../../configs/kie/*/*.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# 关键信息提取模型' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmocr/tree/master/=g' >kie_models.md
sed -e '$a\\n' -s ../../configs/textdet/*/*.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# 文本检测模型' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmocr/tree/master/=g' >textdet_models.md
sed -e '$a\\n' -s ../../configs/textrecog/*/*.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# 文本识别模型' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmocr/tree/master/=g' >textrecog_models.md
sed -e '$a\\n' -s ../../configs/ner/*/*.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# 命名实体识别模型' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmocr/tree/master/=g' >ner_models.md

# replace special symbols in demo.md
cp ../../demo/README_zh-CN.md demo.md
sed -i 's/:heavy_check_mark:/Yes/g' demo.md && sed -i 's/:x:/No/g' demo.md
