#!/usr/bin/env bash

sed -i '$a\\n' ../configs/kie/*/*.md
sed -i '$a\\n' ../configs/textdet/*/*.md
sed -i '$a\\n' ../configs/textrecog/*/*.md
sed -i '$a\\n' ../configs/ner/*/*.md

# gather models
cat ../configs/kie/*/*.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Key Information Extraction Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmocr/tree/master/=g' >kie_models.md
cat ../configs/textdet/*/*.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Text Detection Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmocr/tree/master/=g' >textdet_models.md
cat ../configs/textrecog/*/*.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Text Recognition Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmocr/tree/master/=g' >textrecog_models.md
cat ../configs/ner/*/*.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Named Entity Recognition Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmocr/tree/master/=g' >ner_models.md
# cat ../demo/docs_zh_CN/*_demo.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Demo' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmocr/tree/master/=g' >demo.md

# replace speical symbols in demo.md
sed -i 's/:heavy_check_mark:/Yes/g' demo.md && sed -i 's/:x:/No/g' demo.md
