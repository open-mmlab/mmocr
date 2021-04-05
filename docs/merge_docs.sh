#!/usr/bin/env bash

sed -i '$a\\n' ../configs/kie/*/*.md
sed -i '$a\\n' ../configs/textdet/*/*.md
sed -i '$a\\n' ../configs/textrecog/*/*.md

# gather models
cat ../configs/kie/*/*.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Kie Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmocr/tree/master/=g' >kie_models.md
cat ../configs/textdet/*/*.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Text Detection Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmocr/tree/master/=g' >textdet_models.md
cat ../configs/textrecog/*/*.md | sed "s/md###t/html#t/g" | sed "s/#/#&/" | sed '1i\# Text Recognition Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmocr/tree/master/=g' >textrecog_models.md
