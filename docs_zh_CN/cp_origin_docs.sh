#!/usr/bin/env bash

# Copy *.md files from docs/ if it doesn't have a Chinese translation

for filename in $(find ../docs/ -name '*.md' -printf "%P\n");
do
    mkdir -p $(dirname $filename)
    cp -n ../docs/$filename ./$filename
done
