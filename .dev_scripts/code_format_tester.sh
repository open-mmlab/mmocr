#!/bin/bash

PY_FILES=""
for FILE_NAME in $(git diff --name-only HEAD refactor_dev)
do
    if [ ${FILE_NAME: -3} = ".py" ] && [ ${FILE_NAME:0:6} = "mmocr/" ] && [ -f "$FILE_NAME" ]
    then
        PY_FILES="$PY_FILES $FILE_NAME"
    fi
done

# Only test the coverage when PY_FILES are not empty, otherwise they will test the entire project
if [ ! -z "${PY_FILES}" ]
then
    coverage report --fail-under 80 -m $PY_FILES
    interrogate -v --ignore-init-method --ignore-module --ignore-nested-functions --ignore-regex "__repr__" --fail-under 95 $PY_FILES
fi
