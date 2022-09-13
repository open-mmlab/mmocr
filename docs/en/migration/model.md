# Pretrained Model Migration

Due to the extensive refactoring and fixing of the model structure in the new version, MMOCR 1.x does not support load weights trained by the old version. We have updated the pre-training weights and logs of all models on our website.

In addition, we are working on the development of a weight migration tool for text detection tasks and plan to release it in the near future. Since the text recognition and key information extraction models are too much modified and the migration is lossy, we do not plan to support them accordingly for the time being. If you have specific requirements, please feel free to raise an [Issue](https://github.com/open-mmlab/mmocr/issues).
