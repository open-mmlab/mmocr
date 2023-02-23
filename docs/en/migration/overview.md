# Overview

Along with the release of OpenMMLab 2.0, MMOCR 1.0 made many significant changes, resulting in less redundant, more efficient code and a more consistent overall design. However, these changes break backward compatibility. We understand that with such huge changes, it is not easy for users familiar with the old version to adapt to the new version. Therefore, we prepared a detailed migration guide to make the transition as smooth as possible so that all users can enjoy the productivity benefits of the new MMOCR and the entire OpenMMLab 2.0 ecosystem.

```{warning}
MMOCR 1.0 depends on the new foundational library for training deep learning models [MMEngine](https://github.com/open-mmlab/mmengine), and therefore has an entirely different dependency chain compared with MMOCR 0.x. Even if you have a well-rounded MMOCR 0.x environment before, you still need to create a new python environment for MMOCR 1.0. We provide a detailed [installation guide](../get_started/install.md) for reference.
```

Next, please read the sections according to your requirements.

- If you want to migrate a model trained in version 0.x to use it directly in version 1.0, please read [Pretrained Model Migration](./model.md).
- If you want to train the model, please read [Dataset Migration](./dataset.md) and [Data Transform Migration](./transforms.md).
- If you want to develop on MMOCR, please read [Code Migration](code.md) and [Upstream Library Changes](https://github.com/open-mmlab/mmengine/tree/main/docs/en/migration).

As shown in the following figure, the maintenance plan of MMOCR 1.x version is mainly divided into three stages, namely "RC Period", "Compatibility Period" and "Maintenance Period". For old versions, we will no longer add major new features. Therefore, we strongly recommend users to migrate to MMOCR 1.x version as soon as possible.

![plan](https://user-images.githubusercontent.com/45810070/192927112-70c0108d-58ed-4c77-8a0a-9d9685a48333.png)
