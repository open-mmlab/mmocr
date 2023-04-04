# MMOCR 1.x 更新汇总

此处列出了 MMOCR 1.x 相对于 0.x 版本的重大更新。

1. 架构升级：MMOCR 1.x 是基于 [MMEngine](https://github.com/open-mmlab/mmengine)，提供了一个通用的、强大的执行器，允许更灵活的定制，提供了统一的训练和测试入口。

2. 统一接口：MMOCR 1.x 统一了数据集、模型、评估和可视化的接口和内部逻辑。支持更强的扩展性。

3. 跨项目调用：受益于统一的设计，你可以使用其他OpenMMLab项目中实现的模型，如MMDet。 我们提供了一个例子，说明如何通过MMDetWrapper使用MMDetection的Mask R-CNN。查看我们的文档以了解更多细节。更多的包装器将在未来发布。

4. 更强的可视化：我们提供了一系列可视化工具， 用户现在可以更方便可视化数据。

5. 更多的文档和教程：我们增加了更多的教程，降低用户的学习门槛。

6. 一站式数据准备：准备数据集已经不再是难事。使用我们的 [Dataset Preparer](https://mmocr.readthedocs.io/zh_CN/dev-1.x/user_guides/data_prepare/dataset_preparer.html)，一行命令即可让多个数据集准备就绪。

7. 拥抱更多 `projects/`: 我们推出了 `projects/` 文件夹，用于存放一些实验性的新特性、框架和模型。我们对这个文件夹下的代码规范不作过多要求，力求让社区的所有想法第一时间得到实现和展示。请查看我们的[样例 project](https://github.com/open-mmlab/mmocr/blob/dev-1.x/projects/example_project/) 以了解更多。

8. 更多新模型：MMOCR 1.0 支持了更多模型和模型种类。
