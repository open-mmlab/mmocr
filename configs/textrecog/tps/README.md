# Thin-Plate-Spline (TPS) transformation

## Introduction

[ALGORITHM]

```bibtex
@article{shi2016robust,
  title={Robust Scene Text Recognition with Automatic Rectification},
  author={Shi, Baoguang and Wang, Xinggang and Lyu, Pengyuan and Yao,
  Cong and Bai, Xiang},
  year={2016}
}
```

## About using TPS in other models

- Simply change `cfg.model.preprocessor` from `None` to
```python
dict(
  type='TPSPreprocessor',
  num_fiducial=20,
  img_size=(32, 100),
  rectified_img_size=(32, 100),
  num_img_channel=1
)
