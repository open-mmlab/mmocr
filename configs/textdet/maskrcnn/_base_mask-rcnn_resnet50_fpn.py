_base_ = ['mmdet::_base_/models/mask-rcnn_r50_fpn.py']

file_client_args = dict(backend='disk')

mask_rcnn = _base_.pop('model')
# Adapt Mask R-CNN model to OCR task
mask_rcnn.update(
    dict(
        data_preprocessor=dict(pad_mask=False),
        rpn_head=dict(
            anchor_generator=dict(
                scales=[4], ratios=[0.17, 0.44, 1.13, 2.90, 7.46])),
        roi_head=dict(
            bbox_head=dict(num_classes=1),
            mask_head=dict(num_classes=1),
        )))

model = dict(type='MMDetWrapper', text_repr_type='poly', cfg=mask_rcnn)

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5,
        contrast=0.5),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(1.0, 4.125),
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='TextDetRandomCrop', target_size=(640, 640)),
    dict(type='MMOCR2MMDet', poly2mask=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'flip',
                   'scale_factor', 'flip_direction'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1920, 1920), keep_ratio=True),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
