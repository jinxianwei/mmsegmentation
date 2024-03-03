_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    # pretrained='open-mmlab://resnet101_v1c', 
    pretrained='mmcls://mobilenet_v2',
    # backbone=dict(depth=101),
    backbone=dict(
        _delete_=True,
        type='MobileNetV2',
        widen_factor=1.,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    # 类别信息需要更改为数据集的掩码类别数量 num_classes=21
    decode_head=dict(in_channels=320, c1_in_channels=24, num_classes=21),
    auxiliary_head=dict(in_channels=96, num_classes=21)
    )
