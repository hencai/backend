# dataset settings
dataset_type = 'DOTADataset'
# data_root = "/opt/data/private/LYX/data/split_ms_dota/"
# data_root = "/xiaying_ms/bp/RSOD-dataset/DIOR/split_ms_dota/"
# data_root = "/mnt/bp/Large-Selective-Kernel-Network/data/split_ms_dota/"
data_root = "/mnt/bp/Large-Selective-Kernel-Network/data/split_ss_dota/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# ******* 以下是自己写的  *******
# classes = ("airplane",
#             "airport",
#             "baseballfield",
#             "basketballcourt",
#             "bridge",
#             "chimney",
#             "dam",
#             "Expressway-Service-area",
#             "Expressway-toll-station",
#             "golfcourse",
#             "golffield",
#             "groundtrackfield",
#             "harbor",
#             "overpass",
#             "ship",
#             "stadium",
#             "storagetank",
#             "tenniscourt",
#             "trainstation",
#             "vehicle",
#             "windmill")
# ****************************
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    # 在继承配置中修改，这里修改会被覆盖
    samples_per_gpu=6,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # classes=classes,     # 重写的类别
        ann_file=data_root + 'trainval/annfiles/', img_prefix=data_root + 'trainval/images/',
        # ann_file=data_root + 'trainval/annfiles--fortest/', img_prefix=data_root + 'trainval/images--fortest/',
        # ann_file=data_root + 'train/annfiles--part/', img_prefix=data_root + 'train/images--part/',
        # ann_file=data_root + 'train/annfiles/', img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline),
    val=dict(   # 配置的是epoch之间验证的时候
        type=dataset_type,
        # classes=classes,    # 重写的类别
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline),
    test=dict(   # 配置的是测试的时候
        type=dataset_type,
        # classes=classes,    # 重写的类别
        ann_file=data_root + 'test/annfiles/',
        # ann_file=data_root + 'val/annfiles/',
        # img_prefix=data_root + 'val/images/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline))
