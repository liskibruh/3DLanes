model = dict(
    type= 'mmdet._3DLanes',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean = [],
        std = [],
        bgr_to_rgb=True,
        pad_mask = False,
        pad_size_divisor=1
    ),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        # frozen_stages=1,
        out_indices=(0,1,2,3),
        norm_cfg=dict(
            type='BN',
            requires_grad=True
        ),
        # norm_eval=True,
        style='pytorch', #caffe
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5
        ),
    )