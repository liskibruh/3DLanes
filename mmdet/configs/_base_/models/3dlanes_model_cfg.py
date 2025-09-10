model = dict(
    type= 'mmdet.ThreeD_Lanes',
    backbone=dict(
        type='ResNet',
        depth=50
    ),
    neck=dict(
        type='mmdet.FPN',
        ))