import os
import pickle
from mmdet.registry import MODELS
from mmengine.runner import Runner
from mmengine.config import Config

# load config
cfg = Config.fromfile('/data24t_1/owais.tahir/3DLanes/mmdetection/mmdet/configs/3dlanes/apollo3d_main_cfg.py')
save_pred_pth = cfg.save_pred_pth

# build the model using MMDetection registry
model = MODELS.build(cfg.model)

# build runner manually
runner = Runner(
    model=model,
    test_dataloader=cfg.test_dataloader,
    test_cfg=cfg.test_cfg,
    test_evaluator=cfg.test_evaluator,
    work_dir=cfg.save_pred_pth,
    load_from=cfg.load_from,
)

# run training/test
runner.test()