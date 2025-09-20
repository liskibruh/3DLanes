from mmdet.registry import MODELS
from mmengine.runner import Runner
from mmengine.config import Config

# load config
cfg = Config.fromfile('/data24t_1/owais.tahir/3DLanes/mmdetection/mmdet/configs/3dlanes/apollo3d_main_cfg.py')

# build the model using MMDetection registry
model = MODELS.build(cfg.model)

# build runner manually
runner = Runner(
    model=model,
    train_dataloader=cfg.train_dataloader,
    val_dataloader=cfg.val_dataloader,
    train_cfg=cfg.train_cfg,
    val_cfg=cfg.val_cfg,
    optim_wrapper=cfg.optim_wrapper,
    work_dir=cfg.work_dir,
)

# run training/test
runner.train()
