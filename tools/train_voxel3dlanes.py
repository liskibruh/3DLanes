import argparse
from mmdet.registry import MODELS
from mmengine.runner import Runner
from mmengine.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for Voxel3DLane train pipeline")
    
    parser.add_argument("--config", type=str,
                        help="Path to the main config file")

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    
    args = parse_args()
    
    # load config
    # /data24t_1/owais.tahir/3DLanes/mmdetection/mmdet/configs/3dlanes/apollo3d_main_cfg.py
    cfg = Config.fromfile(args.config)

    # build the model using MMDetection registry
    model = MODELS.build(cfg.model)

    # build runner manually
    runner = Runner(
        model=model,
        train_dataloader=cfg.train_dataloader,
        test_dataloader=cfg.test_dataloader,
        train_cfg=cfg.train_cfg,
        test_cfg=cfg.test_cfg,
        test_evaluator=cfg.test_evaluator,
        optim_wrapper=cfg.optim_wrapper,
        work_dir=cfg.work_dir,
        custom_hooks=cfg.get('custom_hooks', None)
    )

    # run training/test
    runner.train()
