import argparse
from mmdet.registry import MODELS
from mmengine.runner import Runner
from mmengine.config import Config
from mmengine.analysis import parameter_count_table

def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for Voxel3DLane eval pipeline")
    
    parser.add_argument("--config", type=str,
                        help="Path to the main config file")
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # load config
    cfg = Config.fromfile(args.config)
    save_pred_pth = cfg.save_pred_pth

    # build the model
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

    print(parameter_count_table(runner.model, max_depth=3))

    # run training/test
    runner.test()