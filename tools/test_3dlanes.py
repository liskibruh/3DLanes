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
)

# run training/test
runner.test()

# os.makedirs(cfg.save_pred_pth, exist_ok=True)
# for sample in results:
#     im_name = sample.metainfo['img_path'].strip().split('/')[0]
#     ele_pred = sample.pred_3dlanes['ele_pred'].squeeze()
#     out_pkl_pth = cfg.save_pred_pth + '/' + im_name+'.pkl'
#     print(out_pkl_pth)
#     with open(out_pkl_pth, 'wb') as outfile:
#         pickle.dump(ele_pred, outfile)