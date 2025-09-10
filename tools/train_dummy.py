from mmengine.runner import Runner
from mmengine.config import Config
from mmdet.datasets.geo3dlanes import Geo3DLanesDataset

cfg = Config.fromfile('/data24t_1/owais.tahir/3DLanes/mmdetection/mmdet/configs/3dlanes/3dlanes_main_cfg.py')
# runner = Runner.from_cfg(cfg)
# runner.train()

train_dataset = Geo3DLanesDataset(data_root=cfg.data_root,
                                  ann_file=cfg.train_ann_file,
                                  pipeline=cfg.train_pipeline,
                                  backend_args=cfg.backend_args)

# for i, data in enumerate(train_dataset):
#     print(f"type(data): {type(data)}")
#     print(f"data.keys(): {data.keys()}")
#     print(f"="*64)
#     if i>2:
#         break

# print(train_dataset.metainfo)
print(train_dataset[0])
# print(train_dataset.get_data_info(0))