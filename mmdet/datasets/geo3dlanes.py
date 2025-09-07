from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS

@DATASETS.register_module()
class Geo3DLanesDataset(BaseDetDataset):
    METAINFO = {
        'classes': ['lane',],
        'palette': [(0, 255, 0)]
    }

    def parse_data_info(self, raw_data_info):
        # return super().parse_data_info(raw_data_info)
        """Convert a raw sample into standardized format."""
        # data_info = dict(
        #     img_path=raw_data_info['img_path'],
        #     img_shape=(raw_data_info['height'], raw_data_info['width']),
        #     instances=[]
        # )
        # for inst in raw_data_info['instances']:
        #     instance = dict(
        #         lane_points=inst['lane_points'],   # instead of bbox
        #         label=inst['label'],
        #         ignore_flag=inst.get('ignore_flag', 0)
        #     )
        #     data_info['instances'].append(instance)
        # return data_info
        pass
        
    def load_data_list(self):
        #may not need to be overriden (verify)
        pass