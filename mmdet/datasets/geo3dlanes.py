import numpy as np
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS

@DATASETS.register_module()
class Geo3DLanesDataset(BaseDetDataset):
    METAINFO = {
        'classes': ['lane',],
        'palette': [(0, 255, 0)],
        'pose_params_order': ['quat_x', 'quat_y', 'quat_z', 'quat_w', 'trans_x', 'trans_y', 'trans_z'],
        'pose': [0.02542351021739838, 0.01395285328065762, 0.7279922049640486, 0.6849717603851541, 
                 704.5727650179983, 22.671121470092775, -40.598670488654086],
        'calib': {'cam_to_velo': [[-0.9999534974732871, -0.009643464521598512, 8.051683924663496e-05, -0.1361792713644952], 
                                  [-1.2307778763168131e-09, -0.008348949911626491, -0.9999651469103177, -0.37258354515466463], 
                                  [0.009643800648122605, -0.9999186460044607, 0.008348561652491204, -0.30314162226999714], 
                                  [0.0, 0.0, 0.0, 1.0]],
                  'cam_intrinsic': [[958.8315, -0.1807, 934.9871], [0.0, 962.3082, 519.1207], [0.0, 0.0, 1.0]],
                  'distortin':  [-0.3121, 0.1024, 0.00032953, -0.00039793, -0.0158]}
                    
    }

    # def load_data_list(self):
    #     #may not need to be overriden (verify)
    #     pass

    def parse_data_info(self, raw_data_info):
        data_info = raw_data_info.copy()
        data_info.pop('pose') # same for all samples, added to METAINFO
        data_info.pop('calib') # same for all samples, added to METAINFO
        data_info.pop('lane_ann_pth') # not required atm
        data_info.pop('obj_ann_pth') # not required atm
        return data_info
        