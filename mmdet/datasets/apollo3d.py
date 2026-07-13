import os
import numpy as np
import json
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class Apollo3D(BaseDetDataset):
    METAINFO = {
        'classes': ['SingleSolid', 'SingleDash', 'DoubleSolid', 'DoubleDash',
                    'LeftDashRightSolid', 'LeftSolidRightDash', 'Curb', 'Imaginary', 'Other'],
        'palette': [
            [220, 20, 60],    # Red
            [0, 128, 0],      # Green
            [0, 0, 255],      # Blue
            [255, 165, 0],    # Orange
            [128, 0, 128],    # Purple
            [255, 255, 0],    # Yellow
            [0, 255, 255],    # Cyan
            [255, 192, 203],  # Pink
            [128, 128, 128],  # Gray
        ],
        'cam_intrinsic': np.array([
            [2015, 0, 960],
            [0, 2015, 540],
            [0, 0, 1]
        ], dtype=np.float32),
    }

    def __init__(
            self, ann_file: str = '', 
            img_prefix: str = '', 
            id2class: dict = {}, 
            feat_downscale: int = 4,  
            **kwargs
            ):
        self.img_prefix = img_prefix
        self.id2class = id2class
        self.feat_downscale = feat_downscale
        
        super().__init__(ann_file=ann_file, **kwargs)
    
    def build_ground2cam(self, cam_pitch: float, cam_height: float) -> np.ndarray:
        """
        Construct ground-to-camera extrinsic matrix following Apollo's convention.
        Args:
            cam_pitch (float): Camera pitch angle in radians (downward positive).
            cam_height (float): Camera height above ground in meters.
        Returns:
            np.ndarray: 4x4 extrinsic matrix (ground -> camera).
        """
        alpha = np.pi / 2 + cam_pitch

        R = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha),  np.cos(alpha)]
        ])

        T = np.array([0, 0, cam_height]).reshape(3, 1)

        # Homogeneous matrix
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3:] = -R @ T  # subtract height in rotated frame

        return extrinsic
    
    def build_cam2ground(self, cam_pitch: float, cam_height: float) -> np.ndarray:
        # Use pitch directly (downward positive)
        alpha = cam_pitch
        
        R = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha),  np.cos(alpha)]
        ], dtype=np.float64)

        T = np.array([0, cam_height, 0], dtype=np.float64).reshape(3, 1) # this will be commented out?

        extrinsic = np.eye(3, dtype=np.float64)
        extrinsic[:3, :3] = R # R
        extrinsic[:3, 3:] = R @ T # change this to extrinsic[:3, 3:] =  T for ground2cam
        return extrinsic

    def build_cam2vert(self, cam_pitch: float, cam_height: float) -> np.ndarray:

        theta = cam_pitch
        R_X = np.array([
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -np.sin(theta), np.cos(theta)]
        ], dtype=np.float32)

        extrinsic = np.eye(3, dtype=np.float64)
        extrinsic[:3, :3] = R_X
        
        return extrinsic

    def load_data_list(self):
        data_list = []
        with open(self.ann_file, 'r') as infile:
            for i, line in enumerate(infile):
                sample = json.loads(line)
                img_pth = os.path.join(self.img_prefix, sample['img_path'])
                # skip non-existing images
                if not os.path.exists(img_pth):
                    continue

                instances = []
                for j, inst in enumerate(sample['lanes']):
                    instance = {}
                    instance['label'] = self.id2class[sample['labels'][j]]
                    instance['lane'] = inst
                    instances.append(instance)

                ground2cam = self.build_ground2cam(
                    sample['cam_pitch'], sample['cam_height'])
                cam2vert = self.build_cam2vert(
                    sample['cam_pitch'], sample['cam_height'])
                
                cam_intrinsic = self.METAINFO['cam_intrinsic'].copy()
                feat_intrinsic = self.METAINFO['cam_intrinsic'].copy()
                feat_intrinsic[0, 0] /= self.feat_downscale  # fx
                feat_intrinsic[1, 1] /= self.feat_downscale  # fy
                feat_intrinsic[0, 2] /= self.feat_downscale  # cx
                feat_intrinsic[1, 2] /= self.feat_downscale  # cy


                data_list.append(
                    dict(
                        img_path=os.path.join(
                            self.img_prefix, sample['img_path']),
                        img_id=i,
                        cam_height=sample['cam_height'],
                        cam_pitch=sample['cam_pitch'],
                        cam_intrinsic=cam_intrinsic,
                        feat_intrinsic=feat_intrinsic,
                        ground2cam=ground2cam,
                        cam2vert=cam2vert,
                        instances=instances
                    )
                )

        return data_list
