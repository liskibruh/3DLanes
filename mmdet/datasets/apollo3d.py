import os
import numpy as np
import json
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS



@DATASETS.register_module()
class Apollo3D(BaseDetDataset):
    METAINFO = {
        'classes': ['lane', 'no_lane'],
        'palette': [(0, 255, 0), (255, 0, 0)],
        'cam_intrinsic': np.array([
            [2015, 0, 960],
            [0, 2015, 540],
            [0, 0, 1]
        ]),
    }

    def __init__(self, ann_file: str = '', img_prefix: str = '', **kwargs):
        self.img_prefix = img_prefix
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

    def matrix2euler(self, m):
        # order='XYZ'
        d = np.clip
        m = m.reshape(-1)
        a, f, g, k, l, n, e = m[0], m[1], m[2], m[4], m[5], m[7], m[8]
        y = np.arcsin(d(g, -1, 1))
        if 0.99999 > np.abs(g):
            x = np.arctan2(- l, e)
            z = np.arctan2(- f, a)
        else:
            x = np.arctan2(n, k)
            z = 0
        return np.array([x, y, z], dtype=np.float32)

    def build_cam2vert(self, ground2cam_R):
        """
        Builds rotation to convert camera -> vertical BEV coordinates.
        BEV convention: (x=right, z=forward, y=elevation)
        """
        def build_Rx(angle):
            ca, sa = np.cos(angle), np.sin(angle)
            return np.array([[1, 0, 0],
                            [0, ca, sa],
                            [0,-sa, ca]], dtype=np.float32)

        def build_Rz(angle):
            ca, sa = np.cos(angle), np.sin(angle)
            return np.array([[ca,  sa, 0],
                            [-sa, ca, 0],
                            [0,   0,  1]], dtype=np.float32)

        # camera rotation relative to ground
        R_cam_rel_ground = ground2cam_R
        pitch_cam, roll_cam, yaw_cam = self.matrix2euler(R_cam_rel_ground)
        pitch_cam -= 1.5708  # subtract pi/2

        R_X = build_Rx(pitch_cam)
        R_Z = build_Rz(roll_cam)
        R_cam2vert = R_X @ R_Z

        # --- extra swap matrix to reorder axes: (x, y, z) -> (x, z, y) with y flipped ---
        swap_yz = np.array([
            [1, 0, 0],   # x stays right
            [0, 0, 1],   # z stays forward
            [0,-1, 0]    # flip y (down → up) for elevation
        ], dtype=np.float32)

        R_cam2vert = swap_yz @ R_cam2vert
        return R_cam2vert
    
    def build_cam2vert_old(self, ground2cam_R):
        """
        If you only have camera extrinsic (ground -> camera rotation matrix), you can compute
        camera rotation relative to ground directly.
        ground2cam_R: 3x3 rotation matrix that maps ground->camera (i.e. camera_R = ground2cam_R)
        This returns same R_cam2vert and rotated points.
        """
        def build_Rx(angle):
            ca = np.cos(angle); sa = np.sin(angle)
            return np.array([[1, 0, 0],
                            [0, ca, sa],
                            [0, -sa, ca]], dtype=np.float32)

        def build_Rz(angle):
            ca = np.cos(angle); sa = np.sin(angle)
            return np.array([[ca, sa, 0],
                            [-sa, ca, 0],
                            [0, 0, 1]], dtype=np.float32)

        # camera rotation relative to ground is just ground2cam_R
        R_cam_rel_ground = ground2cam_R

        # extract Euler
        pitch_cam, roll_cam, yaw_cam = self.matrix2euler(R_cam_rel_ground)
        pitch_cam -= 1.5708 # pi/2

        R_X = build_Rx(pitch_cam)
        R_Z = build_Rz(roll_cam)
        R_cam2vert = R_X @ R_Z

        # lane_camvert = (R_cam2vert @ lane_cam_pts.T).T
        return R_cam2vert

    def load_data_list(self):
        data_list = []
        with open(self.ann_file, 'r') as infile:
            for i,line in enumerate(infile):
                sample = json.loads(line)

                instances = []
                for inst in sample['laneLines']:
                    instance = {}
                    instance['label'] = 0,
                    instance['lane'] = inst
                    instances.append(instance)

                ground2cam = self.build_ground2cam(sample['cam_pitch'], sample['cam_height'])
                cam2vert = self.build_cam2vert_old(ground2cam)
                # cam2vert = self.build_cam2vert()

                data_list.append(
                    dict(
                        img_path=os.path.join(self.img_prefix, sample['raw_file']),
                        img_id=i,
                        cam_height=sample['cam_height'],
                        cam_pitch=sample['cam_pitch'],
                        cam_intrinsic=self.METAINFO['cam_intrinsic'],
                        ground2cam=ground2cam,
                        cam2vert=cam2vert,
                        instances=instances
                    )
                )

        return data_list

    # def parse_data_info(self, raw_info):
    #     data_info = raw_info.copy()
    #     data_info['cam_intrinsic'] = self.metainfo['camera_intrinsic']
    #     return data_info