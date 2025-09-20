import sys
import numpy as np 
import torch 
import open3d as o3d 
from mmdet.registry import TRANSFORMS, MODELS 
from mmdet.utils.camera_utils import adjust_intrinsic, get_gt_masks, save_masks
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class VoxelGenerator(BaseTransform):
    def __init__(self, base_height=1.1, y_range=0.8,
                 roi_x=(-20, 20), roi_z=(4, 125),
                 grid_res=(0.2, 0.1, 0.5)):
        self.base_height = base_height
        self.y_range = y_range
        self.roi_x = torch.tensor(roi_x, dtype=torch.float32)
        self.roi_z = torch.tensor(roi_z, dtype=torch.float32)
        self.grid_res = torch.tensor(grid_res, dtype=torch.float32)

        self.num_grids_x = int((self.roi_x[1] - self.roi_x[0]) / self.grid_res[0])
        self.num_grids_z = int((self.roi_z[1] - self.roi_z[0]) / self.grid_res[2])
        self.num_grids_y = int((self.y_range * 2) / self.grid_res[1])

        self._build_centers()

    def _build_centers(self):
        # BEV centers
        hori_centers = torch.zeros((self.num_grids_z, self.num_grids_x, 2), dtype=torch.float32)
        hori_centers[:, :, 0] = (torch.arange(self.num_grids_x) * self.grid_res[0] +
                                 self.roi_x[0] + self.grid_res[0] / 2).unsqueeze(0).repeat(self.num_grids_z, 1)
        hori_centers[:, :, 1] = (-torch.arange(self.num_grids_z) * self.grid_res[2] +
                                 self.roi_z[1] - self.grid_res[2] / 2).unsqueeze(1).repeat(1, self.num_grids_x)
        self.map_centers = hori_centers.reshape(-1, 2)

        # 3D voxel centers
        voxel_centers = torch.zeros((self.num_grids_z, self.num_grids_x, self.num_grids_y, 3), dtype=torch.float32)
        voxel_centers[:, :, :, [0, 2]] = hori_centers.unsqueeze(2).repeat(1, 1, self.num_grids_y, 1)
        voxel_centers[:, :, :, 1] = (torch.arange(self.num_grids_y) * self.grid_res[1] +
                                     self.base_height - self.y_range + self.grid_res[1] / 2
                                    ).unsqueeze(0).unsqueeze(0).repeat(self.num_grids_z, self.num_grids_x, 1)
        self.voxel_centers = voxel_centers.reshape(-1, 3)

    def transform(self, results: dict):
        results['voxels_info'] = dict(
            voxel_centers=self.voxel_centers,
            map_centers=self.map_centers,
            roi_x=self.roi_x,
            roi_z=self.roi_z,
            y_range=self.y_range,
            num_grids_x=self.num_grids_x,
            num_grids_z=self.num_grids_z,
            num_grids_y=self.num_grids_y,
            grid_res=self.grid_res,
            base_height=results['cam_height'],
        )
        return results


@TRANSFORMS.register_module()
class LoadLaneMasks(BaseTransform):
    def __init__(self, mask_downscale: int, iterations: int):
        self.mask_downscale = mask_downscale
        self.iterations = iterations

    def transform(self, results: dict) -> dict:
        if not isinstance(results, dict):
            raise TypeError(f"'results' should be dict, got {type(results)}")

        im_h, im_w = results['img_shape']
        mask_h, mask_w = im_h/self.mask_downscale, im_w/self.mask_downscale
        cam_h = results['cam_height']
        # adjust intrinsics and mask dimensions
        if results['img_shape'] != results['ori_shape']:
            results['cam_intrinsic'] = adjust_intrinsic(
                results['cam_intrinsic'], results['ori_shape'], results['img_shape']
            )

        # access voxel info from previous transform
        cam2vert = results['cam2vert']
        voxels_info = results['voxels_info']

        #todo:
            # read the lane line points
            # transform points to vert cam
            # crop roi
            # create gt and ele masks
        print(f"results['img_shape']: {results['img_shape']}")
        print(f"results['cam2vert'].shape: {results['cam2vert'].shape}")
        print(f"results['ground2cam'].shape: {results['ground2cam'].shape}")
        lanes = [inst['lane'] for inst in results['instances']]
        bin_mask, ele_mask = get_gt_masks(lanes, voxels_info, cam2vert, cam_h, iterations=self.iterations)
        print(f"bin_mask.shape: {bin_mask.shape}")
        print(f"ele_mask.shape: {ele_mask.shape}")
        save_masks(bin_mask, ele_mask, voxels_info, results['img_path'], save_dir="debug_masks", idx=results.get("sample_idx", 0))

        return results