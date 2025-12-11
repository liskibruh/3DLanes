import sys
import numpy as np
import torch
import cv2
import open3d as o3d
from typing import Optional

import mmcv
import mmengine.fileio as fileio
from mmdet.registry import TRANSFORMS, MODELS
from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import PixelData
from mmdet.structures import DetDataSample
from mmdet.utils.camera_utils import lanes_on_image #adjust_intrinsic, get_gt_masks, save_masks
from mmdet.utils.camera_utils2 import adjust_intrinsic, get_gt_masks, save_masks


@TRANSFORMS.register_module()
class CustomLoadImageFromFile(LoadImageFromFile):
    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            
            # todo: crop ROI from image
            #       upscale the cropped image using bilinear interp
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

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

    def _build_centers_new(self):
        # BEV centers
        hori_centers = torch.zeros((self.num_grids_z, self.num_grids_x, 2), dtype=torch.float32)
        hori_centers[:, :, 0] = (torch.arange(self.num_grids_x) * self.grid_res[0] +
                                self.roi_x[0] + self.grid_res[0] / 2).unsqueeze(0).repeat(self.num_grids_z, 1)
        hori_centers[:, :, 1] = (torch.arange(self.num_grids_z) * self.grid_res[2] +
                                self.roi_z[0] + self.grid_res[2] / 2).unsqueeze(1).repeat(1, self.num_grids_x)
        self.map_centers = hori_centers.reshape(-1, 2)

        # Generate 3D voxel centers
        voxel_centers = torch.zeros((self.num_grids_z, self.num_grids_x, self.num_grids_y, 3), dtype=torch.float32)
        voxel_centers[:, :, :, [0, 2]] = hori_centers.unsqueeze(2).repeat(1, 1, self.num_grids_y, 1)
        voxel_centers[:, :, :, 1] = (torch.arange(self.num_grids_y) * self.grid_res[1] +
                                    self.base_height - self.y_range + self.grid_res[1] / 2
                                    ).unsqueeze(0).unsqueeze(0).repeat(self.num_grids_z, self.num_grids_x, 1)
        self.voxel_centers = voxel_centers.reshape(-1, 3).transpose(1, 0)

    def _build_centers(self):
        # BEV centers
        hori_centers = torch.zeros((self.num_grids_z, self.num_grids_x, 2), dtype=torch.float32)
        hori_centers[:, :, 0] = (torch.arange(self.num_grids_x) * self.grid_res[0] +
                                 self.roi_x[0] + self.grid_res[0] / 2).unsqueeze(0).repeat(self.num_grids_z, 1)
        hori_centers[:, :, 1] = (-torch.arange(self.num_grids_z) * self.grid_res[2] +
                                 self.roi_z[1] - self.grid_res[2] / 2).unsqueeze(1).repeat(1, self.num_grids_x)
        self.map_centers = hori_centers.reshape(-1, 2)

        # generate the centers of every 3D voxel
        voxel_centers = torch.zeros((self.num_grids_z, self.num_grids_x, self.num_grids_y, 3), dtype=torch.float32)
        voxel_centers[:, :, :, [0, 2]] = hori_centers.unsqueeze(2).repeat(1, 1, self.num_grids_y, 1)
        voxel_centers[:, :, :, 1] = (torch.arange(self.num_grids_y) * self.grid_res[1] +
                                     self.base_height - self.y_range + self.grid_res[1] / 2
                                    ).unsqueeze(0).unsqueeze(0).repeat(self.num_grids_z, self.num_grids_x, 1)
        self.voxel_centers = voxel_centers.reshape(-1, 3).transpose(1,0)

    def transform(self, results: dict):
        feat_intrinsic = results['feat_intrinsic']
        cam2vert = results['cam2vert']
        vert2cam = np.linalg.inv(cam2vert)
        # todo:
        #   vert2cam @ self.voxel_centers
        #   uvz_left = feat_intrinsic @ voxel_centers
        if isinstance(vert2cam, np.ndarray):
            vert2cam = to_tensor(vert2cam).float()
        if isinstance(feat_intrinsic, np.ndarray):
            feat_intrinsic = to_tensor(feat_intrinsic).float()

        voxel_cam = vert2cam @ self.voxel_centers
        # Filter out voxels behind camera (Z <= 0)
        valid_mask = voxel_cam[2, :] > 0.1  # At least 10cm in front
        voxel_cam_valid = voxel_cam[:, valid_mask]

        # voxel_uvz = feat_intrinsic @ voxel_cam
        voxel_uvz = feat_intrinsic @ voxel_cam_valid
        voxel_uv = torch.floor(voxel_uvz[:2, :] / voxel_uvz[2:, :]).type(torch.long)

        # Filter out voxels outside image bounds
        H, W = 270, 480  # Feature map size
        image_mask = (voxel_uv[0, :] >= 0) & (voxel_uv[0, :] < W) & \
                    (voxel_uv[1, :] >= 0) & (voxel_uv[1, :] < H)
    
        voxel_uv_valid = voxel_uv[:, image_mask]

        # Create full index tensor with invalid indices set to 0
        voxel_uv_full = torch.zeros((2, self.voxel_centers.shape[1]), dtype=torch.long)
        valid_indices = torch.where(valid_mask)[0][image_mask]
        voxel_uv_full[:, valid_indices] = voxel_uv_valid

        results['voxels_info'] = dict(
            voxel_uv=voxel_uv_full,
            # voxel_centers=self.voxel_centers,
            # map_centers=self.map_centers,
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
    def __init__(self, iterations: int):
        self.iterations = iterations

    def transform(self, results: dict) -> dict:
        if not isinstance(results, dict):
            raise TypeError(f"'results' should be dict, got {type(results)}")
        
        cam_h = results['cam_height']
        cam_pitch = results['cam_pitch']
        # adjust intrinsics and mask dimensions
        if results['img_shape'] != results['ori_shape']:
            results['cam_intrinsic'] = adjust_intrinsic(
                results['cam_intrinsic'], results['ori_shape'], results['img_shape']
            )

        # access voxel info from previous transform
        cam2vert = results['cam2vert']
        cam2img = results['cam_intrinsic']
        voxels_info = results['voxels_info']

        lanes = [inst['lane'] for inst in results['instances']]
        bin_mask, ele_mask = get_gt_masks(lanes, voxels_info, cam2vert, cam_h, cam_pitch, iterations=self.iterations)
        # lanes_on_image(lanes, cam2img, results['img_path'], idx=results.get("sample_idx", 0))
        # save_masks(lanes, cam2img, bin_mask, ele_mask, voxels_info, results['img_path'], save_dir="debug_masks", idx=results.get("sample_idx", 0))

        # convert to torch tensors for consistency
        if isinstance(bin_mask, np.ndarray):
            bin_mask = torch.from_numpy(bin_mask)
        if isinstance(ele_mask, np.ndarray):
            ele_mask = torch.from_numpy(ele_mask)

        results['gt_bin_mask'] = bin_mask
        results['gt_ele_mask'] = ele_mask

        return results
    

@TRANSFORMS.register_module()
class CropROIimage(BaseTransform):
    def transform(self, results: dict):
        voxels_info = results['voxels_info']
        ground2cam = results['ground2cam']
        cam2img = results['cam_intrinsic']
        roi_x, roi_z, y_range = voxels_info['roi_x'], voxels_info['roi_z'], voxels_info['y_range']

        xmin, ymin, zmin = roi_x[0], -0.5 + results['cam_height'], roi_z[0]
        xmax, ymax, zmax = roi_x[1],  3.0 + results['cam_height'], roi_z[1]

        # corners = np.array([
        #     [xmin, ymin, zmin],
        #     [xmin, ymin, zmax],
        #     [xmin, ymax, zmin],
        #     [xmin, ymax, zmax],
        #     [xmax, ymin, zmin], 
        #     [xmax, ymin, zmax],
        #     [xmax, ymax, zmin],
        #     [xmax, ymax, zmax],
        # ], dtype=np.float32)
        ground_corners = np.array([
            [xmin, 0.0 + results['cam_height'], zmin],
            [xmin, 0.0 + results['cam_height'], zmax],
            [xmax, 0.0 + results['cam_height'], zmin],
            [xmax, 0.0 + results['cam_height'], zmax],
        ], dtype=np.float32)

        # homogeneous coords
        corners_h = np.concatenate([ground_corners, np.ones((ground_corners.shape[0], 1))], axis=1)
        cam_coords = (ground2cam @ corners_h.T).T[:, :3]

        # only keep corners in front of camera
        mask = cam_coords[:, 2] > 1e-3
        cam_coords = cam_coords[mask]
        if cam_coords.shape[0] == 0:
            print(f"skipping")
            return results  # skip if nothing valid

        img_corners = (cam2img @ cam_coords.T).T
        img_corners = (img_corners[:, :2] / img_corners[:, 2:])

        # clip to image size
        H, W = results['img'].shape[:2]
        umin = max(0, int(img_corners[:, 0].min()))
        umax = min(W, int(img_corners[:, 0].max()))
        vmin = max(0, int(img_corners[:, 1].min()))
        vmax = min(H, int(img_corners[:, 1].max()))

        if umin >= umax or vmin >= vmax:
            print(f"degenerate case")
            return results  # degenerate case

        img = results['img']
        cropped_img = img[vmin:vmax, umin:umax, :]

        results['img_roi'] = (umin, vmin, umax, vmax)
        results['img'] = cropped_img
        success = cv2.imwrite('/data24t_1/owais.tahir/3DLanes/mmdetection/tools/debug_masks/cropped_image.png', cropped_img)
        if success:
            return results
        else:
            raise Exception("Failed to write image")


@TRANSFORMS.register_module()
class Pack3DLanesInputs(BaseTransform):
    """Pack inputs for 3D lane detection.

    This transform packs the image and custom ground truth masks (bin_mask, ele_mask)
    into the format expected by the model.

    Args:
        meta_keys (tuple): Keys to be saved in metainfo. Default includes standard
            image metadata plus voxel and camera information.
    """

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'cam_height', 'cam_pitch', 'cam_intrinsic',
                           'cam2vert', 'ground2cam', 'voxels_info')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Packed results with 'inputs' and 'data_samples' keys.
        """
        packed_results = dict()

        # Pack image
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)

            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results['inputs'] = {'img': img}

        data_sample = DetDataSample()

        # Pack ground truth masks as PixelData
        if 'gt_bin_mask' in results \
            and 'gt_ele_mask' in results \
            and 'voxels_info' in results:
            gt_masks_data = dict()

            bin_mask = results['gt_bin_mask']
            ele_mask = results['gt_ele_mask']

            if isinstance(bin_mask, np.ndarray):
                bin_mask = to_tensor(bin_mask)
            if isinstance(ele_mask, np.ndarray):
                ele_mask = to_tensor(ele_mask)

            if len(bin_mask.shape) == 2:
                bin_mask = bin_mask.unsqueeze(0)
            if len(ele_mask.shape) == 2:
                ele_mask = ele_mask.unsqueeze(0)

            gt_masks_data['bin_mask'] = bin_mask
            gt_masks_data['ele_mask'] = ele_mask

            data_sample.gt_lane_masks = PixelData(**gt_masks_data)
            voxel_uv = results['voxels_info']['voxel_uv']

            if isinstance(voxel_uv, np.ndarray):
                voxel_uv = to_tensor(voxel_uv)

            packed_results['inputs']['voxel_proj_index'] = voxel_uv

        # Pack metadata
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)

        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
