import torch

from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class VoxelGenerator:
    def __init__(self,
                 base_height=1.1,
                 y_range=0.8,
                 roi_x=(-20, 20),
                 roi_z=(4, 125),
                 grid_res=(0.2, 0.1, 0.5)):
        """
        Args:
            base_height (float): camera height above ground [m]
            y_range (float): +/- range around base height [m]
            roi_x (tuple): lateral range [m]
            roi_z (tuple): longitudinal range [m]
            grid_res (tuple): resolution (x, y, z) [m]
        """
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
        # horizontal BEV centers
        hori_centers = torch.zeros((self.num_grids_z, self.num_grids_x, 2), dtype=torch.float32)
        hori_centers[:, :, 0] = (torch.arange(self.num_grids_x) * self.grid_res[0] +
                                 self.roi_x[0] + self.grid_res[0] / 2).unsqueeze(0).repeat(self.num_grids_z, 1)
        hori_centers[:, :, 1] = (-torch.arange(self.num_grids_z) * self.grid_res[2] +
                                 self.roi_z[1] - self.grid_res[2] / 2).unsqueeze(1).repeat(1, self.num_grids_x)
        self.map_centers = hori_centers.reshape(-1, 2)

        # full 3D voxel centers
        voxel_centers = torch.zeros((self.num_grids_z, self.num_grids_x, self.num_grids_y, 3), dtype=torch.float32)
        voxel_centers[:, :, :, [0, 2]] = hori_centers.unsqueeze(2).repeat(1, 1, self.num_grids_y, 1)
        voxel_centers[:, :, :, 1] = (torch.arange(self.num_grids_y) * self.grid_res[1] +
                                     self.base_height - self.y_range + self.grid_res[1] / 2
                                    ).unsqueeze(0).unsqueeze(0).repeat(self.num_grids_z, self.num_grids_x, 1)
        self.voxel_centers = voxel_centers.reshape(-1, 3)

    def get_voxels(self):
        """Return voxel centers as (N,3) tensor"""
        return self.voxel_centers

    def get_map_centers(self):
        """Return BEV grid centers as (M,2) tensor"""
        return self.map_centers