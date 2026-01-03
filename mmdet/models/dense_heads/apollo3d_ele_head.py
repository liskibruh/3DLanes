
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.lane3d_submodules import *
from mmdet.registry import MODELS


@MODELS.register_module()
class SegHead(nn.Module):
    def __init__(
            self,
            channel_reshaped,
            inplanes,
            roi_x,
            roi_z,
            grid_res,
            y_range,
            eff_bin,
            num_classes
    ):
        super(SegHead, self).__init__()
        if eff_bin is not None:
            self.effnet_bin = MODELS.build(eff_bin)
        self.channel_reshaped = channel_reshaped
        self.inplanes = inplanes
        self.num_classes = num_classes
        roi_x = torch.tensor(roi_x, dtype=torch.float32)
        roi_z = torch.tensor(roi_z, dtype=torch.float32)
        grid_res = torch.tensor(grid_res, dtype=torch.float32)

        self.num_grids_x = int((roi_x[1] - roi_x[0]) / grid_res[0])
        self.num_grids_z = int((roi_z[1] - roi_z[0]) / grid_res[2])
        self.num_grids_y = int((y_range * 2) / grid_res[1])

        self.first_conv = nn.Sequential(
                                convgn(
                                    self.channel_reshaped,
                                    self.inplanes,
                                    5, 1, 2, 1
                                ),
                                nn.ReLU(inplace=True)
                            )
        
        self.final_conv = convgn(
                            self.num_classes,
                            self.num_classes,          # out_channels
                            3, 1, 1, 1,
                            norm_groups=1,              # out_channels must be divisible by norm_groups
                            )
        
    def forward(self, feat_voxel):
        B = feat_voxel.shape[0]
        feat_bev = feat_voxel.permute(0, 4, 2, 1, 3).reshape(B,
                                                            self.channel_reshaped,
                                                            self.num_grids_z,
                                                            self.num_grids_x)
        feat_bev = self.first_conv(feat_bev)
        feat_bev = self.effnet_bin(feat_bev)
        seg_mask = self.final_conv(feat_bev)
        return seg_mask




@MODELS.register_module()
class EleHead(nn.Module):
    def __init__(
            self, 
            feat_channel, 
            roi_x, 
            roi_z, 
            grid_res, 
            y_range, 
            num_classes, 
            channel_reshaped, 
            inplanes, 
            eff_cla
            ):
        super(EleHead, self).__init__()

        self.feat_channel = feat_channel

        roi_x = torch.tensor(roi_x, dtype=torch.float32)
        roi_z = torch.tensor(roi_z, dtype=torch.float32)
        grid_res = torch.tensor(grid_res, dtype=torch.float32)

        self.num_grids_x = int((roi_x[1] - roi_x[0]) / grid_res[0])
        self.num_grids_z = int((roi_z[1] - roi_z[0]) / grid_res[2])
        self.num_grids_y = int((y_range * 2) / grid_res[1])
        
        self.num_classes = num_classes
        self.channel_reshaped = channel_reshaped
        self.inplanes = inplanes

        # self.channel_adapter = nn.Sequential(
        #     nn.Conv2d(self.channel_reshaped, self.inplanes, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(self.inplanes),
        #     nn.ReLU(inplace=True)
        # )

        self.first_conv = nn.Sequential(
                            convgn(
                                self.channel_reshaped, 
                                self.inplanes, 
                                5, 1, 2, 1
                                ),
                            nn.ReLU(inplace=True)
                            )

        if eff_cla is not None:
            self.effnet_reg = MODELS.build(eff_cla)

        self.final_conv = nn.Sequential(
                            convgn(
                                self.num_classes, 
                                self.num_classes, 
                                3, 1, 1, 1
                                ),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(self.num_classes, self.num_classes, kernel_size=1, stride=1,
                            padding=0, bias=False)
                            )

    def forward(self, feat_voxel):
        # feat_voxel: [B, C, Z, X, Y]
        B = feat_voxel.shape[0]
        #### get the BEV feature.  shape: [B, C_, num_grids_z, num_grids_x]
        feat_bev = feat_voxel.permute(0, 4, 1, 2, 3).reshape(B, 
                                                            self.channel_reshaped, 
                                                            self.num_grids_z, 
                                                            self.num_grids_x)  # [B,Y*C,Z,X]
        feat_bev = self.first_conv(feat_bev)
        feat_bev = self.effnet_reg(feat_bev)
        ele_cla_prob = self.final_conv(feat_bev)  # [B, num_classes, Z, X]

        return ele_cla_prob