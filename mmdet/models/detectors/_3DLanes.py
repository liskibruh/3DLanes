import torch
from mmdet.registry import MODELS, TRANSFORMS
from mmdet.models.detectors.base import BaseDetector

@MODELS.register_module()
class _3DLanes(BaseDetector):
    def __init__(self,
                 data_preprocessor,
                 backbone,
                #  neck=None,
                 bin_head=None,
                 ele_head=None,
                 loss_func=None,
                 cla_res=None,
                 roi_x=None,
                 roi_z=None,
                 grid_res=None,
                 ele_range=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        # self.neck = MODELS.build(neck)
        self.bin_head = MODELS.build(bin_head) if bin_head is not None else None
        self.ele_head = MODELS.build(ele_head) if ele_head is not None else None
        self.loss_func = MODELS.build(loss_func) if loss_func is not None else None

        roi_x = torch.tensor(roi_x, dtype=torch.float32)
        roi_z = torch.tensor(roi_z, dtype=torch.float32)
        grid_res = torch.tensor(grid_res, dtype=torch.float32)
        self.num_grids_x = int((roi_x[1] - roi_x[0]) / grid_res[0])
        self.num_grids_z = int((roi_z[1] - roi_z[0]) / grid_res[2])
        self.num_grids_y = int((ele_range * 2) / grid_res[1])

        self.cla_res = cla_res
        self.ele_range = ele_range
        self.num_classes = int(2 * self.ele_range*100 / self.cla_res)
        ele_values = -torch.arange(self.num_classes, dtype=torch.float32, device='cuda')*self.cla_res + self.ele_range*100 - self.cla_res/2
        self.ele_values = ele_values.reshape(1, self.num_classes, 1, 1)
        

    @property          
    def with_head(self) -> bool:
        return hasattr(self, 'ele_head') and self.ele_head is not None

    def extract_feat(self, batch_inputs):
        x = self.backbone(batch_inputs)
        if self.with_neck:
            return self.neck.forward(x)
        return x

    def loss(self, batch_inputs, batch_data_samples, **kwargs):

        print(f"batch_inputs.keys(): {batch_inputs.keys()}")
        print(f"batch_data_samples[0].keys(): {batch_data_samples[0].keys()}")

        features_left = self.backbone(batch_inputs['img'])
        print(f"features_left.shape: {features_left.shape}")
        B, C, H, W = features_left.shape
        features_left = features_left.reshape(B, C, -1)
    
        if 'voxel_proj_index' in batch_inputs:
            print(f"voxel_proj_index.shape: {batch_inputs['voxel_proj_index'].shape}")
            proj_index_left = batch_inputs['voxel_proj_index']
            # Add bounds checking
            print(f"proj_index_left min/max: {proj_index_left.min()}, {proj_index_left.max()}")
            print(f"Feature map H, W: {H}, {W}")
            
            # Check if indices are within bounds
            x_indices = proj_index_left[:, 0, :]
            y_indices = proj_index_left[:, 1, :]
            print(f"x_indices range: {x_indices.min()} to {x_indices.max()}")
            print(f"y_indices range: {y_indices.min()} to {y_indices.max()}")

        linear_indices = proj_index_left[:, 1, :] * W + proj_index_left[:, 0, :]
        voxel_feat_left = features_left.gather(dim=2, index=linear_indices.unsqueeze(1).expand(-1, C, -1))
        voxel_feat_left = voxel_feat_left.reshape(B, C, self.num_grids_z, self.num_grids_x, self.num_grids_y)

        ele_pred = self.ele_head.forward(voxel_feat_left)

        losses = {}
        losses['loss_dummy'] = torch.tensor(0.0, requires_grad=True)
        return losses

    def loss_old(self, batch_inputs, batch_data_samples, **kwargs):
        batch_imgs = batch_inputs['img']
        feats = self.extract_feat(batch_imgs)

        losses = {}

        if 'voxel_proj_index' in batch_inputs:
            print(f"voxel_proj_index.shape: {batch_inputs['voxel_proj_index'].shape}")
            voxel_proj_index = batch_inputs['voxel_proj_index']
        
        if self.with_head:
            # bin_losses = self.bin_head.loss(feats, batch_data_samples)
            ele_losses = self.ele_head.loss(feats, batch_data_samples)
            # losses.update(bin_losses)
            losses.update(ele_losses)
        else:
            # Placeholder loss for testing
            losses['loss_dummy'] = torch.tensor(0.0, requires_grad=True)
        print(f"="*64)
        print(f"loss() completed")
        print(f"feats.shape: {feats.shape}")
        print(f"="*64)
        
        return losses
    
    def _forward(self, batch_inputs, batch_data_samples, mode):
        voxel_centers = self.voxel_generator.get_voxels()
        map_centers = self.voxel_generator.get_map_centers()

        feats = self.extract_feat(batch_inputs)
        if self.with_head:
            bin = self.bin_head.forward(feats)
            ele = self.ele_head.forward(feats)

        return feats

    def predict(self, batch_inputs, batch_data_samples, **kwargs):
        voxel_centers = self.voxel_generator.get_voxels()
        map_centers = self.voxel_generator.get_map_centers()

        feats = self.extract_feat(batch_inputs)
        if self.with_head:
            bin = self.bin_head.predict(feats)
            ele = self.ele_head.predict(feats)
        return feats