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
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        # self.neck = MODELS.build(neck)
        self.bin_head = MODELS.build(bin_head) if bin_head is not None else None
        self.ele_head = MODELS.build(ele_head) if ele_head is not None else None

    @property          
    def with_head(self) -> bool:
        return hasattr(self, 'head') and self.head is not None

    def extract_feat(self, batch_inputs):
        x = self.backbone(batch_inputs)
        if self.with_neck:
            return self.neck.forward(x)
        return x

    def loss(self, batch_inputs, batch_data_samples, **kwargs):
        batch_imgs = batch_inputs['img']
        feats = self.extract_feat(batch_imgs)
        losses = {}

        if 'voxel_proj_index' in batch_inputs:
            print(f"voxel_proj_index.shape: {batch_inputs['voxel_proj_index'].shape}")
            voxel_proj_index = batch_inputs['voxel_proj_index']
        
        if self.with_head:
            bin_losses = self.bin_head.loss(feats, batch_data_samples)
            ele_losses = self.ele_head.loss(feats, batch_data_samples)
            losses.update(bin_losses)
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