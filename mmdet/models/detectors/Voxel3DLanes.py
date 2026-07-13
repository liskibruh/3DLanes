import numpy as np
import torch
from torch.nn import functional as F
from mmdet.registry import MODELS
from mmdet.models.detectors.base import BaseDetector

@MODELS.register_module()
class Voxel3DLanes(BaseDetector):
    def __init__(self,
                backbone,
                bin_head=None,
                ele_head=None,
                loss_func=None,
                cla_res=None,
                roi_x=None,
                roi_z=None,
                grid_res=None,
                ele_range=None,
                bin_loss=None,
                train_cfg=None,
                test_cfg=None,
                init_cfg=None,
                 **kwargs
        ):
        
        super().__init__(**kwargs)
        self.backbone = MODELS.build(backbone)
        self.bin_head = MODELS.build(bin_head) if bin_head is not None else None
        self.ele_head = MODELS.build(ele_head) if ele_head is not None else None
        self.loss_func = MODELS.build(loss_func) if loss_func is not None else None
        self.bin_loss = MODELS.build(bin_loss) if bin_loss is not None else None
        
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
        
        losses = {}
        # extract ground truth masks
        gt_bin_masks = []
        gt_ele_masks = []
        for data_sample in batch_data_samples:
            if hasattr(data_sample, 'gt_lane_masks'):
                gt_bin_masks.append(data_sample.gt_lane_masks.bin_mask)
                gt_ele_masks.append(data_sample.gt_lane_masks.ele_mask)
        
        # stack masks for batch processing
        if gt_bin_masks:
            gt_bin_mask = torch.stack(gt_bin_masks, dim=0)
            gt_ele_mask = torch.stack(gt_ele_masks, dim=0)
        
        features_left = self.backbone(batch_inputs['img'])
        B, C, H, W = features_left.shape
        features_left = features_left.reshape(B, C, -1)
    
        if 'voxel_proj_index' in batch_inputs:
            proj_index_left = batch_inputs['voxel_proj_index']            
            linear_indices = proj_index_left[:, 1, :] * W + proj_index_left[:, 0, :]
            voxel_feat_left = features_left.gather(dim=2, index=linear_indices.unsqueeze(1).expand(-1, C, -1))
            voxel_feat_left = voxel_feat_left.reshape(B, C, self.num_grids_z, self.num_grids_x, self.num_grids_y)
            
            ele_pred = self.ele_head.forward(voxel_feat_left)
            
            bin_pred = None
            if self.bin_head:
                bin_pred = self.bin_head.forward(voxel_feat_left)
            
            # compute losses
            if gt_bin_masks and self.loss_func is not None:
                ele_loss = self.loss_func(ele_pred, gt_ele_mask, gt_bin_mask)
                losses.update({'ele_loss': ele_loss})
            
            assert bin_pred.shape == gt_bin_mask.shape, \
                f"{bin_pred.shape} vs {gt_bin_mask.shape}"
            
            if bin_pred is not None:
                bin_loss = self.bin_loss(bin_pred.squeeze(1), gt_bin_mask.squeeze(1))
                losses.update({'bin_loss': bin_loss})
            
        else:
            losses['ele_loss'] = torch.tensor(0.0, requires_grad=True)
            losses['bin_loss'] = torch.tensor(0.0, requires_grad=True)
        
        return losses
    
    def predict(self, batch_inputs, batch_data_samples, **kwargs):
        
        features_left = self.backbone(batch_inputs['img'])
        B, C, H_img, W_img = features_left.shape
        features_left = features_left.reshape(B, C, -1)
        
        # get vert2cam for each sample
        batch_vert2cam = []
        for data_sample in batch_data_samples:
            assert 'cam2vert' in data_sample.metainfo_keys(), \
                "missing 'cam2vert' key in data_sample.metainfo_keys()"
            cam2vert = data_sample.metainfo['cam2vert']
            vert2cam = np.linalg.inv(cam2vert)
            batch_vert2cam.append(vert2cam)
            
        proj_index_left = batch_inputs['voxel_proj_index']
        linear_indices = proj_index_left[:, 1, :] * W_img + proj_index_left[:, 0, :]
        voxel_feat_left = features_left.gather(dim=2, index=linear_indices.unsqueeze(1).expand(-1, C, -1))
        voxel_feat_left = voxel_feat_left.reshape(B, C, self.num_grids_z, self.num_grids_x, self.num_grids_y)
        
        ele_logits = self.ele_head.forward(voxel_feat_left)  # (B, num_classes, Z, X)
        bin_logits = self.bin_head.forward(voxel_feat_left)  # (B, 1, Z, X)
        
        bin_prob = torch.sigmoid(bin_logits).squeeze(1)  # (B, Z, X)
        
        # decode elevation logits to continuous height (in meters)
        probs = F.softmax(ele_logits, dim=1)  # (B, classes, Z, X)
        bin_centers = torch.linspace(-self.ele_range, self.ele_range, self.num_classes, device=probs.device)
        bin_centers = bin_centers.view(1, self.num_classes, 1, 1)
        decoded_ele = (probs * bin_centers).sum(dim=1)  # (B, Z, X)
        
        for i in range(B):
            conf_threshold = 0.4 
            mask = (bin_prob[i] > conf_threshold).float()
            masked_ele = decoded_ele[i] * mask
            masked_ele[mask == 0] = 0.0  # todo: should be set to minimum height value in masked_ele
            
            pred_lanes_vert, pred_lanes_cam = self.extract_lanes(masked_ele, mask, \
                                                                batch_vert2cam[i],
                                                                batch_data_samples[i].metainfo['voxels_info'])
            
            batch_data_samples[i].pred_3dlanes= {
                'raw_ele': decoded_ele[i].cpu(),
                'ele_pred': masked_ele.cpu(),         
                'bin_prob': bin_prob[i].cpu(),
                'pred_lanes_vert': pred_lanes_vert,
                'pred_lanes_cam': pred_lanes_cam,        
            }
        
        return batch_data_samples
    
    def extract_lanes(self, ele_mask, bin_mask, vert2cam, voxels_info):
        import scipy.ndimage as ndi
        from skimage.morphology import skeletonize
        from numpy.polynomial.polynomial import polyfit, polyval
        import matplotlib.pyplot as plt
        
        bin_mask = bin_mask.cpu().numpy().astype(np.uint8)
        bin_mask = ndi.binary_dilation(bin_mask, structure=np.ones((3,1)))
        bin_mask = skeletonize(bin_mask)
        labeled, num_lanes = ndi.label(bin_mask, structure=np.ones((3,3)))
        
        lanes_vert, lanes_cam = [], []
        
        for lane_id in range(1, num_lanes + 1):
            z_idx, x_idx = np.where(labeled == lane_id)
            
            if len(z_idx) < 15:   # filtering
                continue
            
            lane_points = []
            for z in np.unique(z_idx):
                xs = x_idx[z_idx == z]
                
                if len(xs) == 0:
                    continue
                
                x_mean = xs.mean()   # or median
                lane_points.append((z, x_mean))
            lane_points = np.array(lane_points)
            
            z_idx = lane_points[:, 0]
            x_idx = lane_points[:, 1]
            
            H, W = ele_mask.shape[0], ele_mask.shape[1]
            res_z = voxels_info['grid_res'][2].detach().cpu().numpy()
            roi_z_min = voxels_info['roi_z'][0].detach().cpu().numpy()
            res_x = voxels_info['grid_res'][0].detach().cpu().numpy()
            roi_x_min = voxels_info['roi_x'][0].detach().cpu().numpy()
            
            # gp
            x = roi_x_min + (x_idx + 0.5) * res_x
            z = roi_z_min + (H - 1 - z_idx - 0.5) * res_z
            
            # cam height adjustment, cms -> ms
            y = ele_mask[z_idx, x_idx].cpu().numpy()
            
            y_vert = y
            
            lane_vert = np.stack([x, y_vert, z], axis=1)
            
            # smooth lane
            z_vals = lane_vert[:, 2]
            x_vals = lane_vert[:, 0]
            
            coeffs = polyfit(z_vals, x_vals, deg=3)
            x_smooth = polyval(z_vals, coeffs)
            
            lane_vert[:, 0] = x_smooth
            lane_vert = lane_vert[np.argsort(lane_vert[:, 2])] # sort along z
            
            lanes_vert.append(lane_vert)
            
            # vert → cam
            lane_cam = (vert2cam @ lane_vert.T).T
            lanes_cam.append(lane_cam)
            
        return lanes_vert, lanes_cam
    
    def _forward(self, batch_inputs, batch_data_samples=None):
        """
        Raw forward function.
        This is NOT used for loss or predict in MMEngine.
        """
        features = self.backbone(batch_inputs['img'])
        return features