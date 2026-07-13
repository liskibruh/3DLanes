import numpy as np
import torch
from torch.nn import functional as F
from mmdet.registry import MODELS, TRANSFORMS
from mmdet.models.detectors.base import BaseDetector

import time

def plot_on_image(pred_lanes, gt_lanes, im_pth, cam_intrin, voxels_info, out_dir):
    import os
    import cv2

    Z_MIN, Z_MAX = voxels_info['roi_z']

    im_pred = cv2.imread(im_pth)
    im_gt = cv2.imread(im_pth)

    for lane_cam in pred_lanes:
        if not isinstance(lane_cam, np.ndarray):
            lane_cam = np.array(lane_cam)
        # lane shape: (N,3)
        proj = cam_intrin@lane_cam.T
        proj = proj.T   # [x1, y1, z1]
                        # [x2, y2, z2]
                        # [x3, y3, z3]
                        # [., ., .,]        (N, 3)

        proj[:, 0]/=proj[:, 2]  # x/z
        proj[:, 1]/=proj[:, 2]  # y/z

        points_uv = proj[:, :2]
        points_uv = points_uv.astype(int)

        for i in range(len(points_uv)-1):
            pt1 = tuple(points_uv[i])
            pt2 = tuple(points_uv[i+1])
            cv2.line(im_pred, pt1, pt2, color=(255, 0, 0), thickness=8)

    for lane_cam in gt_lanes:
        if not isinstance(lane_cam, np.ndarray):
            lane_cam = np.array(lane_cam)
        # lane shape: (N,3)
        valid = (lane_cam[:, 2] >= np.array(Z_MIN)) & (lane_cam[:, 2] <= np.array(Z_MAX))
        lane_cam = lane_cam[valid]

        proj = cam_intrin@lane_cam.T
        proj = proj.T   # [x1, y1, z1]
                        # [x2, y2, z2]
                        # [x3, y3, z3]
                        # [., ., .,]        (N, 3)

        proj[:, 0]/=proj[:, 2]  # x/z
        proj[:, 1]/=proj[:, 2]  # y/z

        points_uv = proj[:, :2]
        points_uv = points_uv.astype(int)

        for i in range(len(points_uv)-1):
            pt1 = tuple(points_uv[i])
            pt2 = tuple(points_uv[i+1])
            cv2.line(im_gt, pt1, pt2, color=(0, 0, 255), thickness=8)

    im_out = cv2.hconcat([im_pred, im_gt])
        
    os.makedirs(out_dir, exist_ok=True)
    fname = "".join([im_pth.strip().split('/')[-2], im_pth.strip().split('/')[-1]])
    out_pth = os.path.join(out_dir, fname)
    success = cv2.imwrite(out_pth, im_out)
    if success:
        print(f"[INFO]: gt and pred lane visualized on image saved at {out_pth}")
    else:
        print(f"[WARNING: gt and pred lane visualized on image saving failed!")

@MODELS.register_module()
class Voxel3DLanes(BaseDetector):
    def __init__(self,
                backbone,
                # neck=None,
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
        # self.neck = MODELS.build(neck)
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

        start_time = time.time()

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
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000.0
        # print(f"[Latency] Train time: {latency_ms:.2f} ms "
        #     f"({latency_ms / len(batch_data_samples):.2f} ms / image)")
        
        return losses
    
    def predict(self, batch_inputs, batch_data_samples, **kwargs):

        start_time = time.time()

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
            
            pred_lanes_vert, pred_lanes_cam, SKIP_ = self.extract_lanes_new(masked_ele, mask, \
                                                                    batch_vert2cam[i],
                                                                    batch_data_samples[i].metainfo['voxels_info'])
            if SKIP_:
                print("[WARNING]: Skipping sample..")
                continue
            # plot_on_image(
            #     pred_lanes=pred_lanes_cam,
            #     gt_lanes=batch_data_samples[i].gt_lanes_cam, 
            #     im_pth=batch_data_samples[i].metainfo['img_path'],
            #     cam_intrin=batch_data_samples[i].metainfo['cam_intrinsic'],
            #     voxels_info=batch_data_samples[i].metainfo['voxels_info'],
            #     out_dir="./gt_and_pred_vis"
            #     )
            
            batch_data_samples[i].pred_3dlanes= {
                'raw_ele': decoded_ele[i].cpu(),
                'ele_pred': masked_ele.cpu(),         
                'bin_prob': bin_prob[i].cpu(),
                'pred_lanes_vert': pred_lanes_vert,
                'pred_lanes_cam': pred_lanes_cam,        
            }
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000.0
        # print(f"[Latency] Inference time: {latency_ms:.2f} ms "
        #     f"({latency_ms / len(batch_data_samples):.2f} ms / image)")

        return batch_data_samples

    def extract_lanes(self, ele_mask, bin_mask, vert2cam, voxels_info):
        import scipy.ndimage as ndi
        from skimage.morphology import skeletonize
        import matplotlib.pyplot as plt
    
        bin_mask = bin_mask.cpu().numpy().astype(np.uint8)
        # plt.imsave('test_img.png', bin_mask, cmap='gray')
        bin_mask = ndi.binary_dilation(bin_mask, structure=np.ones((3,1)))
        bin_mask = skeletonize(bin_mask)
        labeled, num_lanes = ndi.label(bin_mask, structure=np.ones((3,3)))
        # print(f'num_lanes: {num_lanes}')
        lanes_vert, lanes_cam = [], []

        for lane_id in range(1, num_lanes + 1):
            z_idx, x_idx = np.where(labeled == lane_id)

            if len(z_idx) < 7:   # filtering
                continue

            H, W = ele_mask.shape[0], ele_mask.shape[1]
            res_z = voxels_info['grid_res'][2].detach().cpu().numpy()
            roi_z_min = voxels_info['roi_z'][0].detach().cpu().numpy()
            res_x = voxels_info['grid_res'][0].detach().cpu().numpy()
            roi_x_min = voxels_info['roi_x'][0].detach().cpu().numpy()
        
            # cld
            # z = (H - 1 - z_idx) * res_z + roi_z_min + (res_z/2)
            # x = x_idx * res_x + roi_x_min + (res_x/2)

            # gp
            x = roi_x_min + (x_idx + 0.5) * res_x
            z = roi_z_min + (H - 1 - z_idx - 0.5) * res_z

            # standard
            # z = z_idx * voxels_info['grid_res'][2].detach().cpu().numpy()+ voxels_info['roi_z'][0].detach().cpu().numpy()
            # x = x_idx * voxels_info['grid_res'][0].detach().cpu().numpy() + voxels_info['roi_x'][0].detach().cpu().numpy()
            
            y = ele_mask[z_idx, x_idx].cpu().numpy()
            y_vert = voxels_info['base_height'] - (y / 100.0) # cam height adjustment, cms -> ms

            lane_vert = np.stack([x, y_vert, z], axis=1)
            lane_vert = lane_vert[np.argsort(lane_vert[:, 2])] # sort along z
            lanes_vert.append(lane_vert)

            # vert → cam
            lane_cam = (vert2cam @ lane_vert.T).T
            lanes_cam.append(lane_cam)

        return lanes_vert, lanes_cam
    
    def extract_lanes_new(self, ele_mask, bin_mask, vert2cam, voxels_info):
        SKIP_SAMPLE = False
        import scipy.ndimage as ndi
        from skimage.morphology import skeletonize
        from numpy.polynomial.polynomial import polyfit, polyval
        import matplotlib.pyplot as plt
    
        bin_mask = bin_mask.cpu().numpy().astype(np.uint8)
        # plt.imsave('test_img.png', bin_mask, cmap='gray')
        bin_mask = ndi.binary_dilation(bin_mask, structure=np.ones((3,1)))
        bin_mask = skeletonize(bin_mask)
        labeled, num_lanes = ndi.label(bin_mask, structure=np.ones((3,3)))
        # print(f'num_lanes: {num_lanes}')
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

            # for metrics (comment if for plotting)
            y_vert = y

            # for plotting (comment if for metrics)
            # y_vert = voxels_info['base_height'] - (y / 100.0) 
            # y_vert = y / 100.0

            lane_vert = np.stack([x, y_vert, z], axis=1)

            # smooth lane
            z_vals = lane_vert[:, 2]
            x_vals = lane_vert[:, 0]

            coeffs = polyfit(z_vals, x_vals, deg=3)
            x_smooth = polyval(z_vals, coeffs)

            lane_vert[:, 0] = x_smooth
            lane_vert = lane_vert[np.argsort(lane_vert[:, 2])] # sort along z

            # flat z assumption (temporary debug)
            ## the elevation predictions are not good atm
            ## delete/uncomment the following line for using predicted ele insted of flat
            # lane_vert[:, 1] = voxels_info['base_height']

            ## alternatively, we can also ignore lanes with large ele variation
            dy = np.diff(lane_vert[:, 1]) / 100.0 # convert from cms to ms
            if np.sum(np.abs(dy)) > 0.5: # if sum of lane height is > 0.1 ms
                SKIP_SAMPLE=True

            lanes_vert.append(lane_vert)

            # vert → cam
            lane_cam = (vert2cam @ lane_vert.T).T
            lanes_cam.append(lane_cam)

        return lanes_vert, lanes_cam, SKIP_SAMPLE

    def predict_old(self, batch_inputs, batch_data_samples, **kwargs):
        features_left = self.backbone(batch_inputs['img'])
        B, C, H, W = features_left.shape
        features_left = features_left.reshape(B, C, -1)
    
        if 'voxel_proj_index' in batch_inputs:
            proj_index_left = batch_inputs['voxel_proj_index']
            
            # check if indices are within bounds
            x_indices = proj_index_left[:, 0, :]
            y_indices = proj_index_left[:, 1, :]

            linear_indices = proj_index_left[:, 1, :] * W + proj_index_left[:, 0, :]
            voxel_feat_left = features_left.gather(dim=2, index=linear_indices.unsqueeze(1).expand(-1, C, -1))
            voxel_feat_left = voxel_feat_left.reshape(B, C, self.num_grids_z, self.num_grids_x, self.num_grids_y)

            ele_pred = self.ele_head.forward(voxel_feat_left)

        ele_pred = F.softmax(ele_pred, dim=1)
        ele_pred = torch.sum(ele_pred * self.ele_values, dim=1)

        for i, sample in enumerate(batch_data_samples):
            sample.pred_3dlanes = dict(
                ele_pred = ele_pred[i].detach().cpu()
            )
        return batch_data_samples

    def _forward(self, batch_inputs, batch_data_samples=None):
        """
        Raw forward function.
        This is NOT used for loss or predict in MMEngine.
        """
        features = self.backbone(batch_inputs['img'])
        return features

@MODELS.register_module()
class Sparse_3DLanes(BaseDetector):
    def __init__(self,
                backbone,
                # neck=None,
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
        # self.neck = MODELS.build(neck)
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

        start_time = time.time()

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
            gt_bin_mask = torch.stack(gt_bin_masks, dim=0)  # [B, 1, Nz, Nx]
            gt_ele_mask = torch.stack(gt_ele_masks, dim=0)  # [B, 1, Nz, Nx]


        features_left = self.backbone(batch_inputs['img'])  # [B, C, H, W]
        B, C, H, W = features_left.shape

        if 'voxel_proj_index' in batch_inputs:
            proj_index_left = batch_inputs['voxel_proj_index']  # [B, 2, Nz*Nx*Ny]?
            linear_indices = proj_index_left[:, 1, :] * W + proj_index_left[:, 0, :] # linear index formula (M*y+x) => [B, Nz*Nx*Ny]

            gt_bin_bev = gt_bin_mask.squeeze(1) # [B, Nz, Nx]
            gt_bin_3d = gt_bin_bev.unsqueeze(-1).expand(-1, -1, -1, self.num_grids_y)   # [B, Nz, Nx, Ny]
            gt_bin_flat = gt_bin_3d.reshape(B, -1)  # [B, Nz*Nx*Ny]

            feat_left_flat = features_left.reshape(B, C, -1)    # [B, C, H*W]

            num_active_per_batch = gt_bin_flat.sum(dim=1)

        active_lin_indices = []
        active_coords = []
        for b in range(B):
            # filter non-lane indices
            mask_b = gt_bin_flat[b]
            active_idx = mask_b.nonzero(as_tuple=False).squeeze(1)  # [N_active_b]
            active_lin_idx_b = linear_indices[b][active_idx]
            active_lin_indices.append(active_lin_idx_b)

            # recover 3D coordinates
            iz = active_idx // (self.num_grids_x * self.num_grids_y)
            ix = (active_idx % (self.num_grids_x * self.num_grids_y)) // self.num_grids_y
            iy = active_idx % self.num_grids_y
            active_coords_b = torch.stack([iz, ix, iy], dim=1)  # [N_active_b, 3]
            active_coords.append(active_coords_b)

        active_lin_indices = torch.stack(active_lin_indices)    # [B, N_active_b]
        active_coords = torch.stack(active_coords)  # [B, N_active_b, 3]

        # gather feats for active voxels
        voxel_feat_active = feat_left_flat.gather(
            dim=2,
            index=active_lin_indices.unsqueeze(1).expand(-1, C, -1)
        )   # [B, C, max_active]

        print(f"voxel_feat_actives.shape: {voxel_feat_active.shape}")

        all_feats = []
        all_coords = []
        for b in range(B):
            n_active = num_active_per_batch[b].item()

            feats_b = voxel_feat_active[b, :, :n_active] # [C, N_active_b]
            coords_b = active_coords[b, :n_active]    # [N_active_b, 3]
            all_feats.append(feats_b.permute(1, 0))

            # add batch index
            batch_idx = torch.full(
                (n_active, 1),
                b,
                dtype=coords_b.dtype,
                device=coords_b.device
            )

            coords_with_batch = torch.cat([batch_idx, coords_b], dim=1)
            all_coords.append(coords_with_batch)
        
        feats_sparse = torch.cat(all_feats, dim=0) # [N_total, C]
        coords_sparse = torch.cat(all_coords, dim=0) # [N_total, 4]

        print(f"feats_sparse.shape: {feats_sparse.shape}")
        print(f"coords_sparse.shape: {coords_sparse.shape}")
        print(f"="*64)
        exit()

        """
        from torchsparse import SparseTensor

        sparse_input = SparseTensor(
            feats=feats_sparse,
            coords=coords_sparse
        )

        bin_out = self.sparse_bin_head(sparse_input)
        ele_out = self.sparse_ele_head(sparse_input)
        """

    def predict(self, batch_inputs, batch_data_samples, **kwargs):

        start_time = time.time()

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
            
            pred_lanes_vert, pred_lanes_cam, SKIP_ = _3DLanes.extract_lanes_new(masked_ele, mask, \
                                                                    batch_vert2cam[i],
                                                                    batch_data_samples[i].metainfo['voxels_info'])
            if SKIP_:
                print("[WARNING]: Skipping sample..")
                continue
            # plot_on_image(
            #     pred_lanes=pred_lanes_cam,
            #     gt_lanes=batch_data_samples[i].gt_lanes_cam, 
            #     im_pth=batch_data_samples[i].metainfo['img_path'],
            #     cam_intrin=batch_data_samples[i].metainfo['cam_intrinsic'],
            #     voxels_info=batch_data_samples[i].metainfo['voxels_info'],
            #     out_dir="./gt_and_pred_vis"
            #     )
            
            batch_data_samples[i].pred_3dlanes= {
                'raw_ele': decoded_ele[i].cpu(),
                'ele_pred': masked_ele.cpu(),         
                'bin_prob': bin_prob[i].cpu(),
                'pred_lanes_vert': pred_lanes_vert,
                'pred_lanes_cam': pred_lanes_cam,        
            }
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000.0
        print(f"[Latency] Inference time: {latency_ms:.2f} ms "
            f"({latency_ms / len(batch_data_samples):.2f} ms / image)")

        return batch_data_samples

    def _forward(self, batch_inputs, batch_data_samples=None):
        """
        Raw forward function.
        This is NOT used for loss or predict in MMEngine.
        """
        features = self.backbone(batch_inputs['img'])
        return features