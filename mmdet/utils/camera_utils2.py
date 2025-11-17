import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def adjust_intrinsic(K, ori_shape, new_shape, crop_offset=(0,0)):
    H_ori, W_ori = ori_shape
    H_new, W_new = new_shape
    sx, sy = W_new / W_ori, H_new / H_ori

    K_adj = K.copy().astype(np.float32)
    K_adj[0, 0] *= sx
    K_adj[1, 1] *= sy
    K_adj[0, 2] = K[0, 2] * sx - crop_offset[0]
    K_adj[1, 2] = K[1, 2] * sy - crop_offset[1]
    return K_adj

def get_gt_masks(lanes: list, voxels_info: dict, cam2vert: torch.Tensor, cam_h: float, cam_pitch: float, iterations: int, radius: int = 2):
    H, W = voxels_info['num_grids_z'], voxels_info['num_grids_x']
    ele_mask = np.zeros((H, W), dtype=np.float32)   # switched to numpy directly for cv2 ops
    bin_mask = np.zeros((H, W), dtype=np.uint8)

    if isinstance(cam2vert, np.ndarray):
        cam2vert = torch.from_numpy(cam2vert).float()
    else:
        cam2vert = cam2vert.float()

    # project and rasterize into grid
    for lane_id, lane in enumerate(lanes):
        points_cam = torch.tensor(lane, dtype=torch.float32)  # (N, 3)

        print(f"lane_id: {lane_id}, cam_h: {cam_h}, cam_pitch: {cam_pitch}, [Camera Coordinates] \n\ty_values: {points_cam[:, 1]}")
        print(f"")

        # project to vertical space
        points_vert = (cam2vert @ points_cam.T).T
        # points_vert = points_cam.clone().detach()
        print(f"lane_id: {lane_id}, cam_h: {cam_h}, cam_pitch: {cam_pitch}, [Vertical Coordinates] \n\ty_values: {points_vert[:, 1]}")
        print(f"")
        points_vert[:, 1] = (points_vert[:, 1] - cam_h)*100 #cms
        # points_vert[:, 1] = (cam_h - points_vert[:, 1])
        print(f"lane_id: {lane_id}, cam_h: {cam_h}, cam_pitch:  {cam_pitch}, [Vertical Coordinates After Height Addition] \n\ty_values: {points_vert[:, 1]}")
        print(f"")
        points_vert[:, 1] = -points_vert[:, 1]      # invert y
        print(f"lane_id: {lane_id}, cam_h: {cam_h}, cam_pitch: {cam_pitch}, [Vertical Coordinates After Inversion] \n\ty_values: {points_vert[:, 1]}")

        print(f"="*64)

        # apply ROI mask
        mask = (
            (points_vert[:, 0] >= voxels_info['roi_x'][0]) &
            (points_vert[:, 0] <= voxels_info['roi_x'][1]) &
            (points_vert[:, 2] >= voxels_info['roi_z'][0]) &
            (points_vert[:, 2] <= voxels_info['roi_z'][1]) &
            (points_vert[:, 1] >= -voxels_info['y_range']*100) &
            (points_vert[:, 1] <=  voxels_info['y_range']*100)
        )
        points_roi = points_vert[mask]

        if points_roi.shape[0] == 0:
            continue

        x = points_roi[:, 0].numpy()
        y = points_roi[:, 1].numpy()
        z = points_roi[:, 2].numpy()

        for xi, yi, zi in zip(x, y, z):
            idx_x = int((xi - voxels_info['roi_x'][0]) / voxels_info['grid_res'][0])
            idx_z = H - 1 - int((zi - voxels_info['roi_z'][0]) / voxels_info['grid_res'][2])

            if 0 <= idx_x < W and 0 <= idx_z < H:
                # draw filled circles
                cv2.circle(bin_mask, (idx_x, idx_z), radius=radius, color=255, thickness=-1)
                cv2.circle(ele_mask, (idx_x, idx_z), radius=radius, color=float(yi), thickness=-1)

    # spread/dilate to ensure continuity
    if iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (iterations, iterations))
        bin_mask = cv2.dilate(bin_mask, kernel, iterations=1)

        spread_ele = cv2.dilate(ele_mask, kernel, iterations=1)
        ele_mask[bin_mask > 0] = spread_ele[bin_mask > 0]
    
    # fill in empty pixels
    mask_lane = bin_mask > 0
    if np.any(mask_lane):
        ele_mask[~mask_lane] = np.min(ele_mask[mask_lane])

    return bin_mask.astype(bool), ele_mask

def save_masks(lanes, cam2img, bin_mask, ele_mask, voxels_info, im_pth, save_dir="debug_masks", idx=0):
    os.makedirs(save_dir, exist_ok=True)

    H, W = ele_mask.shape
    extent = [
        voxels_info['roi_x'][0], voxels_info['roi_x'][1], 
        voxels_info['roi_z'][0], voxels_info['roi_z'][1]
    ]

    # binary mask
    plt.figure(figsize=(W/50, H/50))
    plt.imshow(bin_mask, cmap='gray', origin='upper', extent=extent, aspect='auto')
    plt.title("Binary Mask")
    plt.savefig(os.path.join(save_dir, f"bin_mask_{idx}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # elevation mask
    plt.figure(figsize=(W/50, H/50))
    plt.imshow(ele_mask, cmap='jet', origin='upper', extent=extent, aspect='auto')
    plt.colorbar(label="Height (cm)")
    plt.title("Elevation Mask")
    plt.savefig(os.path.join(save_dir, f"ele_mask_{idx}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    lanes_on_image(lanes, cam2img, im_pth, idx)


def lanes_on_image(lanes: list, cam2img: np.ndarray, im_pth: str, idx: int):
    print(f"cam2img.shape: {cam2img.shape}")
    im = cv2.imread(im_pth)
    for lane in lanes:
        points_xyz = np.array(lane)
        # ones = np.ones((points_xyz.shape[0], 1))
        # points_homo = np.hstack([points_xyz, ones])
        points_homo = points_xyz
        proj = cam2img@points_homo.T
        proj = proj.T
        
        proj[:, 0] /= proj[:, 2]
        proj[:, 1] /= proj[:, 2]

        points_uv = proj[:, :2]
        points_uv = points_uv.astype(int)
        
        for i in range(len(points_uv)-1):
            pt1 = tuple(points_uv[i])
            pt2 = tuple(points_uv[i+1])
            cv2.line(im, pt1, pt2, color=(0,255,0), thickness=1)

    fname = im_pth.strip().split('/')[-1]

    saved = cv2.imwrite(f'/data24t_1/owais.tahir/3DLanes/mmdetection/tools/debug_masks/lane_image_{idx}.png', im)
    print(f"saved: {saved}")