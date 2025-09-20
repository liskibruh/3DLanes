import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# define the full kernels if not defined
FULL_KERNEL_5 = np.ones((5, 5), dtype=np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), dtype=np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

colors = [[1, 0, 0],  # red
          [0, 1, 0],  # green
          [0, 0, 1],  # blue
          [1, 0, 1],  # purple
          [0, 1, 1],  # cyan
          [1, 0.7, 0],  # orange
          [0.45, 0.17, 0.07]]  # crown

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


def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral', mode='elevation'):
    """Fast, in-place depth completion for a sparse depth map.

    Args:
        depth_map: A 2D numpy array of shape (H, W) containing depth values 
                   at sparse locations (non-zero) and zeros elsewhere.
        max_depth: The maximum depth value used for inversion.
        custom_kernel: The kernel used for the initial dilation.
        extrapolate: If True, extrapolate depth values to the top of the image.
        blur_type: Either 'bilateral' or 'gaussian'. Bilateral preserves edges.
        mode: Either 'elevation' or 'depth'. 'depth' inverts back to the original depth scale, 'elevation' returns depth map as is.

    Returns:
        depth_map: A dense depth map as a 2D numpy array.
    """
    # Inversion: Convert depths so that smaller (closer) depths become larger.
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate with custom (diamond) kernel to spread valid depths.
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing using a full 5x5 kernel.
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty (invalid) pixels with dilated values.
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Optionally extrapolate depth upward.
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]
        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = top_pixel_values[pixel_col_idx]
        empty_pixels = (depth_map < 0.1)
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Apply median blur to smooth out noise.
    depth_map = cv2.medianBlur(depth_map, 5)

    # Apply bilateral or Gaussian blur.
    if blur_type == 'bilateral':
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    if mode=='elevation':
        return depth_map
    # Invert back to original scale.
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map


def interpolate_3d_tool(lanes):
    """
    each lane of input lanes must have at least two points
    input: lanes: [lane_num, ]
    """

    def cubic_interpolate(lane_spec):
        lane_spec = np.array(lane_spec)
        x = lane_spec[:, 0]
        y = lane_spec[:, 1]
        z = lane_spec[:, 2]
        z_min, z_max = min(z), max(z)
        zmin, zmax = np.ceil(z_min * 1000) / 1000, np.floor(z_max * 1000) / 1000
        gt_lane_points_camera = []
        # less than four points cannot do cubic spline interpolate, use linear instead
        if len(lane_spec) < 4:
            f_zx = interp1d(z, x, kind='linear', fill_value='extrapolate')
            f_zy = interp1d(z, y, kind='linear', fill_value='extrapolate')
            for z in np.arange(zmin, zmax, np.around((zmax - zmin) / 6, 3)):
                z = np.around(z, 3)
                x_out = np.around(f_zx(z), 3)
                y_out = np.around(f_zy(z), 3)
                gt_lane_points_camera.append([x_out, y_out, z])
        else:
            f_zx = interp1d(z, x, kind='cubic', fill_value='extrapolate')
            f_zy = interp1d(z, y, kind='cubic', fill_value='extrapolate')
            for z in np.arange(zmin, zmin + np.around((zmax - zmin) / 2, 1), 0.2):
                z = np.around(z, 3)
                x_out = np.around(f_zx(z), 3)
                y_out = np.around(f_zy(z), 3)
                gt_lane_points_camera.append([x_out, y_out, z])
            for z in np.arange(zmin + np.around((zmax - zmin) / 2, 1), zmax, 0.5):
                z = np.around(z, 3)
                x_out = np.around(f_zx(z), 3)
                y_out = np.around(f_zy(z), 3)

                gt_lane_points_camera.append([x_out, y_out, z])
        return gt_lane_points_camera

    lanes_inter = []
    for lane_spec in lanes:
        if len(lane_spec) < 2:
            continue
        lane_spec_inter = cubic_interpolate(lane_spec)
        lanes_inter.append(lane_spec_inter)
    return lanes_inter

def draw_lanes2d_on_img_line(lanes_2d, img_path, save_path):
    image = cv2.imread(img_path)
    for lane_spec in lanes_2d:
        lane_spec = np.array(lane_spec).astype(np.int32)
        lane_spec = lane_spec.reshape((-1, 1, 2))
        cv2.polylines(image, [lane_spec], False, color=(0, 0, 255), thickness=2)
    cv2.imwrite(save_path, image)

def draw_lanes2d_on_img_points(lanes_2d, img_path, save_path):
    image = cv2.imread(img_path)
    for lane_spec in lanes_2d:
        for point in lane_spec:
            try:
                image = cv2.circle(image, (int(point[0]), int(point[1])), radius=3, color=(0, 0, 255), thickness=-1)
            except:
                print('project error:', int(point[0]), int(point[1]))
    cv2.imwrite(save_path, image)

def draw_lanes3d(all_lane_points, save_path):
    """
    visual all lane points in 3D ax plot, in the final plot, the coordinates have changed to normal coordinates
    input:
    all_lane_points: [lane_num, ]  all_lane_points[0]: [N,3]
    save_path: the dir path of saved image
    """
    matplotlib.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    all_lane_points = np.array(all_lane_points)
    # all_lane_points = interpolate_3d_tool(all_lane_points)  #with interpolation
    for i in range(len(all_lane_points)):
        lane_points = all_lane_points[i]
        lane_points = np.array(lane_points)
        x = lane_points[:, 0]
        y = lane_points[:, 1]
        z = lane_points[:, 2]
        t = np.arange(1, lane_points.shape[0] + 1)  # simple assumption that data was sampled in regular steps
        fitx = np.polyfit(t, x, 3)                  # polyfit
        fity = np.polyfit(t, y, 2)
        fitz = np.polyfit(t, z, 2)
        lane = []
        for j in t:
            x_out = np.polyval(fitx, j)
            y_out = np.polyval(fity, j)
            z_out = np.polyval(fitz, j)
            lane.append([x_out, y_out, z_out])
        lane = np.array(lane)
        lane_x, lane_y, lane_z = lane[:, 0], lane[:, 1], lane[:, 2]

        color = colors[np.mod(i, len(colors))]

        ax.plot(lane_x, lane_z, -lane_y, color=color, linewidth=3, label='3D Lane %d' % i)  # normal coordinates
        # ax.scatter(lane_x, lane_z, -lane_y, s=1, c=np.array(color).reshape(1, -1))

    def inverse_z(z, position):
        return "{:.3}".format(-z)

    ax.zaxis.set_major_formatter(FuncFormatter(inverse_z))
    ax.legend()
    plt.savefig(os.path.join(save_path, 'lanes_3d.png'))


def interp_lane(lane: list[list], num_points: int = 30) -> list[list]:
    """
    Interpolate a single lane to have exactly num_points along z-axis.
    
    Args:
        lane: list of [x, y, z] points in camera coordinates
        num_points: desired number of points in output lane
    
    Returns:
        interpolated lane as list of [x, y, z] with length num_points
    """
    lane_spec = np.array(lane)
    x, y, z = lane_spec[:, 0], lane_spec[:, 1], lane_spec[:, 2]

    if len(lane_spec) < 2:
        # Not enough points to interpolate
        return lane

    # Decide interpolation kind
    kind = 'linear' if len(lane_spec) < 4 else 'cubic'

    # Create interpolating functions
    f_zx = interp1d(z, x, kind=kind, fill_value='extrapolate')
    f_zy = interp1d(z, y, kind=kind, fill_value='extrapolate')

    # Generate evenly spaced z values
    z_min, z_max = z.min(), z.max()
    z_new = np.linspace(z_min, z_max, num_points)

    # Interpolate x and y
    x_new = f_zx(z_new)
    y_new = f_zy(z_new)

    lane_new = np.stack([x_new, y_new, z_new], axis=1)
    return lane_new.tolist()


def visualize_lanes(lanes_2d, im, scale_factor):
    h, w = scale_factor
    for lane in lanes_2d:
        lane_scaled = np.array(lane * np.array([w, h], dtype=np.float32))
        lane_scaled = lane_scaled.astype(np.int32)

        # draw circle markers
        for x, y in lane_scaled:
            cv2.circle(im, (x, y), thickness=3, radius=2, color=(255, 0, 0))

        # draw polyline connecting the points
        cv2.polylines(im, [lane_scaled.reshape(-1, 1, 2)], isClosed=False, color=(0, 255, 0), thickness=2)

    cv2.imwrite('/data24t_1/owais.tahir/3DLanes/mmdet/temp_labels/temp.png', im)


def fill_masks(ele_mask, grid_mask, iterations=2):
    ele = ele_mask.unsqueeze(0).unsqueeze(0)
    valid = grid_mask.float().unsqueeze(0).unsqueeze(0)

    # Kernel for 3x3 dilation
    kernel = torch.ones((1, 1, 3, 3), device=ele_mask.device)

    # dilate binary mask
    for _ in range(iterations):
        dilated = (F.conv2d(valid, kernel, padding=1) > 0).float()
        valid = dilated

    # propagate elevation into the dilated regions
    for _ in range(iterations * 2):  # a few smoothing passes
        smoothed = F.conv2d(ele, kernel, padding=1)
        counts   = F.conv2d(valid, kernel, padding=1)

        new_vals = smoothed / (counts + 1e-6)
        update_mask = (counts > 0) & (ele == 0)

        ele[update_mask] = new_vals[update_mask]

    return valid.squeeze(0).squeeze(0).bool(), ele.squeeze(0).squeeze(0)


def get_gt_masks(lanes: list, voxels_info: dict, cam2vert: torch.Tensor, cam_h: float, iterations: int):
    H, W = voxels_info['num_grids_z'], voxels_info['num_grids_x']
    ele_mask = torch.zeros((H, W), dtype=torch.float32)
    grids_count = torch.zeros((H, W), dtype=torch.int32)

    if isinstance(cam2vert, np.ndarray):
        cam2vert = torch.from_numpy(cam2vert).float()
    else:
        cam2vert = cam2vert.float()

    for lane_id, lane in enumerate(lanes):
        points_cam = torch.tensor(lane, dtype=torch.float32)  # (N, 3)

        """
        To extract the true height (relative to ground) of points, you need to first
        transform the lane points to ground coordinate system. then the height dimension
        will tell you the true height of that point relative to the ground.
        Also, invert height: y = -y in bev masks
        """
        
        print(f"\npoints_cam[0]: {points_cam[0]}")
        print(f"voxels_info['y_range']: {voxels_info['y_range']}")
        points_vert = (cam2vert @ points_cam.T).T             # (N, 3), in (x, z, y)

        # Debug ranges before ROI cropping
        print(f"[Lane {lane_id}] Before ROI crop:")
        print(f"  x range (right):   {points_vert[:,0].min():.2f} → {points_vert[:,0].max():.2f}")
        print(f"  z range (forward): {points_vert[:,1].min():.2f} → {points_vert[:,1].max():.2f}")
        print(f"  y range (elev):    {points_vert[:,2].min():.2f} → {points_vert[:,2].max():.2f}")
        print(f"  ROI: x={voxels_info['roi_x']}, "
              f"z={voxels_info['roi_z']}, "
              f"y~[-{voxels_info['y_range']}, {voxels_info['y_range']}]")

        # ROI crop (around ground y=0, not camera height)
        mask = (
            (points_vert[:,0] >= voxels_info['roi_x'][0]) &
            (points_vert[:,0] <= voxels_info['roi_x'][1]) &
            (points_vert[:,1] >= voxels_info['roi_z'][0]) &
            (points_vert[:,1] <= voxels_info['roi_z'][1]) &
            (points_vert[:,2] >= -voxels_info['y_range']) &
            (points_vert[:,2] <=  voxels_info['y_range'])
        )
        points_roi = points_vert[mask]

        if points_roi.shape[0] == 0:
            print("  → No points survived ROI crop")
            continue

        # Explicit naming after crop
        x = points_roi[:, 0]  # right
        z = points_roi[:, 1]  # forward
        y = points_roi[:, 2]  # elevation

        print(f"  After ROI crop: kept {len(points_roi)} points")
        print(f"    x range: {x.min():.2f} → {x.max():.2f}")
        print(f"    z range: {z.min():.2f} → {z.max():.2f}")
        print(f"    y range: {y.min():.2f} → {y.max():.2f}")

        for xi, zi, yi in zip(x, z, y):
            idx_x = int((xi - voxels_info['roi_x'][0]) / voxels_info['grid_res'][0])
            idx_z = H - 1 - int((zi - voxels_info['roi_z'][0]) / voxels_info['grid_res'][2])

            if 0 <= idx_x < W and 0 <= idx_z < H:
                print(f"    Placing point → grid idx_z={idx_z}, idx_x={idx_x}, elev={-yi:.3f}")
                ele_mask[idx_z, idx_x] += (-yi)
                grids_count[idx_z, idx_x] += 1

    # average per grid cell
    grid_mask = grids_count > 0
    ele_mask[grid_mask] /= grids_count[grid_mask]

    bin_mask, ele_mask = fill_masks(ele_mask, grid_mask, iterations=iterations)
    return bin_mask, ele_mask

def save_masks(bin_mask, ele_mask, voxels_info, im_pth, save_dir="debug_masks", idx=0):
    os.makedirs(save_dir, exist_ok=True)

    # Convert tensors to numpy if needed
    if isinstance(bin_mask, np.ndarray) == False:
        bin_mask = bin_mask.cpu().numpy() if hasattr(bin_mask, 'cpu') else np.array(bin_mask)
    if isinstance(ele_mask, np.ndarray) == False:
        ele_mask = ele_mask.cpu().numpy() if hasattr(ele_mask, 'cpu') else np.array(ele_mask)

    # Correct z-axis: map first axis (rows) to roi_z correctly
    H, W = bin_mask.shape
    extent = [
        voxels_info['roi_x'][0], voxels_info['roi_x'][1],  # left, right
        voxels_info['roi_z'][0], voxels_info['roi_z'][1]   # bottom, top
    ]

    # Binary mask
    plt.figure(figsize=(W/50, H/50))
    plt.imshow(bin_mask, cmap='gray', origin='upper', extent=extent, aspect='auto')
    plt.title("Binary Mask")
    plt.xlabel("x (right)")
    plt.ylabel("z (forward)")
    plt.savefig(os.path.join(save_dir, f"bin_mask_{idx}.png"))
    plt.close()

    # Elevation mask
    plt.figure(figsize=(W/50, H/50))
    plt.imshow(ele_mask, cmap='jet', origin='upper', extent=extent, aspect='auto')
    plt.colorbar(label="Height (m)")
    plt.title("Elevation Mask")
    plt.xlabel("x (right)")
    plt.ylabel("z (forward)")
    plt.savefig(os.path.join(save_dir, f"ele_mask_{idx}.png"))
    plt.close()

    # Save image
    im = cv2.imread(im_pth)
    if im is not None:
        cv2.imwrite(os.path.join(save_dir, f"image_{idx}.png"), im)