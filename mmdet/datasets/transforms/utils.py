import os
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

def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion for a sparse depth map.

    Args:
        depth_map: A 2D numpy array of shape (H, W) containing depth values 
                   at sparse locations (non-zero) and zeros elsewhere.
        max_depth: The maximum depth value used for inversion.
        custom_kernel: The kernel used for the initial dilation.
        extrapolate: If True, extrapolate depth values to the top of the image.
        blur_type: Either 'bilateral' or 'gaussian'. Bilateral preserves edges.

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