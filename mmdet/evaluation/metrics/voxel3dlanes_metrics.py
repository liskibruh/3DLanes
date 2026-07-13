import os
import json
import pickle
import numpy as np
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
import torch
import matplotlib.pyplot as plt

from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS

@METRICS.register_module()
class SaveElePredMetric:
    def __init__(self, save_dir, vis_dir):
        self.save_dir = save_dir
        self.vis_dir = vis_dir

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

    def save_vis(self, ele_pred: np.ndarray, out_path: str):
        # ele_pred: (H, W)
        h, w = ele_pred.shape
        Z, X = np.meshgrid(np.arange(w), np.arange(h))

        ele_max = np.max(ele_pred)
        ele_min = np.min(ele_pred)

        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        im = ax.pcolormesh(
            Z, X, ele_pred,
            cmap='plasma',
            vmin=ele_min,
            vmax=ele_max,
            shading='auto'
        )

        ax.set_aspect('equal')
        ax.set_title('Elevation Prediction')
        fig.colorbar(im, ax=ax)

        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            img_path = sample['img_path']
            img_name = os.path.basename(img_path).split('.')[0]

            if sample.get('pred_3dlanes', None) is not None:
                ele_pred = sample['pred_3dlanes']['ele_pred']

                if isinstance(ele_pred, torch.Tensor):
                    ele_pred = ele_pred.detach().cpu().numpy()

                # Optional orientation fix (ONLY if you know it's needed)
                # ele_pred = np.flipud(ele_pred)

                # Save visualization
                vis_path = os.path.join(self.vis_dir, f'{img_name}.png')
                self.save_vis(ele_pred, vis_path)

                # Save raw prediction
                out_pkl = os.path.join(self.save_dir, f'{img_name}.pkl')
                # with open(out_pkl, 'wb') as f:
                #     pickle.dump(ele_pred, f)

    def compute_metrics(self, results):
        return {}


class LaneEval(BaseMetric):
    def __init__(self, mode='2D',
                dist_thresh=0.5,
                format_only=False, # only save preds and gts to txt files
                outfile_prefix=None, # path prefix for saving txt files
                collect_device='cpu',
                prefix=None, # add prefix to metric names
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.mode = mode
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix
        self.dist_thresh = dist_thresh

    def process_2d_lanes(self, data_batch, data_samples):
        for sample in data_samples:
            Z_MAX, Z_MIN = sample['voxels_info']['roi_z']
            gts = dict(lanes=[])
            preds = dict(lanes=[])
            if sample.get('pred_3dlanes', None) is not None:
                gt_lanes = sample['gt_lanes_vert']
                pred_lanes = sample['pred_3dlanes']['pred_lanes_vert'] # was ['pred_3dlanes']['pred_lanes_vert']

                for gt_lane in gt_lanes:
                    if not isinstance(gt_lane, np.ndarray):
                        gt_lane = np.array(gt_lane)
                    valid = (gt_lane[:, 2] >= np.array(Z_MIN)) & (gt_lane[:, 2] <= np.array(Z_MAX))
                    # extract xz (2D) lane
                    gt_lane_x = gt_lane[:, 0]
                    gt_lane_z = gt_lane[:, 2]
                    gt_lane_2d = np.array([gt_lane_x, gt_lane_z])
                    gt_lane_2d = np.sort(gt_lane_2d, axis=-1).T # sort along z axis
                    gts['lanes'].append(gt_lane_2d)
                gts['img_path'] = sample['img_path']

                for pr_lane in pred_lanes:
                    if not isinstance(pr_lane, np.ndarray):
                        pr_lane = pr_lane.cpu().numpy()
                    # extract xz (2D) lane
                    pr_lane_x = pr_lane[:, 0]
                    pr_lane_z = pr_lane[:, 2]
                    pr_lane_2d = np.array([pr_lane_x, pr_lane_z])
                    pr_lane_2d = np.sort(pr_lane_2d, axis=-1).T # sort along z axis
                    preds['lanes'].append(pr_lane_2d)
                preds['img_path'] = sample['img_path']

                # if self.outfile_prefix is not None:
                #     os.makedirs(self.outfile_prefix, exist_ok=True)
                #     gt_pr = {
                #         'ground_truths': {
                #             'lanes': to_json_safe(gts['lanes']),
                #             'img_path': gts.get('img_path', '')
                #         },
                #         'predictions': {
                #             'lanes': to_json_safe(preds['lanes']),
                #             'img_path': preds.get('img_path', '')
                #         }
                #     }
                #     dir_name = "" # todo
                #     im_name = "" # todo
                #     out_pth = os.path.join(self.outfile_prefix, dir_name + "_" + im_name + ".json")

                #     with open(out_pth, 'w') as outfile:
                #         json.dump(gt_pr, outfile, indent=4)
            self.results.append(
                {
                    'gt_lanes': gts['lanes'],
                    'pred_lanes': preds['lanes']
                }
            )
    
    def process_3d_lanes(self, data_batch, data_samples):
        for sample in data_samples:
            if sample.get('pred_3dlanes', None) is not None:
                pass
    
    def process(self, data_batch, data_samples):
        if self.mode=='2D':
            self.process_2d_lanes(data_batch, data_samples)
        elif self.mode=='3D':
            self.process_3d_lanes(data_batch, data_samples)
        else:
            print(f"[ERROR]! 'mode' can only be '2D' or '3D' (case sensitive)")
    
    def compute_metrics(self, results):
        if self.format_only:
            print(f"[INFO]: results stored to {self.outfile_prefix}")
            return OrderedDict()

        if self.mode=='2D':
            all_tp, all_fp, all_fn = 0, 0, 0
            all_errors = []
            z_samples = np.linspace(5, 80, 38) # from 5 meters to 80 meters every ~2 meters
            for res in results:
                gt_lanes = res['gt_lanes']
                pred_lanes = res['pred_lanes']

                if len(gt_lanes) == 0 and len(pred_lanes) == 0:
                    continue

                cost = build_cost_matrix(gt_lanes, pred_lanes, z_samples)
                cost = np.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6) # replace invalid entries with large valid value
                # assert cost.shape == (len(gt_lanes), len(pred_lanes))
                # assert np.isfinite(cost).all()
                row_ind, col_ind = linear_sum_assignment(cost)

                matched_gt = set()
                matched_pr = set()

                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] < self.dist_thresh:
                        all_tp += 1
                        all_errors.append(cost[r, c])
                        matched_gt.add(r)
                        matched_pr.add(c)

                all_fp += len(pred_lanes) - len(matched_pr)
                all_fn += len(gt_lanes) - len(matched_gt)

            precision = all_tp / max(all_tp + all_fp, 1)
            recall = all_tp / max(all_tp + all_fn, 1)
            f1 = 2 * all_tp / max(2 * all_tp + all_fp + all_fn, 1)

            return {
                'TP': all_tp,
                'FP': all_fp,
                'FN': all_fn,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'Mean_Lateral_Error': float(np.mean(all_errors)) if all_errors else 0.0
            }
            
        elif self.mode=='3D':
            pass

@METRICS.register_module()
class LaneEval3D(BaseMetric):
    def __init__(self, mode='2D',
                dist_thresh=0.5,
                format_only=False, # only save preds and gts to txt files
                outfile_prefix=None, # path prefix for saving txt files
                collect_device='cpu',
                prefix=None, # add prefix to metric names
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.mode = mode
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix
        self.dist_thresh = dist_thresh


    def process_2d_lanes(self, data_batch, data_samples):
        for sample in data_samples:
            Z_MAX, Z_MIN = sample['voxels_info']['roi_z']
            gts = dict(lanes=[])
            preds = dict(lanes=[])
            if sample.get('pred_3dlanes', None) is not None:
                gt_lanes = sample['gt_lanes_vert']
                pred_lanes = sample['pred_3dlanes']['pred_lanes_vert']

                for gt_lane in gt_lanes:
                    if not isinstance(gt_lane, np.ndarray):
                        gt_lane = np.array(gt_lane)
                    
                    # Extract X and Z columns to make 2D lane
                    gt_lane_2d = gt_lane[:, [0, 2]]  # Select columns 0 (x) and 2 (z)
                    gt_lane_2d = gt_lane_2d[np.argsort(gt_lane_2d[:, 1])]  # Sort by Z
                    gts['lanes'].append(gt_lane_2d)
                
                gts['img_path'] = sample['img_path']

                for pr_lane in pred_lanes:
                    if not isinstance(pr_lane, np.ndarray):
                        pr_lane = pr_lane.cpu().numpy()
                    
                    pr_lane_2d = pr_lane[:, [0, 2]]
                    pr_lane_2d = pr_lane_2d[np.argsort(pr_lane_2d[:, 1])]
                    preds['lanes'].append(pr_lane_2d)
                
                preds['img_path'] = sample['img_path']

            self.results.append({
                'gt_lanes': gts['lanes'],
                'pred_lanes': preds['lanes']
            })
    
    def process_3d_lanes(self, data_batch, data_samples):
        for sample in data_samples:
            Z_MAX, Z_MIN = sample['voxels_info']['roi_z']
            gts = dict(lanes=[])
            preds = dict(lanes=[])
            if sample.get('pred_3dlanes', None) is not None:
                gt_lanes = sample['gt_lanes_vert']
                pred_lanes = sample['pred_3dlanes']['pred_lanes_vert'] # was ['pred_3dlanes']['pred_lanes_vert']

                for gt_lane in gt_lanes:
                    if not isinstance(gt_lane, np.ndarray):
                        gt_lane = np.array(gt_lane)
                    if gt_lane.shape[1] < 3:
                        print(f"Warning: gt_lane has shape {gt_lane.shape}, skipping")
                        continue
                    valid = (gt_lane[:, 2] >= np.array(Z_MIN)) & (gt_lane[:, 2] <= np.array(Z_MAX))
                    # extract xz (2D) lane
                    # gt_lane_x = gt_lane[:, 0]
                    # gt_lane_y = gt_lane[:, 1]
                    # gt_lane_z = gt_lane[:, 2]
                    # gt_lane_3d = np.array([gt_lane_x, gt_lane_y, gt_lane_z])
                    gt_lane_3d = gt_lane[np.argsort(gt_lane[:, 2])] # sort along z axis
                    # gt_lane_3d = np.sort(gt_lane_3d, axis=-1).T # sort along z axis
                    gts['lanes'].append(gt_lane_3d)
                gts['img_path'] = sample['img_path']

                for pr_lane in pred_lanes:
                    if not isinstance(pr_lane, np.ndarray):
                        pr_lane = pr_lane.cpu().numpy()
                    if pr_lane.shape[1] < 3:
                        print(f"Warning: pr_lane has shape {pr_lane.shape}, skipping")
                        continue
                    pr_lane_3d = pr_lane[np.argsort(pr_lane[:, 2])] # sort rows by z, keeping x,y,z tied per point
                    preds['lanes'].append(pr_lane_3d)
                preds['img_path'] = sample['img_path']

                # if self.outfile_prefix is not None:
                #     os.makedirs(self.outfile_prefix, exist_ok=True)
                #     gt_pr = {
                #         'ground_truths': {
                #             'lanes': to_json_safe(gts['lanes']),
                #             'img_path': gts.get('img_path', '')
                #         },
                #         'predictions': {
                #             'lanes': to_json_safe(preds['lanes']),
                #             'img_path': preds.get('img_path', '')
                #         }
                #     }
                #     dir_name = "" # todo
                #     im_name = "" # todo
                #     out_pth = os.path.join(self.outfile_prefix, dir_name + "_" + im_name + ".json")

                #     with open(out_pth, 'w') as outfile:
                #         json.dump(gt_pr, outfile, indent=4)
            self.results.append(
                {
                    'gt_lanes': gts['lanes'],
                    'pred_lanes': preds['lanes']
                }
            )
    
    def process(self, data_batch, data_samples):
        if self.mode=='2D':
            self.process_2d_lanes(data_batch, data_samples)
        elif self.mode=='3D':
            self.process_3d_lanes(data_batch, data_samples)
        else:
            print(f"[ERROR]! 'mode' can only be '2D' or '3D' (case sensitive)")
    
    def compute_metrics(self, results):
        if self.format_only:
            print(f"[INFO]: results stored to {self.outfile_prefix}")
            return OrderedDict()

        if self.mode=='2D':
            all_tp, all_fp, all_fn = 0, 0, 0
            all_errors = []
            z_samples = np.linspace(5, 80, 38) # from 5 meters to 80 meters every ~2 meters
            for res in results:
                gt_lanes = res['gt_lanes']
                pred_lanes = res['pred_lanes']

                if len(gt_lanes) == 0 and len(pred_lanes) == 0:
                    continue

                cost = build_cost_matrix(gt_lanes, pred_lanes, z_samples)
                cost = np.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6) # replace invalid entries with large valid value
                # assert cost.shape == (len(gt_lanes), len(pred_lanes))
                # assert np.isfinite(cost).all()
                row_ind, col_ind = linear_sum_assignment(cost)

                matched_gt = set()
                matched_pr = set()

                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] < self.dist_thresh:
                        all_tp += 1
                        all_errors.append(cost[r, c])
                        matched_gt.add(r)
                        matched_pr.add(c)

                all_fp += len(pred_lanes) - len(matched_pr)
                all_fn += len(gt_lanes) - len(matched_gt)

            precision = all_tp / max(all_tp + all_fp, 1)
            recall = all_tp / max(all_tp + all_fn, 1)
            f1 = 2 * all_tp / max(2 * all_tp + all_fp + all_fn, 1)

            return {
                'TP': all_tp,
                'FP': all_fp,
                'FN': all_fn,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'Mean_Lateral_Error': float(np.mean(all_errors)) if all_errors else 0.0
            }
            
        elif self.mode=='3D':
            all_tp, all_fp, all_fn = 0, 0, 0
            all_match_costs = []
            all_x_err, all_y_err = [], []

            z_samples = np.linspace(5, 80, 38)  # same as literature

            for res in results:
                gt_lanes = res['gt_lanes']
                pred_lanes = res['pred_lanes']

                if len(gt_lanes) == 0 and len(pred_lanes) == 0:
                    continue

                if len(gt_lanes) == 0:
                    all_fp += len(pred_lanes)
                    continue

                if len(pred_lanes) == 0:
                    all_fn += len(gt_lanes)
                    continue

                cost = build_cost_matrix_for_3d(gt_lanes, pred_lanes, z_samples)
                cost = np.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)

                row_ind, col_ind = linear_sum_assignment(cost)

                matched_gt = set()
                matched_pr = set()

                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] < self.dist_thresh:
                        all_tp += 1
                        all_match_costs.append(cost[r, c])

                        # compute per-point x/y errors at fixed z
                        errs = compute_lane_errors(
                            gt_lanes[r],
                            pred_lanes[c],
                            z_samples
                        )

                        if errs is not None:
                            x_err, y_err = errs
                            all_x_err.extend(x_err)
                            all_y_err.extend(y_err)

                        matched_gt.add(r)
                        matched_pr.add(c)

                all_fp += len(pred_lanes) - len(matched_pr)
                all_fn += len(gt_lanes) - len(matched_gt)

            precision = all_tp / max(all_tp + all_fp, 1)
            recall = all_tp / max(all_tp + all_fn, 1)
            f1 = 2 * all_tp / max(2 * all_tp + all_fp + all_fn, 1)

            return {
                'TP': all_tp,
                'FP': all_fp,
                'FN': all_fn,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'Mean_Lateral_Error': float(np.mean(all_match_costs)) if all_match_costs else 0.0,
                'Mean_X_Error': float(np.mean(all_x_err)) if all_x_err else 0.0,
                'Mean_Y_Error': float(np.mean(all_y_err)) if all_y_err else 0.0
            }


def build_cost_matrix(gt_lanes, pred_lanes, z_samples):
    M, N = len(gt_lanes), len(pred_lanes)
    cost = np.zeros((M, N), dtype=float)

    for i, gt in enumerate(gt_lanes):          # rows = GT
        for j, pr in enumerate(pred_lanes):    # cols = predictions
            cost[i, j] = compute_lane_dist(gt, pr, z_samples)

    return cost

def build_cost_matrix_for_3d(gt_lanes, pred_lanes, z_samples):
    M, N = len(gt_lanes), len(pred_lanes)
    cost = np.zeros((M, N), dtype=float)

    for i, gt in enumerate(gt_lanes):          # rows = GT
        for j, pr in enumerate(pred_lanes):    # cols = predictions
            cost[i, j] = compute_lane_dist_for_3d(gt, pr, z_samples)

    return cost

def compute_lane_dist(pred_lane, gt_lane, z_samples):
    """
    gt_lane, pred_lane: (N, 2) [x, z]
    """
    x_gt, mask_gt = sample_lane_xz(gt_lane, z_samples)
    x_pr, mask_pr = sample_lane_xz(pred_lane, z_samples)

    valid = mask_gt & mask_pr
    if valid.sum() < 3:  # lane sampled at 2m. < 5 means if given gt and pred lanes overlap for 2x5=10m- then match, otherwise skip
        return np.inf

    return np.mean(np.abs(x_gt[valid] - x_pr[valid]))

def compute_lane_dist_for_3d(gt_lane, pred_lane, z_samples):
    """
    gt_lane, pred_lane: (N, 3) [x, y, z]
    """
    x_gt, y_gt, mask_gt = sample_lane_xyz(gt_lane, z_samples)
    x_pr, y_pr, mask_pr = sample_lane_xyz(pred_lane, z_samples)

    valid = mask_gt & mask_pr
    if valid.sum() < 3:  # lane sampled at 2m. < 5 means if given gt and pred lanes overlap for 2x5=10m- then match, otherwise skip
        return np.inf

    return np.mean(np.abs(x_gt[valid] - x_pr[valid]))

def compute_lane_errors(gt_lane, pred_lane, z_samples):
    x_gt, y_gt, m_gt = sample_lane_xyz(gt_lane, z_samples)
    x_pr, y_pr, m_pr = sample_lane_xyz(pred_lane, z_samples)

    valid = m_gt & m_pr
    if valid.sum() < 3:
        return None

    # print(f"y_gt[valid]: {y_gt[valid]}")
    # print(f"y_pr[valid]: {y_pr[valid]}")
    # print(f"="*64)

    x_err = np.abs(x_pr[valid] - x_gt[valid])  # lateral error
    y_err = np.abs(y_pr[valid] - y_gt[valid])/100.0  # height error, converted from cms to ms

    return x_err, y_err

def sample_lane_xz(lane, z_samples):
    """
    lane: (N, 2) array [x, z]
    z_samples: (K,) array

    returns:
        sampled_x: (K,) array
        valid_mask: (K,) boolean
    """
    x = lane[:, 0]
    z = lane[:, 1]

    # ensure sorted by z
    order = np.argsort(z)
    z = z[order]
    x = x[order]

    z_min, z_max = z[0], z[-1]
    valid_mask = (z_samples >= z_min) & (z_samples <= z_max)

    sampled_x = np.full_like(z_samples, fill_value=np.nan, dtype=np.float32)
    sampled_x[valid_mask] = np.interp(z_samples[valid_mask], z, x)

    return sampled_x, valid_mask

def sample_lane_xyz(lane, z_samples):
    """
    lane: (N, 3) [x, y, z] in vertical coords
    """
    x, y, z = lane[:, 0], lane[:, 1], lane[:, 2]

    order = np.argsort(z)
    z, x, y = z[order], x[order], y[order]

    z_min, z_max = z[0], z[-1]
    valid_mask = (z_samples >= z_min) & (z_samples <= z_max)

    xs = np.full_like(z_samples, np.nan, dtype=np.float32)
    ys = np.full_like(z_samples, np.nan, dtype=np.float32)

    xs[valid_mask] = np.interp(z_samples[valid_mask], z, x)
    ys[valid_mask] = np.interp(z_samples[valid_mask], z, y)

    return xs, ys, valid_mask

def to_json_safe(lanes):
    return [lane.tolist() for lane in lanes]