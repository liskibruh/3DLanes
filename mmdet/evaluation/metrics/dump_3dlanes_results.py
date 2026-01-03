import numpy as np
from mmdet.registry import METRICS
import os
import pickle
import matplotlib.pyplot as plt
import torch

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
            with open(out_pkl, 'wb') as f:
                pickle.dump(ele_pred, f)

    def compute_metrics(self, results):
        return {}
