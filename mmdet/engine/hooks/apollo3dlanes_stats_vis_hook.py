import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS

@HOOKS.register_module()
class SaveVisAndStats(Hook):
    def __init__(self,
                stat_out_dir,
                vis_out_dir,
                bin_out_dir,
                num_samples,
                roi_x,
                roi_z):
        os.makedirs(stat_out_dir, exist_ok=True)
        os.makedirs(vis_out_dir, exist_ok=True)
        os.makedirs(bin_out_dir, exist_ok=True)
        self.num_samples = num_samples
        self.roi_x = roi_x
        self.roi_z = roi_z
        self.stats_file = os.path.join(stat_out_dir, 'epoch_stats')
        self.vis_out_dir = vis_out_dir
        self.bin_out_dir = bin_out_dir

    def before_train_epoch(self, runner):
            epoch = runner.epoch
            losses = runner.message_hub.get_info('loss')

            stats = {
                'epoch': epoch,
                'losses': {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()} \
                            if losses is not None else "No Losses Computed"            }
            with open(self.stats_file, 'a') as f:
                json.dump(stats, f)
                f.write('\n')

            test_dataloader = runner.test_dataloader 
            data_batch = next(iter(test_dataloader))

            print(f"type(data_batch): {type(data_batch)}")
            print(f"data_batch.keys(): {data_batch.keys()}")

            # set model to eval mode temporarily
            model = runner.model
            was_training = model.training
            model.eval()

            device = next(model.parameters()).device
            for key in data_batch['inputs']:
                data_batch['inputs'][key] = data_batch['inputs'][key].to(device)
                
            with torch.no_grad():
                data_samples = model.predict(data_batch['inputs'], data_batch['data_samples'])

            # restore training mode
            model.train(mode=was_training)

            for i in range(min(self.num_samples, len(data_samples))):
                sample = data_samples[i]
                img_name = os.path.basename(sample.metainfo['img_path']).split('.')[0]
                ele_pred = sample.pred_3dlanes['ele_pred']
                vis_path = os.path.join(self.vis_out_dir, f'epoch_{epoch}_{img_name}.png')
                self._save_vis(ele_pred, vis_path)

                bin_prob = sample.pred_3dlanes.get('bin_prob', None)
                if bin_prob is not None:
                    bin_vis_path = os.path.join(self.bin_out_dir, f'epoch_{epoch}_{img_name}_bin.png')
                    plt.imsave(bin_vis_path, bin_prob.cpu().numpy(), cmap='gray')

            runner.logger.info(f"Saved visuals and stats for epoch {epoch}")

    def _save_vis(self, ele_pred: np.ndarray, out_path: str):
        h, w = ele_pred.shape
        Z = np.linspace(self.roi_z[0], self.roi_z[1], h)#[-1] to correct the false vert flip in ele mask?
        X = np.linspace(self.roi_x[0], self.roi_x[1], w)#[-1] to correct the false vert flip in ele mask?
        
        # to avoid colorbar skew from zeros
        valid_ele = ele_pred[ele_pred != 0]
        if valid_ele.numel() > 0:
            vmin, vmax = valid_ele.min(), valid_ele.max()
        else:
            vmin, vmax = -1.0, 1.0
        
        # ele_pred = np.flipud(ele_pred).cpu().numpy() to correct the flase vert flip in ele mask?
        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        im = ax.pcolormesh(
            X, Z, ele_pred.cpu().numpy(),
            cmap='jet',
            vmin=vmin,
            vmax=vmax,
            shading='auto'
        )
        ax.set_aspect('equal')
        ax.set_title('Elevation Prediction')
        ax.set_xlabel('x (right)')
        ax.set_ylabel('z (forward)')
        fig.colorbar(im, ax=ax, label='Height (cm)')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
