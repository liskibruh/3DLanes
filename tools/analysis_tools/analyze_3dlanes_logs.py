import re
import matplotlib.pyplot as plt
from pathlib import Path

def moving_average(x, window=50):
    import numpy as np
    x = np.array(x)
    return np.convolve(x, np.ones(window)/window, mode="valid")

# ==========================
# CONFIG
# ==========================
LOG_FILE = "/data24t_1/owais.tahir/3DLanes/mmdetection/mmdet/work_dir/20260103_080819/20260103_080819.log"          # path to your .log file
OUTPUT_FIG = "../loss_curves.png"  # output image
NUM_SAMPLES = 6373              # training samples per epoch

# ==========================
# REGEX PATTERN
# ==========================
pattern = re.compile(
    r"Epoch\(train\)\s+\[(\d+)\]\[\s*(\d+)/\d+\].*?"
    r"loss:\s*([0-9.]+)\s+"
    r"ele_loss:\s*([0-9.]+)\s+"
    r"bin_loss:\s*([0-9.]+)"
)

# ==========================
# PARSE LOG FILE
# ==========================
epochs = []
iters = []
losses = []
ele_losses = []
bin_losses = []

with open(LOG_FILE, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            iteration = int(match.group(2))
            loss = float(match.group(3))
            ele_loss = float(match.group(4))
            bin_loss = float(match.group(5))

            # Convert iteration → fractional epoch
            epoch_progress = epoch + iteration / NUM_SAMPLES

            epochs.append(epoch_progress)
            iters.append(iteration)
            losses.append(loss)
            ele_losses.append(ele_loss)
            bin_losses.append(bin_loss)

print(f"Parsed {len(losses)} log entries")

window = 100 

epochs_s = moving_average(epochs, window)
losses_s = moving_average(losses, window)
ele_losses_s = moving_average(ele_losses, window)
bin_losses_s = moving_average(bin_losses, window)

# ==========================
# PLOT
# ==========================
plt.figure(figsize=(10, 6))
plt.plot(epochs_s, losses_s, label="Total Loss (MA)")
plt.plot(epochs_s, ele_losses_s, label="Elevation Loss (MA)")
plt.plot(epochs_s, bin_losses_s, label="Binary Loss (MA)")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curves")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
plt.close()

print(f"Saved loss curves to {OUTPUT_FIG}")