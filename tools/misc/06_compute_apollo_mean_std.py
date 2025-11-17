import os
import numpy as np
import cv2
from tqdm import tqdm


def compute_mean_std(im_dir):
    """Compute mean and std of a dataset."""
    assert os.path.exists(im_dir), f"{im_dir} does not exist."

    img_files = []
    for root, dirs, files in os.walk(im_dir):
        for file in files:
            if file.endswith(".jpg") \
            or file.endswith(".png"):
                img_files.append(os.path.join(root, file))

    mean_rgb = np.zeros(3)
    std_rgb = np.zeros(3)
    num_images = 0
    for img_file in tqdm(img_files):
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        mean_rgb += np.mean(img, axis=(0, 1))
        std_rgb += np.std(img, axis=(0, 1))
        num_images += 1
    mean_rgb /= num_images
    std_rgb /= num_images
    print(f"Mean: {mean_rgb}")
    print(f"Std: {std_rgb}")

if __name__ == "__main__":
    im_dir = "/data24t_1/owais.tahir/3DLanes/mmdetection/data/Apollo_Sim_3D_Lane_Release/images/"
    compute_mean_std(im_dir)
