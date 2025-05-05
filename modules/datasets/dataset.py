import os
import numpy as np
import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, high_res_dir, low_res_dir):

        self.high_res_dir = high_res_dir    # Folder of high_res_images
        self.low_res_dir = low_res_dir      # Folder of low_res_images

        self.high_res_image_list = [f for f in os.listdir(high_res_dir) if f.endswith(".npy")]   # List of high_res_images

    def __len__(self):
        return len(self.high_res_image_list) * 50 # There exist 50 low_res_images for every high_res_image. TODO: This only works for a hardcoded fix number. Needs Generalization

    def __getitem__(self, i):
        high_res_image_i = i // 50   # Index for finding the right high_res_image 
        low_res_image_i = i % 50    # Index for the low_res_image of the high_res_image

        high_res_path = os.path.join(self.high_res_dir, self.high_res_image_list[high_res_image_i])
        low_res_path = os.path.join(self.low_res_dir, self.high_res_image_list[high_res_image_i].replace(".npy", "_gen.npy")) # low_res_images have a _gen at the end of the name. TODO: This only works for this name. Needs Generalization

        high_res_image_np = np.load(high_res_path)

        low_res_image_stack_np = np.load(low_res_path)  # The low_res_images are in a stack. Thats why you have to pick one. [x, y, image_index]
        low_res_image_np = low_res_image_stack_np[:, :, low_res_image_i]

        return (
            torch.from_numpy(low_res_image_np).float().unsqueeze(0),
            torch.from_numpy(high_res_image_np).float().unsqueeze(0),
        ) # Ready to use for PyTorch!
