# dataset.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import tifffile as tiff

class CTDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, patch_size=96, scale=2, use_multislice=False):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.scale = scale
        self.use_multislice = use_multislice

        self.files = sorted(os.listdir(lr_dir))

    def __len__(self):
        return len(self.files)

    def load_slice(self, idx, directory):
        idx = max(0, min(idx, len(self.files) - 1))  # clamp
        path = os.path.join(directory, self.files[idx])
        img = tiff.imread(path).astype(np.float32)
        return img / 65535.0

    def __getitem__(self, idx):

        if self.use_multislice:
            # Load 3 consecutive LR slices
            lr_prev = self.load_slice(idx - 1, self.lr_dir)
            lr_curr = self.load_slice(idx, self.lr_dir)
            lr_next = self.load_slice(idx + 1, self.lr_dir)

            lr = np.stack([lr_prev, lr_curr, lr_next], axis=0)  # (3, H, W)
        else:
            lr = self.load_slice(idx, self.lr_dir)[None, :, :]  # (1, H, W)

        hr = self.load_slice(idx, self.hr_dir)

        # Crop (same coordinates for all channels)
        _, h, w = lr.shape
        ps = self.patch_size
        x = np.random.randint(0, w - ps)
        y = np.random.randint(0, h - ps)

        lr_patch = lr[:, y:y+ps, x:x+ps]
        hr_patch = hr[
            y*self.scale:(y+ps)*self.scale,
            x*self.scale:(x+ps)*self.scale
        ]

        # To tensor
        lr_patch = torch.from_numpy(lr_patch)
        hr_patch = torch.from_numpy(hr_patch).unsqueeze(0)

        return lr_patch, hr_patch