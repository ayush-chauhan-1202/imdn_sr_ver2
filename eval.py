# eval.py
import torch
from dataset import CTDataset
from model import IMDN
from config import Config
from utils import psnr, ssim

import os
import matplotlib.pyplot as plt
import numpy as np

def save_visual_comparison(lr, sr, hr, save_path, idx):
    """
    lr, sr, hr: tensors with shape [1, 1, H, W] in [0,1]
    """
    lr = lr.squeeze().cpu().numpy()
    sr = sr.squeeze().cpu().numpy()
    hr = hr.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(lr, cmap="gray")
    axes[0].set_title("LR")
    axes[0].axis("off")

    axes[1].imshow(sr, cmap="gray")
    axes[1].set_title("SR (IMDN)")
    axes[1].axis("off")

    axes[2].imshow(hr, cmap="gray")
    axes[2].set_title("HR (GT)")
    axes[2].axis("off")

    os.makedirs(save_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"comparison_{idx:03d}.png"), dpi=200)
    plt.close()

def evaluate(model_path):
    cfg = Config()

    dataset = CTDataset(
        cfg.val_lr_dir,
        cfg.val_hr_dir,
        cfg.patch_size,
        cfg.scale,
        cfg.use_multislice
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    model = IMDN(
        in_channels=3,
        feat=cfg.feat_channels,
        num_blocks=cfg.num_blocks,
        scale=cfg.scale
    )

    model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    model.to(cfg.device)
    model.eval()

    total_psnr, total_ssim = 0, 0

    with torch.no_grad():
        for i, (lr, hr) in enumerate(loader):
            lr, hr = lr.to(cfg.device), hr.to(cfg.device)

            sr = model(lr)

            total_psnr += psnr(sr, hr).item()
            total_ssim += ssim(sr, hr).item()

            if i < 10:
                save_visual_comparison(
                    lr[:, 1:2, :, :],  # center slice only
                    sr,
                    hr,
                    save_path="eval_outputs",
                    idx=i
                )

    print(f"Avg PSNR: {total_psnr / len(loader):.3f}")
    print(f"Avg SSIM: {total_ssim / len(loader):.4f}")
    print("Saved visual comparisons to ./eval_outputs/")


if __name__ == "__main__":
    evaluate("imdn_epoch_190.pth")