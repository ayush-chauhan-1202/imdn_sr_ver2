# config.py
import torch

class Config:
    # Paths
    train_lr_dir = "data/train/LR"
    train_hr_dir = "data/train/HR"
    val_lr_dir   = "data/val/LR"
    val_hr_dir   = "data/val/HR"

    # Training
    scale = 2
    patch_size = 96   # LR patch size
    batch_size = 8
    num_epochs = 10
    lr = 1e-4

    # Model
    in_channels = 3
    feat_channels = 64
    num_blocks = 8

    # Device
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    use_multislice = True   # set True later
    num_input_slices = 3     # 1 or 3

    # loss weights
    w_l1 = 1.0
    w_grad = 0.3
    w_fft = 0.05