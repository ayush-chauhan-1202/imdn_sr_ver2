# train.py
import torch
from torch.utils.data import DataLoader
from dataset import CTDataset
from model import IMDN
from config import Config
from utils import psnr, gradient_loss, fft_loss

def train():
    cfg = Config()

    dataset = CTDataset(cfg.train_lr_dir, cfg.train_hr_dir,
                    cfg.patch_size, cfg.scale,
                    use_multislice=cfg.use_multislice)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = IMDN(cfg.in_channels, cfg.feat_channels, cfg.num_blocks, cfg.scale)
    model.to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.L1Loss()

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0

        for lr, hr in loader:
            lr, hr = lr.to(cfg.device), hr.to(cfg.device)
            print(lr.shape, hr.shape)

            sr = model(lr)

            l1 = loss_fn(sr, hr)
            l_grad = gradient_loss(sr, hr)
            l_fft = fft_loss(sr, hr)

            loss = (
                cfg.w_l1 * l1 +
                cfg.w_grad * l_grad +
                cfg.w_fft * l_fft
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | "f"L1: {l1.item():.4f} | "f"Grad: {l_grad.item():.4f} | "f"FFT: {l_fft.item():.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"imdn_epoch_{epoch}.pth")


if __name__ == "__main__":
    train()