# utils.py
import torch
import torch.nn.functional as F
import math
import torch.fft

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def ssim(pred, target, C1=0.01**2, C2=0.03**2):
    mu_x = pred.mean()
    mu_y = target.mean()

    sigma_x = pred.var()
    sigma_y = target.var()
    sigma_xy = ((pred - mu_x)*(target - mu_y)).mean()

    return ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2)
    )

def gradient_loss(pred, target):
    def gradient(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    dx_p, dy_p = gradient(pred)
    dx_t, dy_t = gradient(target)

    return torch.mean(torch.abs(dx_p - dx_t)) + torch.mean(torch.abs(dy_p - dy_t))


def fft_loss(pred, target):
    # Compute FFT magnitude
    pred_fft = torch.fft.fft2(pred, norm="ortho")
    targ_fft = torch.fft.fft2(target, norm="ortho")

    pred_mag = torch.abs(pred_fft)
    targ_mag = torch.abs(targ_fft)

    return torch.mean(torch.abs(pred_mag - targ_mag))