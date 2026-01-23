# model.py
import torch
import torch.nn as nn

class IMDB(nn.Module):
    def __init__(self, channels):
        super().__init__()

        distill = channels // 2
        remain = channels - distill

        self.c1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.c2 = nn.Conv2d(remain, channels, 3, 1, 1)
        self.c3 = nn.Conv2d(remain, channels, 3, 1, 1)
        self.c4 = nn.Conv2d(remain, distill, 3, 1, 1)

        self.fuse = nn.Conv2d(distill * 4, channels, 1, 1, 0)
        self.act = nn.LeakyReLU(0.05, inplace=True)

    def forward(self, x):
        out_c1 = self.act(self.c1(x))
        d1, r1 = torch.split(out_c1, [out_c1.shape[1]//2, out_c1.shape[1]//2], dim=1)

        out_c2 = self.act(self.c2(r1))
        d2, r2 = torch.split(out_c2, [out_c2.shape[1]//2, out_c2.shape[1]//2], dim=1)

        out_c3 = self.act(self.c3(r2))
        d3, r3 = torch.split(out_c3, [out_c3.shape[1]//2, out_c3.shape[1]//2], dim=1)

        d4 = self.c4(r3)

        out = torch.cat([d1, d2, d3, d4], dim=1)
        out = self.fuse(out)

        return out + x


class IMDN(nn.Module):
    def __init__(self, in_channels=1, feat=64, num_blocks=8, scale=2):
        super().__init__()

        self.head = nn.Conv2d(in_channels, feat, 3, 1, 1)

        self.body = nn.Sequential(*[IMDB(feat) for _ in range(num_blocks)])

        self.tail = nn.Sequential(
            nn.Conv2d(feat, feat * (scale**2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(feat, 1, 3, 1, 1)
        )

    def forward(self, x):
        f = self.head(x)
        f = self.body(f)
        out = self.tail(f)
        return out