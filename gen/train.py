import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

# -----------------------
# Dataset (no resize, no aug)
# -----------------------
class ImageFolder(Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)]
        self.t = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L")  # <-- grayscale
        return self.t(img) * 2 - 1  # [-1,1], shape now [1, H, W]

# -----------------------
# Generator
# -----------------------
class Generator(nn.Module):
    def __init__(self, zc=8, base=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(zc, base, 3, padding=1),
            nn.LeakyReLU(0.2),

            *[nn.Sequential(
                nn.Conv2d(base, base, 3, padding=1),
                nn.LeakyReLU(0.2)
            ) for _ in range(8)],

            nn.Conv2d(base, 1, 1),  # <-- grayscale output
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# -----------------------
# PatchGAN Discriminator
# -----------------------
class Discriminator(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        layers = []
        c = 1  # <-- grayscale input
        for i in range(6):
            layers += [
                nn.Conv2d(c, base * (2**i), 4, 2, 1),
                nn.LeakyReLU(0.2)
            ]
            c = base * (2**i)
        layers.append(nn.Conv2d(c, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------
# Utilities
# -----------------------
def crop_patch(img, size=512):
    _, H, W = img.shape
    y = random.randint(0, H - size)
    x = random.randint(0, W - size)
    return img[:, y:y+size, x:x+size]

def d_loss(real, fake):
    return torch.mean(torch.relu(1 - real)) + torch.mean(torch.relu(1 + fake))

def g_loss(fake):
    return -torch.mean(fake)

# -----------------------
# Training
# -----------------------
device = "cuda"
data = DataLoader(ImageFolder("data"), batch_size=1, shuffle=True)

G = Generator().cuda()
D = Discriminator().cuda()

optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.0, 0.99))
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.0, 0.99))

for epoch in range(200):
    for real in tqdm(data):
        real = real.cuda()

        # sample patch (not augmentation, just locality)
        real_patch = crop_patch(real[0]).unsqueeze(0)

        # noise input
        z = torch.randn(1, 8, *real_patch.shape[2:]).cuda()
        fake = G(z)

        # --- Train D ---
        r = D(real_patch)
        f = D(fake.detach())
        lossD = d_loss(r, f)

        optD.zero_grad()
        lossD.backward()
        optD.step()

        # --- Train G ---
        f = D(fake)
        lossG = g_loss(f)

        optG.zero_grad()
        lossG.backward()
        optG.step()

    print(f"Epoch {epoch} | D: {lossD.item():.3f} | G: {lossG.item():.3f}")

    # save samples
    if epoch % 10 == 0:
        out = (fake.clamp(-1,1)+1)/2
        img = out[0].cpu()           # shape [1, H, W]
        img = (img + 1) / 2          # [0,1]
        transforms.ToPILImage(mode="L")(img).save(f"sample_{epoch}.png")
