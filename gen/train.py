import os, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data"
PATCH_SIZE = 384
EPOCHS = 200
BATCH_SIZE = 1
DEVICE = "cuda"
ZC = 8

# -----------------------------
# Dataset (grayscale, no resize)
# -----------------------------
class ImageFolder(Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)]
        self.t = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L")
        return self.t(img) * 2 - 1  # [-1,1]

def crop_patch(img, size):
    _, H, W = img.shape
    y = random.randint(0, H - size)
    x = random.randint(0, W - size)
    return img[:, y:y+size, x:x+size]

# -----------------------------
# Generator
# -----------------------------
class Generator(nn.Module):
    def __init__(self, zc=8, base=128):
        super().__init__()
        layers = [
            nn.Conv2d(zc, base, 3, padding=1),
            nn.LeakyReLU(0.2)
        ]
        for _ in range(10):
            layers += [
                nn.Conv2d(base, base, 3, padding=1),
                nn.LeakyReLU(0.2)
            ]
        layers += [
            nn.Conv2d(base, 1, 1),
            nn.Tanh()
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

# -----------------------------
# PatchGAN Discriminator (with spectral norm)
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        layers = []
        c = 1
        for i in range(6):
            layers += [
                nn.utils.spectral_norm(nn.Conv2d(c, base*(2**i), 4, 2, 1)),
                nn.LeakyReLU(0.2)
            ]
            c = base*(2**i)
        layers.append(nn.utils.spectral_norm(nn.Conv2d(c, 1, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Losses
# -----------------------------
def d_loss(real, fake):
    return torch.mean(torch.relu(1 - real)) + torch.mean(torch.relu(1 + fake))

def g_loss(fake):
    return -torch.mean(fake)

def r1_penalty(d_out, x):
    grad = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=x,
        create_graph=True
    )[0]
    return grad.pow(2).reshape(grad.size(0), -1).sum(1).mean()

def add_noise(x, sigma=0.03):
    return x + sigma * torch.randn_like(x)

# -----------------------------
# Setup
# -----------------------------
data = DataLoader(ImageFolder(DATA_DIR), batch_size=1, shuffle=True)
G = Generator(ZC).to(DEVICE)
D = Discriminator().to(DEVICE)

optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.0, 0.99))
optD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.99))

os.makedirs("samples", exist_ok=True)

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(EPOCHS):
    for real in tqdm(data, desc=f"Epoch {epoch}"):
        real = real.to(DEVICE)

        real_patch = crop_patch(real[0], PATCH_SIZE).unsqueeze(0)
        real_patch.requires_grad_(True)

        z = torch.randn(1, ZC, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
        fake = G(z)

        # ---- Train D ----
        r = D(add_noise(real_patch))
        f = D(add_noise(fake.detach()))

        lossD = d_loss(r, f)
        r1 = r1_penalty(r, real_patch)
        lossD = lossD + 5.0 * r1

        optD.zero_grad()
        lossD.backward()
        optD.step()

        # ---- Train G ----
        f = D(fake)
        lossG = g_loss(f)

        optG.zero_grad()
        lossG.backward()
        optG.step()

    print(f"Epoch {epoch} | D: {lossD.item():.3f} | G: {lossG.item():.3f}")

    # Save sample
    if epoch % 10 == 0:
        img = (fake.clamp(-1,1) + 1)/2
        transforms.ToPILImage(mode="L")(img[0].cpu()).save(f"samples/sample_{epoch}.png")
