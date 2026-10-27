"""Experiment 6: PyTorch DCGAN trained on face images from a CelebA-style folder."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import os

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _build_loader(data_dir: str | Path, image_size: int, batch_size: int):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from torchvision import transforms

    class RecursiveImageDataset(Dataset):
        def __init__(self, root):
            root = Path(root)
            self.files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            if not self.files:
                raise ValueError(f"No JPG/PNG images were found under {root}.")
        def __len__(self):
            return len(self.files)
        def __getitem__(self, index):
            with Image.open(self.files[index]) as img:
                return self.transform(img.convert("RGB"))

    dataset = RecursiveImageDataset(data_dir)
    if len(dataset) < 2:
        raise ValueError("Please provide at least two face images for adversarial training.")
    # The Generator starts with a 1x1 feature map plus BatchNorm. Avoid a final
    # singleton batch when the dataset size is exactly one more than a multiple.
    remainder = len(dataset) % int(batch_size)
    drop_last = len(dataset) > int(batch_size) and remainder == 1
    return DataLoader(dataset, batch_size=min(int(batch_size), len(dataset)), shuffle=True, num_workers=0, drop_last=drop_last), len(dataset)


def _models(latent_dim: int = 100):
    import torch.nn as nn

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
                nn.BatchNorm2d(512), nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        def forward(self, z):
            return self.net(z)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1, 4, 1, 0, bias=False), nn.Sigmoid(),
            )
        def forward(self, image):
            return self.net(image).view(-1)

    return Generator(), Discriminator()


def train_dcgan(data_dir: str | Path, epochs: int = 1, batch_size: int = 32, max_batches: int = 50, output_dir: str | Path = "outputs/exp06", latent_dim: int = 100) -> dict[str, Any]:
    """Train a compact DCGAN. Use a directory (or extracted ZIP) containing face images.

    `max_batches` caps each epoch for a practical demo. Use None to process all batches.
    """
    import torch
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid, save_image

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, image_count = _build_loader(data_dir, 64, batch_size)
    netG, netD = _models(latent_dim)
    netG, netD = netG.to(device), netD.to(device)

    def weights_init(module):
        name = module.__class__.__name__
        if "Conv" in name:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif "BatchNorm" in name:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    fixed_noise = torch.randn(25, latent_dim, 1, 1, device=device)
    g_losses, d_losses = [], []

    for epoch in range(int(epochs)):
        for batch_idx, real in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            real = real.to(device)
            batch_size_now = real.size(0)
            real_label = torch.ones(batch_size_now, device=device)
            fake_label = torch.zeros(batch_size_now, device=device)

            netD.zero_grad(set_to_none=True)
            output_real = netD(real)
            loss_real = criterion(output_real, real_label)
            noise = torch.randn(batch_size_now, latent_dim, 1, 1, device=device)
            fake = netG(noise)
            output_fake = netD(fake.detach())
            loss_fake = criterion(output_fake, fake_label)
            lossD = loss_real + loss_fake
            lossD.backward()
            optimizerD.step()

            netG.zero_grad(set_to_none=True)
            output = netD(fake)
            lossG = criterion(output, real_label)
            lossG.backward()
            optimizerG.step()
            g_losses.append(float(lossG.item()))
            d_losses.append(float(lossD.item()))

        with torch.no_grad():
            sample = netG(fixed_noise).detach().cpu()
        save_image((sample * 0.5 + 0.5).clamp(0, 1), out / f"generated_epoch_{epoch+1:02d}.png", nrow=5)
        torch.save(netG.state_dict(), out / "generator.pth")
        torch.save(netD.state_dict(), out / "discriminator.pth")

    loss_path = out / "dcgan_losses.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(g_losses, label="Generator loss")
    ax.plot(d_losses, label="Discriminator loss")
    ax.set_xlabel("Training batch")
    ax.set_ylabel("BCE loss")
    ax.set_title("DCGAN training losses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_path, dpi=140)
    plt.close(fig)

    final_path = out / "generated_faces.png"
    with torch.no_grad():
        generated = netG(torch.randn(25, latent_dim, 1, 1, device=device)).detach().cpu()
    save_image((generated * 0.5 + 0.5).clamp(0, 1), final_path, nrow=5)
    return {
        "device": str(device), "training_images": image_count, "generator_path": str(out / "generator.pth"),
        "discriminator_path": str(out / "discriminator.pth"), "generated_grid_path": str(final_path),
        "loss_plot_path": str(loss_path), "generator_losses": g_losses, "discriminator_losses": d_losses,
        "output_dir": str(out),
    }


if __name__ == "__main__":
    raise SystemExit("Use train_dcgan(data_dir=...); upload/extract a CelebA face-image folder first.")
