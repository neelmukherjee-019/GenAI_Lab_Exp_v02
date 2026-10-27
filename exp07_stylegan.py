"""Experiment 7: Compact StyleGAN-inspired face generator for a lab demonstration.

This is an educational style-based GAN with mapping, layer-wise style modulation,
style mixing, and per-layer noise injection. It is not NVIDIA's canonical StyleGAN2-ADA.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any


def _style_models(z_dim: int = 128, w_dim: int = 128):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MappingNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            for i in range(4):
                layers += [nn.Linear(z_dim if i == 0 else w_dim, w_dim), nn.LeakyReLU(0.2)]
            self.net = nn.Sequential(*layers)
        def forward(self, z):
            z = z / (z.norm(dim=1, keepdim=True) + 1e-8) * (z.shape[1] ** 0.5)
            return self.net(z)

    class StyledConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.norm = nn.InstanceNorm2d(out_channels, affine=False)
            self.affine = nn.Linear(w_dim, out_channels * 2)
            self.noise_strength = nn.Parameter(torch.full((1, out_channels, 1, 1), 0.1))
            self.activation = nn.LeakyReLU(0.2)
        def forward(self, x, w, noise_mode="random"):
            x = self.conv(x)
            if noise_mode == "random":
                noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
                x = x + self.noise_strength * noise
            elif noise_mode != "none":
                raise ValueError("noise_mode must be 'random' or 'none'")
            scale, bias = self.affine(w).chunk(2, dim=1)
            x = self.norm(x) * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
            return self.activation(x)

    class StyleGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.z_dim = z_dim
            self.mapping = MappingNetwork()
            self.constant = nn.Parameter(torch.randn(1, 256, 4, 4))
            pairs = [(256, 256), (256, 128), (128, 128), (128, 64), (64, 32)]
            self.blocks = nn.ModuleList([StyledConv(a, b) for a, b in pairs])
            self.to_rgb = nn.Conv2d(32, 3, 1)
        def forward(self, z, mixing_z=None, mix_after=2, noise_mode="random"):
            w1 = self.mapping(z)
            styles = [w1] * len(self.blocks)
            if mixing_z is not None:
                w2 = self.mapping(mixing_z)
                styles = [w1 if i < int(mix_after) else w2 for i in range(len(self.blocks))]
            x = self.constant.expand(z.shape[0], -1, -1, -1)
            for i, block in enumerate(self.blocks):
                if i > 0:
                    x = F.interpolate(x, scale_factor=2, mode="nearest")
                x = block(x, styles[i], noise_mode=noise_mode)
            return torch.tanh(self.to_rgb(x))

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1, 4, 1, 0),
            )
        def forward(self, x):
            return self.net(x).view(-1)

    return StyleGenerator(), Discriminator()


def train_stylegan(data_dir: str | Path, epochs: int = 1, batch_size: int = 16, max_batches: int = 40, output_dir: str | Path = "outputs/exp07", latent_dim: int = 128) -> dict[str, Any]:
    """Train the compact StyleGAN-inspired educational network on an image folder."""
    import torch
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.utils import save_image
    from .exp06_dcgan import _build_loader

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, image_count = _build_loader(data_dir, 64, batch_size)
    generator, discriminator = _style_models(latent_dim, latent_dim)
    generator, discriminator = generator.to(device), discriminator.to(device)

    def init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    fixed_z = torch.randn(25, latent_dim, device=device)
    gen_losses, disc_losses = [], []

    for epoch in range(int(epochs)):
        generator.train(); discriminator.train()
        for batch_idx, real in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            real = real.to(device)
            n = real.shape[0]
            real_targets = torch.ones(n, device=device)
            fake_targets = torch.zeros(n, device=device)

            opt_d.zero_grad(set_to_none=True)
            real_loss = criterion(discriminator(real), real_targets)
            z = torch.randn(n, latent_dim, device=device)
            fake = generator(z)
            fake_loss = criterion(discriminator(fake.detach()), fake_targets)
            d_loss = real_loss + fake_loss
            d_loss.backward(); opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            z = torch.randn(n, latent_dim, device=device)
            fake = generator(z)
            g_loss = criterion(discriminator(fake), real_targets)
            g_loss.backward(); opt_g.step()
            gen_losses.append(float(g_loss.item()))
            disc_losses.append(float(d_loss.item()))

        generator.eval()
        with torch.no_grad():
            samples = generator(fixed_z, noise_mode="random").cpu()
        save_image((samples * 0.5 + 0.5).clamp(0, 1), out / f"stylegan_epoch_{epoch+1:02d}.png", nrow=5)
        torch.save(generator.state_dict(), out / "style_generator.pth")
        torch.save(discriminator.state_dict(), out / "style_discriminator.pth")

    generator.eval()
    with torch.no_grad():
        z1 = torch.randn(4, latent_dim, device=device)
        z2 = torch.randn(4, latent_dim, device=device)
        normal = generator(z1, noise_mode="random").cpu()
        mixed = generator(z1, mixing_z=z2, mix_after=2, noise_mode="random").cpu()
        no_noise = generator(z1, noise_mode="none").cpu()
    demo_path = out / "style_mixing_and_noise.png"
    # 3 rows: normal generation, style mixing, noise disabled (stochastic detail comparison).
    from torchvision.utils import make_grid
    panels = torch.cat([normal, mixed, no_noise], dim=0)
    save_image((panels * 0.5 + 0.5).clamp(0, 1), demo_path, nrow=4)

    loss_path = out / "stylegan_losses.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gen_losses, label="Generator loss")
    ax.plot(disc_losses, label="Discriminator loss")
    ax.set_xlabel("Training batch")
    ax.set_ylabel("Adversarial loss")
    ax.set_title("StyleGAN-inspired training losses")
    ax.legend(); fig.tight_layout(); fig.savefig(loss_path, dpi=140); plt.close(fig)
    return {
        "device": str(device), "training_images": image_count,
        "generator_path": str(out / "style_generator.pth"),
        "discriminator_path": str(out / "style_discriminator.pth"),
        "generated_grid_path": str(out / f"stylegan_epoch_{int(epochs):02d}.png"),
        "style_mixing_noise_path": str(demo_path), "loss_plot_path": str(loss_path),
        "output_dir": str(out),
    }


if __name__ == "__main__":
    raise SystemExit("Use train_stylegan(data_dir=...); upload/extract a face-image folder first.")
