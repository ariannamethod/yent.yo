#!/usr/bin/env python3
"""
Vitriol — WGAN-GP Training
Trains a WGAN-GP on art, exports weights for Go/C inference.

Architecture: DCGAN generator + critic, Wasserstein loss + gradient penalty.
Stable training, no mode collapse.

Usage:
  python train.py --data data/graffiti_train    # train
  python train.py --export checkpoints/ckpt_0500.pt  # export for Go
"""

import argparse
import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image

# Architecture constants
NZ = 128       # latent vector size
NGF = 64       # generator feature maps
NDF = 64       # discriminator (critic) feature maps
NC = 3         # channels (RGB)
IMG_SIZE = 128  # output image size

# WGAN-GP hyperparameters
LAMBDA_GP = 10  # gradient penalty weight
N_CRITIC = 5   # critic steps per generator step

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """DCGAN Generator for 128x128. z (nz,1,1) → (3,128,128)"""
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(NZ, NGF * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 16, NGF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


class Critic(nn.Module):
    """
    WGAN-GP Critic for 128x128. No sigmoid, no BatchNorm (uses LayerNorm).
    (3,128,128) → scalar (Wasserstein distance)
    """
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 128→64
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64→32
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(NDF * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 32→16
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(NDF * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 16→8
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(NDF * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 8→4
            nn.Conv2d(NDF * 8, NDF * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(NDF * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 4→1
            nn.Conv2d(NDF * 16, 1, 4, 1, 0, bias=False),
            # No sigmoid! WGAN outputs raw score
        )

    def forward(self, x):
        return self.main(x).view(-1)


def gradient_penalty(critic, real, fake, device):
    """Compute gradient penalty for WGAN-GP."""
    bs = real.size(0)
    alpha = torch.rand(bs, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolated = critic(interpolated)
    gradients = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(bs, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def train(data_dir, epochs=500, batch_size=16, lr=0.0001, save_every=50,
          checkpoint=None, out_dir="checkpoints"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.1)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)
    print(f"Dataset: {len(dataset)} images, {len(dataloader)} batches")

    G = Generator().to(DEVICE)
    C = Critic().to(DEVICE)

    start_epoch = 0
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=DEVICE)
        G.load_state_dict(ckpt["G"])
        C.load_state_dict(ckpt["C"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {checkpoint} (epoch {start_epoch})")
    else:
        G.apply(weights_init)
        C.apply(weights_init)

    g_params = sum(p.numel() for p in G.parameters())
    c_params = sum(p.numel() for p in C.parameters())
    print(f"Generator: {g_params:,} params ({g_params*4/1e6:.1f}MB)")
    print(f"Critic:    {c_params:,} params ({c_params*4/1e6:.1f}MB)")

    # WGAN-GP uses Adam with specific betas
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_C = optim.Adam(C.parameters(), lr=lr, betas=(0.0, 0.9))

    fixed_noise = torch.randn(16, NZ, 1, 1, device=DEVICE)

    print(f"\nWGAN-GP training on {DEVICE} for {epochs} epochs")
    print(f"  lambda_gp={LAMBDA_GP}, n_critic={N_CRITIC}, lr={lr}")

    for epoch in range(start_epoch, epochs):
        g_loss_sum, c_loss_sum, gp_sum = 0.0, 0.0, 0.0
        g_steps = 0

        data_iter = iter(dataloader)
        n_batches = len(dataloader)
        batch_idx = 0

        while batch_idx < n_batches:
            # === Train Critic N_CRITIC times ===
            for _ in range(N_CRITIC):
                if batch_idx >= n_batches:
                    break
                try:
                    real, _ = next(data_iter)
                except StopIteration:
                    break
                batch_idx += 1

                real = real.to(DEVICE)
                bs = real.size(0)

                C.zero_grad()
                noise = torch.randn(bs, NZ, 1, 1, device=DEVICE)
                fake = G(noise).detach()

                c_real = C(real).mean()
                c_fake = C(fake).mean()
                gp = gradient_penalty(C, real, fake, DEVICE)

                c_loss = c_fake - c_real + LAMBDA_GP * gp
                c_loss.backward()
                opt_C.step()

                c_loss_sum += c_loss.item()
                gp_sum += gp.item()

            # === Train Generator ===
            G.zero_grad()
            noise = torch.randn(batch_size, NZ, 1, 1, device=DEVICE)
            fake = G(noise)
            g_loss = -C(fake).mean()
            g_loss.backward()
            opt_G.step()

            g_loss_sum += g_loss.item()
            g_steps += 1

        # Wasserstein distance ≈ -(c_fake - c_real)
        n = max(batch_idx, 1)
        w_dist = -c_loss_sum / n + gp_sum / n * LAMBDA_GP  # approximate
        print(f"  epoch {epoch+1:4d} | G {g_loss_sum/max(g_steps,1):.4f} | "
              f"C {c_loss_sum/n:.4f} | GP {gp_sum/n:.4f} | W_dist {(c_loss_sum/n):.4f}")

        if (epoch + 1) % save_every == 0 or epoch == 0:
            with torch.no_grad():
                fake_samples = G(fixed_noise)
            save_image(fake_samples, samples_dir / f"epoch_{epoch+1:04d}.png",
                       nrow=4, normalize=True)
            torch.save({
                "epoch": epoch + 1,
                "G": G.state_dict(),
                "C": C.state_dict(),
            }, out_dir / f"ckpt_{epoch+1:04d}.pt")

    print(f"\nTraining complete. Final checkpoint: {out_dir}/ckpt_{epochs:04d}.pt")
    return G


def export_weights(checkpoint_path, out_dir="weights"):
    """Export generator weights to raw binary format for Go/C inference."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["G"] if "G" in ckpt else ckpt

    out_path = out_dir / "vitriol_gen.bin"
    with open(out_path, "wb") as f:
        f.write(b"VTRL")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<IIII", NZ, NGF, NC, IMG_SIZE))

        conv_layers = []
        keys = sorted(state.keys(), key=lambda k: (int(k.split('.')[1]), k.split('.')[2]))

        for key in keys:
            parts = key.split(".")
            idx = int(parts[1])
            param_name = parts[2]
            if param_name == "weight" and "num_batches_tracked" not in key:
                tensor = state[key]
                if len(tensor.shape) == 4:
                    conv_layers.append((idx, tensor))

        n_conv = len(conv_layers)
        n_bn = len([k for k in keys if "running_mean" in k])
        f.write(struct.pack("<I", n_conv + n_bn))

        for idx, weight in conv_layers:
            f.write(struct.pack("<I", 0))  # conv_transpose
            in_ch, out_ch, kH, kW = weight.shape
            f.write(struct.pack("<IIII", in_ch, out_ch, kH, kW))
            f.write(weight.float().numpy().tobytes())

            bn_idx = idx + 1
            bn_weight_key = f"main.{bn_idx}.weight"
            if bn_weight_key in state:
                f.write(struct.pack("<I", 1))  # batchnorm
                bn_w = state[f"main.{bn_idx}.weight"]
                bn_b = state[f"main.{bn_idx}.bias"]
                bn_m = state[f"main.{bn_idx}.running_mean"]
                bn_v = state[f"main.{bn_idx}.running_var"]
                num_features = bn_w.shape[0]
                f.write(struct.pack("<I", num_features))
                f.write(bn_w.float().numpy().tobytes())
                f.write(bn_b.float().numpy().tobytes())
                f.write(bn_m.float().numpy().tobytes())
                f.write(bn_v.float().numpy().tobytes())

    size_mb = out_path.stat().st_size / 1e6
    print(f"Exported generator to {out_path} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vitriol WGAN-GP trainer")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--export", type=str, default=None)
    args = parser.parse_args()

    if args.export:
        export_weights(args.export)
    else:
        train(args.data, epochs=args.epochs, batch_size=args.batch_size,
              lr=args.lr, save_every=args.save_every, checkpoint=args.checkpoint)
