#!/usr/bin/env python3
"""
Vitriol — DCGAN Training
Trains a DCGAN on caricature art, exports weights for Go/C inference.

Architecture: DCGAN 128x128
  Generator: z(128) → 6 TransposeConv2d layers → RGB 128x128
  Discriminator: RGB 128x128 → 6 Conv2d layers → real/fake

Usage:
  python train.py                    # train on data/all/
  python train.py --epochs 200       # more epochs
  python train.py --export weights/  # export only (from checkpoint)
"""

import os
import argparse
import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image

# Architecture constants
NZ = 128       # latent vector size
NGF = 64       # generator feature maps
NDF = 64       # discriminator feature maps
NC = 3         # channels (RGB)
IMG_SIZE = 128  # output image size

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    DCGAN Generator for 128x128.
    z (nz,1,1) → (3,128,128)

    Layer progression:
      4x4 → 8x8 → 16x16 → 32x32 → 64x64 → 128x128
    """
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # z → 4x4
            nn.ConvTranspose2d(NZ, NGF * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 16),
            nn.ReLU(True),
            # 4x4 → 8x8
            nn.ConvTranspose2d(NGF * 16, NGF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            # 8x8 → 16x16
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            # 16x16 → 32x32
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            # 32x32 → 64x64
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            # 64x64 → 128x128
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator for 128x128.
    (3,128,128) → scalar
    """
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 128x128 → 64x64
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64 → 32x32
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32 → 16x16
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 → 8x8
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 → 4x4
            nn.Conv2d(NDF * 8, NDF * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4 → 1x1
            nn.Conv2d(NDF * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)


def train(data_dir, epochs=100, batch_size=32, lr=0.0002, save_every=10,
          checkpoint=None, out_dir="checkpoints"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)

    # Dataset
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)
    print(f"Dataset: {len(dataset)} images, {len(dataloader)} batches")

    # Models
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    start_epoch = 0
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=DEVICE)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {checkpoint} (epoch {start_epoch})")
    else:
        G.apply(weights_init)
        D.apply(weights_init)

    # Count params
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"Generator: {g_params:,} params ({g_params*4/1e6:.1f}MB)")
    print(f"Discriminator: {d_params:,} params ({d_params*4/1e6:.1f}MB)")

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # Fixed noise for visualizing progress
    fixed_noise = torch.randn(16, NZ, 1, 1, device=DEVICE)

    print(f"\nTraining on {DEVICE} for {epochs} epochs...")

    for epoch in range(start_epoch, epochs):
        g_loss_sum, d_loss_sum = 0.0, 0.0

        for i, (real, _) in enumerate(dataloader):
            real = real.to(DEVICE)
            bs = real.size(0)
            real_label = torch.ones(bs, device=DEVICE)
            fake_label = torch.zeros(bs, device=DEVICE)

            # === Train Discriminator ===
            D.zero_grad()
            # Real
            output_real = D(real)
            loss_d_real = criterion(output_real, real_label)
            # Fake
            noise = torch.randn(bs, NZ, 1, 1, device=DEVICE)
            fake = G(noise)
            output_fake = D(fake.detach())
            loss_d_fake = criterion(output_fake, fake_label)
            # Backward
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_D.step()

            # === Train Generator ===
            G.zero_grad()
            output = D(fake)
            loss_g = criterion(output, real_label)
            loss_g.backward()
            opt_G.step()

            g_loss_sum += loss_g.item()
            d_loss_sum += loss_d.item()

        n = len(dataloader)
        print(f"  epoch {epoch+1:4d} | G loss {g_loss_sum/n:.4f} | D loss {d_loss_sum/n:.4f}")

        # Save samples
        if (epoch + 1) % save_every == 0 or epoch == 0:
            with torch.no_grad():
                fake_samples = G(fixed_noise)
            save_image(fake_samples, samples_dir / f"epoch_{epoch+1:04d}.png",
                       nrow=4, normalize=True)

            # Checkpoint
            torch.save({
                "epoch": epoch + 1,
                "G": G.state_dict(),
                "D": D.state_dict(),
            }, out_dir / f"ckpt_{epoch+1:04d}.pt")

    print(f"\nTraining complete. Final checkpoint: {out_dir}/ckpt_{epochs:04d}.pt")
    return G


def export_weights(checkpoint_path, out_dir="weights"):
    """
    Export generator weights to raw binary format for Go/C inference.

    Format: vitriol binary weights v1
      Header: magic(4) + nlayers(4)
      Per layer:
        type(4): 0=conv_transpose, 1=batchnorm
        For conv_transpose: out_ch(4) + in_ch(4) + kH(4) + kW(4) + weight(float32[])
        For batchnorm: num_features(4) + weight(f32[]) + bias(f32[]) + mean(f32[]) + var(f32[])
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["G"] if "G" in ckpt else ckpt

    out_path = out_dir / "vitriol_gen.bin"
    with open(out_path, "wb") as f:
        # Magic: "VTRL"
        f.write(b"VTRL")
        # Version
        f.write(struct.pack("<I", 1))
        # Architecture constants
        f.write(struct.pack("<IIII", NZ, NGF, NC, IMG_SIZE))

        # Export each layer in order
        # Generator structure: pairs of (ConvTranspose2d, BatchNorm2d) + final ConvTranspose2d
        layer_idx = 0
        keys = sorted(state.keys())

        conv_layers = []
        bn_layers = []

        for key in keys:
            parts = key.split(".")
            # main.0.weight, main.1.weight, main.1.bias, main.1.running_mean, ...
            idx = int(parts[1])
            param_name = parts[2]

            if param_name == "weight" and "num_batches_tracked" not in key:
                tensor = state[key]
                if len(tensor.shape) == 4:  # conv weight
                    conv_layers.append((idx, tensor))
                elif len(tensor.shape) == 1 and f"main.{idx}.bias" in state:
                    bn_layers.append(idx)

        # Write layer count (conv + bn pairs)
        n_conv = len(conv_layers)
        n_bn = len([k for k in keys if "running_mean" in k])
        f.write(struct.pack("<I", n_conv + n_bn))

        written = set()
        for idx, weight in conv_layers:
            # ConvTranspose2d weight: (in_ch, out_ch, kH, kW) in PyTorch
            f.write(struct.pack("<I", 0))  # type = conv_transpose
            in_ch, out_ch, kH, kW = weight.shape
            f.write(struct.pack("<IIII", in_ch, out_ch, kH, kW))
            f.write(weight.numpy().tobytes())
            written.add(idx)

            # Check if next layer is BatchNorm
            bn_idx = idx + 1
            bn_weight_key = f"main.{bn_idx}.weight"
            if bn_weight_key in state:
                f.write(struct.pack("<I", 1))  # type = batchnorm
                bn_w = state[f"main.{bn_idx}.weight"]
                bn_b = state[f"main.{bn_idx}.bias"]
                bn_m = state[f"main.{bn_idx}.running_mean"]
                bn_v = state[f"main.{bn_idx}.running_var"]
                num_features = bn_w.shape[0]
                f.write(struct.pack("<I", num_features))
                f.write(bn_w.numpy().tobytes())
                f.write(bn_b.numpy().tobytes())
                f.write(bn_m.numpy().tobytes())
                f.write(bn_v.numpy().tobytes())
                written.add(bn_idx)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Exported generator to {out_path} ({size_mb:.1f}MB)")
    print(f"  Latent dim: {NZ}, Features: {NGF}, Channels: {NC}, Size: {IMG_SIZE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vitriol DCGAN trainer")
    parser.add_argument("--data", type=str, default="data",
                        help="Dataset directory (should contain subdir with images)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--export", type=str, default=None,
                        help="Export weights from checkpoint (no training)")
    args = parser.parse_args()

    if args.export:
        export_weights(args.export)
    else:
        train(args.data, epochs=args.epochs, batch_size=args.batch_size,
              lr=args.lr, save_every=args.save_every, checkpoint=args.checkpoint)
