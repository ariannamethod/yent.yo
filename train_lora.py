#!/usr/bin/env python3
"""Train LoRA style adapter for BK-SDM-Tiny.

Each style = one LoRA file (~5-10MB). Applied at runtime on base model.

Usage:
  python3 train_lora.py <style_name> <image_dir> <model_dir> [output_dir] [steps]

Example:
  python3 train_lora.py caricature data/caricature bk-sdm-tiny-hf lora/ 1000
"""

import sys
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model


class StyleDataset(Dataset):
    """Simple image dataset with fixed caption per style."""

    CAPTIONS = {
        "caricature": [
            "a caricature drawing",
            "a political caricature in the style of Daumier",
            "a satirical illustration with exaggerated features",
            "a pen and ink caricature",
            "a humorous caricature portrait",
        ],
        "graffiti": [
            "a graffiti mural on a wall",
            "street art graffiti",
            "colorful spray paint graffiti",
            "urban graffiti artwork",
            "a graffiti tag on concrete",
        ],
        "propaganda": [
            "a Soviet propaganda poster",
            "a constructivist propaganda poster",
            "a bold propaganda illustration",
            "a revolutionary propaganda poster in red and black",
            "a workers propaganda poster",
        ],
        "pixel": [
            "pixel art",
            "8-bit pixel art sprite",
            "retro pixel art scene",
            "a pixel art character",
            "16-bit pixel art landscape",
        ],
    }

    def __init__(self, image_dir, style_name, size=512):
        self.images = sorted([
            p for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ])
        if not self.images:
            raise ValueError(f"No images found in {image_dir}")

        self.captions = self.CAPTIONS.get(style_name, [f"art in {style_name} style"])
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # → [-1, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.transform(img)
        caption = self.captions[idx % len(self.captions)]
        return img, caption


def train_lora(style_name, image_dir, model_dir, output_dir="lora", num_steps=1000,
               lr=1e-4, batch_size=1, lora_rank=8, gradient_accumulation=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"=== LoRA Training: {style_name} ===")
    print(f"Images: {image_dir}")
    print(f"Model: {model_dir}")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Steps: {num_steps}, LR: {lr}, Rank: {lora_rank}")
    print()

    # Load models
    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_dir, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        model_dir, subfolder="vae", torch_dtype=dtype
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_dir, subfolder="unet", torch_dtype=dtype
    ).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_dir, subfolder="scheduler")

    # Freeze everything except UNet LoRA
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Add LoRA to UNet
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,  # alpha = rank → scaling = 1.0
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                        "proj_in", "proj_out",
                        "ff.net.0.proj", "ff.net.2"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Dataset
    dataset = StyleDataset(image_dir, style_name, size=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"Dataset: {len(dataset)} images")

    # Optimizer — only LoRA params
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)

    # Cosine schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    # Training loop
    print(f"\nTraining {num_steps} steps...")
    unet.train()
    global_step = 0
    running_loss = 0.0

    while global_step < num_steps:
        for images, captions in dataloader:
            if global_step >= num_steps:
                break

            images = images.to(device, dtype=dtype)

            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215

            # Encode text
            tokens = tokenizer(
                list(captions), padding="max_length", max_length=77,
                truncation=True, return_tensors="pt"
            ).input_ids.to(device)
            with torch.no_grad():
                encoder_hidden_states = text_encoder(tokens).last_hidden_state

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (latents.shape[0],), device=device).long()

            # Add noise
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Loss
            loss = F.mse_loss(noise_pred.float(), noise.float())
            loss = loss / gradient_accumulation
            loss.backward()

            if (global_step + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item() * gradient_accumulation
            global_step += 1

            if global_step % 50 == 0 or global_step == 1:
                avg_loss = running_loss / min(global_step, 50)
                running_loss = 0.0
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Step {global_step}/{num_steps}: loss={avg_loss:.4f}, lr={current_lr:.2e}")

    # Save LoRA weights
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{style_name}.safetensors")

    # Extract just the LoRA state dict
    lora_state_dict = {}
    for name, param in unet.named_parameters():
        if param.requires_grad:
            # Clean up PEFT naming
            clean_name = name.replace("base_model.model.", "").replace(".default", "")
            lora_state_dict[clean_name] = param.detach().cpu()

    from safetensors.torch import save_file
    save_file(lora_state_dict, out_path)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nSaved: {out_path} ({size_mb:.1f} MB)")
    print(f"Keys: {len(lora_state_dict)}")
    print(f"Done!")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 train_lora.py <style> <image_dir> <model_dir> [output_dir] [steps]")
        sys.exit(1)

    style = sys.argv[1]
    img_dir = sys.argv[2]
    model_dir = sys.argv[3]
    out_dir = sys.argv[4] if len(sys.argv) > 4 else "lora"
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 1000

    train_lora(style, img_dir, model_dir, out_dir, steps)
