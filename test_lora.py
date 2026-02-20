#!/usr/bin/env python3
"""Test LoRA style: merge into base, export ONNX, generate image.

Usage:
  python3 test_lora.py <lora_file> <model_dir> <onnx_dir> <tokenizer_dir> "prompt" [output.png] [seed] [steps]
"""

import sys
import os
import time
import numpy as np
from PIL import Image


def merge_lora_and_generate(lora_path, model_dir, onnx_dir, tokenizer_dir, prompt,
                             output_path="lora_test.png", seed=42, num_steps=25):
    """Merge LoRA into base UNet, export to ONNX, generate image via ORT."""
    import torch
    from diffusers import UNet2DConditionModel
    from peft import PeftModel, LoraConfig
    from safetensors.torch import load_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # fp32 for ONNX export

    style_name = os.path.splitext(os.path.basename(lora_path))[0]
    print(f"=== Testing LoRA: {style_name} ===")
    print(f"LoRA: {lora_path}")
    print(f"Prompt: {prompt}")

    # Load base UNet
    print("Loading base UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        model_dir, subfolder="unet", torch_dtype=dtype
    ).to(device).eval()

    # Load LoRA weights manually and apply
    print("Applying LoRA...")
    lora_state = load_file(lora_path)

    # Group LoRA A and B matrices by layer
    # Keys: down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight
    # UNet params: down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight
    lora_pairs = {}
    for key, tensor in lora_state.items():
        base = key.rsplit(".lora_", 1)[0]  # everything before .lora_A or .lora_B
        ab = "A" if "lora_A" in key else "B"
        if base not in lora_pairs:
            lora_pairs[base] = {}
        lora_pairs[base][ab] = tensor.to(device, dtype=dtype)

    # Apply: W = W + B @ A (LoRA alpha=rank, so scaling=1.0)
    applied = 0
    for name, param in unet.named_parameters():
        if not name.endswith(".weight"):
            continue
        # UNet param name: down_blocks.0.attentions.0.proj_in.weight
        # LoRA key base:   down_blocks.0.attentions.0.proj_in
        base_name = name.rsplit(".weight", 1)[0]
        if base_name in lora_pairs and "A" in lora_pairs[base_name] and "B" in lora_pairs[base_name]:
            A = lora_pairs[base_name]["A"]
            B = lora_pairs[base_name]["B"]
            # Handle conv (4D: [out, rank, 1, 1]) vs linear (2D: [out, rank])
            if A.dim() == 4:
                # Conv1x1 LoRA: squeeze to 2D, matmul, unsqueeze back
                delta = (B.squeeze(-1).squeeze(-1) @ A.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
            else:
                delta = B @ A
            param.data += delta
            applied += 1

    print(f"  Applied {applied} LoRA layers")

    # Export merged UNet to ONNX
    merged_onnx = os.path.join(onnx_dir, f"unet_{style_name}.onnx")
    print(f"Exporting merged UNet to {merged_onnx}...")

    latent = torch.randn(1, 4, 64, 64, dtype=dtype).to(device)
    timestep = torch.tensor([800], dtype=torch.long).to(device)
    encoder_hidden_states = torch.randn(1, 77, 768, dtype=dtype).to(device)

    with torch.no_grad():
        torch.onnx.export(
            unet, (latent, timestep, encoder_hidden_states),
            merged_onnx,
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["out_sample"],
            dynamic_axes={
                "sample": {0: "batch", 2: "h", 3: "w"},
                "encoder_hidden_states": {0: "batch"},
                "out_sample": {0: "batch", 2: "h", 3: "w"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    size_mb = os.path.getsize(merged_onnx) / 1024 / 1024
    print(f"  Merged UNet: {size_mb:.1f} MB")

    del unet
    torch.cuda.empty_cache()

    # Now generate using ORT (reuse existing CLIP + VAE ONNX)
    print("\nGenerating image via ORT...")
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Import our tokenizer and scheduler from ort_generate
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ort_generate import SimpleTokenizer, DDIMScheduler

    tok = SimpleTokenizer(tokenizer_dir)
    sched = DDIMScheduler()

    clip_sess = ort.InferenceSession(os.path.join(onnx_dir, "clip_text_encoder.onnx"), providers=providers)
    unet_sess = ort.InferenceSession(merged_onnx, providers=providers)
    vae_sess = ort.InferenceSession(os.path.join(onnx_dir, "vae_decoder.onnx"), providers=providers)

    # Encode text
    cond_ids = np.array([tok.encode(prompt)], dtype=np.int64)
    uncond_ids = np.array([tok.encode("")], dtype=np.int64)
    cond_emb = clip_sess.run(None, {"input_ids": cond_ids})[0]
    uncond_emb = clip_sess.run(None, {"input_ids": uncond_ids})[0]

    # Diffusion
    rng = np.random.RandomState(seed)
    latent = rng.randn(1, 4, 64, 64).astype(np.float32)
    timesteps = sched.set_timesteps(num_steps)

    t0 = time.time()
    for i, t in enumerate(timesteps):
        t_arr = np.array([t], dtype=np.int64)
        noise_uncond = unet_sess.run(None, {"sample": latent, "timestep": t_arr, "encoder_hidden_states": uncond_emb})[0]
        noise_cond = unet_sess.run(None, {"sample": latent, "timestep": t_arr, "encoder_hidden_states": cond_emb})[0]
        noise_pred = noise_uncond + 7.5 * (noise_cond - noise_uncond)
        latent = sched.step(noise_pred, t, latent).astype(np.float32)
        if (i+1) % 5 == 0 or i == 0:
            print(f"  Step {i+1}/{num_steps}")

    print(f"Diffusion: {time.time()-t0:.1f}s")

    # VAE decode
    scaled = latent / 0.18215
    image = vae_sess.run(None, {"latent_sample": scaled})[0]

    # Save
    img = image[0]
    img = np.clip((img + 1) / 2, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    Image.fromarray(img).save(output_path)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python3 test_lora.py <lora.safetensors> <model_dir> <onnx_dir> <tokenizer_dir> \"prompt\" [out.png] [seed] [steps]")
        sys.exit(1)

    lora_path = sys.argv[1]
    model_dir = sys.argv[2]
    onnx_dir = sys.argv[3]
    tok_dir = sys.argv[4]
    prompt = sys.argv[5]
    out = sys.argv[6] if len(sys.argv) > 6 else "lora_test.png"
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 42
    steps = int(sys.argv[8]) if len(sys.argv) > 8 else 25

    merge_lora_and_generate(lora_path, model_dir, onnx_dir, tok_dir, prompt, out, seed, steps)
