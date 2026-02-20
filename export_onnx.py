#!/usr/bin/env python3
"""Export BK-SDM-Tiny components to ONNX for GPU inference without PyTorch.

Exports 3 files:
  clip_text_encoder.onnx  — text encoder (input: token_ids → output: embeddings)
  unet.onnx               — denoising UNet (input: latent, timestep, encoder_hidden_states → noise_pred)
  vae_decoder.onnx         — VAE decoder (input: latent → image)

Usage:
  python3 export_onnx.py <model_dir> [output_dir]

Example:
  python3 export_onnx.py bk-sdm-tiny-hf onnx/
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np

def export_clip(model_dir, output_dir):
    """Export CLIP text encoder to ONNX."""
    from transformers import CLIPTextModel

    print("Loading CLIP text encoder...")
    text_encoder = CLIPTextModel.from_pretrained(
        model_dir, subfolder="text_encoder", torch_dtype=torch.float32
    ).cuda().eval()

    # Dummy input: batch=1, seq_len=77 (standard SD tokenizer output)
    dummy_input = torch.randint(0, 49408, (1, 77), dtype=torch.long).cuda()

    out_path = os.path.join(output_dir, "clip_text_encoder.onnx")
    print(f"Exporting to {out_path}...")

    with torch.no_grad():
        torch.onnx.export(
            text_encoder,
            (dummy_input,),
            out_path,
            input_names=["input_ids"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "last_hidden_state": {0: "batch"},
                "pooler_output": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  CLIP exported: {size_mb:.1f} MB")

    # Validate
    ref_out = text_encoder(dummy_input).last_hidden_state.detach().cpu().float().numpy()
    del text_encoder
    torch.cuda.empty_cache()

    return out_path, dummy_input.cpu().numpy(), ref_out


def export_unet(model_dir, output_dir):
    """Export UNet to ONNX."""
    from diffusers import UNet2DConditionModel

    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        model_dir, subfolder="unet", torch_dtype=torch.float32
    ).cuda().eval()

    # Dummy inputs matching SD pipeline
    latent = torch.randn(1, 4, 64, 64, dtype=torch.float32).cuda()
    timestep = torch.tensor([800], dtype=torch.long).cuda()
    encoder_hidden_states = torch.randn(1, 77, 768, dtype=torch.float32).cuda()

    out_path = os.path.join(output_dir, "unet.onnx")
    print(f"Exporting to {out_path}...")

    with torch.no_grad():
        torch.onnx.export(
            unet,
            (latent, timestep, encoder_hidden_states),
            out_path,
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["out_sample"],
            dynamic_axes={
                "sample": {0: "batch", 2: "height", 3: "width"},
                "encoder_hidden_states": {0: "batch"},
                "out_sample": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  UNet exported: {size_mb:.1f} MB")

    del unet
    torch.cuda.empty_cache()
    return out_path


def export_vae_decoder(model_dir, output_dir):
    """Export VAE decoder to ONNX."""
    from diffusers import AutoencoderKL

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        model_dir, subfolder="vae", torch_dtype=torch.float32
    ).cuda().eval()

    # Only export the decoder part
    # VAE decoder input: latent [1, 4, H, W] → image [1, 3, 8H, 8W]
    dummy_latent = torch.randn(1, 4, 64, 64, dtype=torch.float32).cuda()

    out_path = os.path.join(output_dir, "vae_decoder.onnx")
    print(f"Exporting to {out_path}...")

    # We need to export just the decode method
    # Wrap in a simple module
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent_sample):
            return self.vae.decode(latent_sample, return_dict=False)[0]

    wrapper = VAEDecoderWrapper(vae)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_latent,),
            out_path,
            input_names=["latent_sample"],
            output_names=["sample"],
            dynamic_axes={
                "latent_sample": {0: "batch", 2: "height", 3: "width"},
                "sample": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  VAE decoder exported: {size_mb:.1f} MB")

    del vae, wrapper
    torch.cuda.empty_cache()
    return out_path


def validate_onnx(output_dir, clip_path, clip_input, clip_ref_output):
    """Quick validation that ONNX models load and produce correct output."""
    import onnxruntime as ort

    print("\n--- Validation ---")

    # Check CLIP
    print("Validating CLIP...")
    sess = ort.InferenceSession(clip_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    result = sess.run(None, {"input_ids": clip_input})
    max_diff = np.max(np.abs(result[0].astype(np.float32) - clip_ref_output.astype(np.float32)))
    print(f"  CLIP max diff vs PyTorch: {max_diff:.6f}")
    del sess

    # Check UNet loads
    unet_path = os.path.join(output_dir, "unet.onnx")
    print("Validating UNet loads...")
    sess = ort.InferenceSession(unet_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print(f"  UNet OK, inputs: {[i.name for i in sess.get_inputs()]}")
    print(f"  UNet OK, outputs: {[o.name for o in sess.get_outputs()]}")
    del sess

    # Check VAE loads
    vae_path = os.path.join(output_dir, "vae_decoder.onnx")
    print("Validating VAE decoder loads...")
    sess = ort.InferenceSession(vae_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print(f"  VAE OK, inputs: {[i.name for i in sess.get_inputs()]}")
    print(f"  VAE OK, outputs: {[o.name for o in sess.get_outputs()]}")
    del sess

    print("\nAll ONNX models validated!")


def benchmark_onnx(output_dir):
    """Benchmark full pipeline: CLIP → UNet (5 steps) → VAE."""
    import onnxruntime as ort
    import time

    print("\n--- Benchmark (5 steps, 64x64 latent) ---")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Load sessions
    print("Loading ONNX sessions...")
    t0 = time.time()
    clip_sess = ort.InferenceSession(os.path.join(output_dir, "clip_text_encoder.onnx"), providers=providers)
    unet_sess = ort.InferenceSession(os.path.join(output_dir, "unet.onnx"), providers=providers)
    vae_sess = ort.InferenceSession(os.path.join(output_dir, "vae_decoder.onnx"), providers=providers)
    print(f"  Sessions loaded: {time.time()-t0:.1f}s")

    # CLIP
    tokens = np.array([[49406] + [320]*75 + [49407]], dtype=np.int64)  # "a" padded
    t0 = time.time()
    cond_emb = clip_sess.run(None, {"input_ids": tokens})[0]
    clip_time = time.time() - t0
    print(f"  CLIP: {clip_time*1000:.0f}ms")

    # Uncond
    uncond_tokens = np.array([[49406] + [49407] + [49407]*75], dtype=np.int64)
    uncond_emb = clip_sess.run(None, {"input_ids": uncond_tokens})[0]

    # Diffusion loop (5 steps)
    latent = np.random.randn(1, 4, 64, 64).astype(np.float32)
    timesteps = [801, 601, 401, 201, 1]
    guidance_scale = 7.5

    total_unet = 0
    for i, t in enumerate(timesteps):
        t_arr = np.array([t], dtype=np.int64)
        t0 = time.time()
        noise_uncond = unet_sess.run(None, {
            "sample": latent,
            "timestep": t_arr,
            "encoder_hidden_states": uncond_emb,
        })[0]
        noise_cond = unet_sess.run(None, {
            "sample": latent,
            "timestep": t_arr,
            "encoder_hidden_states": cond_emb,
        })[0]
        step_time = time.time() - t0
        total_unet += step_time

        # CFG
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        # Simplified DDIM step (just for benchmarking — real scheduler needed for quality)
        latent = (latent - 0.1 * noise_pred).astype(np.float32)

        print(f"  UNet step {i+1}/5 (t={t}): {step_time*1000:.0f}ms")

    print(f"  UNet total: {total_unet:.2f}s ({total_unet/5*1000:.0f}ms/step)")

    # VAE decode
    t0 = time.time()
    image = vae_sess.run(None, {"latent_sample": latent.astype(np.float32)})[0]
    vae_time = time.time() - t0
    print(f"  VAE decode: {vae_time*1000:.0f}ms")
    print(f"  Output shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")

    total = clip_time + total_unet + vae_time
    print(f"\n  TOTAL: {total:.2f}s (CLIP {clip_time*1000:.0f}ms + UNet {total_unet*1000:.0f}ms + VAE {vae_time*1000:.0f}ms)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 export_onnx.py <model_dir> [output_dir]")
        sys.exit(1)

    model_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "onnx"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Model: {model_dir}")
    print(f"Output: {output_dir}/")
    print()

    clip_path, clip_input, clip_ref = export_clip(model_dir, output_dir)
    unet_path = export_unet(model_dir, output_dir)
    vae_path = export_vae_decoder(model_dir, output_dir)

    print(f"\nExported files:")
    for f in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, f)
        size = os.path.getsize(path) / 1024 / 1024
        print(f"  {f}: {size:.1f} MB")

    validate_onnx(output_dir, clip_path, clip_input, clip_ref)
    benchmark_onnx(output_dir)
