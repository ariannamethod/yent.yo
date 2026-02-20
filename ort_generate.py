#!/usr/bin/env python3
"""Generate images using ONNX Runtime — ZERO PyTorch dependency.

This is the production pipeline. Only needs: onnxruntime-gpu, numpy, Pillow, tokenizers.

Usage:
  python3 ort_generate.py <onnx_dir> <tokenizer_dir> "prompt" [output.png] [seed] [steps] [cfg_scale]

Example:
  python3 ort_generate.py onnx bk-sdm-tiny-hf/tokenizer "a cat on a roof" cat.png 42 25 7.5
"""

import sys
import os
import time
import json
import re
import numpy as np
from PIL import Image


# ---- Tokenizer (minimal CLIP tokenizer, no torch) ----

class SimpleTokenizer:
    """Minimal BPE tokenizer for CLIP (SD 1.x). No torch/transformers needed."""

    def __init__(self, tokenizer_dir):
        vocab_path = os.path.join(tokenizer_dir, "vocab.json")
        merges_path = os.path.join(tokenizer_dir, "merges.txt")

        with open(vocab_path, "r") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_path, "r") as f:
            lines = f.read().strip().split("\n")
            # Skip header if present
            if lines[0].startswith("#"):
                lines = lines[1:]
            self.bpe_ranks = {}
            for i, line in enumerate(lines):
                parts = line.split()
                if len(parts) == 2:
                    self.bpe_ranks[tuple(parts)] = i

        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]|[^\s\w]+""",
            re.IGNORECASE
        )
        self.bos_token_id = 49406  # <|startoftext|>
        self.eos_token_id = 49407  # <|endoftext|>
        self.max_length = 77

    def _bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = set(zip(word[:-1], word[1:]))

        if not pairs:
            return (token + "</w>",)

        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                if j < len(word) - 1 and word[j + 1] == second:
                    new_word.append(first + second)
                    i = j + 2
                else:
                    new_word.append(word[j])
                    i = j + 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = set(zip(word[:-1], word[1:]))

        return word

    def encode(self, text):
        text = text.lower().strip()
        tokens = [self.bos_token_id]

        # Simple word splitting (good enough for SD prompts)
        for word in re.findall(r"[a-z]+|[0-9]+|[^\s]", text):
            bpe_tokens = self._bpe(word)
            for bt in bpe_tokens:
                if bt in self.encoder:
                    tokens.append(self.encoder[bt])

        tokens.append(self.eos_token_id)

        # Pad/truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length - 1] + [self.eos_token_id]
        while len(tokens) < self.max_length:
            tokens.append(self.eos_token_id)

        return tokens


# ---- DDIM Scheduler (pure numpy) ----

class DDIMScheduler:
    def __init__(self, num_train=1000, beta_start=0.00085, beta_end=0.012):
        sqrt_start = np.sqrt(beta_start)
        sqrt_end = np.sqrt(beta_end)
        betas = np.linspace(sqrt_start, sqrt_end, num_train) ** 2
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas)
        self.num_train = num_train

    def set_timesteps(self, num_steps):
        self.num_steps = num_steps
        step_ratio = self.num_train // num_steps
        self.timesteps = [(num_steps - 1 - i) * step_ratio + 1 for i in range(num_steps)]
        return self.timesteps

    def step(self, noise_pred, timestep, sample):
        step_ratio = self.num_train // self.num_steps
        prev_t = timestep - step_ratio

        alpha_t = self.alphas_cumprod[timestep]
        alpha_prev = self.alphas_cumprod[max(prev_t, 0)]

        sqrt_alpha_t = np.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = np.sqrt(1.0 - alpha_t)
        sqrt_alpha_prev = np.sqrt(alpha_prev)
        sqrt_one_minus_alpha_prev = np.sqrt(1.0 - alpha_prev)

        pred_x0 = (sample - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        return sqrt_alpha_prev * pred_x0 + sqrt_one_minus_alpha_prev * noise_pred


# ---- Main pipeline ----

def generate(onnx_dir, tokenizer_dir, prompt, output_path, seed=42, num_steps=25,
             guidance_scale=7.5, latent_size=64):
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    rng = np.random.RandomState(seed)

    print(f"yent.yo — ONNX Runtime pipeline (zero PyTorch)")
    print(f"Prompt: {prompt!r}")
    print(f"Seed: {seed}, Steps: {num_steps}, CFG: {guidance_scale}, Latent: {latent_size}x{latent_size}")
    print()

    # Load tokenizer
    print("Loading tokenizer... ", end="", flush=True)
    t0 = time.time()
    tok = SimpleTokenizer(tokenizer_dir)
    print(f"done ({time.time()-t0:.2f}s)")

    # Tokenize
    cond_tokens = tok.encode(prompt)
    uncond_tokens = tok.encode("")
    cond_ids = np.array([cond_tokens], dtype=np.int64)
    uncond_ids = np.array([uncond_tokens], dtype=np.int64)
    print(f"Cond tokens: {cond_tokens[:8]}... (len={len(cond_tokens)})")

    # Load ONNX sessions
    print("Loading CLIP... ", end="", flush=True)
    t0 = time.time()
    clip_sess = ort.InferenceSession(os.path.join(onnx_dir, "clip_text_encoder.onnx"), providers=providers)
    print(f"done ({time.time()-t0:.2f}s)")

    print("Loading UNet... ", end="", flush=True)
    t0 = time.time()
    unet_sess = ort.InferenceSession(os.path.join(onnx_dir, "unet.onnx"), providers=providers)
    print(f"done ({time.time()-t0:.2f}s)")

    print("Loading VAE... ", end="", flush=True)
    t0 = time.time()
    vae_sess = ort.InferenceSession(os.path.join(onnx_dir, "vae_decoder.onnx"), providers=providers)
    print(f"done ({time.time()-t0:.2f}s)")

    # Phase 1: Text encoding
    print("\n--- Phase 1: Text Encoding ---")
    t0 = time.time()
    cond_emb = clip_sess.run(None, {"input_ids": cond_ids})[0]    # [1, 77, 768]
    uncond_emb = clip_sess.run(None, {"input_ids": uncond_ids})[0]
    print(f"  Text encoding: {time.time()-t0:.3f}s")
    print(f"  cond_emb[0][:3] = [{cond_emb[0,0,0]:.4f}, {cond_emb[0,0,1]:.4f}, {cond_emb[0,0,2]:.4f}]")

    # Phase 2: Diffusion
    print("\n--- Phase 2: Diffusion ---")
    sched = DDIMScheduler()
    timesteps = sched.set_timesteps(num_steps)
    print(f"Timesteps ({len(timesteps)}): [{timesteps[0]} ... {timesteps[-1]}]")

    latent = rng.randn(1, 4, latent_size, latent_size).astype(np.float32)
    print(f"Latent: {latent.shape}, range=[{latent.min():.3f}, {latent.max():.3f}]")

    total_t0 = time.time()
    for i, t in enumerate(timesteps):
        step_t0 = time.time()

        t_arr = np.array([t], dtype=np.int64)
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

        # CFG
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # DDIM step
        latent = sched.step(noise_pred, t, latent).astype(np.float32)

        print(f"  Step {i+1}/{num_steps} (t={t}): {time.time()-step_t0:.3f}s")

    total_diff = time.time() - total_t0
    print(f"\nDiffusion: {total_diff:.2f}s total ({total_diff/num_steps*1000:.0f}ms/step)")

    # Phase 3: VAE decode
    print("\n--- Phase 3: VAE Decoding ---")
    scaled = latent / 0.18215
    t0 = time.time()
    image = vae_sess.run(None, {"latent_sample": scaled})[0]  # [1, 3, H, W]
    print(f"  VAE decode: {time.time()-t0:.3f}s")
    print(f"  Output: {image.shape}, range=[{image.min():.3f}, {image.max():.3f}]")

    # Convert to PIL and save
    img = image[0]  # [3, H, W]
    img = np.clip((img + 1) / 2, 0, 1)  # [-1,1] → [0,1]
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # CHW → HWC
    pil_img = Image.fromarray(img)
    pil_img.save(output_path)
    print(f"\nSaved: {output_path} ({pil_img.size[0]}x{pil_img.size[1]})")

    total = time.time() - total_t0
    print(f"Total generation time: {total:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 ort_generate.py <onnx_dir> <tokenizer_dir> \"prompt\" [output.png] [seed] [steps] [cfg]")
        sys.exit(1)

    onnx_dir = sys.argv[1]
    tokenizer_dir = sys.argv[2]
    prompt = sys.argv[3]
    output = sys.argv[4] if len(sys.argv) > 4 else "ort_output.png"
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
    steps = int(sys.argv[6]) if len(sys.argv) > 6 else 25
    cfg = float(sys.argv[7]) if len(sys.argv) > 7 else 7.5

    generate(onnx_dir, tokenizer_dir, prompt, output, seed, steps, cfg)
