#!/usr/bin/env python3
"""yent.yo image generator — GPU inference via diffusers.
This is the production-ready pipeline while Go pipeline matures.

Usage:
    python3 generate.py "a painting of a cat" output.png [seed] [steps] [cfg_scale]
    python3 generate.py --batch prompts.txt output_dir/ [seed] [steps] [cfg_scale]
"""
import sys
import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

MODEL_ID = 'nota-ai/bk-sdm-tiny'

def load_pipeline(device='cuda'):
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe

def generate(pipe, prompt, seed=42, steps=25, cfg=7.5, width=512, height=512):
    gen = torch.Generator(pipe.device).manual_seed(seed)
    with torch.no_grad():
        result = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=gen,
            width=width,
            height=height,
        )
    return result.images[0]

def main():
    if len(sys.argv) < 3:
        print('Usage: python3 generate.py "prompt" output.png [seed] [steps] [cfg]')
        print('       python3 generate.py --batch prompts.txt output_dir/ [seed] [steps] [cfg]')
        sys.exit(1)
    
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 25
    cfg = float(sys.argv[5]) if len(sys.argv) > 5 else 7.5
    
    print('Loading pipeline...')
    pipe = load_pipeline()
    print('Pipeline loaded')
    
    if sys.argv[1] == '--batch':
        # Batch mode: read prompts from file
        prompts_file = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else 'output/'
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
        steps = int(sys.argv[5]) if len(sys.argv) > 5 else 25
        cfg = float(sys.argv[6]) if len(sys.argv) > 6 else 7.5
        
        os.makedirs(output_dir, exist_ok=True)
        with open(prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        for i, prompt in enumerate(prompts):
            print(f'[{i+1}/{len(prompts)}] {prompt}')
            img = generate(pipe, prompt, seed=seed+i, steps=steps, cfg=cfg)
            out_path = os.path.join(output_dir, f'{i:04d}.png')
            img.save(out_path)
            print(f'  -> {out_path}')
    else:
        # Single mode
        prompt = sys.argv[1]
        output = sys.argv[2]
        print(f'Generating: {prompt}')
        print(f'Seed: {seed}, Steps: {steps}, CFG: {cfg}')
        img = generate(pipe, prompt, seed=seed, steps=steps, cfg=cfg)
        img.save(output)
        print(f'Saved: {output} ({img.size[0]}x{img.size[1]})')

if __name__ == '__main__':
    main()
