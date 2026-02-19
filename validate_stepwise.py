"""Step-by-step diffusion validation at 16x16 latent.
Dumps intermediate values at each step for comparison with Go output."""
import torch
import numpy as np
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

model_id = 'nota-ai/bk-sdm-tiny'
prompt = 'a painting of a cat sitting on a window'
seed = 42
num_steps = 5
guidance_scale = 7.5
latent_h, latent_w = 16, 16

print(f'Prompt: "{prompt}"')
print(f'Seed: {seed}, Steps: {num_steps}, CFG: {guidance_scale}')
print(f'Latent: {latent_h}x{latent_w} -> {latent_h*8}x{latent_w*8} output')
print()

# Load components
print('Loading tokenizer + CLIP...')
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer')
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder='text_encoder', torch_dtype=torch.float32).cuda()

# Encode text
tokens = tokenizer(prompt, padding='max_length', max_length=77, return_tensors='pt')
with torch.no_grad():
    cond_emb = text_encoder(tokens.input_ids.cuda()).last_hidden_state
    uncond_tokens = tokenizer('', padding='max_length', max_length=77, return_tensors='pt')
    uncond_emb = text_encoder(uncond_tokens.input_ids.cuda()).last_hidden_state

print(f'Cond tokens[:8]: {tokens.input_ids[0][:8].tolist()}')
print(f'cond_emb[0][:3] = [{cond_emb[0,0,0]:.4f}, {cond_emb[0,0,1]:.4f}, {cond_emb[0,0,2]:.4f}]')

# Move to float32 for comparison with Go (which uses fp32 arithmetic)
cond_emb_f32 = cond_emb.float()
uncond_emb_f32 = uncond_emb.float()

del text_encoder
torch.cuda.empty_cache()

# Load UNet in fp32 for exact comparison  
print('\nLoading UNet (fp32)...')
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder='unet', torch_dtype=torch.float32).cuda()
unet.eval()

# Scheduler
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
scheduler.set_timesteps(num_steps)
timesteps = scheduler.timesteps.tolist()
print(f'Timesteps: {timesteps}')

# Initial noise - same as Go Box-Muller with seed=42
# Go uses Go's math/rand with seed=42, Python uses torch Generator
# For numerical comparison, create same noise
gen = torch.Generator('cpu').manual_seed(seed)
latent = torch.randn(1, 4, latent_h, latent_w, generator=gen).cuda()
print(f'Latent range: [{latent.min():.3f}, {latent.max():.3f}]')
print(f'Latent[:3]: [{latent[0,0,0,0]:.4f}, {latent[0,0,0,1]:.4f}, {latent[0,0,0,2]:.4f}]')

# Note: Go uses Go stdlib rand (different from torch), so exact noise will differ.
# What we CAN compare: given same noise, do we get same UNet output?
# Save Go-compatible noise for validation
np.save('py_initial_latent.npy', latent.cpu().numpy())

print('\n--- Diffusion Steps ---')
for step_idx, t in enumerate(scheduler.timesteps):
    t_int = t.item()
    
    with torch.no_grad():
        # Run UNet
        latent_model_input = latent  # no scaling for DDIM
        noise_pred_uncond = unet(latent_model_input, t, encoder_hidden_states=uncond_emb_f32).sample
        noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=cond_emb_f32).sample
        
        # CFG
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Scheduler step
        out = scheduler.step(noise_pred, t, latent)
        latent = out.prev_sample
    
    print(f'Step {step_idx+1}/{num_steps} (t={t_int}):')
    print(f'  noise_pred range: [{noise_pred.min():.4f}, {noise_pred.max():.4f}]')
    print(f'  latent range: [{latent.min():.4f}, {latent.max():.4f}]')
    print(f'  latent[:3]: [{latent[0,0,0,0]:.4f}, {latent[0,0,0,1]:.4f}, {latent[0,0,0,2]:.4f}]')

# VAE decode
print('\nLoading VAE...')
del unet
torch.cuda.empty_cache()
vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae', torch_dtype=torch.float32).cuda()
vae.eval()

latent_scaled = latent / 0.18215
with torch.no_grad():
    img = vae.decode(latent_scaled).sample

print(f'Image range: [{img.min():.4f}, {img.max():.4f}]')
print(f'Image shape: {img.shape}')

# Save
img_np = img[0].permute(1, 2, 0).cpu().numpy()
img_np = ((img_np + 1) / 2).clip(0, 1)
from PIL import Image
img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
img_pil.save('py_16x16_output.png')
print(f'Saved py_16x16_output.png ({img_pil.size[0]}x{img_pil.size[1]})')
print('\nDone!')
