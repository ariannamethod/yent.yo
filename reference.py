"""Generate reference image using diffusers pipeline.
Compare this output with Go inference to validate correctness."""
import torch
import numpy as np
from diffusers import StableDiffusionPipeline, DDIMScheduler

model_id = "nota-ai/bk-sdm-tiny"
prompt = "a painting of a cat sitting on a window"
seed = 42
num_steps = 25
guidance_scale = 7.5

print(f'Prompt: "{prompt}"')
print(f'Seed: {seed}, Steps: {num_steps}, Guidance: {guidance_scale}')

# Load pipeline
print('Loading pipeline...')
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to('cuda')
print('Pipeline loaded')

# Generate
print('Generating...')
generator = torch.Generator('cuda').manual_seed(seed)
with torch.no_grad():
    result = pipe(
        prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

img = result.images[0]
img.save('reference_output.png')
print(f'Saved reference_output.png ({img.size[0]}x{img.size[1]})')

# Also save with 5 steps for quick comparison
print('\nGenerating 5-step version...')
generator = torch.Generator('cuda').manual_seed(seed)
with torch.no_grad():
    result5 = pipe(
        prompt,
        num_inference_steps=5,
        guidance_scale=guidance_scale,
        generator=generator,
    )
img5 = result5.images[0]
img5.save('reference_5step.png')
print(f'Saved reference_5step.png ({img5.size[0]}x{img5.size[1]})')

# Dump intermediate values for numerical comparison
print('\nDumping intermediate values...')
from transformers import CLIPTextModel, CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer')
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder='text_encoder', torch_dtype=torch.float32).to('cuda')

tokens = tokenizer(prompt, padding='max_length', max_length=77, return_tensors='pt')
with torch.no_grad():
    text_out = text_encoder(tokens.input_ids.to('cuda'))
    cond_emb = text_out.last_hidden_state

print(f'Cond tokens: {tokens.input_ids[0][:8].tolist()}')
print(f'cond_emb[0][:3] = {cond_emb[0, 0, :3].tolist()}')
print(f'cond_emb shape: {cond_emb.shape}')
print(f'cond_emb range: [{cond_emb.min().item():.4f}, {cond_emb.max().item():.4f}]')

# Save embeddings for Go comparison
np.save('reference_cond_emb.npy', cond_emb.cpu().numpy())
print('Saved reference_cond_emb.npy')

print('\nDone!')
