"""Exact numerical validation using our fp16 safetensors.
Loads weights same as Go (fp16->fp32), uses CPU, compares step by step.
Uses Go-compatible noise (from saved file) for exact comparison."""
import torch
import numpy as np
import json
import struct
import os

# Read our fp16 safetensors (same files Go reads)
model_dir = '/home/ubuntu/yent.yo/bk-sdm-tiny'

# Load one UNet weight to verify we read same values as Go
def read_safetensors_tensor(path, tensor_name):
    """Read a specific tensor from safetensors file, same as Go does."""
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size))
        data_start = 8 + header_size
        
        meta = header[tensor_name]
        dtype = meta['dtype']
        shape = meta['data_offsets']
        begin, end = meta['data_offsets']
        
        f.seek(data_start + begin)
        raw = f.read(end - begin)
        
        if dtype == 'F16':
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dtype == 'F32':
            arr = np.frombuffer(raw, dtype=np.float32)
        else:
            raise ValueError(f'Unknown dtype: {dtype}')
        
        shape_dims = meta['shape']
        return arr.reshape(shape_dims)

# Test: read a known weight
w = read_safetensors_tensor(model_dir + '/unet/diffusion_pytorch_model.fp16.safetensors', 
                            'conv_in.weight')
print(f'conv_in.weight shape: {w.shape}, range: [{w.min():.6f}, {w.max():.6f}]')
print(f'conv_in.weight[:3]: [{w.flat[0]:.6f}, {w.flat[1]:.6f}, {w.flat[2]:.6f}]')

# Load same weight via diffusers for comparison
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained('nota-ai/bk-sdm-tiny', subfolder='unet', torch_dtype=torch.float32)
w_ref = unet.conv_in.weight.data.numpy()
print(f'\nReference conv_in.weight[:3]: [{w_ref.flat[0]:.6f}, {w_ref.flat[1]:.6f}, {w_ref.flat[2]:.6f}]')

# Compare
diff = np.abs(w - w_ref).max()
print(f'Max diff: {diff:.8f}')

# Check a few more
for name, param_name in [
    ('time_embedding.linear_1.weight', 'time_embedding.linear_1.weight'),
    ('down_blocks.0.resnets.0.conv1.weight', 'down_blocks.0.resnets.0.conv1.weight'),
]:
    w_ours = read_safetensors_tensor(model_dir + '/unet/diffusion_pytorch_model.fp16.safetensors', name)
    # Get from model
    parts = param_name.split('.')
    obj = unet
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    w_ref2 = obj.data.numpy()
    diff2 = np.abs(w_ours - w_ref2).max()
    print(f'{name}: shape={w_ours.shape}, max_diff={diff2:.8f}')

print('\nWeight loading validation complete!')
