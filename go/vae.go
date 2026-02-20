package main

import (
	"fmt"
	"math"
)

// VAEDecoder decodes latent [1,4,64,64] → image [1,3,512,512]
// Architecture: post_quant_conv → conv_in → mid_block → 4 up_blocks → conv_norm_out → conv_out
type VAEDecoder struct {
	PostQuantConvW, PostQuantConvB *Tensor // [4,4,1,1]
	ConvInW, ConvInB               *Tensor // [512,4,3,3]

	// Mid block: resnet0 → self-attention → resnet1
	MidResnet0     VAEResNet
	MidAttnNormW   *Tensor // GroupNorm [512]
	MidAttnNormB   *Tensor
	MidAttnQW      *Tensor // [512,512] with bias
	MidAttnQB      *Tensor
	MidAttnKW      *Tensor
	MidAttnKB      *Tensor
	MidAttnVW      *Tensor
	MidAttnVB      *Tensor
	MidAttnOutW    *Tensor
	MidAttnOutB    *Tensor
	MidResnet1     VAEResNet

	// Up blocks (4): 3 resnets each, first 3 have upsamplers
	UpBlocks [4]VAEUpBlock

	// Output
	ConvNormW, ConvNormB *Tensor // GroupNorm [128]
	ConvOutW, ConvOutB   *Tensor // [3,128,3,3]
}

// VAEResNet: GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv + skip (no time embedding)
type VAEResNet struct {
	Norm1W, Norm1B       *Tensor
	Conv1W, Conv1B       *Tensor
	Norm2W, Norm2B       *Tensor
	Conv2W, Conv2B       *Tensor
	ShortcutW, ShortcutB *Tensor // optional 1x1 for channel mismatch
}

type VAEUpBlock struct {
	Resnets      [3]VAEResNet
	HasUpsampler bool
	UpsamplerW   *Tensor
	UpsamplerB   *Tensor
}

func LoadVAEDecoder(st *SafeTensors) (*VAEDecoder, error) {
	v := &VAEDecoder{}

	load := func(name string) *Tensor {
		data, shape, err := st.GetFloat32(name)
		if err != nil {
			return nil
		}
		return TensorFrom(data, shape)
	}

	v.PostQuantConvW = load("post_quant_conv.weight")
	v.PostQuantConvB = load("post_quant_conv.bias")
	v.ConvInW = load("decoder.conv_in.weight")
	v.ConvInB = load("decoder.conv_in.bias")

	// Mid block
	v.MidResnet0 = loadVAEResNet(load, "decoder.mid_block.resnets.0.")
	v.MidAttnNormW = load("decoder.mid_block.attentions.0.group_norm.weight")
	v.MidAttnNormB = load("decoder.mid_block.attentions.0.group_norm.bias")
	v.MidAttnQW = load("decoder.mid_block.attentions.0.to_q.weight")
	v.MidAttnQB = load("decoder.mid_block.attentions.0.to_q.bias")
	v.MidAttnKW = load("decoder.mid_block.attentions.0.to_k.weight")
	v.MidAttnKB = load("decoder.mid_block.attentions.0.to_k.bias")
	v.MidAttnVW = load("decoder.mid_block.attentions.0.to_v.weight")
	v.MidAttnVB = load("decoder.mid_block.attentions.0.to_v.bias")
	v.MidAttnOutW = load("decoder.mid_block.attentions.0.to_out.0.weight")
	v.MidAttnOutB = load("decoder.mid_block.attentions.0.to_out.0.bias")
	v.MidResnet1 = loadVAEResNet(load, "decoder.mid_block.resnets.1.")

	// Up blocks: 4 blocks, first 3 have upsamplers
	hasUpsampler := [4]bool{true, true, true, false}
	for i := 0; i < 4; i++ {
		p := fmt.Sprintf("decoder.up_blocks.%d.", i)
		v.UpBlocks[i].HasUpsampler = hasUpsampler[i]
		for j := 0; j < 3; j++ {
			v.UpBlocks[i].Resnets[j] = loadVAEResNet(load, fmt.Sprintf("%sresnets.%d.", p, j))
		}
		if hasUpsampler[i] {
			v.UpBlocks[i].UpsamplerW = load(p + "upsamplers.0.conv.weight")
			v.UpBlocks[i].UpsamplerB = load(p + "upsamplers.0.conv.bias")
		}
	}

	v.ConvNormW = load("decoder.conv_norm_out.weight")
	v.ConvNormB = load("decoder.conv_norm_out.bias")
	v.ConvOutW = load("decoder.conv_out.weight")
	v.ConvOutB = load("decoder.conv_out.bias")

	return v, nil
}

func loadVAEResNet(load func(string) *Tensor, p string) VAEResNet {
	return VAEResNet{
		Norm1W:    load(p + "norm1.weight"),
		Norm1B:    load(p + "norm1.bias"),
		Conv1W:    load(p + "conv1.weight"),
		Conv1B:    load(p + "conv1.bias"),
		Norm2W:    load(p + "norm2.weight"),
		Norm2B:    load(p + "norm2.bias"),
		Conv2W:    load(p + "conv2.weight"),
		Conv2B:    load(p + "conv2.bias"),
		ShortcutW: load(p + "conv_shortcut.weight"),
		ShortcutB: load(p + "conv_shortcut.bias"),
	}
}

// Decode: latent [1,4,64,64] → image [1,3,512,512]
func (v *VAEDecoder) Decode(latent *Tensor) *Tensor {
	// 1. Post-quant conv: [1,4,64,64] → [1,4,64,64]
	x := Conv2d(latent, v.PostQuantConvW, v.PostQuantConvB, 1, 0)

	// 2. Conv in: [1,4,64,64] → [1,512,64,64]
	x = Conv2d(x, v.ConvInW, v.ConvInB, 1, 1)

	// 3. Mid block: resnet → self-attention → resnet
	x = vaeResnetForward(x, v.MidResnet0)
	x = vaeMidAttention(x, v)
	x = vaeResnetForward(x, v.MidResnet1)

	// 4. Up blocks
	// block 0: 512→512, upsample 64→128
	// block 1: 512→512, upsample 128→256
	// block 2: 512→256, upsample 256→512
	// block 3: 256→128, no upsample
	for i := 0; i < 4; i++ {
		for j := 0; j < 3; j++ {
			x = vaeResnetForward(x, v.UpBlocks[i].Resnets[j])
		}
		if v.UpBlocks[i].HasUpsampler {
			x = Upsample2x(x)
			x = Conv2d(x, v.UpBlocks[i].UpsamplerW, v.UpBlocks[i].UpsamplerB, 1, 1)
		}
	}

	// 5. Output: GroupNorm → SiLU → Conv [128→3]
	x = GroupNorm(x, v.ConvNormW, v.ConvNormB, 32, 1e-6)
	x = SiLU(x)
	x = Conv2d(x, v.ConvOutW, v.ConvOutB, 1, 1)

	return x
}

func vaeResnetForward(x *Tensor, r VAEResNet) *Tensor {
	residual := x

	h := GroupNorm(x, r.Norm1W, r.Norm1B, 32, 1e-6)
	h = SiLU(h)
	h = Conv2d(h, r.Conv1W, r.Conv1B, 1, 1)

	h = GroupNorm(h, r.Norm2W, r.Norm2B, 32, 1e-6)
	h = SiLU(h)
	h = Conv2d(h, r.Conv2W, r.Conv2B, 1, 1)

	if r.ShortcutW != nil {
		residual = Conv2d(residual, r.ShortcutW, r.ShortcutB, 1, 0)
	}

	return Add(h, residual)
}

// vaeMidAttention: single-head self-attention on spatial features
// Input: [1, 512, H, W] → GroupNorm → reshape to [H*W, 512] → Q/K/V → attention → reshape back
func vaeMidAttention(x *Tensor, v *VAEDecoder) *Tensor {
	residual := x

	// GroupNorm
	h := GroupNorm(x, v.MidAttnNormW, v.MidAttnNormB, 32, 1e-6)

	// Reshape to 2D for attention
	C := h.Shape[1] // 512
	H := h.Shape[2]
	W := h.Shape[3]
	h2d := Reshape4Dto2D(h) // [H*W, 512]

	seq := H * W

	// Q, K, V projections (with bias — VAE attention uses biases)
	q := Linear(h2d, v.MidAttnQW, v.MidAttnQB)   // [seq, 512]
	k := Linear(h2d, v.MidAttnKW, v.MidAttnKB)   // [seq, 512]
	val := Linear(h2d, v.MidAttnVW, v.MidAttnVB)  // [seq, 512]

	// Single-head attention (headDim = C = 512)
	scale := float32(1.0 / math.Sqrt(float64(C)))
	out := NewTensor(seq, C)

	if hasAccel {
		// Fused tiled single-head attention in C.
		// At 64×64: seq=4096, dim=512 → score tile = 256×4096 = 4MB (fits L3)
		tileSize := 256
		if seq <= 256 {
			tileSize = seq
		}
		accelTiledAttentionSingle(q.Data, k.Data, val.Data, out.Data,
			seq, C, scale, tileSize)
	} else {
		scores := make([]float32, seq)
		for i := 0; i < seq; i++ {
			maxS := float32(-math.MaxFloat32)
			for j := 0; j < seq; j++ {
				s := float32(0)
				for d := 0; d < C; d++ {
					s += q.Data[i*C+d] * k.Data[j*C+d]
				}
				scores[j] = s * scale
				if scores[j] > maxS {
					maxS = scores[j]
				}
			}
			sumExp := float32(0)
			for j := range scores {
				scores[j] = float32(math.Exp(float64(scores[j] - maxS)))
				sumExp += scores[j]
			}
			for j := range scores {
				scores[j] /= sumExp
			}
			for d := 0; d < C; d++ {
				s := float32(0)
				for j := 0; j < seq; j++ {
					s += scores[j] * val.Data[j*C+d]
				}
				out.Data[i*C+d] = s
			}
		}
	}

	// Output projection
	out = Linear(out, v.MidAttnOutW, v.MidAttnOutB)

	// Reshape back to 4D and add residual
	result := Reshape2Dto4D(out, C, H, W)
	return Add(result, residual)
}
