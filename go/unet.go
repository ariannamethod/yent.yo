package main

import (
	"fmt"
	"math"
)

const attnHeadDim = 8 // BK-SDM-Tiny: attention_head_dim=8


// UNet2D is the BK-SDM-Tiny UNet (no mid block)
// Down: 3 blocks (1 resnet + 1 attn each), blocks 0-1 have downsampler
// Up: 3 blocks (2 resnets + 2 attns each), blocks 0-1 have upsampler
type UNet2D struct {
	ConvInW, ConvInB     *Tensor
	ConvOutW, ConvOutB   *Tensor
	ConvNormW, ConvNormB *Tensor

	TimeLinear1W, TimeLinear1B *Tensor
	TimeLinear2W, TimeLinear2B *Tensor

	DownResnets    [3]ResNetBlock
	DownAttentions [3]TransformerBlock
	DownSamplerW   [2]*Tensor
	DownSamplerB   [2]*Tensor

	UpResnets    [3][2]ResNetBlock
	UpAttentions [3][2]TransformerBlock
	UpSamplerW   [2]*Tensor
	UpSamplerB   [2]*Tensor
}

type ResNetBlock struct {
	Norm1W, Norm1B         *Tensor
	Conv1W, Conv1B         *Tensor
	TimeEmbW, TimeEmbB     *Tensor
	Norm2W, Norm2B         *Tensor
	Conv2W, Conv2B         *Tensor
	ConvShortW, ConvShortB *Tensor // optional 1x1 for channel mismatch
}

type TransformerBlock struct {
	NormW, NormB         *Tensor // GroupNorm on 4D input
	ProjInW, ProjInB     *Tensor // 1x1 conv
	ProjOutW, ProjOutB   *Tensor // 1x1 conv

	Norm1W, Norm1B       *Tensor // LayerNorm before self-attn
	SelfQW               *Tensor
	SelfKW               *Tensor
	SelfVW               *Tensor
	SelfOutW, SelfOutB   *Tensor

	Norm2W, Norm2B       *Tensor // LayerNorm before cross-attn
	CrossQW              *Tensor
	CrossKW              *Tensor // maps from CLIP 768 → dim
	CrossVW              *Tensor
	CrossOutW, CrossOutB *Tensor

	Norm3W, Norm3B         *Tensor // LayerNorm before FF
	FFLinear1W, FFLinear1B *Tensor // [dim*8, dim] for GEGLU
	FFLinear2W, FFLinear2B *Tensor // [dim, dim*4]
}

func LoadUNet(st *SafeTensors) (*UNet2D, error) {
	u := &UNet2D{}

	load := func(name string) *Tensor {
		data, shape, err := st.GetFloat32(name)
		if err != nil {
			return nil
		}
		return TensorFrom(data, shape)
	}

	u.ConvInW = load("conv_in.weight")
	u.ConvInB = load("conv_in.bias")
	u.ConvOutW = load("conv_out.weight")
	u.ConvOutB = load("conv_out.bias")
	u.ConvNormW = load("conv_norm_out.weight")
	u.ConvNormB = load("conv_norm_out.bias")

	u.TimeLinear1W = load("time_embedding.linear_1.weight")
	u.TimeLinear1B = load("time_embedding.linear_1.bias")
	u.TimeLinear2W = load("time_embedding.linear_2.weight")
	u.TimeLinear2B = load("time_embedding.linear_2.bias")

	// Down blocks: 1 resnet + 1 attention each
	for i := 0; i < 3; i++ {
		p := fmt.Sprintf("down_blocks.%d.", i)
		u.DownResnets[i] = loadResNet(load, p+"resnets.0.")
		u.DownAttentions[i] = loadTransformerBlock(load, p+"attentions.0.")
		if i < 2 {
			u.DownSamplerW[i] = load(p + "downsamplers.0.conv.weight")
			u.DownSamplerB[i] = load(p + "downsamplers.0.conv.bias")
		}
	}

	// Up blocks: 2 resnets + 2 attentions each
	for i := 0; i < 3; i++ {
		p := fmt.Sprintf("up_blocks.%d.", i)
		for j := 0; j < 2; j++ {
			u.UpResnets[i][j] = loadResNet(load, fmt.Sprintf("%sresnets.%d.", p, j))
			u.UpAttentions[i][j] = loadTransformerBlock(load, fmt.Sprintf("%sattentions.%d.", p, j))
		}
		if i < 2 {
			u.UpSamplerW[i] = load(p + "upsamplers.0.conv.weight")
			u.UpSamplerB[i] = load(p + "upsamplers.0.conv.bias")
		}
	}

	return u, nil
}

func loadResNet(load func(string) *Tensor, p string) ResNetBlock {
	return ResNetBlock{
		Norm1W: load(p + "norm1.weight"), Norm1B: load(p + "norm1.bias"),
		Conv1W: load(p + "conv1.weight"), Conv1B: load(p + "conv1.bias"),
		TimeEmbW: load(p + "time_emb_proj.weight"), TimeEmbB: load(p + "time_emb_proj.bias"),
		Norm2W: load(p + "norm2.weight"), Norm2B: load(p + "norm2.bias"),
		Conv2W: load(p + "conv2.weight"), Conv2B: load(p + "conv2.bias"),
		ConvShortW: load(p + "conv_shortcut.weight"), ConvShortB: load(p + "conv_shortcut.bias"),
	}
}

func loadTransformerBlock(load func(string) *Tensor, p string) TransformerBlock {
	tb := p + "transformer_blocks.0."
	return TransformerBlock{
		NormW: load(p + "norm.weight"), NormB: load(p + "norm.bias"),
		ProjInW: load(p + "proj_in.weight"), ProjInB: load(p + "proj_in.bias"),
		ProjOutW: load(p + "proj_out.weight"), ProjOutB: load(p + "proj_out.bias"),
		Norm1W: load(tb + "norm1.weight"), Norm1B: load(tb + "norm1.bias"),
		SelfQW: load(tb + "attn1.to_q.weight"),
		SelfKW: load(tb + "attn1.to_k.weight"),
		SelfVW: load(tb + "attn1.to_v.weight"),
		SelfOutW: load(tb + "attn1.to_out.0.weight"), SelfOutB: load(tb + "attn1.to_out.0.bias"),
		Norm2W: load(tb + "norm2.weight"), Norm2B: load(tb + "norm2.bias"),
		CrossQW: load(tb + "attn2.to_q.weight"),
		CrossKW: load(tb + "attn2.to_k.weight"),
		CrossVW: load(tb + "attn2.to_v.weight"),
		CrossOutW: load(tb + "attn2.to_out.0.weight"), CrossOutB: load(tb + "attn2.to_out.0.bias"),
		Norm3W: load(tb + "norm3.weight"), Norm3B: load(tb + "norm3.bias"),
		FFLinear1W: load(tb + "ff.net.0.proj.weight"), FFLinear1B: load(tb + "ff.net.0.proj.bias"),
		FFLinear2W: load(tb + "ff.net.2.weight"), FFLinear2B: load(tb + "ff.net.2.bias"),
	}
}

// Forward: latent [1,4,64,64] + timestep + text_emb [77,768] → noise_pred [1,4,64,64]
//
// Skip connections:
//   Down phase saves 6 skips: conv_in output, then after each (resnet+attn), then after each downsample.
//   Up phase pops skips in reverse, concatenating one per resnet.
func (u *UNet2D) Forward(latent *Tensor, timestep int, textEmb *Tensor) *Tensor {
	// 1. Time embedding (flip_sin_to_cos=true, freq_shift=0)
	temb := timestepEmbedding(timestep, 320)
	temb = Linear(temb, u.TimeLinear1W, u.TimeLinear1B) // [1, 1280]
	temb = SiLU(temb)
	temb = Linear(temb, u.TimeLinear2W, u.TimeLinear2B) // [1, 1280]

	// 2. Conv in: [1,4,64,64] → [1,320,64,64]
	x := Conv2d(latent, u.ConvInW, u.ConvInB, 1, 1)

	// 3. Down blocks — collect skip connections
	skips := []*Tensor{x}

	for i := 0; i < 3; i++ {
		fmt.Printf("      down_%d [%dx%d]...", i, x.Shape[2], x.Shape[3])
		x = resnetForward(x, temb, u.DownResnets[i])
		x = transformerForward(x, textEmb, u.DownAttentions[i])
		skips = append(skips, x)
		if i < 2 {
			x = Conv2d(x, u.DownSamplerW[i], u.DownSamplerB[i], 2, 1)
			skips = append(skips, x)
		}
		fmt.Println(" ok")
	}

	// 4. No mid block (BK-SDM-Tiny compression)

	// 5. Up blocks — each resnet pops and concats one skip
	for i := 0; i < 3; i++ {
		fmt.Printf("      up_%d [%dx%d]...", i, x.Shape[2], x.Shape[3])
		for j := 0; j < 2; j++ {
			skip := skips[len(skips)-1]
			skips = skips[:len(skips)-1]
			x = ConcatChannels(x, skip)
			x = resnetForward(x, temb, u.UpResnets[i][j])
			x = transformerForward(x, textEmb, u.UpAttentions[i][j])
		}
		if i < 2 {
			x = Upsample2x(x)
			x = Conv2d(x, u.UpSamplerW[i], u.UpSamplerB[i], 1, 1)
		}
		fmt.Println(" ok")
	}

	// 6. Output: GroupNorm → SiLU → Conv [320→4]
	x = GroupNorm(x, u.ConvNormW, u.ConvNormB, 32, 1e-5)
	x = SiLU(x)
	x = Conv2d(x, u.ConvOutW, u.ConvOutB, 1, 1)

	return x
}

func resnetForward(x, temb *Tensor, r ResNetBlock) *Tensor {
	residual := x

	// GroupNorm → SiLU → Conv3x3
	h := GroupNorm(x, r.Norm1W, r.Norm1B, 32, 1e-5)
	h = SiLU(h)
	h = Conv2d(h, r.Conv1W, r.Conv1B, 1, 1)

	// Add time embedding: project 1280 → channels, broadcast over spatial
	if r.TimeEmbW != nil {
		tProj := Linear(SiLU(temb), r.TimeEmbW, r.TimeEmbB) // [1, channels]
		channels := h.Shape[1]
		H := h.Shape[2]
		W := h.Shape[3]
		for c := 0; c < channels; c++ {
			tv := tProj.Data[c]
			for hw := 0; hw < H*W; hw++ {
				h.Data[c*H*W+hw] += tv
			}
		}
	}

	// GroupNorm → SiLU → Conv3x3
	h = GroupNorm(h, r.Norm2W, r.Norm2B, 32, 1e-5)
	h = SiLU(h)
	h = Conv2d(h, r.Conv2W, r.Conv2B, 1, 1)

	// Skip connection (1x1 conv if channel mismatch)
	if r.ConvShortW != nil {
		residual = Conv2d(residual, r.ConvShortW, r.ConvShortB, 1, 0)
	}

	return Add(h, residual)
}

func transformerForward(x, textEmb *Tensor, tb TransformerBlock) *Tensor {
	C := x.Shape[1]
	H := x.Shape[2]
	W := x.Shape[3]

	residual := x

	// GroupNorm on 4D → proj_in (1x1 conv)
	x = GroupNorm(x, tb.NormW, tb.NormB, 32, 1e-5)
	x = Conv2d(x, tb.ProjInW, tb.ProjInB, 1, 0)

	// Reshape [1,C,H,W] → [H*W, C] for attention
	hidden := Reshape4Dto2D(x)

	// Self-attention
	normed := LayerNorm(hidden, tb.Norm1W, tb.Norm1B, 1e-5)
	hidden = Add(hidden, sdAttention(normed, normed, tb.SelfQW, tb.SelfKW, tb.SelfVW, tb.SelfOutW, tb.SelfOutB, C))

	// Cross-attention (key/value from CLIP text embeddings)
	normed = LayerNorm(hidden, tb.Norm2W, tb.Norm2B, 1e-5)
	hidden = Add(hidden, sdAttention(normed, textEmb, tb.CrossQW, tb.CrossKW, tb.CrossVW, tb.CrossOutW, tb.CrossOutB, C))

	// Feed-forward: LayerNorm → GEGLU(linear) → linear
	normed = LayerNorm(hidden, tb.Norm3W, tb.Norm3B, 1e-5)
	ff := Linear(normed, tb.FFLinear1W, tb.FFLinear1B) // [seq, dim*8]
	ff = GEGLU(ff)                                      // [seq, dim*4]
	ff = Linear(ff, tb.FFLinear2W, tb.FFLinear2B)       // [seq, dim]
	hidden = Add(hidden, ff)

	// Reshape back to 4D → proj_out (1x1 conv) → residual
	x = Reshape2Dto4D(hidden, C, H, W)
	x = Conv2d(x, tb.ProjOutW, tb.ProjOutB, 1, 0)

	return Add(x, residual)
}

// sdAttention: multi-head attention with headDim=8 (BK-SDM-Tiny)
// qInput: [seqQ, dim], kvInput: [seqKV, kvDim]
// At 64×64 latent: seqQ=4096, 40 heads. Tiled attention keeps score tiles in L3 cache.
func sdAttention(qInput, kvInput *Tensor, qW, kW, vW, outW, outB *Tensor, dim int) *Tensor {
	seqQ := qInput.Shape[0]
	seqKV := kvInput.Shape[0]
	numHeads := dim / attnHeadDim

	q := Linear(qInput, qW, nil)  // [seqQ, dim]
	k := Linear(kvInput, kW, nil) // [seqKV, dim]
	v := Linear(kvInput, vW, nil) // [seqKV, dim]

	scale := float32(1.0 / math.Sqrt(float64(attnHeadDim)))
	out := NewTensor(seqQ, dim)

	if hasAccel {
		// Fused tiled attention in C: deinterleave + tiled matmul + softmax + reinterleave
		// Static buffers, zero Go allocation. Tile size 256 → 4MB score tile fits in L3.
		tileSize := 256
		if seqQ <= 256 {
			tileSize = seqQ // no tiling needed for small sequences
		}
		accelTiledAttention(q.Data, k.Data, v.Data, out.Data,
			seqQ, seqKV, attnHeadDim, numHeads, dim, scale, tileSize)
	} else {
		// Fallback: scalar attention
		for h := 0; h < numHeads; h++ {
			off := h * attnHeadDim
			scores := make([]float32, seqKV)
			for i := 0; i < seqQ; i++ {
				maxScore := float32(-math.MaxFloat32)
				for j := 0; j < seqKV; j++ {
					sum := float32(0)
					for d := 0; d < attnHeadDim; d++ {
						sum += q.Data[i*dim+off+d] * k.Data[j*dim+off+d]
					}
					scores[j] = sum * scale
					if scores[j] > maxScore {
						maxScore = scores[j]
					}
				}
				sumExp := float32(0)
				for j := range scores {
					scores[j] = float32(math.Exp(float64(scores[j] - maxScore)))
					sumExp += scores[j]
				}
				for j := range scores {
					scores[j] /= sumExp
				}
				for d := 0; d < attnHeadDim; d++ {
					sum := float32(0)
					for j := 0; j < seqKV; j++ {
						sum += scores[j] * v.Data[j*dim+off+d]
					}
					out.Data[i*dim+off+d] = sum
				}
			}
		}
	}

	return Linear(out, outW, outB)
}

// timestepEmbedding: sinusoidal embedding with flip_sin_to_cos=true, freq_shift=0
func timestepEmbedding(timestep, dim int) *Tensor {
	halfDim := dim / 2
	emb := NewTensor(1, dim)
	logMax := math.Log(10000.0)

	for i := 0; i < halfDim; i++ {
		freq := math.Exp(-logMax * float64(i) / float64(halfDim))
		angle := float64(timestep) * freq
		// flip_sin_to_cos: first half = cos, second half = sin
		emb.Data[i] = float32(math.Cos(angle))
		emb.Data[halfDim+i] = float32(math.Sin(angle))
	}
	return emb
}
