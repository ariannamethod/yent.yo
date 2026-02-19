package main

import (
	"fmt"
	"math"
)

// CLIPTextEncoder encodes text tokens to embeddings for the UNet
type CLIPTextEncoder struct {
	TokenEmbed    *Tensor // [49408, 768]
	PosEmbed      *Tensor // [77, 768]
	Layers        []CLIPLayer
	FinalLNWeight *Tensor // [768]
	FinalLNBias   *Tensor // [768]
}

type CLIPLayer struct {
	LN1Weight, LN1Bias *Tensor // [768]
	LN2Weight, LN2Bias *Tensor // [768]
	QWeight, QBias     *Tensor // [768, 768]
	KWeight, KBias     *Tensor // [768, 768]
	VWeight, VBias     *Tensor // [768, 768]
	OutWeight, OutBias *Tensor // [768, 768]
	FC1Weight, FC1Bias *Tensor // [3072, 768]
	FC2Weight, FC2Bias *Tensor // [768, 3072]
}

const (
	clipDim     = 768
	clipHeads   = 12
	clipHeadDim = clipDim / clipHeads
	clipLayers  = 12
	clipMaxSeq  = 77
	clipVocab   = 49408
)

// LoadCLIP loads CLIP text encoder from safetensors
func LoadCLIP(st *SafeTensors) (*CLIPTextEncoder, error) {
	clip := &CLIPTextEncoder{}

	load := func(name string) (*Tensor, error) {
		data, shape, err := st.GetFloat32(name)
		if err != nil {
			return nil, err
		}
		return TensorFrom(data, shape), nil
	}

	var err error
	clip.TokenEmbed, err = load("text_model.embeddings.token_embedding.weight")
	if err != nil {
		return nil, err
	}
	clip.PosEmbed, err = load("text_model.embeddings.position_embedding.weight")
	if err != nil {
		return nil, err
	}
	clip.FinalLNWeight, err = load("text_model.final_layer_norm.weight")
	if err != nil {
		return nil, err
	}
	clip.FinalLNBias, err = load("text_model.final_layer_norm.bias")
	if err != nil {
		return nil, err
	}

	clip.Layers = make([]CLIPLayer, clipLayers)
	for i := 0; i < clipLayers; i++ {
		p := fmt.Sprintf("text_model.encoder.layers.%d.", i)
		l := &clip.Layers[i]
		l.LN1Weight, _ = load(p + "layer_norm1.weight")
		l.LN1Bias, _ = load(p + "layer_norm1.bias")
		l.LN2Weight, _ = load(p + "layer_norm2.weight")
		l.LN2Bias, _ = load(p + "layer_norm2.bias")
		l.QWeight, _ = load(p + "self_attn.q_proj.weight")
		l.QBias, _ = load(p + "self_attn.q_proj.bias")
		l.KWeight, _ = load(p + "self_attn.k_proj.weight")
		l.KBias, _ = load(p + "self_attn.k_proj.bias")
		l.VWeight, _ = load(p + "self_attn.v_proj.weight")
		l.VBias, _ = load(p + "self_attn.v_proj.bias")
		l.OutWeight, _ = load(p + "self_attn.out_proj.weight")
		l.OutBias, _ = load(p + "self_attn.out_proj.bias")
		l.FC1Weight, _ = load(p + "mlp.fc1.weight")
		l.FC1Bias, _ = load(p + "mlp.fc1.bias")
		l.FC2Weight, _ = load(p + "mlp.fc2.weight")
		l.FC2Bias, _ = load(p + "mlp.fc2.bias")
	}

	return clip, nil
}

// Encode runs CLIP text encoder: tokens [seq] → hidden_states [seq, 768]
func (clip *CLIPTextEncoder) Encode(tokens []int) *Tensor {
	seq := len(tokens)

	// Embed tokens + positions
	x := NewTensor(seq, clipDim)
	for i, tok := range tokens {
		for j := 0; j < clipDim; j++ {
			x.Data[i*clipDim+j] = clip.TokenEmbed.Data[tok*clipDim+j] + clip.PosEmbed.Data[i*clipDim+j]
		}
	}

	// Causal attention mask
	causalMask := make([]float32, seq*seq)
	for i := 0; i < seq; i++ {
		for j := 0; j < seq; j++ {
			if j > i {
				causalMask[i*seq+j] = float32(-math.MaxFloat32)
			}
		}
	}

	// Transformer layers
	for _, layer := range clip.Layers {
		// Pre-norm + self-attention
		normed := LayerNorm(x, layer.LN1Weight, layer.LN1Bias, 1e-5)
		attnOut := clipSelfAttention(normed, layer, seq, causalMask)
		x = Add(x, attnOut)

		// Pre-norm + FFN
		normed = LayerNorm(x, layer.LN2Weight, layer.LN2Bias, 1e-5)
		ffOut := Linear(normed, layer.FC1Weight, layer.FC1Bias)
		ffOut = QuickGELU(ffOut)
		ffOut = Linear(ffOut, layer.FC2Weight, layer.FC2Bias)
		x = Add(x, ffOut)
	}

	// Final layer norm
	x = LayerNorm(x, clip.FinalLNWeight, clip.FinalLNBias, 1e-5)
	return x
}

func clipSelfAttention(x *Tensor, l CLIPLayer, seq int, causalMask []float32) *Tensor {
	// Project Q, K, V
	q := Linear(x, l.QWeight, l.QBias) // [seq, 768]
	k := Linear(x, l.KWeight, l.KBias)
	v := Linear(x, l.VWeight, l.VBias)

	// Multi-head attention
	out := NewTensor(seq, clipDim)
	scale := float32(1.0 / math.Sqrt(float64(clipHeadDim)))

	for h := 0; h < clipHeads; h++ {
		offset := h * clipHeadDim
		// Compute attention scores
		scores := NewTensor(seq, seq)
		for i := 0; i < seq; i++ {
			for j := 0; j < seq; j++ {
				sum := float32(0)
				for d := 0; d < clipHeadDim; d++ {
					sum += q.Data[i*clipDim+offset+d] * k.Data[j*clipDim+offset+d]
				}
				scores.Data[i*seq+j] = sum*scale + causalMask[i*seq+j]
			}
		}
		// Softmax
		scores = Softmax(scores)
		// Apply to values
		for i := 0; i < seq; i++ {
			for d := 0; d < clipHeadDim; d++ {
				sum := float32(0)
				for j := 0; j < seq; j++ {
					sum += scores.Data[i*seq+j] * v.Data[j*clipDim+offset+d]
				}
				out.Data[i*clipDim+offset+d] = sum
			}
		}
	}

	// Output projection
	return Linear(out, l.OutWeight, l.OutBias)
}
