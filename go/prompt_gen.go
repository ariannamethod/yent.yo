package main

// prompt_gen.go — micro-Yent prompt generator for image generation
//
// Loads micro-Yent (69M, nanollama) and uses it to generate
// text prompts for BK-SDM-Tiny image generation.
// The model generates completions given a seed phrase.

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"yentyo/yent"
)

// PromptGenerator wraps micro-Yent for prompt generation
type PromptGenerator struct {
	model     *yent.LlamaModel
	tokenizer *yent.Tokenizer
	gguf      *yent.GGUFFile
	rng       *rand.Rand
}

// NewPromptGenerator loads micro-Yent from a GGUF file
func NewPromptGenerator(ggufPath string) (*PromptGenerator, error) {
	fmt.Printf("[prompt-gen] loading micro-Yent from %s\n", ggufPath)

	g, err := yent.LoadGGUF(ggufPath)
	if err != nil {
		return nil, fmt.Errorf("load GGUF: %w", err)
	}

	model, err := yent.LoadLlamaModel(g)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}

	tokenizer := yent.NewTokenizer(&g.Meta)

	fmt.Printf("[prompt-gen] micro-Yent loaded: %d layers, %d dim, %d vocab\n",
		model.Config.NumLayers, model.Config.EmbedDim, model.Config.VocabSize)

	return &PromptGenerator{
		model:     model,
		tokenizer: tokenizer,
		gguf:      g,
		rng:       rand.New(rand.NewSource(time.Now().UnixNano())),
	}, nil
}

// Generate creates an image prompt by completing a seed phrase
// seed phrases like: "a painting of", "a photograph of", "graffiti art showing"
func (pg *PromptGenerator) Generate(seedPhrase string, maxTokens int, temperature float32) string {
	tokens := pg.tokenizer.Encode(seedPhrase, false)

	pg.model.Reset()

	// Feed seed tokens
	pos := 0
	for _, tok := range tokens {
		pg.model.Forward(tok, pos)
		pos++
		if pos >= pg.model.Config.SeqLen-1 {
			break
		}
	}

	// Generate completion
	var output []byte
	output = append(output, []byte(seedPhrase)...)

	for i := 0; i < maxTokens; i++ {
		next := pg.sampleTopK(temperature, 40)

		// Stop on EOS or newline (we want single-line prompts)
		if next == pg.tokenizer.EosID {
			break
		}
		piece := pg.tokenizer.DecodeToken(next)
		// Stop on newline — prompt should be one line
		for _, ch := range piece {
			if ch == '\n' {
				goto done
			}
		}

		output = append(output, []byte(piece)...)

		pg.model.Forward(next, pos)
		pos++
		if pos >= pg.model.Config.SeqLen-1 {
			break
		}
	}
done:

	return string(output)
}

// sampleTopK samples from top-k logits
func (pg *PromptGenerator) sampleTopK(temp float32, topK int) int {
	logits := pg.model.State.Logits
	vocab := pg.model.Config.VocabSize

	if temp <= 0 {
		best := 0
		for i := 1; i < vocab; i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return best
	}

	type idxVal struct {
		idx int
		val float32
	}
	top := make([]idxVal, topK)
	for i := 0; i < topK; i++ {
		top[i] = idxVal{-1, -1e30}
	}

	for i := 0; i < vocab; i++ {
		if logits[i] > top[topK-1].val {
			top[topK-1] = idxVal{i, logits[i]}
			for j := topK - 1; j > 0 && top[j].val > top[j-1].val; j-- {
				top[j], top[j-1] = top[j-1], top[j]
			}
		}
	}

	maxVal := top[0].val
	probs := make([]float32, topK)
	var sum float32
	for i := 0; i < topK; i++ {
		if top[i].idx < 0 {
			break
		}
		probs[i] = float32(math.Exp(float64((top[i].val - maxVal) / temp)))
		sum += probs[i]
	}

	r := pg.rng.Float32() * sum
	var cdf float32
	for i := 0; i < topK; i++ {
		cdf += probs[i]
		if r <= cdf {
			return top[i].idx
		}
	}
	return top[0].idx
}

// Free releases the model memory
func (pg *PromptGenerator) Free() {
	pg.model = nil
	pg.tokenizer = nil
	pg.gguf = nil
}
