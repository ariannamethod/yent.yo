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
	"os"
	"strings"
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
	fmt.Fprintf(os.Stderr, "[prompt-gen] loading micro-Yent from %s\n", ggufPath)

	g, err := yent.LoadGGUF(ggufPath)
	if err != nil {
		return nil, fmt.Errorf("load GGUF: %w", err)
	}

	model, err := yent.LoadLlamaModel(g)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}

	tokenizer := yent.NewTokenizer(&g.Meta)

	fmt.Fprintf(os.Stderr, "[prompt-gen] micro-Yent loaded: %d layers, %d dim, %d vocab\n",
		model.Config.NumLayers, model.Config.EmbedDim, model.Config.VocabSize)

	return &PromptGenerator{
		model:     model,
		tokenizer: tokenizer,
		gguf:      g,
		rng:       rand.New(rand.NewSource(time.Now().UnixNano())),
	}, nil
}

// Mood templates: keyword triggers → visual scene starters
// Each has a visual anchor that forces SD-friendly completion
type moodTemplate struct {
	keywords []string
	starters []string // micro-Yent completes after these
}

var moodTemplates = []moodTemplate{
	{[]string{"sad", "alone", "lonely", "cry", "грустн", "одинок"},
		[]string{
			"a lonely figure sinking into",
			"a wilting flower made of",
			"a cracked mirror reflecting",
		}},
	{[]string{"angry", "hate", "stupid", "fuck", "злой", "бесит", "тупой"},
		[]string{
			"an explosion of broken",
			"a screaming face melting into",
			"fists punching through a wall of",
		}},
	{[]string{"love", "heart", "beautiful", "люблю", "сердце", "красив"},
		[]string{
			"two glitching hearts entangled in",
			"a burning rose growing from",
			"a digital embrace dissolving into",
		}},
	{[]string{"bored", "nothing", "whatever", "скучно", "пофиг"},
		[]string{
			"a yawning void eating",
			"a clock melting over",
			"an empty chair staring at",
		}},
	{[]string{"hello", "hi", "hey", "привет", "здорово"},
		[]string{
			"a punk hand waving from",
			"an eye opening through",
			"a door cracking open revealing",
		}},
	{[]string{"duck", "утк"},
		[]string{
			"an angry rubber duck wearing",
			"a militant duck leading an army of",
			"a duck on fire walking through",
		}},
	{[]string{"cat", "кот", "кош"},
		[]string{
			"a punk cat with a mohawk sitting on",
			"a cat with glowing eyes staring at",
			"a cyberpunk cat hacking into",
		}},
	{[]string{"death", "die", "dead", "смерть", "умер"},
		[]string{
			"a skeleton playing guitar on",
			"death itself wearing",
			"bones growing flowers in",
		}},
}

// Default templates when no mood keyword matches
var defaultStarters = []string{
	"a surreal painting of",
	"a punk portrait of",
	"a strange scene showing",
	"a twisted landscape with",
	"a glitching image of",
}

var styleSuffixes = []string{
	", oil painting, vivid colors, detailed brushwork",
	", bold strokes, surreal atmosphere, dramatic",
	", punk aesthetic, neon glow, expressive",
	", dark symbolic illustration, textured, moody",
	", street art style, spray paint, raw energy",
}

// adaptTemperature adjusts temperature based on input characteristics.
// Boring/short input → higher temp (Yent gets creative to compensate).
// Emotional/long input → lower temp (strong signal, stay focused).
func adaptTemperature(input string, baseTemp float32) float32 {
	lower := strings.ToLower(input)
	words := strings.Fields(input)
	nWords := len(words)

	temp := baseTemp

	// Short/lazy input → crank up creativity
	if nWords <= 3 {
		temp += 0.1
	}
	if nWords <= 1 {
		temp += 0.1
	}

	// Boring/generic input → more chaos
	boring := []string{"hello", "hi", "hey", "ok", "test", "привет", "ок"}
	for _, b := range boring {
		if lower == b {
			temp += 0.15
			break
		}
	}

	// Strong emotion → stay focused
	strong := []string{"hate", "love", "die", "kill", "fuck", "death", "ненавижу", "люблю", "смерть"}
	for _, s := range strong {
		if strings.Contains(lower, s) {
			temp -= 0.1
			break
		}
	}

	// Clamp to [0.5, 1.0]
	if temp < 0.5 {
		temp = 0.5
	}
	if temp > 1.0 {
		temp = 1.0
	}

	return temp
}

// React generates an image prompt as Yent's REACTION to user input.
// Hybrid approach: keyword → visual template → micro-Yent fills details → style suffix
// Temperature adapts to input: boring → more chaos, emotional → more focused.
func (pg *PromptGenerator) React(userInput string, maxTokens int, temperature float32) string {
	// Adapt temperature based on input
	temperature = adaptTemperature(userInput, temperature)

	lower := strings.ToLower(userInput)

	// Find matching mood template
	var starter string
	matched := false
	for _, mt := range moodTemplates {
		for _, kw := range mt.keywords {
			if strings.Contains(lower, kw) {
				starter = mt.starters[pg.rng.Intn(len(mt.starters))]
				matched = true
				break
			}
		}
		if matched {
			break
		}
	}
	if !matched {
		starter = defaultStarters[pg.rng.Intn(len(defaultStarters))]
	}

	// Feed user input as context, then the visual starter
	context := fmt.Sprintf(`"%s" — %s`, userInput, starter)
	tokens := pg.tokenizer.Encode(context, true)

	pg.model.Reset()

	pos := 0
	for _, tok := range tokens {
		pg.model.Forward(tok, pos)
		pos++
		if pos >= pg.model.Config.SeqLen-1 {
			break
		}
	}

	// Collect micro-Yent's completion (visual details)
	var completion []byte
	for i := 0; i < maxTokens; i++ {
		next := pg.sampleTopK(temperature, 40)

		if next == pg.tokenizer.EosID {
			break
		}
		piece := pg.tokenizer.DecodeToken(next)

		stop := false
		for _, ch := range piece {
			if ch == '\n' || ch == '"' {
				stop = true
				break
			}
		}
		if stop {
			break
		}

		completion = append(completion, []byte(piece)...)

		// Stop at sentence end after reasonable length
		n := len(completion)
		if n > 15 && (completion[n-1] == '.' || completion[n-1] == '!' || completion[n-1] == ',') {
			break
		}

		pg.model.Forward(next, pos)
		pos++
		if pos >= pg.model.Config.SeqLen-1 {
			break
		}
	}

	// Combine: starter + completion + style
	detail := strings.TrimSpace(string(completion))
	// Trim trailing punctuation for cleaner concat
	detail = strings.TrimRight(detail, ".,;:!?")

	var result string
	if detail != "" {
		result = starter + " " + detail
	} else {
		result = starter + " chaos and defiance"
	}

	suffix := styleSuffixes[pg.rng.Intn(len(styleSuffixes))]
	return result + suffix
}

// Generate creates an image prompt by completing a seed phrase (legacy mode)
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
