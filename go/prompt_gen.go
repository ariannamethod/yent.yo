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
	// Reusable buffers for sampling (avoid per-token allocations)
	topKBuf  []idxVal
	probsBuf []float32
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

// Quality suffix appended to every prompt — fights SD artifacts
const qualitySuffix = ", clean composition, well-defined forms"

// --- Dissonance-based temperature (inspired by Harmonix/HAiKU) ---
//
// Dissonance = how "strange" the input is relative to what the system knows.
// High dissonance → high temperature → creative chaos.
// Low dissonance → low temperature → focused response.
//
// Four pulse metrics:
//   novelty:       fraction of words unknown to our keyword vocabulary
//   entropy:       unique_words / total_words (word diversity)
//   arousal:       emotional keyword density (high arousal → LESS dissonance)
//   lengthPressure: shorter input → higher pressure
//
// dissonance ∈ [0, 1] → temperature ∈ [0.5, 1.0]

// allKnownWords is the union of all mood keywords + boring + strong words.
// Words the system "recognizes" reduce novelty.
var allKnownWords = func() map[string]bool {
	known := map[string]bool{}
	for _, mt := range moodTemplates {
		for _, kw := range mt.keywords {
			known[kw] = true
		}
	}
	for _, w := range []string{"hello", "hi", "hey", "ok", "test", "привет", "ок",
		"the", "a", "an", "is", "am", "are", "i", "you", "me", "my", "we",
		"it", "to", "of", "and", "in", "on", "for", "so", "do", "not", "no",
		"yes", "but", "or", "if", "at", "by", "up", "out", "all", "just",
		"like", "what", "why", "how", "when", "who", "this", "that", "with",
		"from", "have", "has", "had", "was", "were", "will", "would", "can",
		"could", "should", "want", "need", "know", "think", "feel", "make",
		"go", "get", "see", "say", "tell", "give", "take", "come", "some",
		"very", "really", "much", "too", "more", "about", "your", "our",
		"его", "на", "не", "и", "в", "я", "ты", "мне", "меня"} {
		known[w] = true
	}
	return known
}()

// arousalWords trigger focused (low-dissonance) responses
var arousalWords = map[string]bool{
	"hate": true, "love": true, "die": true, "kill": true, "fuck": true,
	"death": true, "dead": true, "cry": true, "sad": true, "angry": true,
	"beautiful": true, "alone": true, "lonely": true, "miss": true, "hurt": true,
	"ненавижу": true, "люблю": true, "смерть": true, "плачу": true, "больно": true,
}

// computeDissonance measures how "strange" the input is to the system.
// Returns dissonance ∈ [0, 1].
func computeDissonance(input string) float32 {
	lower := strings.ToLower(input)
	words := strings.Fields(lower)
	nWords := len(words)
	if nWords == 0 {
		return 1.0 // empty input = max dissonance
	}

	// Novelty: fraction of words unknown to system
	unknownCount := 0
	for _, w := range words {
		if !allKnownWords[w] {
			unknownCount++
		}
	}
	novelty := float32(unknownCount) / float32(nWords)

	// Entropy: word diversity (unique / total)
	unique := map[string]bool{}
	for _, w := range words {
		unique[w] = true
	}
	entropy := float32(len(unique)) / float32(nWords)

	// Arousal: emotional keyword density (high → focused → low dissonance)
	// Check both exact word match and substring match (Russian stems),
	// but deduplicate to avoid double-counting
	arousalMatched := map[string]bool{}
	for _, w := range words {
		if arousalWords[w] {
			arousalMatched[w] = true
		}
	}
	for aw := range arousalWords {
		if strings.Contains(lower, aw) {
			arousalMatched[aw] = true
		}
	}
	arousal := float32(len(arousalMatched)) / float32(nWords+1)
	if arousal > 1.0 {
		arousal = 1.0
	}

	// Length pressure: shorter → more pressure
	lengthPressure := float32(1.0) / float32(nWords)
	if lengthPressure > 1.0 {
		lengthPressure = 1.0
	}

	// Combine: novelty and length push dissonance up, arousal pulls it down
	dissonance := 0.30*novelty + 0.25*entropy + 0.25*lengthPressure + 0.20*(1.0-arousal)

	// Clamp
	if dissonance < 0 {
		dissonance = 0
	}
	if dissonance > 1 {
		dissonance = 1
	}

	return dissonance
}

// adaptTemperature maps dissonance to temperature.
// Dissonance ∈ [0, 1] → temperature ∈ [0.5, 1.0].
// Inspired by Harmonix/HAiKU pressure system.
func adaptTemperature(input string, baseTemp float32) float32 {
	d := computeDissonance(input)

	// Map: dissonance 0 → 0.5, dissonance 1 → 1.0
	temp := 0.5 + d*0.5

	// Blend with base temp (respect caller's hint)
	temp = 0.6*temp + 0.4*baseTemp

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
	// Compute dissonance and adapt temperature
	dissonance := computeDissonance(userInput)
	temperature = adaptTemperature(userInput, temperature)
	fmt.Fprintf(os.Stderr, "[react] input=%q d=%.2f T=%.2f\n", userInput, dissonance, temperature)

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
	// Capped at 512 bytes to prevent unbounded growth
	var completion []byte
	const maxCompletionBytes = 512
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

		if len(completion)+len(piece) > maxCompletionBytes {
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
	return result + suffix + qualitySuffix
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

// idxVal holds token index + logit value for top-k sampling
type idxVal struct {
	idx int
	val float32
}

// sampleTopK samples from top-k logits (reuses buffers to avoid per-token allocations)
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

	// Reuse or grow buffers
	if cap(pg.topKBuf) < topK {
		pg.topKBuf = make([]idxVal, topK)
		pg.probsBuf = make([]float32, topK)
	}
	top := pg.topKBuf[:topK]
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
	probs := pg.probsBuf[:topK]
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
