package main

// prompt_gen.go — micro-Yent prompt generator for image generation
//
// Loads micro-Yent (nanollama) and uses it to generate
// text prompts for BK-SDM-Tiny image generation.
//
// Dissonance system adapted from Harmonix/HAiKU:
//   Trigram-based Jaccard similarity, pulse adjustments,
//   boredom detection, cloud morphing.
//   Temperature range: [0.3, 1.5] (HAiKU-level)

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

	// HAiKU cloud: word weights that grow/decay across interactions
	cloud        map[string]float32
	lastTrigrams map[string]bool // previous interaction trigrams (for Jaccard)
	boredomCount int             // consecutive low-dissonance interactions
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
		cloud:     make(map[string]float32),
	}, nil
}

// --- Oppositional reaction templates ---
// Yent REACTS AGAINST the input, not describes it.
// "утка" → "сам ты утка" energy → visual pushback

type reactionTemplate struct {
	keywords []string
	starters []string // oppositional visual reactions
}

var reactionTemplates = []reactionTemplate{
	{[]string{"sad", "alone", "lonely", "cry", "грустн", "одинок"},
		[]string{
			"a mirror throwing your sadness back at",
			"tears that refuse to fall, frozen in",
			"a hand slapping away the self-pity from",
		}},
	{[]string{"angry", "hate", "stupid", "fuck", "злой", "бесит", "тупой"},
		[]string{
			"a hand pushing back through broken",
			"your own rage reflected in shattered",
			"the middle finger of the universe pointing at",
		}},
	{[]string{"love", "heart", "beautiful", "люблю", "сердце", "красив"},
		[]string{
			"love eating itself alive in",
			"a heart that bites the hand reaching for",
			"beauty rotting from the inside through",
		}},
	{[]string{"bored", "nothing", "whatever", "скучно", "пофиг"},
		[]string{
			"your boredom staring back with contempt from",
			"the void yawning at your attempt to fill",
			"nothing mocking the one who summoned",
		}},
	{[]string{"hello", "hi", "hey", "привет", "здорово"},
		[]string{
			"an eye that doesn't want to see you opening through",
			"a door slamming shut in the face of",
			"a greeting that curdles into",
		}},
	{[]string{"duck", "утк"},
		[]string{
			"the duck judging you harder than you judged",
			"a bird that knows more than you waddling through",
			"your own reflection quacking back from",
		}},
	{[]string{"cat", "кот", "кош"},
		[]string{
			"a cat that has already forgotten you staring through",
			"eyes that see through your pretense glowing in",
			"the indifference of something that never needed you sitting in",
		}},
	{[]string{"death", "die", "dead", "смерть", "умер"},
		[]string{
			"death laughing at your fear of",
			"bones dancing on the grave of your certainty in",
			"the dead refusing to stay dead crawling through",
		}},
}

// Default oppositional starters — when no keyword matches
var defaultStarters = []string{
	"a mirror cracking under the weight of",
	"the wrong answer to a question nobody asked painted in",
	"your words dissolving before they reach",
	"a wall that heard everything and says nothing in",
	"the shape of what you meant but couldn't say standing in",
}

// Style suffixes — match known styles BK-SDM-Tiny handles well
var styleSuffixes = []string{
	", Picasso late period, distorted figures, bold lines",
	", social realism, workers, dramatic lighting",
	", street art, spray paint, concrete wall, graffiti",
	", caricature, exaggerated features, ink and wash",
	", propaganda poster, bold red and black, stark contrast",
	", oil painting, thick impasto, raw brushstrokes",
}

// ═══════════════════════════════════════════════════════════════
// HAiKU-level Dissonance System
// Adapted from github.com/ariannamethod/harmonix/haiku
// ═══════════════════════════════════════════════════════════════

// extractTrigrams extracts character trigrams from text (HAiKU-style)
func extractTrigrams(text string) map[string]bool {
	lower := strings.ToLower(text)
	words := strings.Fields(lower)
	trigrams := make(map[string]bool)

	// Word-level trigrams (sliding window of 3 words)
	for i := 0; i+2 < len(words); i++ {
		tri := words[i] + " " + words[i+1] + " " + words[i+2]
		trigrams[tri] = true
	}
	// Also add bigrams for short inputs
	for i := 0; i+1 < len(words); i++ {
		bi := words[i] + " " + words[i+1]
		trigrams[bi] = true
	}
	// Single words as fallback
	for _, w := range words {
		trigrams[w] = true
	}

	return trigrams
}

// jaccardSimilarity computes Jaccard similarity between two trigram sets
func jaccardSimilarity(a, b map[string]bool) float32 {
	if len(a) == 0 && len(b) == 0 {
		return 0
	}
	intersection := 0
	for k := range a {
		if b[k] {
			intersection++
		}
	}
	union := len(a) + len(b) - intersection
	if union == 0 {
		return 0
	}
	return float32(intersection) / float32(union)
}

// arousalWords trigger focused (low-dissonance) responses
var arousalWords = map[string]bool{
	"hate": true, "love": true, "die": true, "kill": true, "fuck": true,
	"death": true, "dead": true, "cry": true, "sad": true, "angry": true,
	"beautiful": true, "alone": true, "lonely": true, "miss": true, "hurt": true,
	"pain": true, "suffer": true, "burn": true, "scream": true, "bleed": true,
	"ненавижу": true, "люблю": true, "смерть": true, "плачу": true, "больно": true,
	"горю": true, "кричу": true, "страдаю": true,
}

// PulseSnapshot — lightweight state vector (HAiKU)
type PulseSnapshot struct {
	Novelty float32 // how new is the input (1 - word overlap)
	Arousal float32 // emotional keyword density
	Entropy float32 // word diversity
}

// computeDissonance measures how "strange" the input is to the system.
// HAiKU-level: trigram Jaccard + pulse adjustments + boredom detection.
// Returns dissonance ∈ [0, 1] and pulse snapshot.
func (pg *PromptGenerator) computeDissonance(input string) (float32, PulseSnapshot) {
	lower := strings.ToLower(input)
	words := strings.Fields(lower)
	nWords := len(words)
	if nWords == 0 {
		return 1.0, PulseSnapshot{Novelty: 1.0, Entropy: 1.0}
	}

	// Extract trigrams
	trigrams := extractTrigrams(input)

	// Base dissonance: 1 - Jaccard similarity with previous interaction
	var similarity float32
	if pg.lastTrigrams != nil {
		similarity = jaccardSimilarity(trigrams, pg.lastTrigrams)
	}
	dissonance := 1.0 - similarity

	// Pulse: novelty (cloud-based, not static word list)
	unknownCount := 0
	for _, w := range words {
		if pg.cloud[w] < 0.1 { // word not in cloud or decayed
			unknownCount++
		}
	}
	novelty := float32(unknownCount) / float32(nWords)

	// Pulse: entropy (word diversity)
	unique := make(map[string]bool)
	for _, w := range words {
		unique[w] = true
	}
	entropy := float32(len(unique)) / float32(nWords)

	// Pulse: arousal (emotional keyword density)
	arousalCount := 0
	for _, w := range words {
		if arousalWords[w] {
			arousalCount++
		}
	}
	// Also check substrings for Russian stems
	for aw := range arousalWords {
		if strings.Contains(lower, aw) {
			arousalCount++
		}
	}
	arousal := float32(arousalCount) / float32(nWords+1)
	if arousal > 1.0 {
		arousal = 1.0
	}

	pulse := PulseSnapshot{
		Novelty: novelty,
		Arousal: arousal,
		Entropy: entropy,
	}

	// HAiKU pulse adjustments
	if entropy > 0.7 {
		dissonance *= 1.2 // high entropy → more dissonance
	}
	if arousal > 0.6 {
		dissonance *= 1.15 // high arousal → more dissonance (unlike old code!)
	}
	if novelty > 0.7 {
		dissonance *= 1.1 // high novelty → more dissonance
	}

	// Trigram overlap reduces dissonance (system "recognizes" patterns)
	trigramOverlap := 0
	if pg.lastTrigrams != nil {
		for k := range trigrams {
			if pg.lastTrigrams[k] {
				trigramOverlap++
			}
		}
	}
	if trigramOverlap > 0 {
		dissonance *= 0.7
	}

	// Boredom detection: repeated low dissonance → force creativity
	if dissonance < 0.3 {
		pg.boredomCount++
		if pg.boredomCount >= 2 {
			// Boredom penalty: force high dissonance
			dissonance = 0.7 + float32(pg.boredomCount)*0.1
			fmt.Fprintf(os.Stderr, "[dissonance] BOREDOM detected (%d repeats), forcing d=%.2f\n",
				pg.boredomCount, dissonance)
		}
	} else {
		pg.boredomCount = 0
	}

	// Clamp
	if dissonance < 0 {
		dissonance = 0
	}
	if dissonance > 1 {
		dissonance = 1
	}

	// Cloud morphing: active words grow, all words decay
	for _, w := range words {
		pg.cloud[w] = pg.cloud[w]*1.1 + 0.1 // active: boost
	}
	for w, v := range pg.cloud {
		pg.cloud[w] = v * 0.99 // dormant: decay
		if pg.cloud[w] < 0.01 {
			delete(pg.cloud, w) // garbage collect dead words
		}
	}

	// Store trigrams for next interaction
	pg.lastTrigrams = trigrams

	return dissonance, pulse
}

// adaptTemperature maps dissonance to temperature.
// HAiKU range: dissonance ∈ [0, 1] → temperature ∈ [0.3, 1.5]
func (pg *PromptGenerator) adaptTemperature(input string, baseTemp float32) float32 {
	d, _ := pg.computeDissonance(input)

	// HAiKU mapping: d=0 → T=0.3, d=1 → T=1.5
	temp := 0.3 + d*1.2

	// Blend with base temp (40% caller hint)
	temp = 0.6*temp + 0.4*float32(baseTemp)

	// Clamp to HAiKU range
	if temp < 0.3 {
		temp = 0.3
	}
	if temp > 1.5 {
		temp = 1.5
	}

	return temp
}

// React generates an image prompt as Yent's REACTION to user input.
// Oppositional: Yent pushes back, not describes.
// Temperature adapts via HAiKU dissonance.
func (pg *PromptGenerator) React(userInput string, maxTokens int, temperature float32) string {
	// Compute dissonance and adapt temperature
	dissonance, pulse := pg.computeDissonance(userInput)
	temperature = pg.adaptTemperature(userInput, temperature)
	fmt.Fprintf(os.Stderr, "[react] input=%q d=%.2f T=%.2f pulse=[n=%.2f a=%.2f e=%.2f] boredom=%d\n",
		userInput, dissonance, temperature, pulse.Novelty, pulse.Arousal, pulse.Entropy, pg.boredomCount)

	lower := strings.ToLower(userInput)

	// Find matching reaction template (oppositional)
	var starter string
	matched := false
	for _, rt := range reactionTemplates {
		for _, kw := range rt.keywords {
			if strings.Contains(lower, kw) {
				starter = rt.starters[pg.rng.Intn(len(rt.starters))]
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

	// Feed user input as context with oppositional framing + visual grounding
	context := fmt.Sprintf(`Describe a painting. Be specific: people, objects, colors, actions. No abstractions.
"%s" — Yent reacts with sarcasm and drama: %s`, userInput, starter)
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

// Roast generates a verbal reaction to mock the user (for commentator role)
func (pg *PromptGenerator) Roast(userInput string, maxTokens int, temperature float32) string {
	context := fmt.Sprintf(`User said: "%s"
Yent (cynical, mocking): `, userInput)
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

	var output []byte
	for i := 0; i < maxTokens; i++ {
		next := pg.sampleTopK(temperature, 40)

		if next == pg.tokenizer.EosID {
			break
		}
		piece := pg.tokenizer.DecodeToken(next)

		stop := false
		for _, ch := range piece {
			if ch == '\n' {
				stop = true
				break
			}
		}
		if stop && len(output) > 10 {
			break
		}

		output = append(output, []byte(piece)...)

		if len(output) > 300 {
			break
		}

		pg.model.Forward(next, pos)
		pos++
		if pos >= pg.model.Config.SeqLen-1 {
			break
		}
	}

	return strings.TrimSpace(string(output))
}

// Generate creates an image prompt by completing a seed phrase (legacy mode)
func (pg *PromptGenerator) Generate(seedPhrase string, maxTokens int, temperature float32) string {
	tokens := pg.tokenizer.Encode(seedPhrase, false)

	pg.model.Reset()

	pos := 0
	for _, tok := range tokens {
		pg.model.Forward(tok, pos)
		pos++
		if pos >= pg.model.Config.SeqLen-1 {
			break
		}
	}

	var output []byte
	output = append(output, []byte(seedPhrase)...)

	for i := 0; i < maxTokens; i++ {
		next := pg.sampleTopK(temperature, 40)

		if next == pg.tokenizer.EosID {
			break
		}
		piece := pg.tokenizer.DecodeToken(next)
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
