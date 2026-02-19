package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"
)

// CLIPTokenizer implements BPE tokenization for CLIP
type CLIPTokenizer struct {
	Vocab  map[string]int
	Merges []MergePair
	BOS    int // 49406
	EOS    int // 49407
	MaxLen int // 77
}

type MergePair struct {
	A, B string
}

// LoadTokenizer loads CLIP BPE tokenizer from directory
func LoadTokenizer(dir string) (*CLIPTokenizer, error) {
	// Load vocab
	vocabData, err := os.ReadFile(dir + "/vocab.json")
	if err != nil {
		return nil, fmt.Errorf("read vocab: %w", err)
	}
	vocab := make(map[string]int)
	if err := json.Unmarshal(vocabData, &vocab); err != nil {
		return nil, fmt.Errorf("parse vocab: %w", err)
	}

	// Load merges
	mergesData, err := os.ReadFile(dir + "/merges.txt")
	if err != nil {
		return nil, fmt.Errorf("read merges: %w", err)
	}
	lines := strings.Split(string(mergesData), "\n")
	var merges []MergePair
	for _, line := range lines {
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		parts := strings.SplitN(line, " ", 2)
		if len(parts) == 2 {
			merges = append(merges, MergePair{A: parts[0], B: parts[1]})
		}
	}

	bos, ok := vocab["<|startoftext|>"]
	if !ok {
		return nil, fmt.Errorf("missing BOS token")
	}
	eos, ok := vocab["<|endoftext|>"]
	if !ok {
		return nil, fmt.Errorf("missing EOS token")
	}

	return &CLIPTokenizer{
		Vocab:  vocab,
		Merges: merges,
		BOS:    bos,
		EOS:    eos,
		MaxLen: 77,
	}, nil
}

// Encode tokenizes text and returns token IDs padded to MaxLen
func (t *CLIPTokenizer) Encode(text string) []int {
	text = strings.ToLower(strings.TrimSpace(text))

	// Split into words
	words := splitIntoWords(text)

	// BPE encode each word
	var tokens []int
	tokens = append(tokens, t.BOS)

	for _, word := range words {
		word = word + "</w>"
		// Initial: each character is a token
		parts := make([]string, 0, len(word))
		i := 0
		for i < len(word) {
			if strings.HasPrefix(word[i:], "</w>") {
				parts = append(parts, "</w>")
				i += 4
			} else {
				parts = append(parts, string(word[i]))
				i++
			}
		}

		// Apply BPE merges
		for _, merge := range t.Merges {
			newParts := make([]string, 0, len(parts))
			j := 0
			for j < len(parts) {
				if j+1 < len(parts) && parts[j] == merge.A && parts[j+1] == merge.B {
					newParts = append(newParts, merge.A+merge.B)
					j += 2
				} else {
					newParts = append(newParts, parts[j])
					j++
				}
			}
			parts = newParts
			if len(parts) == 1 {
				break
			}
		}

		// Look up token IDs
		for _, part := range parts {
			if id, ok := t.Vocab[part]; ok {
				tokens = append(tokens, id)
			}
		}
	}

	tokens = append(tokens, t.EOS)

	// Pad or truncate to MaxLen
	if len(tokens) > t.MaxLen {
		tokens = tokens[:t.MaxLen]
		tokens[t.MaxLen-1] = t.EOS
	}
	for len(tokens) < t.MaxLen {
		tokens = append(tokens, t.EOS)
	}

	return tokens
}

func splitIntoWords(text string) []string {
	var words []string
	var current []rune
	for _, r := range text {
		if unicode.IsSpace(r) || unicode.IsPunct(r) {
			if len(current) > 0 {
				words = append(words, string(current))
				current = current[:0]
			}
			if unicode.IsPunct(r) {
				words = append(words, string(r))
			}
		} else {
			current = append(current, r)
		}
	}
	if len(current) > 0 {
		words = append(words, string(current))
	}
	return words
}
