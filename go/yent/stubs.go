package yent

// stubs.go — minimal stubs for types not needed in yent.yo
// GammaEssence is unused here but referenced by LlamaModel

type GammaEssence struct {
	EmbedDim  int
	NumTokens int
}

func (g *GammaEssence) ApplyToEmbedding(emb []float32, token int) {
	// no-op in yent.yo context
}
