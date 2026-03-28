package main

// server.go — HTTP server for yent.yo web UI
//
// Endpoints:
//   GET  /           — serves ui.html
//   GET  /health     — model info
//   POST /react      — user input → dual yent reaction + image generation
//   GET  /image/:id  — serve generated images

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image/png"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// Server holds the dual yent and SD model references
type Server struct {
	dy         *DualYent
	sdModelDir string
	mu         sync.Mutex // serialize generation requests
	rng        *rand.Rand
	images     map[string][]byte // id → PNG bytes (in-memory cache)
	imagesMu   sync.RWMutex
}

// ReactRequest is the JSON body for /react
type ReactRequest struct {
	Input       string  `json:"input"`
	Temperature float64 `json:"temperature,omitempty"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
}

// ReactResponse is the JSON response from /react
type ReactResponse struct {
	Prompt     string  `json:"prompt"`
	YentWords  string  `json:"yent_words"`
	Roast      string  `json:"roast"`
	ArtistID   string  `json:"artist_id"`
	ImageURL   string  `json:"image_url,omitempty"`
	ImageB64   string  `json:"image_b64,omitempty"`
	Dissonance float64 `json:"dissonance"`
	Temp       float64 `json:"temperature"`
	ElapsedMs  int64   `json:"elapsed_ms"`
}

// HealthResponse is the JSON response from /health
type HealthResponse struct {
	Version string `json:"version"`
	ModelA  string `json:"model_a"`
	ModelB  string `json:"model_b"`
	SDModel string `json:"sd_model"`
	Ready   bool   `json:"ready"`
}

func startServer(sdModelDir, microPath, nanoPath, port string) {
	fmt.Fprintf(os.Stderr, "[server] loading dual yent...\n")

	dy, err := NewDualYent(microPath, nanoPath)
	if err != nil {
		fatal("dual yent: %v", err)
	}

	srv := &Server{
		dy:         dy,
		sdModelDir: sdModelDir,
		rng:        rand.New(rand.NewSource(time.Now().UnixNano())),
		images:     make(map[string][]byte),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", srv.handleUI)
	mux.HandleFunc("/health", srv.handleHealth)
	mux.HandleFunc("/react", srv.handleReact)
	mux.HandleFunc("/image/", srv.handleImage)

	addr := ":" + port
	fmt.Fprintf(os.Stderr, "[server] listening on http://localhost%s\n", addr)
	fmt.Fprintf(os.Stderr, "[server] SD model: %s\n", sdModelDir)
	fmt.Fprintf(os.Stderr, "[server] ready.\n")

	if err := http.ListenAndServe(addr, mux); err != nil {
		fatal("server: %v", err)
	}
}

func (s *Server) handleUI(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(uiHTML))
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	resp := HealthResponse{
		Version: yentYoVersion,
		ModelA:  fmt.Sprintf("%d layers, %d dim", s.dy.A.model.Config.NumLayers, s.dy.A.model.Config.EmbedDim),
		ModelB:  fmt.Sprintf("%d layers, %d dim", s.dy.B.model.Config.NumLayers, s.dy.B.model.Config.EmbedDim),
		SDModel: s.sdModelDir,
		Ready:   true,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleReact(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req ReactRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json: "+err.Error(), http.StatusBadRequest)
		return
	}

	if req.Input == "" {
		http.Error(w, "input required", http.StatusBadRequest)
		return
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = 30
	}
	if req.Temperature <= 0 {
		req.Temperature = 0.8
	}

	// Serialize generation (models aren't thread-safe)
	s.mu.Lock()
	defer s.mu.Unlock()

	start := time.Now()

	// Dual yent react
	result := s.dy.React(req.Input, req.MaxTokens, float32(req.Temperature))

	// Compute dissonance for display
	d, _ := s.dy.A.computeDissonance(req.Input)
	temp := s.dy.A.adaptTemperature(req.Input, float32(req.Temperature))

	resp := ReactResponse{
		Prompt:     result.Prompt,
		YentWords:  result.YentWords,
		Roast:      result.Roast,
		ArtistID:   result.ArtistID,
		Dissonance: float64(d),
		Temp:       float64(temp),
		ElapsedMs:  time.Since(start).Milliseconds(),
	}

	// Try to generate image (if SD model available)
	imgData := s.tryGenerateImage(result.Prompt)
	if imgData != nil {
		// Store and return as base64
		id := fmt.Sprintf("%d", time.Now().UnixNano())
		s.imagesMu.Lock()
		s.images[id] = imgData
		s.imagesMu.Unlock()

		resp.ImageURL = "/image/" + id
		resp.ImageB64 = base64.StdEncoding.EncodeToString(imgData)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleImage(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/image/")
	s.imagesMu.RLock()
	data, ok := s.images[id]
	s.imagesMu.RUnlock()

	if !ok {
		http.NotFound(w, r)
		return
	}

	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Cache-Control", "max-age=3600")
	w.Write(data)
}

// tryGenerateImage attempts diffusion. Returns PNG bytes or nil.
func (s *Server) tryGenerateImage(prompt string) []byte {
	// Check if SD model directory exists and has tokenizer
	tokDir := s.sdModelDir + "/tokenizer/vocab.json"
	if _, err := os.Stat(tokDir); err != nil {
		fmt.Fprintf(os.Stderr, "[server] SD model not available (%s), skipping image generation\n", s.sdModelDir)
		return nil
	}

	prompt = strings.TrimSpace(prompt)
	if len(prompt) > 200 {
		prompt = prompt[:200]
	}

	seed := s.rng.Int63()
	tmpPath := fmt.Sprintf("/tmp/yentyo_%d.png", time.Now().UnixNano())
	defer os.Remove(tmpPath)

	// Run diffusion — this may call fatal(), so we need to be careful
	// For now, only run if we verified the model exists above
	runDiffusion(s.sdModelDir, prompt, tmpPath, seed, 30, 64, 7.5)

	data, err := os.ReadFile(tmpPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[server] no image generated: %v\n", err)
		return nil
	}
	return data
}

// pngToBytes encodes an image to PNG bytes (for in-memory responses)
func pngToBytes(img interface{ Bounds() interface{ Dx() int } }) []byte {
	return nil // fallback — actual encoding happens in tryGenerateImage
}

// Unused but kept for potential streaming
var _ = png.Encode