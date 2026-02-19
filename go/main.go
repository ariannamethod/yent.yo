package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"time"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("yent.yo — Text-to-Image (Pure Go, Zero Dependencies)")
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("  yentyo <sd_model_dir> [prompt] [output.png] [seed] [steps] [latent_size]")
		fmt.Println("  yentyo <sd_model_dir> --yent <micro_yent.gguf> [seed_phrase] [output.png] [seed]")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("  yentyo bk-sdm-tiny \"a cat on a roof\" cat.png 42 25 64")
		fmt.Println("  yentyo bk-sdm-tiny --yent micro-yent-f16.gguf \"a painting of\" auto.png 42")
		os.Exit(0)
	}

	modelDir := os.Args[1]

	// Check for --yent mode
	if len(os.Args) > 2 && os.Args[2] == "--yent" {
		runWithYent(modelDir)
		return
	}

	// Direct prompt mode
	prompt := "a painting of a cat"
	outPath := "yentyo_output.png"
	seed := int64(42)
	numSteps := 25
	latentSize := 64
	guidanceScale := float32(7.5)

	if len(os.Args) > 2 {
		prompt = os.Args[2]
	}
	if len(os.Args) > 3 {
		outPath = os.Args[3]
	}
	if len(os.Args) > 4 {
		fmt.Sscanf(os.Args[4], "%d", &seed)
	}
	if len(os.Args) > 5 {
		fmt.Sscanf(os.Args[5], "%d", &numSteps)
	}
	if len(os.Args) > 6 {
		fmt.Sscanf(os.Args[6], "%d", &latentSize)
	}

	runDiffusion(modelDir, prompt, outPath, seed, numSteps, latentSize, guidanceScale)
}

// runWithYent uses micro-Yent to generate prompt, then runs diffusion
func runWithYent(sdModelDir string) {
	if len(os.Args) < 4 {
		fatal("--yent requires: <micro_yent.gguf> [seed_phrase] [output.png] [seed]")
	}

	yentPath := os.Args[3]
	seedPhrase := "a painting of"
	outPath := "yentyo_auto.png"
	seed := int64(time.Now().UnixNano())

	if len(os.Args) > 4 {
		seedPhrase = os.Args[4]
	}
	if len(os.Args) > 5 {
		outPath = os.Args[5]
	}
	if len(os.Args) > 6 {
		fmt.Sscanf(os.Args[6], "%d", &seed)
	}

	// Phase 0: Generate prompt with micro-Yent
	fmt.Print("\n--- Phase 0: Prompt Generation (micro-Yent) ---\n")
	start := time.Now()

	pg, err := NewPromptGenerator(yentPath)
	if err != nil {
		fatal("prompt generator: %v", err)
	}

	// Generate prompt
	prompt := pg.Generate(seedPhrase, 30, 0.8)
	// Clean up: trim to reasonable length, no trailing spaces
	prompt = strings.TrimSpace(prompt)
	if len(prompt) > 200 {
		prompt = prompt[:200]
	}

	pg.Free()
	runtime.GC()

	fmt.Printf("Generated prompt: %q (%.1fs)\n", prompt, time.Since(start).Seconds())

	// Run diffusion with generated prompt
	runDiffusion(sdModelDir, prompt, outPath, seed, 25, 64, 7.5)
}

func runDiffusion(modelDir, prompt, outPath string, seed int64, numSteps, latentSize int, guidanceScale float32) {
	fmt.Printf("Model: %s\n", modelDir)
	fmt.Printf("Prompt: %q\n", prompt)
	fmt.Printf("Seed: %d, Steps: %d, Guidance: %.1f, Latent: %dx%d\n", seed, numSteps, guidanceScale, latentSize, latentSize)

	// ===== PHASE 1: Text Encoding =====
	fmt.Print("\n--- Phase 1: Text Encoding ---\n")

	fmt.Print("Loading tokenizer... ")
	start := time.Now()
	tokenizer, err := LoadTokenizer(modelDir + "/tokenizer")
	if err != nil {
		fatal("tokenizer: %v", err)
	}
	fmt.Printf("done (%v)\n", time.Since(start))

	condTokens := tokenizer.Encode(prompt)
	uncondTokens := tokenizer.Encode("")
	fmt.Printf("Cond tokens: %v... (len=%d)\n", condTokens[:min(8, len(condTokens))], len(condTokens))

	fmt.Print("Loading CLIP... ")
	start = time.Now()
	clipST, err := OpenSafeTensors(modelDir + "/text_encoder/model.fp16.safetensors")
	if err != nil {
		fatal("clip load: %v", err)
	}
	clipModel, err := LoadCLIP(clipST)
	if err != nil {
		fatal("clip parse: %v", err)
	}
	fmt.Printf("done (%v)\n", time.Since(start))

	fmt.Print("Encoding text... ")
	start = time.Now()
	condEmb := clipModel.Encode(condTokens)
	uncondEmb := clipModel.Encode(uncondTokens)
	fmt.Printf("done (%v)\n", time.Since(start))
	fmt.Printf("  cond_emb[0][:3] = [%.4f, %.4f, %.4f]\n",
		condEmb.Data[0], condEmb.Data[1], condEmb.Data[2])

	// Free CLIP
	clipModel = nil
	clipST = nil
	runtime.GC()
	fmt.Println("CLIP freed")

	// ===== PHASE 2: Diffusion =====
	fmt.Print("\n--- Phase 2: Diffusion ---\n")

	fmt.Print("Loading UNet... ")
	start = time.Now()
	unetST, err := OpenSafeTensors(modelDir + "/unet/diffusion_pytorch_model.fp16.safetensors")
	if err != nil {
		fatal("unet load: %v", err)
	}
	unet, err := LoadUNet(unetST)
	if err != nil {
		fatal("unet parse: %v", err)
	}
	unetST = nil
	runtime.GC()
	fmt.Printf("done (%v)\n", time.Since(start))

	// Scheduler
	sched := NewDDIMScheduler(1000, 0.00085, 0.012)
	timesteps := sched.SetTimesteps(numSteps)
	fmt.Printf("Timesteps (%d): [%d ... %d]\n", len(timesteps), timesteps[0], timesteps[len(timesteps)-1])

	// Initial noise
	latent := randomLatent(1, 4, latentSize, latentSize, seed)
	fmt.Printf("Latent: [%d,%d,%d,%d], range=[%.3f, %.3f]\n",
		latent.Shape[0], latent.Shape[1], latent.Shape[2], latent.Shape[3],
		tensorMin(latent), tensorMax(latent))

	// Diffusion loop
	fmt.Println()
	totalStart := time.Now()
	for step, t := range timesteps {
		stepStart := time.Now()

		noiseUncond := unet.Forward(latent, t, uncondEmb)
		noiseCond := unet.Forward(latent, t, condEmb)

		noisePred := NewTensor(noiseUncond.Shape...)
		for i := range noisePred.Data {
			noisePred.Data[i] = noiseUncond.Data[i] + guidanceScale*(noiseCond.Data[i]-noiseUncond.Data[i])
		}

		latent = sched.Step(noisePred, t, latent)

		fmt.Printf("  Step %d/%d (t=%d): %.1fs\n",
			step+1, numSteps, t, time.Since(stepStart).Seconds())
	}
	fmt.Printf("\nDiffusion: %.1fs total\n", time.Since(totalStart).Seconds())

	unet = nil
	runtime.GC()
	fmt.Println("UNet freed")

	// ===== PHASE 3: VAE Decoding =====
	fmt.Print("\n--- Phase 3: VAE Decoding ---\n")

	latent = Scale(latent, float32(1.0/0.18215))

	fmt.Print("Loading VAE decoder... ")
	start = time.Now()
	vaeST, err := OpenSafeTensors(modelDir + "/vae/diffusion_pytorch_model.fp16.safetensors")
	if err != nil {
		fatal("vae load: %v", err)
	}
	vae, err := LoadVAEDecoder(vaeST)
	if err != nil {
		fatal("vae parse: %v", err)
	}
	vaeST = nil
	runtime.GC()
	fmt.Printf("done (%v)\n", time.Since(start))

	fmt.Print("Decoding... ")
	start = time.Now()
	img := vae.Decode(latent)
	fmt.Printf("done (%v)\n", time.Since(start))
	fmt.Printf("  Output: [%d,%d,%d,%d], range=[%.3f, %.3f]\n",
		img.Shape[0], img.Shape[1], img.Shape[2], img.Shape[3],
		tensorMin(img), tensorMax(img))

	// Save PNG
	fmt.Printf("Saving %s... ", outPath)
	if err := savePNG(img, outPath); err != nil {
		fatal("save: %v", err)
	}
	fmt.Println("done!")
}

func randomLatent(n, c, h, w int, seed int64) *Tensor {
	rng := rand.New(rand.NewSource(seed))
	t := NewTensor(n, c, h, w)
	for i := 0; i < len(t.Data)-1; i += 2 {
		u1 := rng.Float64()
		u2 := rng.Float64()
		for u1 == 0 {
			u1 = rng.Float64()
		}
		r := math.Sqrt(-2 * math.Log(u1))
		theta := 2 * math.Pi * u2
		t.Data[i] = float32(r * math.Cos(theta))
		t.Data[i+1] = float32(r * math.Sin(theta))
	}
	if len(t.Data)%2 == 1 {
		u1 := rng.Float64()
		u2 := rng.Float64()
		for u1 == 0 {
			u1 = rng.Float64()
		}
		t.Data[len(t.Data)-1] = float32(math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2))
	}
	return t
}

func savePNG(tensor *Tensor, path string) error {
	H := tensor.Shape[2]
	W := tensor.Shape[3]
	rgba := image.NewRGBA(image.Rect(0, 0, W, H))

	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			r := tensor.Data[0*H*W+y*W+x]
			g := tensor.Data[1*H*W+y*W+x]
			b := tensor.Data[2*H*W+y*W+x]
			rgba.Set(x, y, color.RGBA{
				R: clampByte((r + 1) / 2),
				G: clampByte((g + 1) / 2),
				B: clampByte((b + 1) / 2),
				A: 255,
			})
		}
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, rgba)
}

func clampByte(v float32) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 1 {
		return 255
	}
	return uint8(v * 255)
}

func tensorMin(t *Tensor) float32 {
	m := t.Data[0]
	for _, v := range t.Data[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func tensorMax(t *Tensor) float32 {
	m := t.Data[0]
	for _, v := range t.Data[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func fatal(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "Error: "+format+"\n", args...)
	os.Exit(1)
}
