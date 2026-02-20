package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"os"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// ORTPipeline runs BK-SDM-Tiny inference via ONNX Runtime (GPU or CPU).
// Zero PyTorch dependency — only needs .onnx files + libonnxruntime.so
type ORTPipeline struct {
	clipSession *ort.DynamicAdvancedSession
	unetSession *ort.DynamicAdvancedSession
	vaeSession  *ort.DynamicAdvancedSession
	scheduler   *DDIMScheduler
	tokenizer   *Tokenizer
}

// NewORTPipeline loads all ONNX models and creates inference sessions.
// onnxDir should contain: clip_text_encoder.onnx, unet.onnx, vae_decoder.onnx
// ortLibPath is path to libonnxruntime.so (or .dylib)
// useCUDA enables GPU acceleration
func NewORTPipeline(onnxDir, modelDir, ortLibPath string, useCUDA bool) (*ORTPipeline, error) {
	// Initialize ONNX Runtime
	ort.SetSharedLibraryPath(ortLibPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("ORT init: %w", err)
	}

	// Session options
	var sessionOpts []ort.NewAdvancedSessionOption
	if useCUDA {
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err != nil {
			fmt.Printf("Warning: CUDA not available (%v), falling back to CPU\n", err)
		} else {
			defer cudaOpts.Destroy()
			sessionOpts = append(sessionOpts, ort.WithCUDAProviderOptions(cudaOpts))
		}
	}

	p := &ORTPipeline{}

	// Load tokenizer
	fmt.Print("Loading tokenizer... ")
	start := time.Now()
	tok, err := LoadTokenizer(modelDir + "/tokenizer")
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}
	p.tokenizer = tok
	fmt.Printf("done (%v)\n", time.Since(start))

	// Load CLIP
	fmt.Print("Loading CLIP ONNX... ")
	start = time.Now()
	p.clipSession, err = ort.NewDynamicAdvancedSession(
		onnxDir+"/clip_text_encoder.onnx",
		[]string{"input_ids"},
		[]string{"last_hidden_state", "pooler_output"},
		sessionOpts...,
	)
	if err != nil {
		return nil, fmt.Errorf("CLIP session: %w", err)
	}
	fmt.Printf("done (%v)\n", time.Since(start))

	// Load UNet
	fmt.Print("Loading UNet ONNX... ")
	start = time.Now()
	p.unetSession, err = ort.NewDynamicAdvancedSession(
		onnxDir+"/unet.onnx",
		[]string{"sample", "timestep", "encoder_hidden_states"},
		[]string{"out_sample"},
		sessionOpts...,
	)
	if err != nil {
		return nil, fmt.Errorf("UNet session: %w", err)
	}
	fmt.Printf("done (%v)\n", time.Since(start))

	// Load VAE decoder
	fmt.Print("Loading VAE ONNX... ")
	start = time.Now()
	p.vaeSession, err = ort.NewDynamicAdvancedSession(
		onnxDir+"/vae_decoder.onnx",
		[]string{"latent_sample"},
		[]string{"sample"},
		sessionOpts...,
	)
	if err != nil {
		return nil, fmt.Errorf("VAE session: %w", err)
	}
	fmt.Printf("done (%v)\n", time.Since(start))

	// Scheduler
	p.scheduler = NewDDIMScheduler(1000, 0.00085, 0.012)

	return p, nil
}

// Generate creates an image from a text prompt.
func (p *ORTPipeline) Generate(prompt string, seed int64, numSteps, latentSize int, guidanceScale float32, outPath string) error {
	fmt.Printf("\nPrompt: %q\n", prompt)
	fmt.Printf("Seed: %d, Steps: %d, Guidance: %.1f, Latent: %dx%d\n",
		seed, numSteps, guidanceScale, latentSize, latentSize)

	// Phase 1: Text encoding
	fmt.Print("\n--- Phase 1: Text Encoding ---\n")
	start := time.Now()

	condTokens := p.tokenizer.Encode(prompt)
	uncondTokens := p.tokenizer.Encode("")

	condEmb, err := p.encodeText(condTokens)
	if err != nil {
		return fmt.Errorf("cond encoding: %w", err)
	}
	uncondEmb, err := p.encodeText(uncondTokens)
	if err != nil {
		return fmt.Errorf("uncond encoding: %w", err)
	}
	fmt.Printf("  Text encoding: %v\n", time.Since(start))
	fmt.Printf("  cond_emb[0][:3] = [%.4f, %.4f, %.4f]\n",
		condEmb[0], condEmb[1], condEmb[2])

	// Phase 2: Diffusion
	fmt.Print("\n--- Phase 2: Diffusion ---\n")
	timesteps := p.scheduler.SetTimesteps(numSteps)
	fmt.Printf("Timesteps (%d): [%d ... %d]\n", len(timesteps), timesteps[0], timesteps[len(timesteps)-1])

	// Initial noise
	latent := makeNoise(1, 4, latentSize, latentSize, seed)

	totalStart := time.Now()
	for step, t := range timesteps {
		stepStart := time.Now()

		// UNet forward: unconditional + conditional
		noiseUncond, err := p.runUNet(latent, int64(t), uncondEmb, latentSize)
		if err != nil {
			return fmt.Errorf("unet uncond step %d: %w", step, err)
		}
		noiseCond, err := p.runUNet(latent, int64(t), condEmb, latentSize)
		if err != nil {
			return fmt.Errorf("unet cond step %d: %w", step, err)
		}

		// Classifier-free guidance
		noisePred := make([]float32, len(noiseUncond))
		for i := range noisePred {
			noisePred[i] = noiseUncond[i] + guidanceScale*(noiseCond[i]-noiseUncond[i])
		}

		// Scheduler step
		latent = p.schedulerStep(noisePred, t, latent, latentSize)

		fmt.Printf("  Step %d/%d (t=%d): %.1fs\n",
			step+1, numSteps, t, time.Since(stepStart).Seconds())
	}
	fmt.Printf("\nDiffusion: %.1fs total\n", time.Since(totalStart).Seconds())

	// Phase 3: VAE Decode
	fmt.Print("\n--- Phase 3: VAE Decoding ---\n")

	// Scale latent for VAE: latent / 0.18215
	scaledLatent := make([]float32, len(latent))
	for i := range latent {
		scaledLatent[i] = latent[i] / 0.18215
	}

	start = time.Now()
	imgData, imgH, imgW, err := p.decodeVAE(scaledLatent, latentSize)
	if err != nil {
		return fmt.Errorf("VAE decode: %w", err)
	}
	fmt.Printf("  VAE decode: %v\n", time.Since(start))
	fmt.Printf("  Output: [1,3,%d,%d]\n", imgH, imgW)

	// Save PNG
	fmt.Printf("Saving %s... ", outPath)
	if err := saveORTPNG(imgData, imgH, imgW, outPath); err != nil {
		return fmt.Errorf("save: %w", err)
	}
	fmt.Println("done!")

	return nil
}

// encodeText runs CLIP text encoder on token IDs
func (p *ORTPipeline) encodeText(tokens []int) ([]float32, error) {
	// Convert to int64 for ONNX
	tokenIDs := make([]int64, len(tokens))
	for i, t := range tokens {
		tokenIDs[i] = int64(t)
	}

	inputTensor, err := ort.NewTensor(ort.NewShape(1, int64(len(tokens))), tokenIDs)
	if err != nil {
		return nil, fmt.Errorf("input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputs, err := p.clipSession.Run([]ort.ArbitraryTensor{inputTensor})
	if err != nil {
		return nil, fmt.Errorf("CLIP run: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			o.Destroy()
		}
	}()

	// First output is last_hidden_state [1, 77, 512]
	hiddenState, ok := outputs[0].(*ort.Tensor[float16])
	if ok {
		// fp16 output — convert to float32
		data := hiddenState.GetData()
		result := make([]float32, len(data))
		for i, v := range data {
			result[i] = float32(v)
		}
		return result, nil
	}
	// Try float32
	hiddenState32, ok := outputs[0].(*ort.Tensor[float32])
	if ok {
		data := hiddenState32.GetData()
		result := make([]float32, len(data))
		copy(result, data)
		return result, nil
	}

	return nil, fmt.Errorf("unexpected CLIP output type")
}

// runUNet runs one UNet forward pass
func (p *ORTPipeline) runUNet(latent []float32, timestep int64, textEmb []float32, latentSize int) ([]float32, error) {
	// Convert latent to fp16 for ONNX
	latentFp16 := float32ToFloat16(latent)
	sampleTensor, err := ort.NewTensor(
		ort.NewShape(1, 4, int64(latentSize), int64(latentSize)),
		latentFp16,
	)
	if err != nil {
		return nil, fmt.Errorf("sample tensor: %w", err)
	}
	defer sampleTensor.Destroy()

	tsTensor, err := ort.NewTensor(ort.NewShape(1), []int64{timestep})
	if err != nil {
		return nil, fmt.Errorf("timestep tensor: %w", err)
	}
	defer tsTensor.Destroy()

	embFp16 := float32ToFloat16(textEmb)
	embTensor, err := ort.NewTensor(ort.NewShape(1, 77, 768), embFp16)
	if err != nil {
		return nil, fmt.Errorf("emb tensor: %w", err)
	}
	defer embTensor.Destroy()

	outputs, err := p.unetSession.Run([]ort.ArbitraryTensor{sampleTensor, tsTensor, embTensor})
	if err != nil {
		return nil, fmt.Errorf("UNet run: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			o.Destroy()
		}
	}()

	// Output is [1, 4, latentSize, latentSize] in fp16
	outFp16, ok := outputs[0].(*ort.Tensor[float16])
	if ok {
		data := outFp16.GetData()
		result := make([]float32, len(data))
		for i, v := range data {
			result[i] = float32(v)
		}
		return result, nil
	}
	out32, ok := outputs[0].(*ort.Tensor[float32])
	if ok {
		data := out32.GetData()
		result := make([]float32, len(data))
		copy(result, data)
		return result, nil
	}

	return nil, fmt.Errorf("unexpected UNet output type")
}

// decodeVAE runs the VAE decoder
func (p *ORTPipeline) decodeVAE(latent []float32, latentSize int) ([]float32, int, int, error) {
	latentFp16 := float32ToFloat16(latent)
	inputTensor, err := ort.NewTensor(
		ort.NewShape(1, 4, int64(latentSize), int64(latentSize)),
		latentFp16,
	)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("VAE input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputs, err := p.vaeSession.Run([]ort.ArbitraryTensor{inputTensor})
	if err != nil {
		return nil, 0, 0, fmt.Errorf("VAE run: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			o.Destroy()
		}
	}()

	// Output is [1, 3, H, W] — image
	imgH := latentSize * 8
	imgW := latentSize * 8

	outFp16, ok := outputs[0].(*ort.Tensor[float16])
	if ok {
		data := outFp16.GetData()
		result := make([]float32, len(data))
		for i, v := range data {
			result[i] = float32(v)
		}
		return result, imgH, imgW, nil
	}
	out32, ok := outputs[0].(*ort.Tensor[float32])
	if ok {
		data := out32.GetData()
		result := make([]float32, len(data))
		copy(result, data)
		return result, imgH, imgW, nil
	}

	return nil, 0, 0, fmt.Errorf("unexpected VAE output type")
}

// schedulerStep applies DDIM step on flat float32 arrays
func (p *ORTPipeline) schedulerStep(noisePred []float32, timestep int, sample []float32, latentSize int) []float32 {
	// Reuse the existing scheduler logic
	t := NewTensor(1, 4, latentSize, latentSize)
	copy(t.Data, sample)
	np := NewTensor(1, 4, latentSize, latentSize)
	copy(np.Data, noisePred)
	result := p.scheduler.Step(np, timestep, t)
	return result.Data
}

func (p *ORTPipeline) Destroy() {
	if p.clipSession != nil {
		p.clipSession.Destroy()
	}
	if p.unetSession != nil {
		p.unetSession.Destroy()
	}
	if p.vaeSession != nil {
		p.vaeSession.Destroy()
	}
	ort.DestroyEnvironment()
}

// ---- Helpers ----

// float16 type for onnxruntime_go
type float16 = uint16

// float32ToFloat16 converts float32 slice to IEEE 754 fp16
func float32ToFloat16(data []float32) []float16 {
	result := make([]float16, len(data))
	for i, v := range data {
		result[i] = f32ToF16(v)
	}
	return result
}

// f32ToF16 converts a single float32 to float16 (IEEE 754)
func f32ToF16(f float32) float16 {
	bits := math.Float32bits(f)
	sign := (bits >> 31) & 1
	exp := int((bits>>23)&0xFF) - 127
	frac := bits & 0x7FFFFF

	if exp > 15 {
		// Overflow → infinity
		return float16(sign<<15 | 0x7C00)
	}
	if exp < -14 {
		// Underflow → zero (or denorm, but we'll just zero)
		return float16(sign << 15)
	}

	// Normal number
	e16 := uint16(exp + 15)
	f16 := uint16(frac >> 13)
	return float16(uint16(sign)<<15 | e16<<10 | f16)
}

// makeNoise generates Gaussian noise as flat float32 slice
func makeNoise(n, c, h, w int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	size := n * c * h * w
	data := make([]float32, size)
	for i := 0; i < size-1; i += 2 {
		u1 := rng.Float64()
		u2 := rng.Float64()
		for u1 == 0 {
			u1 = rng.Float64()
		}
		r := math.Sqrt(-2 * math.Log(u1))
		theta := 2 * math.Pi * u2
		data[i] = float32(r * math.Cos(theta))
		data[i+1] = float32(r * math.Sin(theta))
	}
	if size%2 == 1 {
		u1 := rng.Float64()
		u2 := rng.Float64()
		for u1 == 0 {
			u1 = rng.Float64()
		}
		data[size-1] = float32(math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2))
	}
	return data
}

// saveORTPNG saves [1,3,H,W] flat float32 data as PNG
func saveORTPNG(data []float32, H, W int, path string) error {
	rgba := image.NewRGBA(image.Rect(0, 0, W, H))

	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			r := data[0*H*W+y*W+x]
			g := data[1*H*W+y*W+x]
			b := data[2*H*W+y*W+x]
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
