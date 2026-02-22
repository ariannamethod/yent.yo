//go:build ort

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
	"unsafe"

	ort "github.com/yalue/onnxruntime_go"
)

func init() {
	runDiffusion = runDiffusionORT
}

func runDiffusionORT(modelDir, prompt, outPath string, seed int64, numSteps, latentSize int, guidanceScale float32) {
	fmt.Printf("[ORT] Model: %s\n", modelDir)
	fmt.Printf("[ORT] Prompt: %q\n", prompt)
	fmt.Printf("[ORT] Seed: %d, Steps: %d, Guidance: %.1f, Latent: %dx%d\n",
		seed, numSteps, guidanceScale, latentSize, latentSize)

	// ONNX dir: env override or auto-detect (prefer int8)
	onnxDir := os.Getenv("ONNX_DIR")
	if onnxDir == "" {
		onnxDir = modelDir + "/onnx_int8"
		if _, err := os.Stat(onnxDir + "/unet.onnx"); err != nil {
			onnxDir = modelDir + "/onnx_fp16"
		}
	}
	fmt.Printf("[ORT] ONNX dir: %s\n", onnxDir)

	// Auto-detect ORT library
	ortLib := findORTLibrary()
	if ortLib == "" {
		fatal("libonnxruntime not found. Install: brew install onnxruntime")
	}
	fmt.Printf("[ORT] Library: %s\n", ortLib)

	pipeline, err := NewORTPipeline(onnxDir, modelDir, ortLib)
	if err != nil {
		fatal("ORT pipeline: %v", err)
	}
	defer pipeline.Destroy()

	if err := pipeline.Generate(prompt, seed, numSteps, latentSize, guidanceScale, outPath); err != nil {
		fatal("generate: %v", err)
	}
}

// findORTLibrary looks for libonnxruntime in common locations
func findORTLibrary() string {
	candidates := []string{
		"/usr/local/lib/libonnxruntime.dylib",
		"/usr/local/Cellar/onnxruntime/1.24.2/lib/libonnxruntime.dylib",
		"/opt/homebrew/lib/libonnxruntime.dylib",
		"/usr/lib/libonnxruntime.so",
		"/usr/local/lib/libonnxruntime.so",
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

// ORTPipeline runs BK-SDM-Tiny inference via ONNX Runtime (CPU).
type ORTPipeline struct {
	clipSession *ort.DynamicAdvancedSession
	unetSession *ort.DynamicAdvancedSession
	vaeSession  *ort.DynamicAdvancedSession
	scheduler   *DDIMScheduler
	tokenizer   *CLIPTokenizer

	// Input data types detected from ONNX models
	clipInputType ort.TensorElementDataType
	unetInputType ort.TensorElementDataType // for sample + encoder_hidden_states
	vaeInputType  ort.TensorElementDataType
}

// NewORTPipeline loads all ONNX models and creates inference sessions.
func NewORTPipeline(onnxDir, modelDir, ortLibPath string) (*ORTPipeline, error) {
	ort.SetSharedLibraryPath(ortLibPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("ORT init: %w", err)
	}

	// Session options
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("session options: %w", err)
	}
	defer opts.Destroy()
	opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll)

	// CUDA only when YENT_GPU=1 (avoids cuDNN version crashes)
	usedGPU := false
	if os.Getenv("YENT_GPU") == "1" {
		cudaOpts, cudaErr := ort.NewCUDAProviderOptions()
		if cudaErr == nil {
			err = opts.AppendExecutionProviderCUDA(cudaOpts)
			cudaOpts.Destroy()
			if err == nil {
				fmt.Println("[ORT] Using CUDA execution provider")
				usedGPU = true
			} else {
				fmt.Printf("[ORT] CUDA init failed (%v), falling back to CPU\n", err)
			}
		} else {
			fmt.Printf("[ORT] CUDA not available (%v), using CPU\n", cudaErr)
		}
	}
	if !usedGPU {
		fmt.Println("[ORT] Using CPU")
		opts.SetIntraOpNumThreads(4)
		opts.SetInterOpNumThreads(1)
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

	// Inspect and load CLIP
	clipPath := onnxDir + "/clip_text_encoder.onnx"
	fmt.Print("Loading CLIP ONNX... ")
	start = time.Now()
	clipInputs, clipOutputs, err := ort.GetInputOutputInfo(clipPath)
	if err != nil {
		return nil, fmt.Errorf("CLIP info: %w", err)
	}
	fmt.Printf("\n  CLIP inputs: ")
	for _, in := range clipInputs {
		fmt.Printf("%s(%v %v) ", in.Name, in.DataType, in.Dimensions)
	}
	fmt.Printf("\n  CLIP outputs: ")
	clipOutNames := make([]string, len(clipOutputs))
	for i, out := range clipOutputs {
		fmt.Printf("%s(%v %v) ", out.Name, out.DataType, out.Dimensions)
		clipOutNames[i] = out.Name
	}
	fmt.Println()
	p.clipInputType = clipInputs[0].DataType

	p.clipSession, err = ort.NewDynamicAdvancedSession(
		clipPath,
		[]string{"input_ids"},
		clipOutNames,
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("CLIP session: %w", err)
	}
	fmt.Printf("  CLIP loaded (%v)\n", time.Since(start))

	// Inspect and load UNet
	unetPath := onnxDir + "/unet.onnx"
	fmt.Print("Loading UNet ONNX... ")
	start = time.Now()
	unetInputs, unetOutputs, err := ort.GetInputOutputInfo(unetPath)
	if err != nil {
		return nil, fmt.Errorf("UNet info: %w", err)
	}
	fmt.Printf("\n  UNet inputs: ")
	unetInNames := make([]string, len(unetInputs))
	for i, in := range unetInputs {
		fmt.Printf("%s(%v %v) ", in.Name, in.DataType, in.Dimensions)
		unetInNames[i] = in.Name
	}
	fmt.Printf("\n  UNet outputs: ")
	unetOutNames := make([]string, len(unetOutputs))
	for i, out := range unetOutputs {
		fmt.Printf("%s(%v %v) ", out.Name, out.DataType, out.Dimensions)
		unetOutNames[i] = out.Name
	}
	fmt.Println()
	// Find the sample input type (first input that isn't timestep)
	for _, in := range unetInputs {
		if in.Name == "sample" {
			p.unetInputType = in.DataType
			break
		}
	}

	p.unetSession, err = ort.NewDynamicAdvancedSession(
		unetPath,
		unetInNames,
		unetOutNames,
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("UNet session: %w", err)
	}
	fmt.Printf("  UNet loaded (%v)\n", time.Since(start))

	// Inspect and load VAE
	vaePath := onnxDir + "/vae_decoder.onnx"
	fmt.Print("Loading VAE ONNX... ")
	start = time.Now()
	vaeInputs, vaeOutputs, err := ort.GetInputOutputInfo(vaePath)
	if err != nil {
		return nil, fmt.Errorf("VAE info: %w", err)
	}
	fmt.Printf("\n  VAE inputs: ")
	for _, in := range vaeInputs {
		fmt.Printf("%s(%v %v) ", in.Name, in.DataType, in.Dimensions)
	}
	fmt.Printf("\n  VAE outputs: ")
	vaeOutNames := make([]string, len(vaeOutputs))
	for i, out := range vaeOutputs {
		fmt.Printf("%s(%v %v) ", out.Name, out.DataType, out.Dimensions)
		vaeOutNames[i] = out.Name
	}
	fmt.Println()
	p.vaeInputType = vaeInputs[0].DataType

	p.vaeSession, err = ort.NewDynamicAdvancedSession(
		vaePath,
		[]string{"latent_sample"},
		vaeOutNames,
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("VAE session: %w", err)
	}
	fmt.Printf("  VAE loaded (%v)\n", time.Since(start))

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
	condEmb, err := p.encodeText(condTokens)
	if err != nil {
		return fmt.Errorf("cond encoding: %w", err)
	}

	// Only encode unconditional if using CFG
	var uncondEmb []float32
	useCFG := guidanceScale > 1.0
	if useCFG {
		uncondTokens := p.tokenizer.Encode("")
		uncondEmb, err = p.encodeText(uncondTokens)
		if err != nil {
			return fmt.Errorf("uncond encoding: %w", err)
		}
	}
	fmt.Printf("  Text encoding: %v (CFG=%v)\n", time.Since(start), useCFG)
	if len(condEmb) >= 3 {
		fmt.Printf("  cond_emb[0][:3] = [%.4f, %.4f, %.4f]\n",
			condEmb[0], condEmb[1], condEmb[2])
	}

	// Phase 2: Diffusion
	fmt.Print("\n--- Phase 2: Diffusion ---\n")
	timesteps := p.scheduler.SetTimesteps(numSteps)
	fmt.Printf("Timesteps (%d): [%d ... %d]\n", len(timesteps), timesteps[0], timesteps[len(timesteps)-1])

	latent := makeNoise(1, 4, latentSize, latentSize, seed)

	totalStart := time.Now()
	for step, t := range timesteps {
		stepStart := time.Now()

		var noisePred []float32
		if useCFG {
			noiseUncond, err := p.runUNet(latent, int64(t), uncondEmb, latentSize)
			if err != nil {
				return fmt.Errorf("unet uncond step %d: %w", step, err)
			}
			noiseCond, err := p.runUNet(latent, int64(t), condEmb, latentSize)
			if err != nil {
				return fmt.Errorf("unet cond step %d: %w", step, err)
			}
			noisePred = make([]float32, len(noiseUncond))
			for i := range noisePred {
				noisePred[i] = noiseUncond[i] + guidanceScale*(noiseCond[i]-noiseUncond[i])
			}
		} else {
			// No CFG — single UNet pass
			noisePred, err = p.runUNet(latent, int64(t), condEmb, latentSize)
			if err != nil {
				return fmt.Errorf("unet step %d: %w", step, err)
			}
		}

		latent = p.schedulerStep(noisePred, t, latent, latentSize)

		fmt.Printf("  Step %d/%d (t=%d): %.1fs\n",
			step+1, numSteps, t, time.Since(stepStart).Seconds())
	}
	fmt.Printf("\nDiffusion: %.1fs total\n", time.Since(totalStart).Seconds())

	// Phase 3: VAE Decode
	fmt.Print("\n--- Phase 3: VAE Decoding ---\n")

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

	fmt.Printf("Saving %s... ", outPath)
	if err := saveORTPNG(imgData, imgH, imgW, outPath); err != nil {
		return fmt.Errorf("save: %w", err)
	}
	fmt.Println("done!")

	return nil
}

// makeTensorValue creates an ORT Value from float32 data, converting to fp16 if needed.
func makeTensorValue(data []float32, shape ort.Shape, dtype ort.TensorElementDataType) (ort.Value, error) {
	switch dtype {
	case ort.TensorElementDataTypeFloat:
		return ort.NewTensor(shape, data)
	case ort.TensorElementDataTypeFloat16:
		// Convert float32 → fp16 bytes
		fp16Bytes := float32SliceToFP16Bytes(data)
		return ort.NewCustomDataTensor(shape, fp16Bytes, ort.TensorElementDataTypeFloat16)
	default:
		// Try float32 as default
		return ort.NewTensor(shape, data)
	}
}

// extractFloat32 extracts float32 data from an ORT output Value.
func extractFloat32(v ort.Value) ([]float32, error) {
	// Try float32 first
	if t, ok := v.(*ort.Tensor[float32]); ok {
		src := t.GetData()
		result := make([]float32, len(src))
		copy(result, src)
		return result, nil
	}
	// Try Tensor[uint16] (fp16 from patched ort library)
	if t, ok := v.(*ort.Tensor[uint16]); ok {
		src := t.GetData()
		result := make([]float32, len(src))
		for i, bits := range src {
			result[i] = fp16ToFloat32(bits)
		}
		return result, nil
	}
	// Try CustomDataTensor (fp16)
	if t, ok := v.(*ort.CustomDataTensor); ok {
		raw := t.GetData()
		n := len(raw) / 2
		result := make([]float32, n)
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint16(raw[i*2 : i*2+2])
			result[i] = fp16ToFloat32(bits)
		}
		return result, nil
	}
	return nil, fmt.Errorf("unsupported output tensor type %T", v)
}

// encodeText runs CLIP text encoder on token IDs
func (p *ORTPipeline) encodeText(tokens []int) ([]float32, error) {
	tokenIDs := make([]int64, len(tokens))
	for i, t := range tokens {
		tokenIDs[i] = int64(t)
	}

	inputTensor, err := ort.NewTensor(ort.NewShape(1, int64(len(tokens))), tokenIDs)
	if err != nil {
		return nil, fmt.Errorf("input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Run with nil outputs — ORT allocates them
	outputs := make([]ort.Value, 2) // last_hidden_state, pooler_output
	err = p.clipSession.Run([]ort.Value{inputTensor}, outputs)
	if err != nil {
		return nil, fmt.Errorf("CLIP run: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	// First output is last_hidden_state
	return extractFloat32(outputs[0])
}

// runUNet runs one UNet forward pass
func (p *ORTPipeline) runUNet(latent []float32, timestep int64, textEmb []float32, latentSize int) ([]float32, error) {
	sampleTensor, err := makeTensorValue(latent,
		ort.NewShape(1, 4, int64(latentSize), int64(latentSize)),
		p.unetInputType)
	if err != nil {
		return nil, fmt.Errorf("sample tensor: %w", err)
	}
	defer sampleTensor.Destroy()

	tsTensor, err := ort.NewTensor(ort.NewShape(1), []int64{timestep})
	if err != nil {
		return nil, fmt.Errorf("timestep tensor: %w", err)
	}
	defer tsTensor.Destroy()

	embTensor, err := makeTensorValue(textEmb,
		ort.NewShape(1, 77, 768),
		p.unetInputType)
	if err != nil {
		return nil, fmt.Errorf("emb tensor: %w", err)
	}
	defer embTensor.Destroy()

	outputs := make([]ort.Value, 1) // out_sample
	err = p.unetSession.Run([]ort.Value{sampleTensor, tsTensor, embTensor}, outputs)
	if err != nil {
		return nil, fmt.Errorf("UNet run: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	return extractFloat32(outputs[0])
}

// decodeVAE runs the VAE decoder
func (p *ORTPipeline) decodeVAE(latent []float32, latentSize int) ([]float32, int, int, error) {
	inputTensor, err := makeTensorValue(latent,
		ort.NewShape(1, 4, int64(latentSize), int64(latentSize)),
		p.vaeInputType)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("VAE input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputs := make([]ort.Value, 1) // sample
	err = p.vaeSession.Run([]ort.Value{inputTensor}, outputs)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("VAE run: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	imgH := latentSize * 8
	imgW := latentSize * 8
	data, err := extractFloat32(outputs[0])
	return data, imgH, imgW, err
}

// schedulerStep applies DDIM step on flat float32 arrays
func (p *ORTPipeline) schedulerStep(noisePred []float32, timestep int, sample []float32, latentSize int) []float32 {
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

// ---- fp16 helpers ----

// float32SliceToFP16Bytes converts []float32 to raw fp16 bytes (little-endian)
func float32SliceToFP16Bytes(data []float32) []byte {
	result := make([]byte, len(data)*2)
	for i, v := range data {
		bits := f32ToF16(v)
		binary.LittleEndian.PutUint16(result[i*2:], bits)
	}
	return result
}

// f32ToF16 converts a single float32 to float16 (IEEE 754)
func f32ToF16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 31) & 1
	exp := int((bits>>23)&0xFF) - 127
	frac := bits & 0x7FFFFF

	if exp == 128 {
		// Inf/NaN
		if frac != 0 {
			return uint16(sign<<15 | 0x7C00 | 1) // NaN
		}
		return uint16(sign<<15 | 0x7C00) // Inf
	}
	if exp > 15 {
		return uint16(sign<<15 | 0x7C00) // overflow → Inf
	}
	if exp < -24 {
		return uint16(sign << 15) // underflow → zero
	}
	if exp < -14 {
		// Denormalized
		frac |= 0x800000
		shift := uint(-14 - exp)
		frac >>= (shift + 13)
		return uint16(sign<<15) | uint16(frac)
	}

	e16 := uint16(exp + 15)
	f16 := uint16(frac >> 13)
	return uint16(sign)<<15 | e16<<10 | f16
}

// fp16ToFloat32 converts IEEE 754 fp16 bits to float32
func fp16ToFloat32(bits uint16) float32 {
	sign := uint32(bits>>15) & 1
	exp := uint32(bits>>10) & 0x1F
	frac := uint32(bits) & 0x3FF

	if exp == 31 {
		if frac != 0 {
			return float32(math.NaN())
		}
		if sign == 1 {
			return float32(math.Inf(-1))
		}
		return float32(math.Inf(1))
	}
	if exp == 0 {
		if frac == 0 {
			if sign == 1 {
				return math.Float32frombits(1 << 31) // -0
			}
			return 0
		}
		// Denormalized
		f := float32(frac) / 1024.0 * float32(math.Pow(2, -14))
		if sign == 1 {
			return -f
		}
		return f
	}

	e32 := exp - 15 + 127
	f32 := frac << 13
	return math.Float32frombits(sign<<31 | e32<<23 | f32)
}

// ---- noise & image helpers ----

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

func saveORTPNG(data []float32, H, W int, path string) error {
	rgba := float32ToRGBA(data, H, W)

	// Apply post-processing if yentWords available
	if postProcessWords != "" {
		rgba = PostProcess(rgba, postProcessWords, postProcessRoast)
	}

	return saveProcessedPNG(rgba, path)
}

// Ensure unsafe is used (needed for potential future CGO interop)
var _ = unsafe.Sizeof(0)
