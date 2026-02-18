// Vitriol — Pure Go inference for DCGAN generator
// Generates 128x128 caricature images from random noise
// Zero dependencies except stdlib
//
// Binary weight format (vitriol_gen.bin):
//   Header: "VTRL"(4) + version(4) + nz(4) + ngf(4) + nc(4) + img_size(4) + nlayers(4)
//   Per layer:
//     type(4): 0=conv_transpose, 1=batchnorm
//     conv_transpose: in_ch(4) + out_ch(4) + kH(4) + kW(4) + weight(float32[])
//     batchnorm: num_features(4) + weight(f32[]) + bias(f32[]) + mean(f32[]) + var(f32[])

package main

import (
	"encoding/binary"
	"fmt"
	"image"
	"image/png"
	"math"
	"math/rand"
	"os"
	"time"
)

// Tensor is a 4D tensor [N, C, H, W] stored as flat float32
type Tensor struct {
	Data []float32
	N, C, H, W int
}

func NewTensor(n, c, h, w int) *Tensor {
	return &Tensor{
		Data: make([]float32, n*c*h*w),
		N: n, C: c, H: h, W: w,
	}
}

func (t *Tensor) At(n, c, h, w int) float32 {
	return t.Data[((n*t.C+c)*t.H+h)*t.W+w]
}

func (t *Tensor) Set(n, c, h, w int, v float32) {
	t.Data[((n*t.C+c)*t.H+h)*t.W+w] = v
}

// ConvTranspose2d layer (no bias, as in DCGAN)
type ConvTranspose2d struct {
	Weight          *Tensor // (InCh, OutCh, KH, KW)
	InCh, OutCh     int
	KH, KW          int
	Stride, Padding int
}

// BatchNorm2d layer
type BatchNorm2d struct {
	Weight, Bias       []float32
	RunningMean, RunningVar []float32
	NumFeatures        int
	Eps                float32
}

// Generator holds all layers
type Generator struct {
	Layers  []interface{} // ConvTranspose2d or BatchNorm2d
	NZ      int
	NGF     int
	NC      int
	ImgSize int
}

// ConvTranspose2d forward: (1, inCh, hIn, wIn) → (1, outCh, hOut, wOut)
// hOut = (hIn-1)*stride - 2*padding + kH
func (conv *ConvTranspose2d) Forward(input *Tensor) *Tensor {
	hOut := (input.H-1)*conv.Stride - 2*conv.Padding + conv.KH
	wOut := (input.W-1)*conv.Stride - 2*conv.Padding + conv.KW
	output := NewTensor(1, conv.OutCh, hOut, wOut)

	// For each output channel
	for oc := 0; oc < conv.OutCh; oc++ {
		// For each input channel
		for ic := 0; ic < conv.InCh; ic++ {
			// For each input position
			for ih := 0; ih < input.H; ih++ {
				for iw := 0; iw < input.W; iw++ {
					val := input.At(0, ic, ih, iw)
					// Scatter to output through kernel
					for kh := 0; kh < conv.KH; kh++ {
						for kw := 0; kw < conv.KW; kw++ {
							oh := ih*conv.Stride + kh - conv.Padding
							ow := iw*conv.Stride + kw - conv.Padding
							if oh >= 0 && oh < hOut && ow >= 0 && ow < wOut {
								w := conv.Weight.At(ic, oc, kh, kw)
								old := output.At(0, oc, oh, ow)
								output.Set(0, oc, oh, ow, old+val*w)
							}
						}
					}
				}
			}
		}
	}
	return output
}

// BatchNorm2d forward (eval mode)
func (bn *BatchNorm2d) Forward(input *Tensor) *Tensor {
	output := NewTensor(input.N, input.C, input.H, input.W)
	for c := 0; c < input.C; c++ {
		scale := bn.Weight[c] / float32(math.Sqrt(float64(bn.RunningVar[c]+bn.Eps)))
		shift := bn.Bias[c] - bn.RunningMean[c]*scale
		for h := 0; h < input.H; h++ {
			for w := 0; w < input.W; w++ {
				v := input.At(0, c, h, w)
				output.Set(0, c, h, w, v*scale+shift)
			}
		}
	}
	return output
}

// ReLU in-place
func ReLU(t *Tensor) *Tensor {
	for i := range t.Data {
		if t.Data[i] < 0 {
			t.Data[i] = 0
		}
	}
	return t
}

// Tanh in-place
func Tanh(t *Tensor) *Tensor {
	for i := range t.Data {
		t.Data[i] = float32(math.Tanh(float64(t.Data[i])))
	}
	return t
}

// LoadGenerator loads weights from binary file
func LoadGenerator(path string) (*Generator, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read header
	magic := make([]byte, 4)
	if _, err := f.Read(magic); err != nil {
		return nil, err
	}
	if string(magic) != "VTRL" {
		return nil, fmt.Errorf("invalid magic: %s", magic)
	}

	var version, nz, ngf, nc, imgSize, nLayers uint32
	binary.Read(f, binary.LittleEndian, &version)
	binary.Read(f, binary.LittleEndian, &nz)
	binary.Read(f, binary.LittleEndian, &ngf)
	binary.Read(f, binary.LittleEndian, &nc)
	binary.Read(f, binary.LittleEndian, &imgSize)
	binary.Read(f, binary.LittleEndian, &nLayers)

	gen := &Generator{
		NZ:      int(nz),
		NGF:     int(ngf),
		NC:      int(nc),
		ImgSize: int(imgSize),
	}

	for i := 0; i < int(nLayers); i++ {
		var layerType uint32
		binary.Read(f, binary.LittleEndian, &layerType)

		switch layerType {
		case 0: // ConvTranspose2d
			var inCh, outCh, kH, kW uint32
			binary.Read(f, binary.LittleEndian, &inCh)
			binary.Read(f, binary.LittleEndian, &outCh)
			binary.Read(f, binary.LittleEndian, &kH)
			binary.Read(f, binary.LittleEndian, &kW)

			size := int(inCh * outCh * kH * kW)
			weight := NewTensor(int(inCh), int(outCh), int(kH), int(kW))
			for j := 0; j < size; j++ {
				binary.Read(f, binary.LittleEndian, &weight.Data[j])
			}

			conv := &ConvTranspose2d{
				Weight:  weight,
				InCh:    int(inCh),
				OutCh:   int(outCh),
				KH:      int(kH),
				KW:      int(kW),
				Stride:  2,
				Padding: 1,
			}
			// First layer has stride=1, padding=0
			if len(gen.Layers) == 0 {
				conv.Stride = 1
				conv.Padding = 0
			}
			gen.Layers = append(gen.Layers, conv)

		case 1: // BatchNorm2d
			var numFeatures uint32
			binary.Read(f, binary.LittleEndian, &numFeatures)
			nf := int(numFeatures)

			bn := &BatchNorm2d{
				Weight:      make([]float32, nf),
				Bias:        make([]float32, nf),
				RunningMean: make([]float32, nf),
				RunningVar:  make([]float32, nf),
				NumFeatures: nf,
				Eps:         1e-5,
			}
			for j := 0; j < nf; j++ {
				binary.Read(f, binary.LittleEndian, &bn.Weight[j])
			}
			for j := 0; j < nf; j++ {
				binary.Read(f, binary.LittleEndian, &bn.Bias[j])
			}
			for j := 0; j < nf; j++ {
				binary.Read(f, binary.LittleEndian, &bn.RunningMean[j])
			}
			for j := 0; j < nf; j++ {
				binary.Read(f, binary.LittleEndian, &bn.RunningVar[j])
			}
			gen.Layers = append(gen.Layers, bn)
		}
	}

	return gen, nil
}

// Generate produces a 128x128 RGB image from random noise
func (gen *Generator) Generate(seed int64) *Tensor {
	rng := rand.New(rand.NewSource(seed))

	// Create latent vector z: (1, NZ, 1, 1)
	z := NewTensor(1, gen.NZ, 1, 1)
	for i := range z.Data {
		z.Data[i] = float32(rng.NormFloat64())
	}

	// Forward pass through all layers
	x := z
	for i, layer := range gen.Layers {
		switch l := layer.(type) {
		case *ConvTranspose2d:
			x = l.Forward(x)
		case *BatchNorm2d:
			x = l.Forward(x)
			x = ReLU(x) // ReLU after every batchnorm
		}
		// Tanh after last layer (which is a conv, no batchnorm follows)
		if i == len(gen.Layers)-1 {
			x = Tanh(x)
		}
	}
	return x
}

// SavePNG saves a tensor as a PNG image
func SavePNG(t *Tensor, path string) error {
	img := image.NewRGBA(image.Rect(0, 0, t.W, t.H))
	for y := 0; y < t.H; y++ {
		for x := 0; x < t.W; x++ {
			// Tensor is in [-1, 1], convert to [0, 255]
			r := clampByte((t.At(0, 0, y, x) + 1) * 127.5)
			g := clampByte((t.At(0, 1, y, x) + 1) * 127.5)
			b := clampByte((t.At(0, 2, y, x) + 1) * 127.5)
			idx := (y*t.W + x) * 4
			img.Pix[idx+0] = r
			img.Pix[idx+1] = g
			img.Pix[idx+2] = b
			img.Pix[idx+3] = 255
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func clampByte(v float32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Vitriol — DCGAN Caricature Generator")
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("  vitriol <weights.bin> [output.png] [seed]")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("  vitriol weights/vitriol_gen.bin")
		fmt.Println("  vitriol weights/vitriol_gen.bin caricature.png")
		fmt.Println("  vitriol weights/vitriol_gen.bin caricature.png 42")
		os.Exit(0)
	}

	weightsPath := os.Args[1]
	outPath := "vitriol_output.png"
	seed := time.Now().UnixNano()

	if len(os.Args) > 2 {
		outPath = os.Args[2]
	}
	if len(os.Args) > 3 {
		fmt.Sscanf(os.Args[3], "%d", &seed)
	}

	fmt.Printf("Loading weights from %s...\n", weightsPath)
	gen, err := LoadGenerator(weightsPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading weights: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Generator: nz=%d, ngf=%d, nc=%d, img=%dx%d, layers=%d\n",
		gen.NZ, gen.NGF, gen.NC, gen.ImgSize, gen.ImgSize, len(gen.Layers))

	fmt.Printf("Generating (seed=%d)...\n", seed)
	start := time.Now()
	output := gen.Generate(seed)
	elapsed := time.Since(start)
	fmt.Printf("Generated %dx%d image in %v\n", output.W, output.H, elapsed)

	if err := SavePNG(output, outPath); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Saved to %s\n", outPath)
}
