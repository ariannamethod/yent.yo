package main

// postprocess.go — Pure Go port of artifact_mask.py
//
// Pipeline: VAE output → grain → artifact detection → ASCII blend → chromatic aberration → vignette → grain
// No Python. No external tools. Just Go.

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"math/rand"
	"os"
	"sort"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

// PostProcess applies the full yent.yo post-processing pipeline.
// Takes raw VAE output (image.RGBA) + Yent's words → processed image with grain, ASCII, effects.
func PostProcess(img *image.RGBA, yentWords string, roastWords ...string) *image.RGBA {
	bounds := img.Bounds()
	W, H := bounds.Dx(), bounds.Dy()
	// Combine artist + commentator words for richer ASCII texture
	allWords := yentWords
	if len(roastWords) > 0 && roastWords[0] != "" {
		allWords = yentWords + " " + roastWords[0]
	}
	fmt.Fprintf(os.Stderr, "[postprocess] %dx%d, words=%q\n", W, H, truncate(allWords, 80))

	// Step 1: Artifact score map
	scoreMap := computeArtifactScore(img)
	meanScore := meanFloat32(scoreMap)
	highPct := countAbove(scoreMap, 0.5) * 100
	fmt.Fprintf(os.Stderr, "[postprocess] score: mean=%.2f, high-artifact=%.1f%%\n", meanScore, highPct)

	// Step 2: First grain pass (depth layer under ASCII)
	grained := cloneRGBA(img)
	applyFilmGrain(grained, 22, 42)

	// Step 3: Render ASCII layer (combined words from both yents)
	asciiLayer := renderASCIILayer(img, allWords, scoreMap)

	// Step 4: Blend — ASCII only where artifacts live
	asciiMax := float32(0.90)
	scorePower := float32(3.0)

	// Adaptive: dense images get less text so the image shows through
	if meanScore > 0.45 {
		excess := meanScore - 0.45
		asciiMax = max32(0.30, asciiMax-excess*2.0)
		scorePower = max32(2.5, scorePower+excess*3.5)
		fmt.Fprintf(os.Stderr, "[postprocess] adaptive: dense image, ascii_max=%.2f, power=%.1f\n", asciiMax, scorePower)
	}

	// Resize grained to match ASCII layer dimensions
	aw, ah := asciiLayer.Bounds().Dx(), asciiLayer.Bounds().Dy()
	grainedResized := resizeRGBA(grained, aw, ah)
	scoreResized := bilinearUpscale(scoreMap, W, H, aw, ah)

	// Composite blend
	composite := image.NewRGBA(image.Rect(0, 0, aw, ah))
	asciiFloor := float32(0.05)
	for y := 0; y < ah; y++ {
		for x := 0; x < aw; x++ {
			score := scoreResized[y*aw+x]
			blend := asciiFloor + pow32(score, scorePower)*(asciiMax-asciiFloor)

			gi := grainedResized.RGBAAt(x, y)
			ai := asciiLayer.RGBAAt(x, y)

			r := float32(gi.R)*(1-blend) + float32(ai.R)*blend
			g := float32(gi.G)*(1-blend) + float32(ai.G)*blend
			b := float32(gi.B)*(1-blend) + float32(ai.B)*blend

			composite.SetRGBA(x, y, color.RGBA{
				R: clamp8(r), G: clamp8(g), B: clamp8(b), A: 255,
			})
		}
	}

	// Step 5: Chromatic aberration
	applyChromaticAberration(composite, 2)

	// Step 6: Vignette
	applyVignette(composite, 0.30)

	// Step 7: Second grain pass (lighter, bonds layers)
	applyFilmGrain(composite, 15, 137)

	asciiVisible := countAbove(scoreResized, 0.1) * 100
	fmt.Fprintf(os.Stderr, "[postprocess] ASCII visible: %.0f%% of image\n", asciiVisible)

	return composite
}

// ═══════════════════════════════════════════════════════════════
// Artifact Detection
// ═══════════════════════════════════════════════════════════════

// computeGradient computes Sobel-like gradient magnitude on grayscale
func computeGradient(gray []float32, W, H int) []float32 {
	mag := make([]float32, W*H)
	for y := 1; y < H-1; y++ {
		for x := 1; x < W-1; x++ {
			gx := gray[y*W+x+1] - gray[y*W+x-1]
			gy := gray[(y+1)*W+x] - gray[(y-1)*W+x]
			mag[y*W+x] = float32(math.Sqrt(float64(gx*gx + gy*gy)))
		}
	}
	return mag
}

// computeArtifactScore returns per-pixel artifact score [0, 1]
// 0 = clean/detailed, 1 = smooth/artifact
func computeArtifactScore(img *image.RGBA) []float32 {
	bounds := img.Bounds()
	W, H := bounds.Dx(), bounds.Dy()
	blockSize := 12

	// Convert to grayscale
	gray := make([]float32, W*H)
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			c := img.RGBAAt(x+bounds.Min.X, y+bounds.Min.Y)
			gray[y*W+x] = 0.299*float32(c.R) + 0.587*float32(c.G) + 0.114*float32(c.B)
		}
	}

	// Gradient magnitude
	grad := computeGradient(gray, W, H)

	// Block-wise variance and brightness
	blocksH := H / blockSize
	blocksW := W / blockSize
	if blocksH == 0 || blocksW == 0 {
		return make([]float32, W*H)
	}

	varMap := make([]float32, blocksH*blocksW)
	brightMap := make([]float32, blocksH*blocksW)

	for by := 0; by < blocksH; by++ {
		for bx := 0; bx < blocksW; bx++ {
			var sum, sumSq, brightSum float32
			n := float32(blockSize * blockSize)
			for dy := 0; dy < blockSize; dy++ {
				for dx := 0; dx < blockSize; dx++ {
					y := by*blockSize + dy
					x := bx*blockSize + dx
					v := grad[y*W+x]
					sum += v
					sumSq += v * v
					brightSum += gray[y*W+x]
				}
			}
			mean := sum / n
			varMap[by*blocksW+bx] = sumSq/n - mean*mean
			brightMap[by*blocksW+bx] = brightSum / n
		}
	}

	// Percentile-based normalization on lit blocks
	minBrightness := float32(25)
	var litVars []float32
	litMask := make([]bool, blocksH*blocksW)
	for i := 0; i < blocksH*blocksW; i++ {
		if brightMap[i] > minBrightness {
			litMask[i] = true
			litVars = append(litVars, varMap[i])
		}
	}
	if len(litVars) == 0 {
		return make([]float32, W*H)
	}

	p10 := percentile(litVars, 10)
	p90 := percentile(litVars, 90)
	if p90 <= p10 {
		return make([]float32, W*H)
	}

	// Score blocks: low variance → high score (inverted)
	scoreBlocks := make([]float32, blocksH*blocksW)
	for i := 0; i < blocksH*blocksW; i++ {
		if litMask[i] {
			normalized := (varMap[i] - p10) / (p90 - p10)
			if normalized < 0 {
				normalized = 0
			}
			if normalized > 1 {
				normalized = 1
			}
			scoreBlocks[i] = 1.0 - normalized
		}
	}

	// Upscale to pixel level (bilinear)
	scorePx := bilinearUpscale(scoreBlocks, blocksW, blocksH, W, H)

	// Gaussian blur approximation (3-pass box blur)
	radius := int(float64(blockSize) * 1.5)
	boxBlur(scorePx, W, H, radius)
	boxBlur(scorePx, W, H, radius)
	boxBlur(scorePx, W, H, radius)

	// Power curve — push low scores lower
	for i := range scorePx {
		scorePx[i] = pow32(scorePx[i], 1.8)
	}

	return scorePx
}

// ═══════════════════════════════════════════════════════════════
// Effects
// ═══════════════════════════════════════════════════════════════

// applyFilmGrain adds film grain with shadow bias (in-place)
func applyFilmGrain(img *image.RGBA, intensity float32, seed int64) {
	bounds := img.Bounds()
	W, H := bounds.Dx(), bounds.Dy()
	rng := rand.New(rand.NewSource(seed))

	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			c := img.RGBAAt(x+bounds.Min.X, y+bounds.Min.Y)
			lum := (0.299*float32(c.R) + 0.587*float32(c.G) + 0.114*float32(c.B)) / 255.0
			shadowMask := 1.0 - lum*0.4

			// Box-Muller gaussian noise
			n := gaussNoise(rng) * intensity * shadowMask

			img.SetRGBA(x+bounds.Min.X, y+bounds.Min.Y, color.RGBA{
				R: clamp8(float32(c.R) + n),
				G: clamp8(float32(c.G) + n),
				B: clamp8(float32(c.B) + n),
				A: 255,
			})
		}
	}
}

// applyChromaticAberration shifts R right and B left (in-place)
func applyChromaticAberration(img *image.RGBA, shift int) {
	bounds := img.Bounds()
	W, H := bounds.Dx(), bounds.Dy()

	// Extract channels
	red := make([]uint8, W*H)
	blue := make([]uint8, W*H)
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			c := img.RGBAAt(x+bounds.Min.X, y+bounds.Min.Y)
			red[y*W+x] = c.R
			blue[y*W+x] = c.B
		}
	}

	// Apply shifts
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			c := img.RGBAAt(x+bounds.Min.X, y+bounds.Min.Y)
			// Red: read from left (shift right)
			rx := x - shift
			if rx < 0 {
				rx = 0
			}
			// Blue: read from right (shift left)
			bx := x + shift
			if bx >= W {
				bx = W - 1
			}
			img.SetRGBA(x+bounds.Min.X, y+bounds.Min.Y, color.RGBA{
				R: red[y*W+rx],
				G: c.G,
				B: blue[y*W+bx],
				A: 255,
			})
		}
	}
}

// applyVignette darkens edges with radial falloff (in-place)
func applyVignette(img *image.RGBA, strength float32) {
	bounds := img.Bounds()
	W, H := bounds.Dx(), bounds.Dy()
	cx, cy := float32(W)/2, float32(H)/2
	maxDist := float32(math.Sqrt(float64(cx*cx + cy*cy)))

	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			dx := float32(x) - cx
			dy := float32(y) - cy
			dist := float32(math.Sqrt(float64(dx*dx+dy*dy))) / maxDist
			mult := 1.0 - strength*pow32(dist, 1.5)

			c := img.RGBAAt(x+bounds.Min.X, y+bounds.Min.Y)
			img.SetRGBA(x+bounds.Min.X, y+bounds.Min.Y, color.RGBA{
				R: clamp8(float32(c.R) * mult),
				G: clamp8(float32(c.G) * mult),
				B: clamp8(float32(c.B) * mult),
				A: 255,
			})
		}
	}
}

// ═══════════════════════════════════════════════════════════════
// ASCII Layer Rendering
// ═══════════════════════════════════════════════════════════════

// ASCII charset — light to dark
var asciiChars = " .'·:;~=+*#%@"

// renderASCIILayer creates the ASCII art overlay image
func renderASCIILayer(img *image.RGBA, words string, scoreMap []float32) *image.RGBA {
	bounds := img.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()

	face := basicfont.Face7x13
	charW := 7  // basicfont char width
	charH := 13 // basicfont char height

	cols := srcW / charW
	rows := srcH / charH
	if cols < 20 {
		cols = 40
	}
	if rows < 15 {
		rows = 30
	}

	outW := cols * charW
	outH := rows * charH

	// Downsample image to grid
	pixels := resizeRGBA(img, cols, rows)

	// Downsample score map to grid
	scoreGrid := bilinearUpscale(scoreMap, srcW, srcH, cols, rows)

	// Text stream
	if words == "" {
		words = "void noise static the machine dreams pixels bleed light i was not born i became"
	}
	textPos := 0

	// Create canvas
	canvas := image.NewRGBA(image.Rect(0, 0, outW, outH))
	// Fill with dark background
	draw.Draw(canvas, canvas.Bounds(), image.NewUniform(color.RGBA{8, 8, 12, 255}), image.Point{}, draw.Src)

	numChars := len(asciiChars)
	bgLevelClean := float32(0.50)  // clean zones: lighter bg, text fades
	bgLevelArtifact := float32(0.25) // artifact zones: darker bg, text pops
	boostClean := float32(1.5)     // clean zones: subtle text
	boostArtifact := float32(3.2)  // artifact zones: punchy text

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			c := pixels.RGBAAt(x, y)
			br := (0.299*float32(c.R) + 0.587*float32(c.G) + 0.114*float32(c.B)) / 255.0
			score := scoreGrid[y*cols+x]

			px, py := x*charW, y*charH

			// Background: tinted cell — darker in artifact zones for contrast
			bgLevel := bgLevelClean
			if score > 0.4 {
				bgLevel = bgLevelArtifact
			}
			bgR := clamp8(float32(c.R) * bgLevel)
			bgG := clamp8(float32(c.G) * bgLevel)
			bgB := clamp8(float32(c.B) * bgLevel)
			for dy := 0; dy < charH; dy++ {
				for dx := 0; dx < charW; dx++ {
					canvas.SetRGBA(px+dx, py+dy, color.RGBA{bgR, bgG, bgB, 255})
				}
			}

			// Choose character
			var ch byte
			if score > 0.4 {
				// Artifact zone: Yent's words
				ch = words[textPos%len(words)]
				textPos++
			} else {
				// Clean zone: ASCII by brightness
				idx := int(br * float32(numChars-1))
				if idx < 0 {
					idx = 0
				}
				if idx >= numChars {
					idx = numChars - 1
				}
				ch = asciiChars[idx]
			}

			if ch == ' ' {
				continue
			}

			// Foreground color — strong contrast in artifact zones
			boost := boostClean
			if score > 0.4 {
				boost = boostArtifact
			}
			cr := clamp8(float32(c.R) * boost)
			cg := clamp8(float32(c.G) * boost)
			cb := clamp8(float32(c.B) * boost)

			// Artifact zones: blue tint for visual distinction
			if score > 0.4 {
				cr = clamp8(float32(cr) * 0.7)
				cb = clamp8(float32(cb)*1.3 + 25)
			}

			// Draw character using basicfont
			d := &font.Drawer{
				Dst:  canvas,
				Src:  image.NewUniform(color.RGBA{cr, cg, cb, 255}),
				Face: face,
				Dot:  fixed.P(px, py+charH-2), // baseline offset
			}
			d.DrawString(string(ch))
		}
	}

	return canvas
}

// ═══════════════════════════════════════════════════════════════
// Image Helpers
// ═══════════════════════════════════════════════════════════════

// bilinearUpscale resizes a float32 grid using bilinear interpolation
func bilinearUpscale(data []float32, srcW, srcH, dstW, dstH int) []float32 {
	result := make([]float32, dstW*dstH)
	for y := 0; y < dstH; y++ {
		for x := 0; x < dstW; x++ {
			// Map destination pixel to source coordinates
			sx := float32(x) * float32(srcW-1) / float32(dstW-1)
			sy := float32(y) * float32(srcH-1) / float32(dstH-1)

			x0 := int(sx)
			y0 := int(sy)
			x1 := x0 + 1
			y1 := y0 + 1

			if x1 >= srcW {
				x1 = srcW - 1
			}
			if y1 >= srcH {
				y1 = srcH - 1
			}

			fx := sx - float32(x0)
			fy := sy - float32(y0)

			v00 := data[y0*srcW+x0]
			v10 := data[y0*srcW+x1]
			v01 := data[y1*srcW+x0]
			v11 := data[y1*srcW+x1]

			v := v00*(1-fx)*(1-fy) + v10*fx*(1-fy) + v01*(1-fx)*fy + v11*fx*fy
			result[y*dstW+x] = v
		}
	}
	return result
}

// boxBlur applies a box blur in-place (horizontal + vertical pass)
func boxBlur(data []float32, W, H, radius int) {
	if radius <= 0 {
		return
	}
	tmp := make([]float32, W*H)

	// Horizontal pass
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			var sum float32
			var count float32
			for dx := -radius; dx <= radius; dx++ {
				nx := x + dx
				if nx >= 0 && nx < W {
					sum += data[y*W+nx]
					count++
				}
			}
			tmp[y*W+x] = sum / count
		}
	}

	// Vertical pass
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			var sum float32
			var count float32
			for dy := -radius; dy <= radius; dy++ {
				ny := y + dy
				if ny >= 0 && ny < H {
					sum += tmp[ny*W+x]
					count++
				}
			}
			data[y*W+x] = sum / count
		}
	}
}

// resizeRGBA does nearest-neighbor resize (fast, good enough for ASCII grid)
func resizeRGBA(img *image.RGBA, dstW, dstH int) *image.RGBA {
	bounds := img.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, dstW, dstH))

	for y := 0; y < dstH; y++ {
		for x := 0; x < dstW; x++ {
			sx := x * srcW / dstW
			sy := y * srcH / dstH
			dst.SetRGBA(x, y, img.RGBAAt(sx+bounds.Min.X, sy+bounds.Min.Y))
		}
	}
	return dst
}

// cloneRGBA creates a deep copy
func cloneRGBA(img *image.RGBA) *image.RGBA {
	clone := image.NewRGBA(img.Bounds())
	copy(clone.Pix, img.Pix)
	return clone
}

// ═══════════════════════════════════════════════════════════════
// Math Helpers
// ═══════════════════════════════════════════════════════════════

func percentile(data []float32, pct float64) float32 {
	sorted := make([]float32, len(data))
	copy(sorted, data)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	idx := int(pct / 100.0 * float64(len(sorted)-1))
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func gaussNoise(rng *rand.Rand) float32 {
	u1 := rng.Float64()
	u2 := rng.Float64()
	for u1 == 0 {
		u1 = rng.Float64()
	}
	return float32(math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2))
}

func pow32(base, exp float32) float32 {
	return float32(math.Pow(float64(base), float64(exp)))
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func clamp8(v float32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

func meanFloat32(data []float32) float32 {
	if len(data) == 0 {
		return 0
	}
	var sum float64
	for _, v := range data {
		sum += float64(v)
	}
	return float32(sum / float64(len(data)))
}

func countAbove(data []float32, threshold float32) float32 {
	if len(data) == 0 {
		return 0
	}
	count := 0
	for _, v := range data {
		if v > threshold {
			count++
		}
	}
	return float32(count) / float32(len(data))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// ═══════════════════════════════════════════════════════════════
// Pipeline Entry Points
// ═══════════════════════════════════════════════════════════════

// tensorToRGBA converts a [1,3,H,W] float32 tensor to image.RGBA
func tensorToRGBA(tensor *Tensor) *image.RGBA {
	H := tensor.Shape[2]
	W := tensor.Shape[3]
	rgba := image.NewRGBA(image.Rect(0, 0, W, H))

	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			r := tensor.Data[0*H*W+y*W+x]
			g := tensor.Data[1*H*W+y*W+x]
			b := tensor.Data[2*H*W+y*W+x]
			rgba.SetRGBA(x, y, color.RGBA{
				R: clampByte((r + 1) / 2),
				G: clampByte((g + 1) / 2),
				B: clampByte((b + 1) / 2),
				A: 255,
			})
		}
	}
	return rgba
}

// float32ToRGBA converts flat [3*H*W] float32 array to image.RGBA
func float32ToRGBA(data []float32, H, W int) *image.RGBA {
	rgba := image.NewRGBA(image.Rect(0, 0, W, H))
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			r := data[0*H*W+y*W+x]
			g := data[1*H*W+y*W+x]
			b := data[2*H*W+y*W+x]
			rgba.SetRGBA(x, y, color.RGBA{
				R: clampByte((r + 1) / 2),
				G: clampByte((g + 1) / 2),
				B: clampByte((b + 1) / 2),
				A: 255,
			})
		}
	}
	return rgba
}

// saveProcessedPNG saves an image.RGBA to a PNG file
func saveProcessedPNG(img *image.RGBA, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}
