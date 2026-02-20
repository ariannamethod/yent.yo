package main

import (
	"math"
)

// Tensor is an n-dimensional float32 array
type Tensor struct {
	Data  []float32
	Shape []int
}

func NewTensor(shape ...int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	return &Tensor{Data: make([]float32, size), Shape: shape}
}

func TensorFrom(data []float32, shape []int) *Tensor {
	return &Tensor{Data: data, Shape: shape}
}

func (t *Tensor) Numel() int {
	n := 1
	for _, s := range t.Shape {
		n *= s
	}
	return n
}

func (t *Tensor) Clone() *Tensor {
	d := make([]float32, len(t.Data))
	copy(d, t.Data)
	return &Tensor{Data: d, Shape: append([]int{}, t.Shape...)}
}

// --- Basic operations ---

// Linear: y = x @ W^T + b, x: [batch, in], W: [out, in], b: [out] → [batch, out]
func Linear(x, weight, bias *Tensor) *Tensor {
batch := x.Shape[0]
	inDim := x.Shape[1]
	outDim := weight.Shape[0]
	out := NewTensor(batch, outDim)

	if hasAccel {
		// BLAS: out = x @ W^T → treat as out^T = W @ x^T
		// But simpler: sgemm(batch, outDim, inDim, x, W^T)
		// W is [outDim, inDim] row-major = W^T in col-major sense
		// We want C[batch, outDim] = X[batch, inDim] @ W^T[inDim, outDim]
		// = sgemm(M=batch, N=outDim, K=inDim, A=X, B=W^T)
		// But W is stored as [outDim, inDim], so W^T is what we need
		// We can use: C = alpha * A @ B^T by calling sgemm with CblasTrans on B
		// For simplicity, just use accelSgemm with transposed weight
		// Actually: rewrite as batched dot products via sgemm
		// C[M,N] = A[M,K] @ B[K,N] where B = W^T
		// W is [outDim, inDim], W^T is [inDim, outDim]
		// So we need to transpose W. But sgemm can handle this.
		// Use accelBatchedMatmulNT(1, batch, outDim, inDim, 1.0, x, W, out)
		// That computes: out = 1.0 * x @ W^T ← exactly what we need!
		accelBatchedMatmulNT(1, batch, outDim, inDim, 1.0, x.Data, weight.Data, out.Data)
		if bias != nil {
			for b := 0; b < batch; b++ {
				for o := 0; o < outDim; o++ {
					out.Data[b*outDim+o] += bias.Data[o]
				}
			}
		}
	} else {
		for b := 0; b < batch; b++ {
			for o := 0; o < outDim; o++ {
				sum := float32(0)
				for i := 0; i < inDim; i++ {
					sum += x.Data[b*inDim+i] * weight.Data[o*inDim+i]
				}
				if bias != nil {
					sum += bias.Data[o]
				}
				out.Data[b*outDim+o] = sum
			}
		}
	}
	return out
}

// Conv2d: input [N,Cin,H,W], weight [Cout,Cin,kH,kW], bias [Cout], stride, padding
func Conv2d(input, weight, bias *Tensor, stride, padding int) *Tensor {
N := input.Shape[0]
	Cin := input.Shape[1]
	Hin := input.Shape[2]
	Win := input.Shape[3]
	Cout := weight.Shape[0]
	kH := weight.Shape[2]
	kW := weight.Shape[3]
	Hout := (Hin+2*padding-kH)/stride + 1
	Wout := (Win+2*padding-kW)/stride + 1

	out := NewTensor(N, Cout, Hout, Wout)

	if hasAccel {
		spatialOut := Hout * Wout
		for n := 0; n < N; n++ {
			inputOffset := n * Cin * Hin * Win
			outputOffset := n * Cout * Hout * Wout
			if kH == 1 && kW == 1 && stride == 1 && padding == 0 {
				// 1×1 conv = matrix multiply, no im2col needed
				// output[Cout, spatial] = weight[Cout, Cin] @ input[Cin, spatial]
				accelSgemm(Cout, spatialOut, Cin, 1.0,
					weight.Data, input.Data[inputOffset:],
					0.0, out.Data[outputOffset:outputOffset+Cout*spatialOut])
				if bias != nil {
					for co := 0; co < Cout; co++ {
						b := bias.Data[co]
						off := outputOffset + co*spatialOut
						for i := 0; i < spatialOut; i++ {
							out.Data[off+i] += b
						}
					}
				}
			} else {
				var biasData []float32
				if bias != nil {
					biasData = bias.Data
				}
				accelConv2d(
					input.Data[inputOffset:],
					Cin, Hin, Win,
					weight.Data, biasData, Cout,
					kH, kW, stride, padding,
					out.Data[outputOffset:outputOffset+Cout*spatialOut],
				)
			}
		}
	} else {
		for n := 0; n < N; n++ {
			for co := 0; co < Cout; co++ {
				for oh := 0; oh < Hout; oh++ {
					for ow := 0; ow < Wout; ow++ {
						sum := float32(0)
						for ci := 0; ci < Cin; ci++ {
							for kh := 0; kh < kH; kh++ {
								for kw := 0; kw < kW; kw++ {
									ih := oh*stride - padding + kh
									iw := ow*stride - padding + kw
									if ih >= 0 && ih < Hin && iw >= 0 && iw < Win {
										inIdx := ((n*Cin+ci)*Hin+ih)*Win + iw
										wIdx := ((co*Cin+ci)*kH+kh)*kW + kw
										sum += input.Data[inIdx] * weight.Data[wIdx]
									}
								}
							}
						}
						if bias != nil {
							sum += bias.Data[co]
						}
						out.Data[((n*Cout+co)*Hout+oh)*Wout+ow] = sum
					}
				}
			}
		}
	}
	return out
}

// GroupNorm: x [N,C,H,W], weight [C], bias [C], num_groups, eps
func GroupNorm(x, weight, bias *Tensor, numGroups int, eps float32) *Tensor {
N := x.Shape[0]
	C := x.Shape[1]
	H := x.Shape[2]
	W := x.Shape[3]
	groupSize := C / numGroups
	out := NewTensor(N, C, H, W)

	if hasAccel {
		for n := 0; n < N; n++ {
			nOff := n * C * H * W
			var wData, bData []float32
			if weight != nil {
				wData = weight.Data
				bData = bias.Data
			}
			for g := 0; g < numGroups; g++ {
				accelGroupNorm(
					x.Data[nOff:], out.Data[nOff:],
					wData, bData,
					g*groupSize, (g+1)*groupSize, H, W, eps)
			}
		}
		return out
	}

	for n := 0; n < N; n++ {
		for g := 0; g < numGroups; g++ {
			mean := float32(0)
			count := groupSize * H * W
			for c := g * groupSize; c < (g+1)*groupSize; c++ {
				for h := 0; h < H; h++ {
					for w := 0; w < W; w++ {
						mean += x.Data[((n*C+c)*H+h)*W+w]
					}
				}
			}
			mean /= float32(count)

			variance := float32(0)
			for c := g * groupSize; c < (g+1)*groupSize; c++ {
				for h := 0; h < H; h++ {
					for w := 0; w < W; w++ {
						d := x.Data[((n*C+c)*H+h)*W+w] - mean
						variance += d * d
					}
				}
			}
			variance /= float32(count)

			invStd := float32(1.0 / math.Sqrt(float64(variance+eps)))
			for c := g * groupSize; c < (g+1)*groupSize; c++ {
				for h := 0; h < H; h++ {
					for w := 0; w < W; w++ {
						idx := ((n*C+c)*H+h)*W + w
						v := (x.Data[idx] - mean) * invStd
						if weight != nil {
							v = v*weight.Data[c] + bias.Data[c]
						}
						out.Data[idx] = v
					}
				}
			}
		}
	}
	return out
}

// LayerNorm: x [batch, dim], weight [dim], bias [dim], eps
func LayerNorm(x, weight, bias *Tensor, eps float32) *Tensor {
batch := x.Shape[0]
	dim := x.Shape[1]
	out := NewTensor(batch, dim)

	for b := 0; b < batch; b++ {
		mean := float32(0)
		for i := 0; i < dim; i++ {
			mean += x.Data[b*dim+i]
		}
		mean /= float32(dim)

		variance := float32(0)
		for i := 0; i < dim; i++ {
			d := x.Data[b*dim+i] - mean
			variance += d * d
		}
		variance /= float32(dim)

		invStd := float32(1.0 / math.Sqrt(float64(variance+eps)))
		for i := 0; i < dim; i++ {
			v := (x.Data[b*dim+i] - mean) * invStd
			if weight != nil {
				v = v*weight.Data[i] + bias.Data[i]
			}
			out.Data[b*dim+i] = v
		}
	}
	return out
}

// SiLU activation: x * sigmoid(x)
func SiLU(x *Tensor) *Tensor {
out := NewTensor(x.Shape...)
	if hasAccel {
		accelSiLU(x.Data, out.Data, len(x.Data))
		return out
	}
	for i, v := range x.Data {
		out.Data[i] = v * float32(1.0/(1.0+math.Exp(-float64(v))))
	}
	return out
}

// QuickGELU: x * sigmoid(1.702 * x) — used by CLIP
func QuickGELU(x *Tensor) *Tensor {
	out := NewTensor(x.Shape...)
	for i, v := range x.Data {
		out.Data[i] = v * float32(1.0/(1.0+math.Exp(-1.702*float64(v))))
	}
	return out
}

// GEGLU: split input in half, first half * gelu(second half) — used in UNet FF
func GEGLU(x *Tensor) *Tensor {
	batch := x.Shape[0]
	dim := x.Shape[1]
	halfDim := dim / 2
	out := NewTensor(batch, halfDim)

	for b := 0; b < batch; b++ {
		for i := 0; i < halfDim; i++ {
			hidden := x.Data[b*dim+i]          // first half: passed through
			gate := x.Data[b*dim+halfDim+i]    // second half: gated
			// hidden * GELU(gate)
			gelu := gate * float32(0.5*(1.0+math.Erf(float64(gate)/math.Sqrt2)))
			out.Data[b*halfDim+i] = hidden * gelu
		}
	}
	return out
}

// Softmax along last dimension: x [batch, seq]
func Softmax(x *Tensor) *Tensor {
	batch := x.Shape[0]
	dim := x.Shape[1]
	out := NewTensor(batch, dim)

	for b := 0; b < batch; b++ {
		maxVal := float32(-math.MaxFloat32)
		for i := 0; i < dim; i++ {
			if x.Data[b*dim+i] > maxVal {
				maxVal = x.Data[b*dim+i]
			}
		}
		sumExp := float32(0)
		for i := 0; i < dim; i++ {
			e := float32(math.Exp(float64(x.Data[b*dim+i] - maxVal)))
			out.Data[b*dim+i] = e
			sumExp += e
		}
		for i := 0; i < dim; i++ {
			out.Data[b*dim+i] /= sumExp
		}
	}
	return out
}

// Add two tensors element-wise (must have same size)
func Add(a, b *Tensor) *Tensor {
out := NewTensor(a.Shape...)
	if hasAccel {
		accelVadd(a.Data, b.Data, out.Data, len(a.Data))
	} else {
		for i := range a.Data {
			out.Data[i] = a.Data[i] + b.Data[i]
		}
	}
	return out
}

// Scale tensor by scalar
func Scale(x *Tensor, s float32) *Tensor {
	out := NewTensor(x.Shape...)
	for i := range x.Data {
		out.Data[i] = x.Data[i] * s
	}
	return out
}

// Upsample2x: nearest-neighbor 2x upsampling for [N,C,H,W]
func Upsample2x(x *Tensor) *Tensor {
	N := x.Shape[0]
	C := x.Shape[1]
	H := x.Shape[2]
	W := x.Shape[3]
	out := NewTensor(N, C, H*2, W*2)

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					v := x.Data[((n*C+c)*H+h)*W+w]
					out.Data[((n*C+c)*(H*2)+(h*2))*(W*2)+(w*2)] = v
					out.Data[((n*C+c)*(H*2)+(h*2))*(W*2)+(w*2+1)] = v
					out.Data[((n*C+c)*(H*2)+(h*2+1))*(W*2)+(w*2)] = v
					out.Data[((n*C+c)*(H*2)+(h*2+1))*(W*2)+(w*2+1)] = v
				}
			}
		}
	}
	return out
}

// Downsample2x via stride-2 conv is just Conv2d with stride=2

// Concat tensors along channel dim: [N,C1,H,W] + [N,C2,H,W] → [N,C1+C2,H,W]
func ConcatChannels(a, b *Tensor) *Tensor {
	N := a.Shape[0]
	C1 := a.Shape[1]
	C2 := b.Shape[1]
	H := a.Shape[2]
	W := a.Shape[3]
	out := NewTensor(N, C1+C2, H, W)

	for n := 0; n < N; n++ {
		for c := 0; c < C1; c++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					out.Data[((n*(C1+C2)+c)*H+h)*W+w] = a.Data[((n*C1+c)*H+h)*W+w]
				}
			}
		}
		for c := 0; c < C2; c++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					out.Data[((n*(C1+C2)+(C1+c))*H+h)*W+w] = b.Data[((n*C2+c)*H+h)*W+w]
				}
			}
		}
	}
	return out
}

// Reshape4Dto2D: [N,C,H,W] → [N, C*H*W] or [1,C,H,W] → [H*W, C]
func Reshape4Dto2D(x *Tensor) *Tensor {
	C := x.Shape[1]
	H := x.Shape[2]
	W := x.Shape[3]
	seq := H * W
	// Transpose to [seq, C] for attention
	out := NewTensor(seq, C)
	for h := 0; h < H; h++ {
		for w := 0; w < W; w++ {
			for c := 0; c < C; c++ {
				out.Data[(h*W+w)*C+c] = x.Data[((0*C+c)*H+h)*W+w]
			}
		}
	}
	return out
}

// Reshape2Dto4D: [seq, C] → [1, C, H, W]
func Reshape2Dto4D(x *Tensor, C, H, W int) *Tensor {
	out := NewTensor(1, C, H, W)
	for h := 0; h < H; h++ {
		for w := 0; w < W; w++ {
			for c := 0; c < C; c++ {
				out.Data[((0*C+c)*H+h)*W+w] = x.Data[(h*W+w)*C+c]
			}
		}
	}
	return out
}
