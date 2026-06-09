//go:build notorch

// accel_notorch.go — the SD compute layer on notorch instead of the in-repo
// accel.c. Built with `-tags notorch`; links the system-wide libnotorch
// (/opt/homebrew/lib). Every accel* entry point keeps its exact signature, so
// tensor.go / unet.go / vae.go / clip.go are untouched — the diffusion runs on
// the Arianna Method's own tensor stack, and ORT is no longer needed at all.
//
// Heavy ops route to notorch's nt_* kernels (conv2d, group_norm, attention,
// GEMM); the trivial element-wise ops (silu, vadd, softmax) stay inline Go.
// Multi-head attention loops nt_attention over heads with strided per-head
// copies — correctness first; the accel.c version's tiling is a speed detail.
package main

/*
#cgo CFLAGS: -I/opt/homebrew/include
#cgo darwin LDFLAGS: -L/opt/homebrew/lib -lnotorch -framework Accelerate -lm
#cgo linux LDFLAGS: -L/opt/homebrew/lib -lnotorch -lopenblas -lm
#include <stdint.h>
void nt_blas_mm(float *C, const float *A, const float *B, int m, int k, int n);
void nt_blas_mmT(float *C, const float *A, const float *BT, int m, int k, int n);
int  nt_conv2d(float *out, const float *in, const float *weight, const float *bias,
               int Cin, int Hin, int Win, int Cout, int kH, int kW, int stride, int padding);
int  nt_group_norm(float *out, const float *in, const float *gamma, const float *beta,
                   int C, int H, int W, int num_groups, float eps);
int  nt_attention(float *out, const float *Q, const float *K, const float *V, int T, int S, int d);
*/
import "C"

import (
	"math"
	"unsafe"
)

const hasAccel = true

func fptr(s []float32) *C.float { return (*C.float)(unsafe.Pointer(&s[0])) }

// sgemm: C_ = alpha * A[M,K] @ B[K,N] + beta * C_.  notorch's nt_blas_mm does
// the plain A@B; alpha/beta are folded in Go (alpha==1 && beta==0 is the hot path).
func accelSgemm(M, N, K int, alpha float32, A, B []float32, beta float32, C_ []float32) {
	if beta == 0 && alpha == 1 {
		C.nt_blas_mm(fptr(C_), fptr(A), fptr(B), C.int(M), C.int(K), C.int(N))
		return
	}
	tmp := make([]float32, M*N)
	C.nt_blas_mm(fptr(tmp), fptr(A), fptr(B), C.int(M), C.int(K), C.int(N))
	for i := 0; i < M*N; i++ {
		C_[i] = alpha*tmp[i] + beta*C_[i]
	}
}

// conv2d: weight[Cout,Cin*kH*kW] @ im2col(input) + bias -> notorch nt_conv2d.
func accelConv2d(input []float32, Cin, Hin, Win int,
	weight []float32, bias []float32, Cout int,
	kH, kW, stride, padding int, output []float32) {
	var biasPtr *C.float
	if bias != nil {
		biasPtr = fptr(bias)
	}
	C.nt_conv2d(fptr(output), fptr(input), fptr(weight), biasPtr,
		C.int(Cin), C.int(Hin), C.int(Win), C.int(Cout),
		C.int(kH), C.int(kW), C.int(stride), C.int(padding))
}

// batchedMatmulNT: C[b] = scale * A[b][M,K] @ B[b][N,K]^T.
func accelBatchedMatmulNT(batch, M, N, K int, scale float32, A, B, C_ []float32) {
	szA, szB, szC := M*K, N*K, M*N
	for b := 0; b < batch; b++ {
		C.nt_blas_mmT(fptr(C_[b*szC:]), fptr(A[b*szA:]), fptr(B[b*szB:]),
			C.int(M), C.int(K), C.int(N))
	}
	if scale != 1 {
		for i := 0; i < batch*szC; i++ {
			C_[i] *= scale
		}
	}
}

// batchedMatmulNN: C[b] = A[b][M,K] @ B[b][K,N].
func accelBatchedMatmulNN(batch, M, N, K int, A, B, C_ []float32) {
	szA, szB, szC := M*K, K*N, M*N
	for b := 0; b < batch; b++ {
		C.nt_blas_mm(fptr(C_[b*szC:]), fptr(A[b*szA:]), fptr(B[b*szB:]),
			C.int(M), C.int(K), C.int(N))
	}
}

// softmaxRows: in-place row softmax (numerically stable). Pure Go — trivial.
func accelSoftmaxRows(scores []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		row := scores[r*cols : r*cols+cols]
		mx := row[0]
		for _, v := range row {
			if v > mx {
				mx = v
			}
		}
		var sum float32
		for i, v := range row {
			e := float32(math.Exp(float64(v - mx)))
			row[i] = e
			sum += e
		}
		inv := 1.0 / sum
		for i := range row {
			row[i] *= inv
		}
	}
}

// silu: out = x * sigmoid(x). Pure Go.
func accelSiLU(x, out []float32, n int) {
	for i := 0; i < n; i++ {
		v := x[i]
		out[i] = v / (1.0 + float32(math.Exp(float64(-v))))
	}
}

// groupNorm: normalize channels [chStart,chEnd) as a single group. notorch's
// nt_group_norm(num_groups=1) over the channel sub-range, with affine offset.
func accelGroupNorm(input, output []float32, weight, bias []float32,
	chStart, chEnd, H, W int, eps float32) {
	spatial := H * W
	C_ := chEnd - chStart
	off := chStart * spatial
	var gPtr, bPtr *C.float
	if weight != nil {
		gPtr = fptr(weight[chStart:])
		bPtr = fptr(bias[chStart:])
	}
	C.nt_group_norm(fptr(output[off:]), fptr(input[off:]), gPtr, bPtr,
		C.int(C_), C.int(H), C.int(W), C.int(1), C.float(eps))
}

// vadd: C = A + B. Pure Go.
func accelVadd(A, B, C_ []float32, n int) {
	for i := 0; i < n; i++ {
		C_[i] = A[i] + B[i]
	}
}

// headAttention runs scaled dot-product attention per head via notorch's
// single-head nt_attention. Q[seqQ,dim], K/V[seqKV,dim], dim = numHeads*headDim,
// head h occupies columns [h*headDim,(h+1)*headDim). mask (or nil) is additive,
// broadcast over Q rows: mask[seqKV] added to the pre-softmax scores.
func headAttention(Q, K, V, output []float32, seqQ, seqKV, headDim, numHeads, dim int, mask []float32) {
	qh := make([]float32, seqQ*headDim)
	kh := make([]float32, seqKV*headDim)
	vh := make([]float32, seqKV*headDim)
	oh := make([]float32, seqQ*headDim)
	for h := 0; h < numHeads; h++ {
		c0 := h * headDim
		for i := 0; i < seqQ; i++ {
			copy(qh[i*headDim:(i+1)*headDim], Q[i*dim+c0:i*dim+c0+headDim])
		}
		for i := 0; i < seqKV; i++ {
			copy(kh[i*headDim:(i+1)*headDim], K[i*dim+c0:i*dim+c0+headDim])
			copy(vh[i*headDim:(i+1)*headDim], V[i*dim+c0:i*dim+c0+headDim])
		}
		if mask == nil {
			C.nt_attention(fptr(oh), fptr(qh), fptr(kh), fptr(vh),
				C.int(seqQ), C.int(seqKV), C.int(headDim))
		} else {
			maskedAttention(oh, qh, kh, vh, mask, seqQ, seqKV, headDim)
		}
		for i := 0; i < seqQ; i++ {
			copy(output[i*dim+c0:i*dim+c0+headDim], oh[i*headDim:(i+1)*headDim])
		}
	}
}

// maskedAttention: scaled dot-product with an additive per-key mask (CLIP causal).
func maskedAttention(out, Q, K, V, mask []float32, T, S, d int) {
	scale := 1.0 / float32(math.Sqrt(float64(d)))
	scores := make([]float32, S)
	for t := 0; t < T; t++ {
		qrow := Q[t*d : t*d+d]
		mx := float32(math.Inf(-1))
		for s := 0; s < S; s++ {
			var dot float32
			krow := K[s*d : s*d+d]
			for j := 0; j < d; j++ {
				dot += qrow[j] * krow[j]
			}
			sc := dot*scale + mask[s]
			scores[s] = sc
			if sc > mx {
				mx = sc
			}
		}
		var sum float32
		for s := 0; s < S; s++ {
			e := float32(math.Exp(float64(scores[s] - mx)))
			scores[s] = e
			sum += e
		}
		inv := 1.0 / sum
		orow := out[t*d : t*d+d]
		for j := range orow {
			orow[j] = 0
		}
		for s := 0; s < S; s++ {
			w := scores[s] * inv
			vrow := V[s*d : s*d+d]
			for j := 0; j < d; j++ {
				orow[j] += w * vrow[j]
			}
		}
	}
}

func accelTiledAttention(Q, K, V, output []float32,
	seqQ, seqKV, headDim, numHeads, dim int, scale float32, tileSize int) {
	headAttention(Q, K, V, output, seqQ, seqKV, headDim, numHeads, dim, nil)
}

func accelTiledAttentionMasked(Q, K, V, output, mask []float32,
	seqQ, seqKV, headDim, numHeads, dim int, scale float32, tileSize int) {
	headAttention(Q, K, V, output, seqQ, seqKV, headDim, numHeads, dim, mask)
}

func accelTiledAttentionSingle(Q, K, V, output []float32,
	seq, dim int, scale float32, tileSize int) {
	C.nt_attention(fptr(output), fptr(Q), fptr(K), fptr(V),
		C.int(seq), C.int(seq), C.int(dim))
}
