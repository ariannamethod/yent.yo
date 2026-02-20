//go:build darwin

package main

// #cgo CFLAGS: -DACCELERATE_NEW_LAPACK -O2
// #cgo LDFLAGS: -framework Accelerate
// #include "accel.h"
import "C"
import "unsafe"

// sgemm: C_ = alpha * A @ B + beta * C_  (row-major)
func accelSgemm(M, N, K int, alpha float32, A, B []float32, beta float32, C_ []float32) {
	C.accel_sgemm(C.int(M), C.int(N), C.int(K),
		C.float(alpha), (*C.float)(unsafe.Pointer(&A[0])), (*C.float)(unsafe.Pointer(&B[0])),
		C.float(beta), (*C.float)(unsafe.Pointer(&C_[0])))
}

// conv2d: output = weight @ im2col(input) + bias
func accelConv2d(input []float32, Cin, Hin, Win int,
	weight []float32, bias []float32, Cout int,
	kH, kW, stride, padding int,
	output []float32) {
	var biasPtr *C.float
	if bias != nil {
		biasPtr = (*C.float)(unsafe.Pointer(&bias[0]))
	}
	C.accel_conv2d(
		(*C.float)(unsafe.Pointer(&input[0])), C.int(Cin), C.int(Hin), C.int(Win),
		(*C.float)(unsafe.Pointer(&weight[0])), biasPtr, C.int(Cout),
		C.int(kH), C.int(kW), C.int(stride), C.int(padding),
		(*C.float)(unsafe.Pointer(&output[0])))
}

// batchedMatmulNT: C[b] = scale * A[b] @ B[b]^T
func accelBatchedMatmulNT(batch, M, N, K int, scale float32, A, B, C_ []float32) {
	C.accel_batched_matmul_nt(C.int(batch), C.int(M), C.int(N), C.int(K),
		C.float(scale),
		(*C.float)(unsafe.Pointer(&A[0])), (*C.float)(unsafe.Pointer(&B[0])),
		(*C.float)(unsafe.Pointer(&C_[0])))
}

// batchedMatmulNN: C[b] = A[b] @ B[b]
func accelBatchedMatmulNN(batch, M, N, K int, A, B, C_ []float32) {
	C.accel_batched_matmul_nn(C.int(batch), C.int(M), C.int(N), C.int(K),
		(*C.float)(unsafe.Pointer(&A[0])), (*C.float)(unsafe.Pointer(&B[0])),
		(*C.float)(unsafe.Pointer(&C_[0])))
}

// softmaxRows: apply softmax to each row of [rows, cols] matrix in-place
func accelSoftmaxRows(scores []float32, rows, cols int) {
	C.accel_softmax_rows((*C.float)(unsafe.Pointer(&scores[0])), C.int(rows), C.int(cols))
}

// silu: out = x * sigmoid(x)
func accelSiLU(x, out []float32, n int) {
	C.accel_silu((*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&out[0])), C.int(n))
}

// groupNorm: normalize one group of channels
func accelGroupNorm(input, output []float32, weight, bias []float32,
	chStart, chEnd, H, W int, eps float32) {
	var wPtr, bPtr *C.float
	if weight != nil {
		wPtr = (*C.float)(unsafe.Pointer(&weight[0]))
		bPtr = (*C.float)(unsafe.Pointer(&bias[0]))
	}
	C.accel_group_norm(
		(*C.float)(unsafe.Pointer(&input[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
		wPtr, bPtr,
		C.int(chStart), C.int(chEnd), C.int(H), C.int(W), C.float(eps))
}

// vadd: C = A + B element-wise
func accelVadd(A, B, C_ []float32, n int) {
	C.accel_vadd((*C.float)(unsafe.Pointer(&A[0])), (*C.float)(unsafe.Pointer(&B[0])),
		(*C.float)(unsafe.Pointer(&C_[0])), C.int(n))
}

// tiledAttention: fused multi-head attention with tiling over Q rows.
// All buffers are static C-side — zero Go allocation.
func accelTiledAttention(Q, K, V, output []float32,
	seqQ, seqKV, headDim, numHeads, dim int, scale float32, tileSize int) {
	C.accel_tiled_attention(
		(*C.float)(unsafe.Pointer(&Q[0])),
		(*C.float)(unsafe.Pointer(&K[0])),
		(*C.float)(unsafe.Pointer(&V[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.int(seqQ), C.int(seqKV), C.int(headDim), C.int(numHeads), C.int(dim),
		C.float(scale), C.int(tileSize))
}

// tiledAttentionMasked: same but with additive mask (causal mask for CLIP)
func accelTiledAttentionMasked(Q, K, V, output, mask []float32,
	seqQ, seqKV, headDim, numHeads, dim int, scale float32, tileSize int) {
	C.accel_tiled_attention_masked(
		(*C.float)(unsafe.Pointer(&Q[0])),
		(*C.float)(unsafe.Pointer(&K[0])),
		(*C.float)(unsafe.Pointer(&V[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
		(*C.float)(unsafe.Pointer(&mask[0])),
		C.int(seqQ), C.int(seqKV), C.int(headDim), C.int(numHeads), C.int(dim),
		C.float(scale), C.int(tileSize))
}

// tiledAttentionSingle: single-head tiled attention (VAE mid-block)
func accelTiledAttentionSingle(Q, K, V, output []float32,
	seq, dim int, scale float32, tileSize int) {
	C.accel_tiled_attention_single(
		(*C.float)(unsafe.Pointer(&Q[0])),
		(*C.float)(unsafe.Pointer(&K[0])),
		(*C.float)(unsafe.Pointer(&V[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.int(seq), C.int(dim), C.float(scale), C.int(tileSize))
}

const hasAccel = true
