#ifndef ACCEL_H
#define ACCEL_H

#include <stdint.h>

// BLAS matrix multiply: C = alpha * A @ B + beta * C
// A: [M x K], B: [K x N], C: [M x N]  (row-major)
void accel_sgemm(int M, int N, int K,
                 float alpha, const float *A, const float *B,
                 float beta, float *C);

// im2col: unfold Conv2d input into a column matrix for GEMM
// input: [Cin, Hin, Win], output: [Cin*kH*kW, Hout*Wout]
void accel_im2col(const float *input, int Cin, int Hin, int Win,
                  int kH, int kW, int stride, int padding,
                  float *col);

// Conv2d via im2col + GEMM: output = weight @ im2col(input) + bias
// input: [Cin, Hin, Win], weight: [Cout, Cin*kH*kW], bias: [Cout] or NULL
// output: [Cout, Hout, Wout]
void accel_conv2d(const float *input, int Cin, int Hin, int Win,
                  const float *weight, const float *bias, int Cout,
                  int kH, int kW, int stride, int padding,
                  float *output);

// Batched matmul for attention: C[b] = A[b] @ B[b]^T * scale
// A: [batch, M, K], B: [batch, N, K], C: [batch, M, N]
void accel_batched_matmul_nt(int batch, int M, int N, int K,
                             float scale, const float *A, const float *B,
                             float *C);

// Batched matmul: C[b] = A[b] @ B[b]
// A: [batch, M, K], B: [batch, K, N], C: [batch, M, N]
void accel_batched_matmul_nn(int batch, int M, int N, int K,
                             const float *A, const float *B,
                             float *C);

// Vectorized softmax: apply softmax to each row of [rows, cols] matrix in-place
void accel_softmax_rows(float *scores, int rows, int cols);

// Vectorized SiLU: out = x * sigmoid(x), n elements
void accel_silu(const float *x, float *out, int n);

// Vectorized GroupNorm for one group of channels
void accel_group_norm(const float *input, float *output,
                      const float *weight, const float *bias,
                      int channels_start, int channels_end,
                      int H, int W, float eps);

// Vectorized element-wise add: C = A + B, n elements
void accel_vadd(const float *A, const float *B, float *C, int n);

// Vectorized scalar mul + add: C = A * scalar + B
void accel_vsma(const float *A, float scalar, const float *B, float *C, int n);

// ---- Fused tiled attention ----
// Full multi-head attention in C: deinterleave + tiled Q×K^T + softmax + ×V + reinterleave
// Q: [seq_q, dim] (interleaved heads), K/V: [seq_kv, dim], output: [seq_q, dim]
// Tiles over Q rows (tile_size) so score matrix fits in L3 cache.
// All buffers allocated once as static — zero malloc per call.
void accel_tiled_attention(
    const float *Q, const float *K, const float *V, float *output,
    int seq_q, int seq_kv, int head_dim, int num_heads, int dim,
    float scale, int tile_size);

// Same but with additive mask (e.g. causal mask for CLIP)
// mask: [seq_q, seq_kv], added to scores before softmax. NULL = no mask.
void accel_tiled_attention_masked(
    const float *Q, const float *K, const float *V, float *output,
    const float *mask,
    int seq_q, int seq_kv, int head_dim, int num_heads, int dim,
    float scale, int tile_size);

// Single-head tiled attention (VAE mid-block): Q/K/V are [seq, dim] contiguous
// No deinterleave needed. Tiles over Q rows.
void accel_tiled_attention_single(
    const float *Q, const float *K, const float *V, float *output,
    int seq, int dim, float scale, int tile_size);

#endif
