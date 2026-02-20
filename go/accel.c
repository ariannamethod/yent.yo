// accel.c — BLAS-accelerated kernels for UNet inference
//
// Replaces naive Go loops with hardware-optimized BLAS:
//   Conv2d:  8 nested loops → im2col + single cblas_sgemm
//   Linear:  3 nested loops → single cblas_sgemm
//   Attention: batched matmul → strided cblas_sgemm
//
// macOS: Apple Accelerate (built-in, zero deps)
// Linux: OpenBLAS (apt install libopenblas-dev)

#include "accel.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef __APPLE__
  #ifndef ACCELERATE_NEW_LAPACK
    #define ACCELERATE_NEW_LAPACK
  #endif
  #include <Accelerate/Accelerate.h>
#else
  #include <cblas.h>
#endif

// ---- BLAS matmul ----

void accel_sgemm(int M, int N, int K,
                 float alpha, const float *A, const float *B,
                 float beta, float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
}

// ---- im2col ----

void accel_im2col(const float *input, int Cin, int Hin, int Win,
                  int kH, int kW, int stride, int padding,
                  float *col) {
    int Hout = (Hin + 2 * padding - kH) / stride + 1;
    int Wout = (Win + 2 * padding - kW) / stride + 1;
    int col_cols = Hout * Wout;

    for (int c = 0; c < Cin; c++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int row = (c * kH + kh) * kW + kw;
                for (int oh = 0; oh < Hout; oh++) {
                    for (int ow = 0; ow < Wout; ow++) {
                        int ih = oh * stride - padding + kh;
                        int iw = ow * stride - padding + kw;
                        float val = 0.0f;
                        if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win) {
                            val = input[(c * Hin + ih) * Win + iw];
                        }
                        col[row * col_cols + oh * Wout + ow] = val;
                    }
                }
            }
        }
    }
}

// ---- Conv2d = im2col + GEMM ----

// Static im2col buffer — reused across calls to avoid malloc/free overhead.
// At 64×64 with 320 channels and 3×3 kernel: 2880 × 4096 = ~47MB
static float *s_col_buf = NULL;
static size_t s_col_cap = 0;

void accel_conv2d(const float *input, int Cin, int Hin, int Win,
                  const float *weight, const float *bias, int Cout,
                  int kH, int kW, int stride, int padding,
                  float *output) {
    int Hout = (Hin + 2 * padding - kH) / stride + 1;
    int Wout = (Win + 2 * padding - kW) / stride + 1;
    int col_rows = Cin * kH * kW;
    int col_cols = Hout * Wout;
    size_t col_size = (size_t)col_rows * col_cols;

    // Grow static buffer if needed (never shrinks)
    if (col_size > s_col_cap) {
        free(s_col_buf);
        s_col_buf = (float *)malloc(col_size * sizeof(float));
        s_col_cap = col_size;
    }

    accel_im2col(input, Cin, Hin, Win, kH, kW, stride, padding, s_col_buf);

    // GEMM: output[Cout, Hout*Wout] = weight[Cout, Cin*kH*kW] @ col[Cin*kH*kW, Hout*Wout]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Cout, col_cols, col_rows,
                1.0f, weight, col_rows,
                s_col_buf, col_cols,
                0.0f, output, col_cols);

    // Add bias
    if (bias != NULL) {
        for (int co = 0; co < Cout; co++) {
            float b = bias[co];
            float *row = output + co * col_cols;
            for (int i = 0; i < col_cols; i++) {
                row[i] += b;
            }
        }
    }
}

// ---- Batched matmul for attention ----

void accel_batched_matmul_nt(int batch, int M, int N, int K,
                             float scale, const float *A, const float *B,
                             float *C) {
    int strideA = M * K;
    int strideB = N * K;
    int strideC = M * N;

    for (int b = 0; b < batch; b++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K,
                    scale,
                    A + b * strideA, K,
                    B + b * strideB, K,
                    0.0f,
                    C + b * strideC, N);
    }
}

void accel_batched_matmul_nn(int batch, int M, int N, int K,
                             const float *A, const float *B,
                             float *C) {
    int strideA = M * K;
    int strideB = K * N;
    int strideC = M * N;

    for (int b = 0; b < batch; b++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1.0f,
                    A + b * strideA, K,
                    B + b * strideB, N,
                    0.0f,
                    C + b * strideC, N);
    }
}

// ---- Vectorized softmax (row-wise) ----
// scores: [rows, cols], softmax applied to each row in-place

#ifdef __APPLE__

void accel_softmax_rows(float *scores, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float *row = scores + i * cols;

        // max
        float maxVal;
        vDSP_maxv(row, 1, &maxVal, cols);

        // row -= max (for numerical stability)
        float negMax = -maxVal;
        vDSP_vsadd(row, 1, &negMax, row, 1, cols);

        // exp(row) — vvexpf is Accelerate's vectorized exp
        int n = cols;
        vvexpf(row, row, &n);

        // sum
        float sumVal;
        vDSP_sve(row, 1, &sumVal, cols);

        // row /= sum
        vDSP_vsdiv(row, 1, &sumVal, row, 1, cols);
    }
}

// Vectorized SiLU: out = x * sigmoid(x) = x / (1 + exp(-x))
void accel_silu(const float *x, float *out, int n) {
    // neg = -x
    float negOne = -1.0f;
    vDSP_vsmul(x, 1, &negOne, out, 1, n);

    // out = exp(-x)
    vvexpf(out, out, &n);

    // out = 1 + exp(-x)
    float one = 1.0f;
    vDSP_vsadd(out, 1, &one, out, 1, n);

    // out = x / (1 + exp(-x))
    vDSP_vdiv(out, 1, x, 1, out, 1, n);
}

// Vectorized GroupNorm for one group
void accel_group_norm(const float *input, float *output,
                      const float *weight, const float *bias,
                      int channels_start, int channels_end,
                      int H, int W, float eps) {
    int group_channels = channels_end - channels_start;
    int spatial = H * W;
    int count = group_channels * spatial;

    // Compute mean
    float mean;
    vDSP_meanv(input + channels_start * spatial, 1, &mean, count);

    // Compute variance: sum((x - mean)^2) / count
    // Using vDSP_measqv for mean of squares, then variance = measq - mean^2
    float meanSq;
    vDSP_measqv(input + channels_start * spatial, 1, &meanSq, count);
    float variance = meanSq - mean * mean;

    float invStd = 1.0f / sqrtf(variance + eps);

    // Normalize and apply affine: out = (x - mean) * invStd * weight + bias
    for (int c = channels_start; c < channels_end; c++) {
        const float *in_ptr = input + c * spatial;
        float *out_ptr = output + c * spatial;
        float negMean = -mean;
        float scale = invStd;

        // out = x - mean
        vDSP_vsadd(in_ptr, 1, &negMean, out_ptr, 1, spatial);
        // out *= invStd
        vDSP_vsmul(out_ptr, 1, &scale, out_ptr, 1, spatial);

        if (weight != NULL) {
            float w = weight[c];
            float b = bias[c];
            // out = out * w + b
            vDSP_vsmsa(out_ptr, 1, &w, &b, out_ptr, 1, spatial);
        }
    }
}

void accel_vadd(const float *A, const float *B, float *C, int n) {
    vDSP_vadd(A, 1, B, 1, C, 1, n);
}

void accel_vsma(const float *A, float scalar, const float *B, float *C, int n) {
    // C = A * scalar + B
    vDSP_vsma(A, 1, &scalar, B, 1, C, 1, n);
}

#else /* Linux fallback for vsma */

// Linux fallback — scalar implementations
void accel_softmax_rows(float *scores, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float *row = scores + i * cols;
        float maxVal = row[0];
        for (int j = 1; j < cols; j++) {
            if (row[j] > maxVal) maxVal = row[j];
        }
        float sumExp = 0;
        for (int j = 0; j < cols; j++) {
            row[j] = expf(row[j] - maxVal);
            sumExp += row[j];
        }
        float inv = 1.0f / sumExp;
        for (int j = 0; j < cols; j++) {
            row[j] *= inv;
        }
    }
}

void accel_silu(const float *x, float *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void accel_group_norm(const float *input, float *output,
                      const float *weight, const float *bias,
                      int channels_start, int channels_end,
                      int H, int W, float eps) {
    int group_channels = channels_end - channels_start;
    int spatial = H * W;
    int count = group_channels * spatial;

    float mean = 0;
    for (int c = channels_start; c < channels_end; c++) {
        for (int i = 0; i < spatial; i++) {
            mean += input[c * spatial + i];
        }
    }
    mean /= count;

    float variance = 0;
    for (int c = channels_start; c < channels_end; c++) {
        for (int i = 0; i < spatial; i++) {
            float d = input[c * spatial + i] - mean;
            variance += d * d;
        }
    }
    variance /= count;

    float invStd = 1.0f / sqrtf(variance + eps);
    for (int c = channels_start; c < channels_end; c++) {
        for (int i = 0; i < spatial; i++) {
            float v = (input[c * spatial + i] - mean) * invStd;
            if (weight != NULL) {
                v = v * weight[c] + bias[c];
            }
            output[c * spatial + i] = v;
        }
    }
}

void accel_vadd(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i] + B[i];
}

void accel_vsma(const float *A, float scalar, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i] * scalar + B[i];
}

#endif /* platform-specific ops */

// ============================================================
// TILED ATTENTION — platform-independent, uses cblas_sgemm
// ============================================================
// Key insight: at 64×64 latent with 40 heads, the score matrix per head
// is 4096×4096 = 64MB — way bigger than L3 cache (6-16MB).
// By tiling over Q rows (tile=256 → 4MB per tile), scores stay in cache.
// All buffers are static: zero malloc during inference.

static float *s_attn_kH = NULL;   // [max_seq_kv * max_head_dim]
static float *s_attn_vH = NULL;   // [max_seq_kv * max_head_dim]
static float *s_attn_qT = NULL;   // [tile_size * max_head_dim]
static float *s_attn_scores = NULL; // [tile_size * max_seq_kv]
static float *s_attn_outT = NULL;   // [tile_size * max_head_dim]
static size_t s_attn_kv_cap = 0;
static size_t s_attn_tile_cap = 0;

static void ensure_attn_bufs(int seq_kv, int head_dim, int tile_size) {
    size_t kv_need = (size_t)seq_kv * head_dim;
    if (kv_need > s_attn_kv_cap) {
        free(s_attn_kH); free(s_attn_vH);
        s_attn_kH = (float *)malloc(kv_need * sizeof(float));
        s_attn_vH = (float *)malloc(kv_need * sizeof(float));
        s_attn_kv_cap = kv_need;
    }
    size_t tile_scores = (size_t)tile_size * seq_kv;
    size_t tile_qo = (size_t)tile_size * head_dim;
    size_t tile_need = tile_scores > tile_qo ? tile_scores : tile_qo;
    if (tile_need > s_attn_tile_cap) {
        free(s_attn_qT); free(s_attn_scores); free(s_attn_outT);
        s_attn_qT = (float *)malloc(tile_qo * sizeof(float));
        s_attn_scores = (float *)malloc(tile_scores * sizeof(float));
        s_attn_outT = (float *)malloc(tile_qo * sizeof(float));
        s_attn_tile_cap = tile_need;
    }
}

void accel_tiled_attention(
    const float *Q, const float *K, const float *V, float *output,
    int seq_q, int seq_kv, int head_dim, int num_heads, int dim,
    float scale, int tile_size)
{
    accel_tiled_attention_masked(Q, K, V, output, NULL,
        seq_q, seq_kv, head_dim, num_heads, dim, scale, tile_size);
}

void accel_tiled_attention_masked(
    const float *Q, const float *K, const float *V, float *output,
    const float *mask,
    int seq_q, int seq_kv, int head_dim, int num_heads, int dim,
    float scale, int tile_size)
{
    ensure_attn_bufs(seq_kv, head_dim, tile_size);

    for (int h = 0; h < num_heads; h++) {
        int off = h * head_dim;

        // Deinterleave K, V for this head: [seq_kv, dim] → [seq_kv, head_dim]
        for (int j = 0; j < seq_kv; j++) {
            memcpy(s_attn_kH + j * head_dim, K + j * dim + off, head_dim * sizeof(float));
            memcpy(s_attn_vH + j * head_dim, V + j * dim + off, head_dim * sizeof(float));
        }

        // Process Q in tiles
        for (int ts = 0; ts < seq_q; ts += tile_size) {
            int te = ts + tile_size;
            if (te > seq_q) te = seq_q;
            int tile_rows = te - ts;

            // Deinterleave Q tile: [tile_rows, dim] → [tile_rows, head_dim]
            for (int i = 0; i < tile_rows; i++) {
                memcpy(s_attn_qT + i * head_dim, Q + (ts + i) * dim + off, head_dim * sizeof(float));
            }

            // scores[tile_rows, seq_kv] = scale * qT @ kH^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        tile_rows, seq_kv, head_dim,
                        scale, s_attn_qT, head_dim, s_attn_kH, head_dim,
                        0.0f, s_attn_scores, seq_kv);

            // Add mask if present (causal mask for CLIP)
            if (mask != NULL) {
                for (int i = 0; i < tile_rows; i++) {
                    const float *mask_row = mask + (ts + i) * seq_kv;
                    float *score_row = s_attn_scores + i * seq_kv;
                    for (int j = 0; j < seq_kv; j++) {
                        score_row[j] += mask_row[j];
                    }
                }
            }

            // Softmax each row
            accel_softmax_rows(s_attn_scores, tile_rows, seq_kv);

            // outT[tile_rows, head_dim] = scores @ vH
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        tile_rows, head_dim, seq_kv,
                        1.0f, s_attn_scores, seq_kv, s_attn_vH, head_dim,
                        0.0f, s_attn_outT, head_dim);

            // Interleave back: [tile_rows, head_dim] → output[tile_rows, dim]
            for (int i = 0; i < tile_rows; i++) {
                memcpy(output + (ts + i) * dim + off, s_attn_outT + i * head_dim, head_dim * sizeof(float));
            }
        }
    }
}

// Single-head attention (VAE mid-block): Q/K/V already contiguous [seq, dim]
// No deinterleave needed. Just tile the score matrix.
static float *s_attn1_scores = NULL;
static float *s_attn1_out = NULL;
static size_t s_attn1_cap = 0;

void accel_tiled_attention_single(
    const float *Q, const float *K, const float *V, float *output,
    int seq, int dim, float scale, int tile_size)
{
    size_t scores_need = (size_t)tile_size * seq;
    size_t out_need = (size_t)tile_size * dim;
    size_t need = scores_need > out_need ? scores_need : out_need;
    if (need > s_attn1_cap) {
        free(s_attn1_scores); free(s_attn1_out);
        s_attn1_scores = (float *)malloc(scores_need * sizeof(float));
        s_attn1_out = (float *)malloc(out_need * sizeof(float));
        s_attn1_cap = need;
    }

    for (int ts = 0; ts < seq; ts += tile_size) {
        int te = ts + tile_size;
        if (te > seq) te = seq;
        int tile_rows = te - ts;

        // scores[tile_rows, seq] = scale * Q_tile @ K^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    tile_rows, seq, dim,
                    scale, Q + ts * dim, dim, K, dim,
                    0.0f, s_attn1_scores, seq);

        // Softmax each row
        accel_softmax_rows(s_attn1_scores, tile_rows, seq);

        // out[tile_rows, dim] = scores @ V
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    tile_rows, dim, seq,
                    1.0f, s_attn1_scores, seq, V, dim,
                    0.0f, output + ts * dim, dim);
    }
}
