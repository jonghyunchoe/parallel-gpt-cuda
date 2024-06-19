#pragma once

#include <vector>

#include "tensor.h"

using std::vector;

#define MATH_PI 3.1415926535f
#define LEN_INPUT 16
#define LEN_OUTPUT 8

/* Elementwise operations */
void add(float *inout, float *x, int N);

/* Matmul operations */
void linear(float *in, float *w, float *b, float *out,
            int M, int N, int K,
            int lda, int ldb, int ldc, float *sum_out, bool gelu);
void linear(float *in, float *w, float *b, float *out,
            int M, int N, int K, int lda, int ldb, int ldc,
            int B, int ldab, int ldcb, float *sum_out, bool gelu);
void lm_head(float *in1, float *in2, float *out, int B, int L, int V, int H);

/* Other operations */
void layer_norm(float *in, float *gamma, float *beta, float *out, int N, int H);
void top1_sampling(float *in, int *out, int N, int V, int ldin);
void attention(float *q, float *kv, float *o,
               int h_dim, int n_head, int batch_size, int q_len, int kv_len);
void token_pos_embedding(int *in, float *wte, float *wpe, float *out,
                         int B, int L, int H, int pos, int V);
