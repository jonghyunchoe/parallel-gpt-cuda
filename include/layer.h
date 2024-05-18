#pragma once

#include <vector>

#include "tensor.h"

using std::vector;

#define MATH_PI 3.1415926535f

/* Elementwise operations */
void gelu(float *d_inout, size_t N);
void add(float *d_inout, float *d_x, size_t N);
void scaling(float *d_inout, float scale, size_t N);

/* Matmul operations */
void linear(float *d_in, float *d_w, float *d_b, float *d_out, size_t M, size_t K, size_t N);
void matmul(float *d_in1, float *d_in2, float *d_out, size_t M, size_t K, size_t N);

/* Data movement operations */
void copy(float *d_in, float *d_out, size_t N);
void transpose(float *d_in, float *d_out, size_t M, size_t N);
void split_qkv(float *d_in, float *d_out, size_t s, size_t H);
void split_head(float *d_in, float *d_out, size_t n_head, size_t s, size_t H);
void concat_head(float *d_in, float *d_out, size_t n_head, size_t s, size_t H_);
void extract_qkv(float *d_in, size_t head_idx, size_t n_head, float *d_q, float *d_k, float *d_v, size_t s, size_t H_);
void merge_head(float *d_in, size_t head_idx, size_t n_head, float *d_out, size_t s, size_t H_);
void token_pos_embedding(int *d_in, float *d_wte, float *d_wpe, float *d_out, size_t s, size_t H);

/* Other operations */
void softmax(float *d_inout, size_t s, size_t V);
void layer_norm(float *d_inout, float *d_gamma, float *d_beta, size_t s, size_t H);
void generate_mask(float *d_inout, size_t s);
int top1_sampling(float *d_in, size_t V);