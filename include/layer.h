#pragma once

#include <vector>

#include "tensor.h"

using std::vector;

#define MATH_PI 3.1415926535f

/* Elementwise operations */
void gelu(Tensor *inout);
void add(Tensor *inout, Tensor *x);
void add_cuda(Tensor *inout, Tensor *x);
void scaling(Tensor *inout, float scale);

/* Matmul operations */
void linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void matmul(Tensor *in1, Tensor *in2, Tensor *out);

/* Data movement operations */
void copy(Tensor *in, Tensor *out);
void transpose(Tensor *in, Tensor *out);
void split_qkv(Tensor *in, Tensor *out);
void split_head(Tensor *in, size_t n_head, Tensor *out);
void concat_head(Tensor *in, Tensor *out);
void extract_qkv(Tensor *in, size_t head_idx, size_t n_head, Tensor *q,
                 Tensor *k, Tensor *v);
void merge_head(Tensor *in, size_t head_idx, size_t n_head, Tensor *out);
void token_pos_embedding(vector<int> in, Parameter *wte, Parameter *wpe,
                        Tensor *out);

/* Other operations */
void softmax(Tensor *inout);
void layer_norm(Tensor *inout, Tensor *gamma, Tensor *beta);
void generate_mask(Tensor *inout);
int top1_sampling(Tensor *in);

/* Operations that receive device pointers */
void token_pos_embedding(int *d_in, float *d_wte, float *d_wpe, float *d_out, size_t s, size_t H);
void gelu(float *d_inout, size_t N);
void softmax(float *d_inout, size_t s, size_t V);
void layer_norm(float *d_inout, float *d_gamma, float *d_beta, size_t s, size_t H, float eps);
void linear(float *d_in, float *d_w, float *d_b, float *d_out, size_t M, size_t K, size_t N);
void matmul(float *d_in1, float *d_in2, float *d_out, size_t M, size_t K, size_t N);
void transpose(float *d_in, float *d_out, size_t M, size_t N);
void scaling(float *d_inout, float scale, size_t N);
void generate_mask(float *d_inout, size_t s);
void copy(float *d_in, float *d_out, size_t N);
void add(float *d_inout, float *d_x, size_t N);
void split_qkv(float *d_in, float *d_out, size_t s, size_t H);
void split_head(float *d_in, float *d_out, size_t n_head, size_t s, size_t H);
void extract_qkv(float *d_in, float *d_q, float *d_k, float *d_v, size_t head_idx, size_t n_head, size_t s, size_t H_);
void merge_head(float *d_in, float *d_out, size_t head_idx, size_t s, size_t H_);
void concat_head(float *d_in, float *d_out, size_t n_head, size_t s, size_t H_);
void top1_sampling(float *d_in, int *d_out, size_t s, size_t V);

/* Batch operations that receive device pointers */ 
void batch_token_pos_embedding(int *d_in, float *d_wte, float *d_wpe, float *d_out, size_t batch_size, size_t s, size_t H);
void batch_gelu(float *d_inout, size_t batch_size, size_t N);
void batch_softmax(float *d_inout, size_t batch_size, size_t s, size_t V);
void batch_layer_norm(float *d_inout, float *d_gamma, float *d_beta, size_t batch_size, size_t s, size_t H, float eps);
void batch_linear(float *d_in, float *d_w, float *d_b, float *d_out, size_t batch_size, size_t M, size_t K, size_t N);
void batch_matmul(float *d_in1, float *d_in2, float *d_out, size_t batch_size, size_t M, size_t K, size_t N);
void batch_transpose(float *d_in, float *d_out, size_t batch_size, size_t M, size_t N);
void batch_scaling(float *d_inout, float scale, size_t batch_size, size_t N);
void batch_generate_mask(float *d_inout, size_t batch_size, size_t s);
void batch_copy(float *d_in, float *d_out, size_t batch_size, size_t N);
void batch_add(float *d_inout, float *d_x, size_t batch_size, size_t N);
void batch_split_qkv(float *d_in, float *d_out, size_t batch_size, size_t s, size_t H);
void batch_split_head(float *d_in, float *d_out, size_t batch_size, size_t n_head, size_t s, size_t H);
void batch_extract_qkv(float *d_in, float *d_q, float *d_k, float *d_v, size_t batch_size, size_t head_idx, size_t n_head, size_t s, size_t H_);
void batch_merge_head(float *d_in, float *d_out, size_t batch_size, size_t n_head, size_t head_idx, size_t s, size_t H_);
void batch_concat_head(float *d_in, float *d_out, size_t batch_size, size_t n_head, size_t s, size_t H_);
void batch_top1_sampling(float *d_in, int *d_out, size_t batch_size, size_t n_token, size_t position, size_t s, size_t V);

void batch_matmul_final(float *d_in1, float *d_in2, float *d_out, size_t batch_size, size_t M, size_t K, size_t N);
void batch_update_cache(float *kv, float *kv_cache, size_t batch_size, size_t seq_len, size_t head_dim, size_t t);