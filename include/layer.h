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
