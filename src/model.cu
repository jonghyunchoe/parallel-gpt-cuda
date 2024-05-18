#include <mpi.h>

#include <cmath>
#include <cstdio>

#include "layer.h"
#include "model.h"

Parameter *attn_b[NUM_LAYER], *attn_w[NUM_LAYER];
Parameter *proj_b[NUM_LAYER], *proj_w[NUM_LAYER];
Parameter *ln_1_b[NUM_LAYER], *ln_1_g[NUM_LAYER];
Parameter *ln_2_b[NUM_LAYER], *ln_2_g[NUM_LAYER];
Parameter *mlp1_b[NUM_LAYER], *mlp1_w[NUM_LAYER];
Parameter *mlp2_b[NUM_LAYER], *mlp2_w[NUM_LAYER];
Parameter *ln_f_b, *ln_f_g;
Parameter *wpe, *wte;

float *d_attn_b[NUM_LAYER], *d_attn_w[NUM_LAYER];
float *d_proj_b[NUM_LAYER], *d_proj_w[NUM_LAYER];
float *d_ln_1_b[NUM_LAYER], *d_ln_1_g[NUM_LAYER];
float *d_ln_2_b[NUM_LAYER], *d_ln_2_g[NUM_LAYER];
float *d_mlp1_b[NUM_LAYER], *d_mlp1_w[NUM_LAYER];
float *d_mlp2_b[NUM_LAYER], *d_mlp2_w[NUM_LAYER];
float *d_ln_f_b, *d_ln_f_g;
float *d_wpe, *d_wte;

void alloc_and_set_parameters(float *param) {
  size_t pos = 0;
  int order[] = {
      0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9,
  };
  for (int i = 0; i < NUM_LAYER; i++) {
    attn_b[order[i]] = new Parameter({3 * HIDDEN_DIM}, param + pos);
    pos += OFFSET1;
    attn_w[order[i]] = new Parameter({HIDDEN_DIM, 3 * HIDDEN_DIM}, param + pos);
    pos += OFFSET2;
    proj_b[order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    proj_w[order[i]] = new Parameter({HIDDEN_DIM, HIDDEN_DIM}, param + pos);
    pos += OFFSET4;
    ln_1_b[order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    ln_1_g[order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    ln_2_b[order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    ln_2_g[order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    mlp1_b[order[i]] = new Parameter({4 * HIDDEN_DIM}, param + pos);
    pos += OFFSET5;
    mlp1_w[order[i]] = new Parameter({HIDDEN_DIM, 4 * HIDDEN_DIM}, param + pos);
    pos += OFFSET6;
    mlp2_b[order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    mlp2_w[order[i]] = new Parameter({4 * HIDDEN_DIM, HIDDEN_DIM}, param + pos);
    pos += OFFSET6;
  }
  ln_f_b = new Parameter({HIDDEN_DIM}, param + pos);
  pos += OFFSET3;
  ln_f_g = new Parameter({HIDDEN_DIM}, param + pos);
  pos += OFFSET3;
  wpe = new Parameter({MAX_SEQ_LEN, HIDDEN_DIM}, param + pos);
  pos += OFFSET7;
  wte = new Parameter({NUM_VOCAB, HIDDEN_DIM}, param + pos);
  pos += OFFSET8;

  for (int i = 0; i < NUM_LAYER; i++) {
    cudaMalloc(&d_attn_b[order[i]], 3 * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_attn_w[order[i]], HIDDEN_DIM * 3 * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_proj_b[order[i]], HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_proj_w[order[i]], HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_ln_1_b[order[i]], HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_ln_1_g[order[i]], HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_ln_2_b[order[i]], HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_ln_2_g[order[i]], HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_mlp1_b[order[i]], 4 * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_mlp1_w[order[i]], HIDDEN_DIM * 4 * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_mlp2_b[order[i]], HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_mlp2_w[order[i]], 4 * HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
  }
  cudaMalloc(&d_ln_f_b, HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_ln_f_g, HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_wpe, MAX_SEQ_LEN * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_wte, NUM_VOCAB * HIDDEN_DIM * sizeof(float));

  for (int i = 0; i < NUM_LAYER; i++) {
    cudaMemcpy(d_attn_b[order[i]], attn_b[order[i]]->buf, 3 * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_attn_w[order[i]], attn_w[order[i]]->buf, HIDDEN_DIM * 3 * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj_b[order[i]], proj_b[order[i]]->buf, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj_w[order[i]], proj_w[order[i]]->buf, HIDDEN_DIM * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln_1_b[order[i]], ln_1_b[order[i]]->buf, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln_1_g[order[i]], ln_1_g[order[i]]->buf, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln_2_b[order[i]], ln_2_b[order[i]]->buf, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln_2_g[order[i]], ln_2_g[order[i]]->buf, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp1_b[order[i]], mlp1_b[order[i]]->buf, 4 * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp1_w[order[i]], mlp1_w[order[i]]->buf, HIDDEN_DIM * 4 * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp2_b[order[i]], mlp2_b[order[i]]->buf, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp2_w[order[i]], mlp2_w[order[i]]->buf, 4 * HIDDEN_DIM * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
  }
  cudaMemcpy(d_ln_f_b, ln_f_b->buf, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ln_f_g, ln_f_g->buf, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wpe, wpe->buf, MAX_SEQ_LEN * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wte, wte->buf, NUM_VOCAB * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
}

void free_parameters() {
  for (int i = 0; i < NUM_LAYER; i++) {
    delete attn_b[i];
    delete attn_w[i];
    delete proj_b[i];
    delete proj_w[i];
    delete ln_1_b[i];
    delete ln_1_g[i];
    delete ln_2_b[i];
    delete ln_2_g[i];
    delete mlp1_b[i];
    delete mlp1_w[i];
    delete mlp2_b[i];
    delete mlp2_w[i];
  }
  delete ln_f_b;
  delete ln_f_g;
  delete wpe;
  delete wte;

  for (int i = 0; i < NUM_LAYER; i++) {
    cudaFree(d_attn_b[i]);
    cudaFree(d_attn_w[i]);
    cudaFree(d_proj_b[i]);
    cudaFree(d_proj_w[i]);
    cudaFree(d_ln_1_b[i]);
    cudaFree(d_ln_1_g[i]);
    cudaFree(d_ln_2_b[i]);
    cudaFree(d_ln_2_g[i]);
    cudaFree(d_mlp1_b[i]);
    cudaFree(d_mlp1_w[i]);
    cudaFree(d_mlp2_b[i]);
    cudaFree(d_mlp2_w[i]);
  }
  cudaFree(d_ln_f_b);
  cudaFree(d_ln_f_g);
  cudaFree(d_wpe);
  cudaFree(d_wte);
}

Activation *embd_a, *ffn_proj_a;
Activation *mha_qkv_proj_a, *mha_out_a, *mha_split_qkv_a, *mha_split_head_a,
    *mha_mask_a, *mha_merge_head_a, *mha_q_a, *mha_k_a, *mha_v_a,
    *mha_attn_out_a, *mha_concat_head_a;
Activation *attn_score_a, *k_transposed_a;
Activation *wte_transposed_a, *residual_a, *logit_a;
Activation *transformer_block_a;

float *d_embd_a, *d_ffn_proj_a;
float *d_mha_qkv_proj_a, *d_mha_out_a, *d_mha_split_qkv_a, *d_mha_split_head_a,
      *d_mha_mask_a, *d_mha_merge_head_a, *d_mha_q_a, *d_mha_k_a, *d_mha_v_a,
      *d_mha_attn_out_a, *d_mha_concat_head_a;
float *d_attn_score_a, *d_k_transposed_a;
float *d_wte_transposed_a, *d_residual_a, *d_logit_a;
float *d_transformer_block_a;

void alloc_activations(size_t prompt_size) {
  embd_a = new Activation({prompt_size, HIDDEN_DIM});

  ffn_proj_a = new Activation({prompt_size, 4 * HIDDEN_DIM});

  mha_qkv_proj_a = new Activation({prompt_size, 3 * HIDDEN_DIM});
  mha_out_a = new Activation({prompt_size, HIDDEN_DIM});
  mha_split_qkv_a = new Activation({3, prompt_size, HIDDEN_DIM});
  mha_split_head_a =
      new Activation({3, NUM_HEAD, prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_mask_a = new Activation({prompt_size, prompt_size});
  mha_merge_head_a =
      new Activation({NUM_HEAD, prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_q_a = new Activation({prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_k_a = new Activation({prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_v_a = new Activation({prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_attn_out_a = new Activation({prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_concat_head_a = new Activation({prompt_size, HIDDEN_DIM});

  attn_score_a = new Activation({prompt_size, prompt_size});
  k_transposed_a = new Activation({HIDDEN_DIM / NUM_HEAD, prompt_size});

  wte_transposed_a = new Activation({HIDDEN_DIM, NUM_VOCAB});

  residual_a = new Activation({prompt_size, HIDDEN_DIM});
  logit_a = new Activation({prompt_size, NUM_VOCAB});
  transformer_block_a = new Activation({prompt_size, HIDDEN_DIM});

  cudaMalloc(&d_embd_a, prompt_size * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_ffn_proj_a, prompt_size * 4 * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_mha_qkv_proj_a, prompt_size * 3 * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_mha_out_a, prompt_size * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_mha_split_qkv_a, 3 * prompt_size * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_mha_split_head_a, 3 * NUM_HEAD * prompt_size * (HIDDEN_DIM / NUM_HEAD) * sizeof(float));
  cudaMalloc(&d_mha_mask_a, prompt_size * prompt_size * sizeof(float));
  cudaMalloc(&d_mha_merge_head_a, NUM_HEAD * prompt_size * (HIDDEN_DIM / NUM_HEAD) * sizeof(float));
  cudaMalloc(&d_mha_q_a, prompt_size * (HIDDEN_DIM / NUM_HEAD) * sizeof(float));
  cudaMalloc(&d_mha_k_a, prompt_size * (HIDDEN_DIM / NUM_HEAD) * sizeof(float));
  cudaMalloc(&d_mha_v_a, prompt_size * (HIDDEN_DIM / NUM_HEAD) * sizeof(float));
  cudaMalloc(&d_mha_attn_out_a, prompt_size * (HIDDEN_DIM / NUM_HEAD) * sizeof(float));
  cudaMalloc(&d_mha_concat_head_a, prompt_size * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_attn_score_a, prompt_size * prompt_size * sizeof(float));
  cudaMalloc(&d_k_transposed_a, (HIDDEN_DIM / NUM_HEAD) * prompt_size * sizeof(float));
  cudaMalloc(&d_wte_transposed_a, HIDDEN_DIM * NUM_VOCAB * sizeof(float));
  cudaMalloc(&d_residual_a, prompt_size * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_logit_a, prompt_size * NUM_VOCAB * sizeof(float));
  cudaMalloc(&d_transformer_block_a, prompt_size * HIDDEN_DIM * sizeof(float));

  cudaMemcpy(d_embd_a, embd_a->buf, prompt_size * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ffn_proj_a, ffn_proj_a->buf, prompt_size * 4 * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mha_qkv_proj_a, mha_qkv_proj_a->buf, prompt_size * 3 * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mha_out_a, mha_out_a->buf, prompt_size * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mha_split_qkv_a, mha_split_qkv_a->buf, 3 * prompt_size * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mha_split_head_a, mha_split_head_a->buf, 3 * NUM_HEAD * prompt_size * (HIDDEN_DIM / NUM_HEAD) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mha_mask_a, mha_mask_a->buf, prompt_size * prompt_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mha_merge_head_a, mha_merge_head_a->buf, NUM_HEAD * prompt_size * (HIDDEN_DIM / NUM_HEAD) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mha_q_a, mha_q_a->buf, prompt_size * (HIDDEN_DIM / NUM_HEAD) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mha_k_a, mha_k_a->buf, prompt_size * (HIDDEN_DIM / NUM_HEAD) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mha_v_a, mha_v_a->buf, prompt_size * (HIDDEN_DIM / NUM_HEAD) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mha_attn_out_a, mha_attn_out_a->buf, prompt_size * (HIDDEN_DIM / NUM_HEAD) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mha_concat_head_a, mha_concat_head_a->buf, prompt_size * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_attn_score_a, attn_score_a->buf, prompt_size * prompt_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k_transposed_a, k_transposed_a->buf, (HIDDEN_DIM / NUM_HEAD) * prompt_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wte_transposed_a, wte_transposed_a->buf, HIDDEN_DIM * NUM_VOCAB * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_residual_a, residual_a->buf, prompt_size * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_logit_a, logit_a->buf, prompt_size * NUM_VOCAB * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_transformer_block_a, transformer_block_a->buf, prompt_size * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
}

void free_activations() {
  delete embd_a;
  delete ffn_proj_a;
  delete mha_qkv_proj_a;
  delete mha_out_a;
  delete mha_split_qkv_a;
  delete mha_split_head_a;
  delete mha_mask_a;
  delete mha_merge_head_a;
  delete mha_q_a;
  delete mha_k_a;
  delete mha_v_a;
  delete mha_attn_out_a;
  delete mha_concat_head_a;
  delete attn_score_a;
  delete k_transposed_a;
  delete wte_transposed_a;
  delete residual_a;
  delete logit_a;
  delete transformer_block_a;

  cudaFree(d_embd_a);
  cudaFree(d_ffn_proj_a);
  cudaFree(d_mha_qkv_proj_a);
  cudaFree(d_mha_out_a);
  cudaFree(d_mha_split_qkv_a);
  cudaFree(d_mha_split_head_a);
  cudaFree(d_mha_mask_a);
  cudaFree(d_mha_merge_head_a);
  cudaFree(d_mha_q_a);
  cudaFree(d_mha_k_a);
  cudaFree(d_mha_v_a);
  cudaFree(d_mha_attn_out_a);
  cudaFree(d_mha_concat_head_a);
  cudaFree(d_attn_score_a);
  cudaFree(d_k_transposed_a);
  cudaFree(d_wte_transposed_a);
  cudaFree(d_residual_a);
  cudaFree(d_logit_a);
  cudaFree(d_transformer_block_a);
}

/* (Position-wise) Feed-Forward Network
 * @param [in1]     in: [seq_len, HIDDEN_DIM]
 * @param [in2] mlp1_w: [HIDDEN_DIM, 4*HIDDEN_DIM]
 * @param [in3] mlp1_b: [4*HIDDEN_DIM]
 * @param [in4] mlp2_w: [4*HIDDEN_DIM, HIDDEN_DIM]
 * @param [in5] mlp2_b: [HIDDEN_DIM]
 * @param [out]    out: [seq_len, HIDDEN_DIM]
 */
void ffn(float *d_in, float *d_mlp1_w, float *d_mlp1_b, float *d_mlp2_w, float *d_mlp2_b, float *d_out, size_t seq_len, size_t hidden_dim) {
  /* Projection Up:
    [seq_len, HIDDEN_DIM] -> [seq_len, 4*HIDDEN_DIM] */
  linear(d_in, d_mlp1_w, d_mlp1_b, d_ffn_proj_a, seq_len, HIDDEN_DIM, 4 * HIDDEN_DIM);

  /* GELU */
  gelu(d_ffn_proj_a, seq_len * 4 * HIDDEN_DIM);

  /* Projection Down:
    [seq_len, 4*HIDDEN_DIM] -> [seq_len, HIDDEN_DIM] */
  linear(d_ffn_proj_a, d_mlp2_w, d_mlp2_b, d_out, seq_len, 4 * HIDDEN_DIM, HIDDEN_DIM);
}

/* Attention
 * @param [in1]    q: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in2]    k: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in3]    v: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in4] mask: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [out]  out: [seq_len, HIDDEN_DIM/NUM_HEAD]
 */
void attention(float *d_q, float *d_k, float *d_v, float *d_mask,
               float *d_out, size_t seq_len, size_t head_dim) {
  /* Get Attention score by q @ k */
  transpose(d_k, d_k_transposed_a, seq_len, head_dim);
  matmul(d_q, d_k_transposed_a, d_attn_score_a, seq_len, head_dim, seq_len);

  /* Scaling */
  scaling(d_attn_score_a, (1.0 / sqrt(head_dim)), seq_len * seq_len);

  /* Masking */
  add(d_attn_score_a, d_mask, seq_len * seq_len);

  /* Softmax */
  softmax(d_attn_score_a, seq_len, seq_len);

  /* Attention score @ v */
  matmul(d_attn_score_a, d_v, d_out, seq_len, seq_len, head_dim);
}

/* (Masked) Multi-Head Self Attention
 * @param [in1]     in: [seq_len, HIDDEN_DIM]
 * @param [in2] attn_b: [3*HIDDEN_DIM]
 * @param [in3] attn_w: [HIDDEN_DIM, 3*HIDDEN_DIM]
 * @param [in4] proj_b: [HIDDEN_DIM]
 * @param [in5] proj_w: [HIDDEN_DIM, HIDDEN_DIM]
 * @param [out]    out: [seq_len, HIDDEN_DIM]
 */
void mha(float *d_in, float *d_attn_b, float *d_attn_w,
         float *d_proj_b, float *d_proj_w, float *d_out, size_t seq_len, size_t hidden_dim, size_t num_head) {
  size_t head_dim = HIDDEN_DIM / NUM_HEAD;

  /* QKV projection:
    [seq_len, HIDDEN_DIM] ->
    [seq_len, 3*HIDDEN_DIM] */
  linear(d_in, d_attn_w, d_attn_b, d_mha_qkv_proj_a, seq_len, HIDDEN_DIM, 3 * HIDDEN_DIM);

  /* Split into Q, K, V:
    [seq_len, 3*HIDDEN_DIM] ->
    [3, seq_len, HIDDEN_DIM] */
  split_qkv(d_mha_qkv_proj_a, d_mha_split_qkv_a, seq_len, HIDDEN_DIM);

  /* Split into multiple heads:
    [3, seq_len, HIDDEN_DIM] ->
    [3, NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
  split_head(d_mha_split_qkv_a, d_mha_split_head_a, NUM_HEAD, seq_len, HIDDEN_DIM);

  /* Generate mask to hide future inputs */
  generate_mask(d_mha_mask_a, seq_len);

  /* Perform Attention over each head:
    [NUM_HEAD, 3, seq_len, HIDDEN_DIM/NUM_HEAD] ->
    [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
  for (size_t idx = 0; idx < NUM_HEAD; idx++) {
    /* Extract Q, K, V from qkv_head */
    extract_qkv(d_mha_split_head_a, idx, NUM_HEAD, d_mha_q_a, d_mha_k_a, d_mha_v_a, seq_len, head_dim);

    /* Attention */
    attention(d_mha_q_a, d_mha_k_a, d_mha_v_a, d_mha_mask_a, d_mha_attn_out_a, seq_len, head_dim);

    /* Merge each head's attn output
      [seq_len, HIDDEN_DIM/NUM_HEAD] ->
      [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
    merge_head(d_mha_attn_out_a, idx, NUM_HEAD, d_mha_merge_head_a, seq_len, head_dim);
  }

  /* Concat each heads:
    [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] ->
    [seq_len, HIDDEN_DIM] */
  concat_head(d_mha_merge_head_a, d_mha_concat_head_a, NUM_HEAD, seq_len, head_dim);

  /* OUT projection:
    [seq_len, HIDDEN_DIM] -> [seq_len, HIDDEN_DIM] */
  linear(d_mha_concat_head_a, d_proj_w, d_proj_b, d_out, seq_len, HIDDEN_DIM, HIDDEN_DIM);
}

/* Transformer Block
 * @param [in1]      in: [seq_len, HIDDEN_DIM]
 * @param [in2]  attn_b: [3*HIDDEN_DIM]
 * @param [in3]  attn_w: [HIDDEN_DIM, 3*HIDDEN_DIM]
 * @param [in4]  proj_b: [HIDDEN_DIM]
 * @param [in5]  proj_w: [HIDDEN_DIM, HIDDEN_DIM]
 * @param [in6]  ln_1_b: [HIDDEN_DIM]
 * @param [in7]  ln_1_g: [HIDDEN_DIM]
 * @param [in8]  ln_2_b: [HIDDEN_DIM]
 * @param [in9]  ln_2_g: [HIDDEN_DIM]
 * @param [in10] mlp1_b: [4*HIDDEN_DIM]
 * @param [in11] mlp1_w: [HIDDEN_DIM, 4*HIDDEN_DIM]
 * @param [in12] mlp2_b: [HIDDEN_DIM]
 * @param [in13] mlp2_w: [4*HIDDEN_DIM, HIDDEN_DIM]
 * @param [out]     out: [seq_len, HIDDEN_DIM]
 */
void transformer_block(float *d_in, float *d_attn_b, float *d_attn_w,
                       float *d_proj_b, float *d_proj_w, float *d_ln_1_b,
                       float *d_ln_1_g, float *d_ln_2_b, float *d_ln_2_g,
                       float *d_mlp1_b, float *d_mlp1_w, float *d_mlp2_b,
                       float *d_mlp2_w, float *d_out, size_t seq_len, size_t hidden_dim, size_t num_head) {
  /* Copy Residual */
  copy(d_in, d_residual_a, seq_len * HIDDEN_DIM);

  /* Layer Normalization */
  layer_norm(d_in, d_ln_1_g, d_ln_1_b, seq_len, HIDDEN_DIM);

  /* Masked Multi-Head Self-Attention */
  mha(d_in, d_attn_b, d_attn_w, d_proj_b, d_proj_w, d_mha_out_a, seq_len, HIDDEN_DIM, NUM_HEAD);

  /* Add Residual */
  add(d_mha_out_a, d_residual_a, seq_len * HIDDEN_DIM);

  /* Copy Residual */
  copy(d_mha_out_a, d_residual_a, seq_len * HIDDEN_DIM);

  /* Layer Normalization */
  layer_norm(d_mha_out_a, d_ln_2_g, d_ln_2_b, seq_len, HIDDEN_DIM);

  /* Position-wise Feed-Forward Network */
  ffn(d_mha_out_a, d_mlp1_w, d_mlp1_b, d_mlp2_w, d_mlp2_b, d_out, seq_len, HIDDEN_DIM);

  /* Add Residual */
  add(d_out, d_residual_a, seq_len * HIDDEN_DIM);
}

/* [Model Computation: Token Generation] */
void generate_tokens(int *input, int *output, size_t n_prompt, size_t n_token) {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    int *d_input, *d_output; 
    size_t input_size = n_prompt * tokens_per_prompt * sizeof(int);
    size_t output_size = n_prompt * n_token * sizeof(int);
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);

    /* Outer loop: generate tokens for each prompt */
    for (size_t p = 0; p < n_prompt; p++) {
      int prompt_size = tokens_per_prompt;

      /* Initialize input prompt */
      // vector<int> input_prompt(prompt_size);
      // memcpy(input_prompt.data(), input + p * prompt_size,
      //        prompt_size * sizeof(int));
      
      int *d_input_prompt;
      cudaMalloc(&d_input_prompt, prompt_size * sizeof(int));

      cudaMemcpy(d_input_prompt, d_input + p * tokens_per_prompt, prompt_size * sizeof(int), cudaMemcpyDeviceToDevice);


      /* Inner loop: generate next token */
      for (size_t t = 0; t < n_token; t++) {
        /* Initialize activations */
        alloc_activations(prompt_size);

        /* Token + Positional Embedding */
        token_pos_embedding(d_input_prompt, d_wte, d_wpe, d_embd_a, prompt_size, HIDDEN_DIM);

        /* Forward path of Transformer blocks */
        for (size_t l = 0; l < NUM_LAYER; l++) {
          transformer_block(d_embd_a, d_attn_b[l], d_attn_w[l], d_proj_b[l], d_proj_w[l],
                            d_ln_1_b[l], d_ln_1_g[l], d_ln_2_b[l], d_ln_2_g[l],
                            d_mlp1_b[l], d_mlp1_w[l], d_mlp2_b[l], d_mlp2_w[l],
                            d_transformer_block_a, prompt_size, HIDDEN_DIM, NUM_HEAD);

          /* Copy output to embd_a for next block */
          copy(d_transformer_block_a, d_embd_a, prompt_size * HIDDEN_DIM);
        }

        /* Final Layer Normalization */
        layer_norm(d_embd_a, d_ln_f_g, d_ln_f_b, prompt_size, HIDDEN_DIM);

        /* Projection to vocab. dimension */
        float *d_wte_transposed;
        cudaMalloc(&d_wte_transposed, NUM_VOCAB * HIDDEN_DIM * sizeof(float));
        transpose(d_wte, d_wte_transposed, NUM_VOCAB, HIDDEN_DIM);
        matmul(d_embd_a, d_wte_transposed, d_logit_a, prompt_size, HIDDEN_DIM, NUM_VOCAB);

        /* Greedy sampling (only last timestep is considered) */
        int next_token_id = top1_sampling(d_logit_a + (prompt_size - 1) * NUM_VOCAB, NUM_VOCAB);
        // int next_token_id = top1_sampling(logit_a);

        /* Update input prompt and prompt size */
        cudaMemcpy(d_input_prompt + prompt_size, &next_token_id, sizeof(int), cudaMemcpyHostToDevice);
        // input_prompt.push_back(next_token_id);
        prompt_size += 1;

        /* Store generated token to output */
        cudaMemcpy(d_output + p * n_token + t, &next_token_id, sizeof(int), cudaMemcpyDeviceToDevice);
        // output[p * n_token + t] = next_token_id;

        /* Finalize activations for next token generation */
        free_activations();
      }
      cudaFree(d_input_prompt);
    }

    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
  }
}
