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
}

Activation *embd_a, *ffn_proj_a;
Activation *mha_qkv_proj_a, *mha_out_a, *mha_split_qkv_a, *mha_split_head_a,
    *mha_mask_a, *mha_merge_head_a, *mha_q_a, *mha_k_a, *mha_v_a,
    *mha_attn_out_a, *mha_concat_head_a;
Activation *attn_score_a, *k_transposed_a;
Activation *wte_transposed_a, *residual_a, *logit_a;
Activation *transformer_block_a;

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
}

/* (Position-wise) Feed-Forward Network
 * @param [in1]     in: [seq_len, HIDDEN_DIM]
 * @param [in2] mlp1_w: [HIDDEN_DIM, 4*HIDDEN_DIM]
 * @param [in3] mlp1_b: [4*HIDDEN_DIM]
 * @param [in4] mlp2_w: [4*HIDDEN_DIM, HIDDEN_DIM]
 * @param [in5] mlp2_b: [HIDDEN_DIM]
 * @param [out]    out: [seq_len, HIDDEN_DIM]
 */
void ffn(Activation *in, Parameter *mlp1_w, Parameter *mlp1_b,
         Parameter *mlp2_w, Parameter *mlp2_b, Activation *out) {
  /* Projection Up:
    [seq_len, HIDDEN_DIM] -> [seq_len, 4*HIDDEN_DIM] */
  linear(in, mlp1_w, mlp1_b, ffn_proj_a);

  /* GELU */
  gelu(ffn_proj_a);

  /* Projection Down:
    [seq_len, 4*HIDDEN_DIM] -> [seq_len, HIDDEN_DIM] */
  linear(ffn_proj_a, mlp2_w, mlp2_b, out);
}

/* Attention
 * @param [in1]    q: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in2]    k: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in3]    v: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in4] mask: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [out]  out: [seq_len, HIDDEN_DIM/NUM_HEAD]
 */
void attention(Activation *q, Activation *k, Activation *v, Activation *mask,
               Activation *out) {
  /* Get Attention score by q @ k */
  transpose(k, k_transposed_a);
  matmul(q, k_transposed_a, attn_score_a);

  /* Scaling */
  scaling(attn_score_a, (1.0 / sqrt(k->shape[1])));

  /* Masking */
  add(attn_score_a, mask);

  /* Softmax */
  softmax(attn_score_a);

  /* Attention score @ v */
  matmul(attn_score_a, v, out);
}

/* (Masked) Multi-Head Self Attention
 * @param [in1]     in: [seq_len, HIDDEN_DIM]
 * @param [in2] attn_b: [3*HIDDEN_DIM]
 * @param [in3] attn_w: [HIDDEN_DIM, 3*HIDDEN_DIM]
 * @param [in4] proj_b: [HIDDEN_DIM]
 * @param [in5] proj_w: [HIDDEN_DIM, HIDDEN_DIM]
 * @param [out]    out: [seq_len, HIDDEN_DIM]
 */
void mha(Activation *in, Parameter *attn_b, Parameter *attn_w,
         Parameter *proj_b, Parameter *proj_w, Activation *out) {
  /* QKV projection:
    [seq_len, HIDDEN_DIM] ->
    [seq_len, 3*HIDDEN_DIM] */
  linear(in, attn_w, attn_b, mha_qkv_proj_a);

  /* Split into Q, K, V:
    [seq_len, 3*HIDDEN_DIM] ->
    [3, seq_len, HIDDEN_DIM] */
  split_qkv(mha_qkv_proj_a, mha_split_qkv_a);

  /* Split into multiple heads:
    [3, seq_len, HIDDEN_DIM] ->
    [3, NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
  split_head(mha_split_qkv_a, NUM_HEAD, mha_split_head_a);

  /* Generate mask to hide future inputs */
  generate_mask(mha_mask_a);

  /* Perform Attention over each head:
    [NUM_HEAD, 3, seq_len, HIDDEN_DIM/NUM_HEAD] ->
    [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
  for (size_t idx = 0; idx < NUM_HEAD; idx++) {
    /* Extract Q, K, V from qkv_head */
    extract_qkv(mha_split_head_a, idx, NUM_HEAD, mha_q_a, mha_k_a, mha_v_a);

    /* Attention */
    attention(mha_q_a, mha_k_a, mha_v_a, mha_mask_a, mha_attn_out_a);

    /* Merge each head's attn output
      [seq_len, HIDDEN_DIM/NUM_HEAD] ->
      [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
    merge_head(mha_attn_out_a, idx, NUM_HEAD, mha_merge_head_a);
  }

  /* Concat each heads:
    [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] ->
    [seq_len, HIDDEN_DIM] */
  concat_head(mha_merge_head_a, mha_concat_head_a);

  /* OUT projection:
    [seq_len, HIDDEN_DIM] -> [seq_len, HIDDEN_DIM] */
  linear(mha_concat_head_a, proj_w, proj_b, out);
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
void transformer_block(Activation *in, Parameter *attn_b, Parameter *attn_w,
                       Parameter *proj_b, Parameter *proj_w, Parameter *ln_1_b,
                       Parameter *ln_1_g, Parameter *ln_2_b, Parameter *ln_2_g,
                       Parameter *mlp1_b, Parameter *mlp1_w, Parameter *mlp2_b,
                       Parameter *mlp2_w, Activation *out) {
  /* Copy Residual */
  copy(in, residual_a);

  /* Layer Normalization */
  layer_norm(in, ln_1_g, ln_1_b);

  /* Masked Multi-Head Self-Attention */
  mha(in, attn_b, attn_w, proj_b, proj_w, mha_out_a);

  /* Add Residual */
  add(mha_out_a, residual_a);

  /* Copy Residual */
  copy(mha_out_a, residual_a);

  /* Layer Normalization */
  layer_norm(mha_out_a, ln_2_g, ln_2_b);

  /* Position-wise Feed-Forward Network */
  ffn(mha_out_a, mlp1_w, mlp1_b, mlp2_w, mlp2_b, out);

  /* Add Residual */
  add(out, residual_a);
}

/* [Model Computation: Token Generation] */
void generate_tokens(int *input, int *output, size_t n_prompt, size_t n_token) {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    /* Outer loop: generate tokens for each prompt */
    for (size_t p = 0; p < n_prompt; p++) {
      int prompt_size = tokens_per_prompt;

      /* Initialize input prompt */
      vector<int> input_prompt(prompt_size);
      memcpy(input_prompt.data(), input + p * prompt_size,
             prompt_size * sizeof(int));

      /* Inner loop: generate next token */
      for (size_t t = 0; t < n_token; t++) {
        /* Initialize activations */
        alloc_activations(prompt_size);

        /* Token + Positional Embedding */
        token_pos_embedding(input_prompt, wte, wpe, embd_a);

        /* Forward path of Transformer blocks */
        for (size_t l = 0; l < NUM_LAYER; l++) {
          transformer_block(embd_a, attn_b[l], attn_w[l], proj_b[l], proj_w[l],
                            ln_1_b[l], ln_1_g[l], ln_2_b[l], ln_2_g[l],
                            mlp1_b[l], mlp1_w[l], mlp2_b[l], mlp2_w[l],
                            transformer_block_a);

          /* Copy output to embd_a for next block */
          copy(transformer_block_a, embd_a);
        }

        /* Final Layer Normalization */
        layer_norm(embd_a, ln_f_g, ln_f_b);

        /* Projection to vocab. dimension */
        transpose(wte, wte_transposed_a);
        matmul(embd_a, wte_transposed_a, logit_a);

        /* Greedy sampling (only last timestep is considered) */
        int next_token_id = top1_sampling(logit_a);

        /* Update input prompt and prompt size */
        input_prompt.push_back(next_token_id);
        prompt_size += 1;

        /* Store generated token to output */
        output[p * n_token + t] = next_token_id;

        /* Finalize activations for next token generation */
        free_activations();
      }
    }
  }
}
