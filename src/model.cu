#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <chrono>

#include "layer.h"
#include "model.h"

#define BATCH_SIZE 512
#define ALLOC_VOCAB ((NUM_VOCAB + 127) / 128) * 128

int local_rank;

Parameter *attn_b[NUM_LAYER], *attn_w[NUM_LAYER];
Parameter *proj_b[NUM_LAYER], *proj_w[NUM_LAYER];
Parameter *ln_1_b[NUM_LAYER], *ln_1_g[NUM_LAYER];
Parameter *ln_2_b[NUM_LAYER], *ln_2_g[NUM_LAYER];
Parameter *mlp1_b[NUM_LAYER], *mlp1_w[NUM_LAYER];
Parameter *mlp2_b[NUM_LAYER], *mlp2_w[NUM_LAYER];
Parameter *ln_f_b, *ln_f_g;
Parameter *wpe, *wte;

void alloc_activations();
void free_activations();

void alloc_and_set_parameters(float *param) {
  char *local_rank_str = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (local_rank_str == nullptr) {
    fprintf(stderr, "Error: OMPI_COMM_WORLD_LOCAL_RANK is not set\n");
    exit(1);
  }
  local_rank = atoi(local_rank_str);

  cudaSetDevice(local_rank);
  cudaDeviceSynchronize();

  size_t pos = 0;
  int order[] = { 0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9 };

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
  wte = new Parameter({(HIDDEN_DIM + 1), ALLOC_VOCAB});
  float *tmp = (float *)calloc((HIDDEN_DIM + 1) * ALLOC_VOCAB, sizeof(float));
  for (int i = 0; i < NUM_VOCAB; i++)
    for (int j = 0; j < HIDDEN_DIM; j++)
      tmp[j * ALLOC_VOCAB + i] = param[pos + i * HIDDEN_DIM + j];
  cudaMemcpy(wte->buf, tmp, (HIDDEN_DIM + 1) * ALLOC_VOCAB * sizeof(float), cudaMemcpyHostToDevice);
  pos += OFFSET8;
  free(tmp);

  alloc_activations();
}

void free_parameters() {
  free_activations();

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

Activation *embd_a, *ffn_proj_a, *matmul_reduce;
Activation *mha_q_proj_a, *mha_out_a, *mha_attn_out_a;
Activation *residual_a, *logit_a, *out_a;

Cache::Cache() {
  kv = new Activation({BATCH_SIZE, LEN_INPUT + LEN_OUTPUT, 2 * HIDDEN_DIM});
}

Cache::~Cache() {
  delete kv;
}

void Cache::clear() {
  len = 0;
}

void Cache::append(Activation *x, Parameter *attn_b, Parameter *attn_w) {
  linear(x->buf, attn_w->buf + HIDDEN_DIM, attn_b->buf + HIDDEN_DIM, kv->buf + len * 2 * HIDDEN_DIM,
    BATCH_SIZE, 2 * HIDDEN_DIM, HIDDEN_DIM, x->shape[1] * HIDDEN_DIM, 3 * HIDDEN_DIM, kv->shape[1] * 2 * HIDDEN_DIM,
    x->shape[1], HIDDEN_DIM, 2 * HIDDEN_DIM, matmul_reduce->buf, false);
  len += x->shape[1];
}

Cache *kv_cache[NUM_LAYER];

void alloc_activations() {
  embd_a = new Activation({BATCH_SIZE, LEN_INPUT, HIDDEN_DIM});
  ffn_proj_a = new Activation({BATCH_SIZE, LEN_INPUT, 4 * HIDDEN_DIM});
  matmul_reduce = new Activation({6, 512 * 768 * 2});

  mha_q_proj_a = new Activation({BATCH_SIZE, LEN_INPUT, HIDDEN_DIM});
  mha_out_a = new Activation({BATCH_SIZE, LEN_INPUT, HIDDEN_DIM});
  mha_attn_out_a = new Activation({BATCH_SIZE, LEN_INPUT, HIDDEN_DIM});

  residual_a = new Activation({BATCH_SIZE, LEN_INPUT, HIDDEN_DIM});
  logit_a = new Activation({BATCH_SIZE, ALLOC_VOCAB});
  out_a = new Activation({BATCH_SIZE, LEN_INPUT});

  for (int i = 0; i < NUM_LAYER; i++) kv_cache[i] = new Cache();
}

void free_activations() {
  delete embd_a;
  delete ffn_proj_a;
  delete matmul_reduce;
  delete mha_q_proj_a;
  delete mha_out_a;
  delete mha_attn_out_a;
  delete residual_a;
  delete logit_a;
  delete out_a;

  for (int i = 0; i < NUM_LAYER; i++) delete kv_cache[i];
}

void set_length(size_t seq_len) {
  if (seq_len != LEN_INPUT && seq_len != 1) {
    fprintf(stderr, "Error: seq_len must be %d or %d\n", LEN_INPUT, 1);
    exit(1);
  }

  embd_a->shape[1] = seq_len;
  ffn_proj_a->shape[1] = seq_len;
  mha_q_proj_a->shape[1] = seq_len;
  mha_out_a->shape[1] = seq_len;
  mha_attn_out_a->shape[1] = seq_len;
  residual_a->shape[1] = seq_len;
  out_a->shape[1] = seq_len;
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
  /* Projection Up + GELU:
    [seq_len, HIDDEN_DIM] -> [seq_len, 4*HIDDEN_DIM] */
  linear(in->buf, mlp1_w->buf, mlp1_b->buf, ffn_proj_a->buf,
    in->shape[0] * in->shape[1], 4 * HIDDEN_DIM,
    HIDDEN_DIM, HIDDEN_DIM, 4 * HIDDEN_DIM, 4 * HIDDEN_DIM, matmul_reduce->buf, true);

  /* Projection Down:
    [seq_len, 4*HIDDEN_DIM] -> [seq_len, HIDDEN_DIM] */
  linear(ffn_proj_a->buf, mlp2_w->buf, mlp2_b->buf, out->buf,
    ffn_proj_a->shape[0] * ffn_proj_a->shape[1], HIDDEN_DIM, 4 * HIDDEN_DIM,
    4 * HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, matmul_reduce->buf, false);
}

/* (Masked) Multi-Head Self Attention
 * @param [in1]     in: [seq_len, HIDDEN_DIM]
 * @param [in2] attn_b: [3*HIDDEN_DIM]
 * @param [in3] attn_w: [HIDDEN_DIM, 3*HIDDEN_DIM]
 * @param [in4] proj_b: [HIDDEN_DIM]
 * @param [in5] proj_w: [HIDDEN_DIM, HIDDEN_DIM]
 * @param [out]    out: [seq_len, HIDDEN_DIM]
 */
void mha(Activation *in, Cache *kv, Parameter *attn_b, Parameter *attn_w,
         Parameter *proj_b, Parameter *proj_w, Activation *out) {
  /* Q projection */
  linear(in->buf, attn_w->buf, attn_b->buf, mha_q_proj_a->buf,
    in->shape[0] * in->shape[1], HIDDEN_DIM, HIDDEN_DIM,
    HIDDEN_DIM, 3 * HIDDEN_DIM, HIDDEN_DIM, matmul_reduce->buf, false);

  /* KV projection and update cache */
  kv->append(in, attn_b, attn_w);

  /* Perform Attention over each head */
  attention(mha_q_proj_a->buf, kv->kv->buf, mha_attn_out_a->buf,
            HIDDEN_DIM, NUM_HEAD, BATCH_SIZE, mha_q_proj_a->shape[1], kv->len);

  /* OUT projection:
    [seq_len, HIDDEN_DIM] -> [seq_len, HIDDEN_DIM] */
  linear(mha_attn_out_a->buf, proj_w->buf, proj_b->buf, out->buf,
    mha_attn_out_a->shape[0] * mha_attn_out_a->shape[1], HIDDEN_DIM, HIDDEN_DIM,
    HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, matmul_reduce->buf, false);
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
void transformer_block(Activation *in, Cache *kv, Parameter *attn_b, Parameter *attn_w,
                       Parameter *proj_b, Parameter *proj_w, Parameter *ln_1_b,
                       Parameter *ln_1_g, Parameter *ln_2_b, Parameter *ln_2_g,
                       Parameter *mlp1_b, Parameter *mlp1_w, Parameter *mlp2_b,
                       Parameter *mlp2_w, Activation *out) {
  /* Layer Normalization */
  layer_norm(in->buf, ln_1_g->buf, ln_1_b->buf, residual_a->buf, in->shape[0] * in->shape[1], HIDDEN_DIM);

  /* Masked Multi-Head Self-Attention */
  mha(residual_a, kv, attn_b, attn_w, proj_b, proj_w, mha_out_a);

  /* Add Residual */
  add(mha_out_a->buf, in->buf, mha_out_a->num_elem());

  /* Layer Normalization */
  layer_norm(mha_out_a->buf, ln_2_g->buf, ln_2_b->buf, residual_a->buf, mha_out_a->shape[0] * mha_out_a->shape[1], HIDDEN_DIM);

  /* Position-wise Feed-Forward Network */
  ffn(residual_a, mlp1_w, mlp1_b, mlp2_w, mlp2_b, out);

  /* Add Residual */
  add(out->buf, mha_out_a->buf, mha_out_a->num_elem());
}

void run_model(int *input, int *output, size_t n_prompt, size_t n_token) {
  if (n_prompt % BATCH_SIZE != 0) {
    fprintf(stderr, "Error: n_prompt must be a multiple of %d\n", BATCH_SIZE);
    exit(1);
  }

  /* Outer loop: generate tokens for each prompt */
  for (size_t p = 0; p < n_prompt; p += BATCH_SIZE) {
    /* Initialize input prompt */
    set_length(LEN_INPUT);
    cudaMemcpy(out_a->buf, input + p * LEN_INPUT, LEN_INPUT * BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    /* Inner loop: generate next token */
    for (size_t t = 0; t < n_token; t++) {
      /* Token + Positional Embedding */
      int pos = t == 0 ? 0 : LEN_INPUT + t - 1;
      token_pos_embedding((int *)out_a->buf, wte->buf, wpe->buf, embd_a->buf, BATCH_SIZE, out_a->shape[1], HIDDEN_DIM, pos, ALLOC_VOCAB);

      /* Forward path of Transformer blocks */
      for (size_t l = 0; l < NUM_LAYER; l++)
        transformer_block(embd_a, kv_cache[l], attn_b[l], attn_w[l], proj_b[l], proj_w[l],
                          ln_1_b[l], ln_1_g[l], ln_2_b[l], ln_2_g[l],
                          mlp1_b[l], mlp1_w[l], mlp2_b[l], mlp2_w[l],
                          embd_a);

      /* Final Layer Normalization */
      layer_norm(embd_a->buf, ln_f_g->buf, ln_f_b->buf, embd_a->buf, embd_a->shape[0] * embd_a->shape[1], HIDDEN_DIM);

      /* Projection to vocab. dimension */
      lm_head(embd_a->buf, wte->buf, logit_a->buf, BATCH_SIZE, embd_a->shape[1], ALLOC_VOCAB, HIDDEN_DIM);

      /* Greedy sampling */
      top1_sampling(logit_a->buf, (int *)out_a->buf, BATCH_SIZE, NUM_VOCAB, ALLOC_VOCAB);

      int input_prompt[BATCH_SIZE];
      cudaMemcpy(input_prompt, out_a->buf, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
      for (size_t i = 0; i < BATCH_SIZE; i++)
        output[(p + i) * n_token + t] = input_prompt[i];

      set_length(1);
    }
    
    for (int l = 0; l < NUM_LAYER; l++)
      kv_cache[l]->clear();
  }
}

/* [Model Computation: Token Generation] */
void generate_tokens(int *input, int *output, size_t n_prompt, size_t n_token) {
  cudaSetDevice(local_rank);
  cudaDeviceSynchronize();

  // get MPI rank and size
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (n_token != LEN_OUTPUT) {
    fprintf(stderr, "Error: n_token must be %d\n", LEN_OUTPUT);
    exit(1);
  }

  if (tokens_per_prompt != LEN_INPUT) {
    fprintf(stderr, "Error: tokens_per_prompt must be %d\n", LEN_INPUT);
    exit(1);
  }

  // allocate memory for input and output
  size_t n_prompt_node = n_prompt / mpi_size;
  if (mpi_rank != 0) {
    input = (int *)malloc(n_prompt_node * tokens_per_prompt * sizeof(int));
    output = (int *)malloc(n_prompt_node * n_token * sizeof(int));
  }

  // scatter input to all nodes
  MPI_Scatter(input, n_prompt_node * tokens_per_prompt, MPI_INT, input,
              n_prompt_node * tokens_per_prompt, MPI_INT, 0, MPI_COMM_WORLD);

  // generate tokens from model
  run_model(input, output, n_prompt_node, n_token);

  // gather output from all nodes
  MPI_Gather(output, n_prompt_node * n_token, MPI_INT, output,
             n_prompt_node * n_token, MPI_INT, 0, MPI_COMM_WORLD);

  // free memory of input and output
  if (mpi_rank != 0) {
    free(input);
    free(output);
  }
}
