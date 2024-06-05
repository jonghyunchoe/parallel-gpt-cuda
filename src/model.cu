#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>

#include "layer.h"
#include "model.h"

#define BATCH_SIZE 128
#define NGPU 4  // Number of GPUs per node

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

void print_device_pointer(float* d_ptr, size_t N, int gpu_id) {
  float* h_ptr = (float*)malloc(N * sizeof(float));
  cudaSetDevice(gpu_id);
  cudaMemcpy(h_ptr, d_ptr, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < N; i += 8) {
    for (size_t j = 0; j < 8; j++) {
      if ((i+j) < N)
        printf("%lf ", h_ptr[i+j]);
    }
    printf("\n");
  }
  printf("\n");
  free(h_ptr);
}

// Parameters declaration
Parameter *attn_b[NUM_LAYER], *attn_w[NUM_LAYER];
Parameter *proj_b[NUM_LAYER], *proj_w[NUM_LAYER];
Parameter *ln_1_b[NUM_LAYER], *ln_1_g[NUM_LAYER];
Parameter *ln_2_b[NUM_LAYER], *ln_2_g[NUM_LAYER];
Parameter *mlp1_b[NUM_LAYER], *mlp1_w[NUM_LAYER];
Parameter *mlp2_b[NUM_LAYER], *mlp2_w[NUM_LAYER];
Parameter *ln_f_b, *ln_f_g;
Parameter *wpe, *wte;

float *d_attn_b[NUM_LAYER][NGPU], *d_attn_w[NUM_LAYER][NGPU];
float *d_proj_b[NUM_LAYER][NGPU], *d_proj_w[NUM_LAYER][NGPU];
float *d_ln_1_b[NUM_LAYER][NGPU], *d_ln_1_g[NUM_LAYER][NGPU];
float *d_ln_2_b[NUM_LAYER][NGPU], *d_ln_2_g[NUM_LAYER][NGPU];
float *d_mlp1_b[NUM_LAYER][NGPU], *d_mlp1_w[NUM_LAYER][NGPU];
float *d_mlp2_b[NUM_LAYER][NGPU], *d_mlp2_w[NUM_LAYER][NGPU];
float *d_ln_f_b[NGPU], *d_ln_f_g[NGPU];
float *d_wpe[NGPU], *d_wte[NGPU];

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

void alloc_and_set_device_parameters() {
  int order[] = {
      0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9,
  };

  for (int gpu_id = 0; gpu_id < NGPU; gpu_id++) {
    cudaSetDevice(gpu_id);

    for (int i = 0; i < NUM_LAYER; i++) {
      cudaMalloc(&d_attn_b[order[i]][gpu_id], attn_b[order[i]]->num_elem() * sizeof(float));
      cudaMalloc(&d_attn_w[order[i]][gpu_id], attn_w[order[i]]->num_elem() * sizeof(float));
      cudaMalloc(&d_proj_b[order[i]][gpu_id], proj_b[order[i]]->num_elem() * sizeof(float));
      cudaMalloc(&d_proj_w[order[i]][gpu_id], proj_w[order[i]]->num_elem() * sizeof(float));
      cudaMalloc(&d_ln_1_b[order[i]][gpu_id], ln_1_b[order[i]]->num_elem() * sizeof(float));
      cudaMalloc(&d_ln_1_g[order[i]][gpu_id], ln_1_g[order[i]]->num_elem() * sizeof(float));
      cudaMalloc(&d_ln_2_b[order[i]][gpu_id], ln_2_b[order[i]]->num_elem() * sizeof(float));
      cudaMalloc(&d_ln_2_g[order[i]][gpu_id], ln_2_g[order[i]]->num_elem() * sizeof(float));
      cudaMalloc(&d_mlp1_b[order[i]][gpu_id], mlp1_b[order[i]]->num_elem() * sizeof(float));
      cudaMalloc(&d_mlp1_w[order[i]][gpu_id], mlp1_w[order[i]]->num_elem() * sizeof(float));
      cudaMalloc(&d_mlp2_b[order[i]][gpu_id], mlp2_b[order[i]]->num_elem() * sizeof(float));
      cudaMalloc(&d_mlp2_w[order[i]][gpu_id], mlp2_w[order[i]]->num_elem() * sizeof(float));
    }
    cudaMalloc(&d_ln_f_b[gpu_id], ln_f_b->num_elem() * sizeof(float));
    cudaMalloc(&d_ln_f_g[gpu_id], ln_f_g->num_elem() * sizeof(float));
    cudaMalloc(&d_wpe[gpu_id], wpe->num_elem() * sizeof(float));
    cudaMalloc(&d_wte[gpu_id], wte->num_elem() * sizeof(float));
    
    // Copy data to device
    for (int i = 0; i < NUM_LAYER; i++) {
      cudaMemcpy(d_attn_b[order[i]][gpu_id], attn_b[order[i]]->buf, attn_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_attn_w[order[i]][gpu_id], attn_w[order[i]]->buf, attn_w[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_proj_b[order[i]][gpu_id], proj_b[order[i]]->buf, proj_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_proj_w[order[i]][gpu_id], proj_w[order[i]]->buf, proj_w[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ln_1_b[order[i]][gpu_id], ln_1_b[order[i]]->buf, ln_1_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ln_1_g[order[i]][gpu_id], ln_1_g[order[i]]->buf, ln_1_g[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ln_2_b[order[i]][gpu_id], ln_2_b[order[i]]->buf, ln_2_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ln_2_g[order[i]][gpu_id], ln_2_g[order[i]]->buf, ln_2_g[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_mlp1_b[order[i]][gpu_id], mlp1_b[order[i]]->buf, mlp1_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_mlp1_w[order[i]][gpu_id], mlp1_w[order[i]]->buf, mlp1_w[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_mlp2_b[order[i]][gpu_id], mlp2_b[order[i]]->buf, mlp2_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_mlp2_w[order[i]][gpu_id], mlp2_w[order[i]]->buf, mlp2_w[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_ln_f_b[gpu_id], ln_f_b->buf, ln_f_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln_f_g[gpu_id], ln_f_g->buf, ln_f_g->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wpe[gpu_id], wpe->buf, wpe->num_elem() * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_wte[gpu_id], wte->buf, wte->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  }
}

void free_parameters() {
  for (int gpu_id = 0; gpu_id < NGPU; gpu_id++) {
    cudaSetDevice(gpu_id);

    for (int i = 0; i < NUM_LAYER; i++) {
      cudaFree(d_attn_b[i][gpu_id]);
      cudaFree(d_attn_w[i][gpu_id]);
      cudaFree(d_proj_b[i][gpu_id]);
      cudaFree(d_proj_w[i][gpu_id]);
      cudaFree(d_ln_1_b[i][gpu_id]);
      cudaFree(d_ln_1_g[i][gpu_id]);
      cudaFree(d_ln_2_b[i][gpu_id]);
      cudaFree(d_ln_2_g[i][gpu_id]);
      cudaFree(d_mlp1_b[i][gpu_id]);
      cudaFree(d_mlp1_w[i][gpu_id]);
      cudaFree(d_mlp2_b[i][gpu_id]);
      cudaFree(d_mlp2_w[i][gpu_id]);
    }
    cudaFree(d_ln_f_b[gpu_id]);
    cudaFree(d_ln_f_g[gpu_id]);
    cudaFree(d_wpe[gpu_id]);
    cudaFree(d_wte[gpu_id]);
  }
}

float *d_embd_a[NGPU], *d_ffn_proj_a[NGPU];
float *d_mha_qkv_proj_a[NGPU], *d_mha_out_a[NGPU], *d_mha_split_qkv_a[NGPU],
    *d_mha_split_head_a[NGPU], *d_mha_mask_a[NGPU], *d_mha_merge_head_a[NGPU], *d_mha_q_a[NGPU],
    *d_mha_k_a[NGPU], *d_mha_v_a[NGPU], *d_mha_attn_out_a[NGPU], *d_mha_concat_head_a[NGPU];
float *d_attn_score_a[NGPU], *d_k_transposed_a[NGPU];
float *d_wte_transposed_a[NGPU], *d_residual_a[NGPU], *d_logit_a[NGPU];
float *d_transformer_block_a[NGPU];

void alloc_activations(size_t prompt_size, int gpu_id) {
  cudaSetDevice(gpu_id);

  cudaMalloc(&d_embd_a[gpu_id], prompt_size * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_ffn_proj_a[gpu_id], prompt_size * 4 * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_mha_qkv_proj_a[gpu_id], prompt_size * 3 * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_mha_out_a[gpu_id], prompt_size * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_mha_split_qkv_a[gpu_id], 3 * prompt_size * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_mha_split_head_a[gpu_id], 3 * NUM_HEAD * prompt_size * HIDDEN_DIM / NUM_HEAD * sizeof(float));
  cudaMalloc(&d_mha_mask_a[gpu_id], prompt_size * prompt_size * sizeof(float));
  cudaMalloc(&d_mha_merge_head_a[gpu_id], NUM_HEAD * prompt_size * HIDDEN_DIM / NUM_HEAD * sizeof(float));
  cudaMalloc(&d_mha_q_a[gpu_id], prompt_size * HIDDEN_DIM / NUM_HEAD * sizeof(float));
  cudaMalloc(&d_mha_k_a[gpu_id], prompt_size * HIDDEN_DIM / NUM_HEAD * sizeof(float));
  cudaMalloc(&d_mha_v_a[gpu_id], prompt_size * HIDDEN_DIM / NUM_HEAD * sizeof(float));
  cudaMalloc(&d_mha_attn_out_a[gpu_id], prompt_size * HIDDEN_DIM / NUM_HEAD * sizeof(float));
  cudaMalloc(&d_mha_concat_head_a[gpu_id], prompt_size * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_attn_score_a[gpu_id], prompt_size * prompt_size * sizeof(float));
  cudaMalloc(&d_k_transposed_a[gpu_id], HIDDEN_DIM / NUM_HEAD * prompt_size * sizeof(float));
  cudaMalloc(&d_wte_transposed_a[gpu_id], HIDDEN_DIM * NUM_VOCAB * sizeof(float));
  cudaMalloc(&d_residual_a[gpu_id], prompt_size * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_logit_a[gpu_id], prompt_size * NUM_VOCAB * sizeof(float));
  cudaMalloc(&d_transformer_block_a[gpu_id], prompt_size * HIDDEN_DIM * sizeof(float));
}

void free_activations(int gpu_id) {
  cudaSetDevice(gpu_id);

  cudaFree(d_embd_a[gpu_id]);
  cudaFree(d_ffn_proj_a[gpu_id]);
  cudaFree(d_mha_qkv_proj_a[gpu_id]);
  cudaFree(d_mha_out_a[gpu_id]);
  cudaFree(d_mha_split_qkv_a[gpu_id]);
  cudaFree(d_mha_split_head_a[gpu_id]);
  cudaFree(d_mha_mask_a[gpu_id]);
  cudaFree(d_mha_merge_head_a[gpu_id]);
  cudaFree(d_mha_q_a[gpu_id]);
  cudaFree(d_mha_k_a[gpu_id]);
  cudaFree(d_mha_v_a[gpu_id]);
  cudaFree(d_mha_attn_out_a[gpu_id]);
  cudaFree(d_mha_concat_head_a[gpu_id]);
  cudaFree(d_attn_score_a[gpu_id]);
  cudaFree(d_k_transposed_a[gpu_id]);
  cudaFree(d_wte_transposed_a[gpu_id]);
  cudaFree(d_residual_a[gpu_id]);
  cudaFree(d_logit_a[gpu_id]);
  cudaFree(d_transformer_block_a[gpu_id]);
}

// Adapted functions for multi-GPU
void ffn(float *d_in, float *d_mlp1_w, float *d_mlp1_b,
         float *d_mlp2_w, float *d_mlp2_b, float *d_out, size_t seq_len, size_t batch_size, int gpu_id) {
    cudaSetDevice(gpu_id);

    batch_linear(d_in, d_mlp1_w, d_mlp1_b, d_ffn_proj_a[gpu_id], batch_size, seq_len, HIDDEN_DIM, 4 * HIDDEN_DIM);
    batch_gelu(d_ffn_proj_a[gpu_id], batch_size, seq_len * 4 * HIDDEN_DIM);
    batch_linear(d_ffn_proj_a[gpu_id], d_mlp2_w, d_mlp2_b, d_out, batch_size, seq_len, 4 * HIDDEN_DIM, HIDDEN_DIM);
}

void attention(float *d_q, float *d_k, float *d_v, float *d_mask, float *d_out, size_t seq_len, size_t head_dim, size_t batch_size, int gpu_id) {
    cudaSetDevice(gpu_id);

    batch_transpose(d_k, d_k_transposed_a[gpu_id], batch_size, seq_len, head_dim);
    batch_matmul(d_q, d_k_transposed_a[gpu_id], d_attn_score_a[gpu_id], batch_size, seq_len, head_dim, seq_len);
    batch_scaling(d_attn_score_a[gpu_id], 1.0 / sqrt(head_dim), batch_size, seq_len * seq_len);
    batch_add(d_attn_score_a[gpu_id], d_mask, batch_size, seq_len * seq_len);
    batch_softmax(d_attn_score_a[gpu_id], batch_size, seq_len, seq_len);
    batch_matmul(d_attn_score_a[gpu_id], d_v, d_out, batch_size, seq_len, seq_len, head_dim);
}

void mha(float *d_in, float *d_attn_b, float *d_attn_w,
         float *d_proj_b, float *d_proj_w, float *d_out, size_t seq_len, size_t batch_size, int gpu_id) {
    cudaSetDevice(gpu_id);

    batch_linear(d_in, d_attn_w, d_attn_b, d_mha_qkv_proj_a[gpu_id], batch_size, seq_len, HIDDEN_DIM, 3 * HIDDEN_DIM);
    batch_split_qkv(d_mha_qkv_proj_a[gpu_id], d_mha_split_qkv_a[gpu_id], batch_size, seq_len, 3 * HIDDEN_DIM);
    batch_split_head(d_mha_split_qkv_a[gpu_id], d_mha_split_head_a[gpu_id], batch_size, NUM_HEAD, seq_len, HIDDEN_DIM);
    batch_generate_mask(d_mha_mask_a[gpu_id], batch_size, seq_len);

    for (size_t idx = 0; idx < NUM_HEAD; idx++) {
        batch_extract_qkv(d_mha_split_head_a[gpu_id], d_mha_q_a[gpu_id], d_mha_k_a[gpu_id], d_mha_v_a[gpu_id], batch_size, idx, NUM_HEAD, seq_len, HIDDEN_DIM / NUM_HEAD);
        attention(d_mha_q_a[gpu_id], d_mha_k_a[gpu_id], d_mha_v_a[gpu_id], d_mha_mask_a[gpu_id], d_mha_attn_out_a[gpu_id], seq_len, HIDDEN_DIM / NUM_HEAD, batch_size, gpu_id);
        batch_merge_head(d_mha_attn_out_a[gpu_id], d_mha_merge_head_a[gpu_id], batch_size, NUM_HEAD, idx, seq_len, HIDDEN_DIM / NUM_HEAD);
    }

    batch_concat_head(d_mha_merge_head_a[gpu_id], d_mha_concat_head_a[gpu_id], batch_size, NUM_HEAD, seq_len, HIDDEN_DIM / NUM_HEAD);
    batch_linear(d_mha_concat_head_a[gpu_id], d_proj_w, d_proj_b, d_out, batch_size, seq_len, HIDDEN_DIM, HIDDEN_DIM);
}

void transformer_block(float *d_in, float *d_attn_b, float *d_attn_w,
                       float *d_proj_b, float *d_proj_w, float *d_ln_1_b,
                       float *d_ln_1_g, float *d_ln_2_b, float *d_ln_2_g,
                       float *d_mlp1_b, float *d_mlp1_w, float *d_mlp2_b,
                       float *d_mlp2_w, float *d_out, size_t seq_len, size_t batch_size, int gpu_id) { 
    cudaSetDevice(gpu_id);

    batch_copy(d_in, d_residual_a[gpu_id], batch_size, seq_len * HIDDEN_DIM);
    batch_layer_norm(d_in, d_ln_1_g, d_ln_1_b, batch_size, seq_len, HIDDEN_DIM, 1e-5);
    mha(d_in, d_attn_b, d_attn_w, d_proj_b, d_proj_w, d_mha_out_a[gpu_id], seq_len, batch_size, gpu_id);
    batch_add(d_mha_out_a[gpu_id], d_residual_a[gpu_id], batch_size, seq_len * HIDDEN_DIM);
    batch_copy(d_mha_out_a[gpu_id], d_residual_a[gpu_id], batch_size, seq_len * HIDDEN_DIM);
    batch_layer_norm(d_mha_out_a[gpu_id], d_ln_2_g, d_ln_2_b, batch_size, seq_len, HIDDEN_DIM, 1e-5);
    ffn(d_mha_out_a[gpu_id], d_mlp1_w, d_mlp1_b, d_mlp2_w, d_mlp2_b, d_out, seq_len, batch_size, gpu_id);
    batch_add(d_out, d_residual_a[gpu_id], batch_size, seq_len * HIDDEN_DIM);
}

void generate_tokens(int *input, int *output, size_t n_prompt, size_t n_token) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (mpi_rank != 0) {
        input = (int *)malloc(n_prompt * tokens_per_prompt * sizeof(int));
        output = (int *)malloc(n_prompt * n_token * sizeof(int));
    }

    MPI_Bcast(input, n_prompt * tokens_per_prompt, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(output, n_prompt * n_token, MPI_INT, 0, MPI_COMM_WORLD);

    size_t prompts_per_node = (n_prompt + mpi_size - 1) / mpi_size;
    size_t start_prompt = mpi_rank * prompts_per_node;
    size_t end_prompt = MIN(start_prompt + prompts_per_node, n_prompt);

    alloc_and_set_device_parameters();

    for (size_t p = start_prompt; p < end_prompt; p += BATCH_SIZE) {
        int batch_size = MIN(BATCH_SIZE, end_prompt - p);
        int prompt_size = tokens_per_prompt;

        std::vector<int> input_prompt(batch_size * prompt_size);
        memcpy(input_prompt.data(), input + p * prompt_size, batch_size * prompt_size * sizeof(int));

        int *d_input_prompt[NGPU];
        int *d_out[NGPU];

        for (int gpu_id = 0; gpu_id < NGPU; gpu_id++) {
            cudaSetDevice(gpu_id);
            CHECK_CUDA(cudaMalloc(&d_input_prompt[gpu_id], batch_size * (prompt_size + n_token - 1) * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_out[gpu_id], batch_size * sizeof(int)));
            alloc_activations(batch_size * (prompt_size + n_token - 1), gpu_id);
        }

        for (size_t t = 0; t < n_token; t++) {
            for (int gpu_id = 0; gpu_id < NGPU; gpu_id++) {
                cudaSetDevice(gpu_id);
                size_t start_idx = gpu_id * (batch_size / NGPU);
                size_t end_idx = (gpu_id == NGPU - 1) ? batch_size : (gpu_id + 1) * (batch_size / NGPU);
                size_t gpu_batch_size = end_idx - start_idx;

                CHECK_CUDA(cudaMemcpy(d_input_prompt[gpu_id], input_prompt.data() + start_idx * prompt_size, gpu_batch_size * prompt_size * sizeof(int), cudaMemcpyHostToDevice));

                batch_token_pos_embedding(d_input_prompt[gpu_id], d_wte[gpu_id], d_wpe[gpu_id], d_embd_a[gpu_id], gpu_batch_size, prompt_size, HIDDEN_DIM);

                for (size_t l = 0; l < NUM_LAYER; l++) {
                    transformer_block(d_embd_a[gpu_id], d_attn_b[l][gpu_id], d_attn_w[l][gpu_id], d_proj_b[l][gpu_id], d_proj_w[l][gpu_id],
                                      d_ln_1_b[l][gpu_id], d_ln_1_g[l][gpu_id], d_ln_2_b[l][gpu_id], d_ln_2_g[l][gpu_id],
                                      d_mlp1_b[l][gpu_id], d_mlp1_w[l][gpu_id], d_mlp2_b[l][gpu_id], d_mlp2_w[l][gpu_id],
                                      d_transformer_block_a[gpu_id], prompt_size, gpu_batch_size, gpu_id);
                    batch_copy(d_transformer_block_a[gpu_id], d_embd_a[gpu_id], gpu_batch_size, prompt_size * HIDDEN_DIM);
                }

                batch_layer_norm(d_embd_a[gpu_id], d_ln_f_g[gpu_id], d_ln_f_b[gpu_id], gpu_batch_size, prompt_size, HIDDEN_DIM, 1e-5);
                transpose(d_wte[gpu_id], d_wte_transposed_a[gpu_id], wte->shape[0], wte->shape[1]);
                batch_matmul_final(d_embd_a[gpu_id], d_wte_transposed_a[gpu_id], d_logit_a[gpu_id], gpu_batch_size, prompt_size, HIDDEN_DIM, wte->shape[0]);
                batch_top1_sampling(d_logit_a[gpu_id], d_out[gpu_id], gpu_batch_size, prompt_size, NUM_VOCAB);
            }

            std::vector<int> next_token_ids(batch_size);
            for (int gpu_id = 0; gpu_id < NGPU; gpu_id++) {
                size_t start_idx = gpu_id * (batch_size / NGPU);
                size_t end_idx = (gpu_id == NGPU - 1) ? batch_size : (gpu_id + 1) * (batch_size / NGPU);
                size_t gpu_batch_size = end_idx - start_idx;

                CHECK_CUDA(cudaMemcpy(next_token_ids.data() + start_idx, d_out[gpu_id], gpu_batch_size * sizeof(int), cudaMemcpyDeviceToHost));
            }

            for (size_t i = 0; i < batch_size; i++) {
                size_t insert_position = (i + 1) * prompt_size + i;
                input_prompt.insert(input_prompt.begin() + insert_position, next_token_ids[i]);
                output[(p + i) * n_token + t] = next_token_ids[i];
            }
            prompt_size += 1;
        }

        for (int gpu_id = 0; gpu_id < NGPU; gpu_id++) {
            cudaSetDevice(gpu_id);
            CHECK_CUDA(cudaFree(d_input_prompt[gpu_id]));
            CHECK_CUDA(cudaFree(d_out[gpu_id]));
            free_activations(gpu_id);
        }
    }

    if (mpi_rank == 0) {
        std::vector<int> final_output(n_prompt * n_token);
        MPI_Gather(output, prompts_per_node * n_token, MPI_INT, final_output.data(), prompts_per_node * n_token, MPI_INT, 0, MPI_COMM_WORLD);
        memcpy(output, final_output.data(), n_prompt * n_token * sizeof(int));
    } else {
        MPI_Gather(output + start_prompt * n_token, prompts_per_node * n_token, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }
}
