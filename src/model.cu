#include <mpi.h>

#include <cmath>
#include <cstdio>

#include "layer.h"
#include "model.h"

#define BATCH_SIZE 128

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

void print_device_pointer(float* d_ptr, size_t N) {
  float* h_ptr = (float*)malloc(N * sizeof(float));
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


void print_device_pointer(float* d_ptr, size_t N, size_t stride) {
  float* h_ptr = (float*)malloc((N) * sizeof(float));
  cudaMemcpy(h_ptr, d_ptr + stride * sizeof(float), N * sizeof(float), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < N; i += 8) {
    for (size_t j = 0; j < 8; j++) {
      if ((i+j) < N)
        printf("%lf ", h_ptr[i+j+stride]);
    }
    printf("\n");
  }
  printf("\n");
  free(h_ptr);
}

void print_batch_device_pointer(float* d_ptr, size_t N, size_t batch_stride) {
  float* h_ptr = (float*)malloc((N + batch_stride)* sizeof(float));
  cudaMemcpy(h_ptr, d_ptr, (N + batch_stride) * sizeof(float), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < N; i += 8) {
    for (size_t j = 0; j < 8; j++) {
      if ((i+j) < N)
        printf("%lf ", h_ptr[i+j]);
    }
    printf("\n");
  }

  printf("\n");
  printf("next prompt: \n"); 

  for (size_t i = 0; i < N; i += 8) {
    for (size_t j = 0; j < 8; j++) {
      if ((i+j) < N)
        printf("%lf ", h_ptr[i+j + batch_stride]);
    }
    printf("\n");
  }

  printf("\n");
  free(h_ptr);
}

void print_batch_device_pointer(int* d_ptr, size_t N, size_t batch_stride) {
  float* h_ptr = (float*)malloc((N + batch_stride)* sizeof(int));
  cudaMemcpy(h_ptr, d_ptr, (N + batch_stride) * sizeof(int), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < N; i += 8) {
    for (size_t j = 0; j < 8; j++) {
      if ((i+j) < N)
        std::cout << h_ptr[i+j] << " ";
    }
    printf("\n");
  }

  printf("\n");
  printf("next prompt: \n"); 

  for (size_t i = 0; i < N; i += 8) {
    for (size_t j = 0; j < 8; j++) {
      if ((i+j) < N)
        std::cout << h_ptr[i+j + batch_stride] << " ";
    }
    printf("\n");
  }

  printf("\n");
  free(h_ptr);
}


void print_device_pointer(int* d_ptr, size_t N) {
  int* h_ptr = (int*)malloc(N * sizeof(int));
  cudaMemcpy(h_ptr, d_ptr, N * sizeof(int), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < N; ++i) {
      std::cout << h_ptr[i] << " ";
  }
  std::cout << std::endl;
  free(h_ptr);
}

Parameter *attn_b[NUM_LAYER], *attn_w[NUM_LAYER];
Parameter *proj_b[NUM_LAYER], *proj_w[NUM_LAYER];
Parameter *ln_1_b[NUM_LAYER], *ln_1_g[NUM_LAYER];
Parameter *ln_2_b[NUM_LAYER], *ln_2_g[NUM_LAYER];
Parameter *mlp1_b[NUM_LAYER], *mlp1_w[NUM_LAYER];
Parameter *mlp2_b[NUM_LAYER], *mlp2_w[NUM_LAYER];
Parameter *ln_f_b, *ln_f_g;
Parameter *wpe, *wte;

/* Device pointers */
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

  // /* Allocate device memory */
  // for (int i = 0; i < NUM_LAYER; i++) {
  //   cudaMalloc(&d_attn_b[order[i]], attn_b[order[i]]->num_elem() * sizeof(float));
  //   cudaMalloc(&d_attn_w[order[i]], attn_w[order[i]]->num_elem() * sizeof(float));
  //   cudaMalloc(&d_proj_b[order[i]], proj_b[order[i]]->num_elem() * sizeof(float));
  //   cudaMalloc(&d_proj_w[order[i]], proj_w[order[i]]->num_elem() * sizeof(float));
  //   cudaMalloc(&d_ln_1_b[order[i]], ln_1_b[order[i]]->num_elem() * sizeof(float));
  //   cudaMalloc(&d_ln_1_g[order[i]], ln_1_g[order[i]]->num_elem() * sizeof(float));
  //   cudaMalloc(&d_ln_2_b[order[i]], ln_2_b[order[i]]->num_elem() * sizeof(float));
  //   cudaMalloc(&d_ln_2_g[order[i]], ln_2_g[order[i]]->num_elem() * sizeof(float));
  //   cudaMalloc(&d_mlp1_b[order[i]], mlp1_b[order[i]]->num_elem() * sizeof(float));
  //   cudaMalloc(&d_mlp1_w[order[i]], mlp1_w[order[i]]->num_elem() * sizeof(float));
  //   cudaMalloc(&d_mlp2_b[order[i]], mlp2_b[order[i]]->num_elem() * sizeof(float));
  //   cudaMalloc(&d_mlp2_w[order[i]], mlp2_w[order[i]]->num_elem() * sizeof(float));
  // }
  // cudaMalloc(&d_ln_f_b, ln_f_b->num_elem() * sizeof(float));
  // cudaMalloc(&d_ln_f_g, ln_f_g->num_elem() * sizeof(float));
  // cudaMalloc(&d_wpe, wpe->num_elem() * sizeof(float));
  // cudaMalloc(&d_wte, wte->num_elem() * sizeof(float));
  
  // /* Copy data to device */
  // for (int i = 0; i < NUM_LAYER; i++) {
  //   cudaMemcpy(d_attn_b[order[i]], attn_b[order[i]]->buf, attn_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  //   cudaMemcpy(d_attn_w[order[i]], attn_w[order[i]]->buf, attn_w[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  //   cudaMemcpy(d_proj_b[order[i]], proj_b[order[i]]->buf, proj_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  //   cudaMemcpy(d_proj_w[order[i]], proj_w[order[i]]->buf, proj_w[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  //   cudaMemcpy(d_ln_1_b[order[i]], ln_1_b[order[i]]->buf, ln_1_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  //   cudaMemcpy(d_ln_1_g[order[i]], ln_1_g[order[i]]->buf, ln_1_g[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  //   cudaMemcpy(d_ln_2_b[order[i]], ln_2_b[order[i]]->buf, ln_2_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  //   cudaMemcpy(d_ln_2_g[order[i]], ln_2_g[order[i]]->buf, ln_2_g[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  //   cudaMemcpy(d_mlp1_b[order[i]], mlp1_b[order[i]]->buf, mlp1_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  //   cudaMemcpy(d_mlp1_w[order[i]], mlp1_w[order[i]]->buf, mlp1_w[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  //   cudaMemcpy(d_mlp2_b[order[i]], mlp2_b[order[i]]->buf, mlp2_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  //   cudaMemcpy(d_mlp2_w[order[i]], mlp2_w[order[i]]->buf, mlp2_w[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  // }
  // cudaMemcpy(d_ln_f_b, ln_f_b->buf, ln_f_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_ln_f_g, ln_f_g->buf, ln_f_g->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_wpe, wpe->buf, wpe->num_elem() * sizeof(float), cudaMemcpyHostToDevice); 
  // cudaMemcpy(d_wte, wte->buf, wte->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
}

void alloc_and_set_device_parameters() {
  int order[] = {
      0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9,
  };

  /* Allocate device memory */
  for (int i = 0; i < NUM_LAYER; i++) {
    cudaMalloc(&d_attn_b[order[i]], attn_b[order[i]]->num_elem() * sizeof(float));
    cudaMalloc(&d_attn_w[order[i]], attn_w[order[i]]->num_elem() * sizeof(float));
    cudaMalloc(&d_proj_b[order[i]], proj_b[order[i]]->num_elem() * sizeof(float));
    cudaMalloc(&d_proj_w[order[i]], proj_w[order[i]]->num_elem() * sizeof(float));
    cudaMalloc(&d_ln_1_b[order[i]], ln_1_b[order[i]]->num_elem() * sizeof(float));
    cudaMalloc(&d_ln_1_g[order[i]], ln_1_g[order[i]]->num_elem() * sizeof(float));
    cudaMalloc(&d_ln_2_b[order[i]], ln_2_b[order[i]]->num_elem() * sizeof(float));
    cudaMalloc(&d_ln_2_g[order[i]], ln_2_g[order[i]]->num_elem() * sizeof(float));
    cudaMalloc(&d_mlp1_b[order[i]], mlp1_b[order[i]]->num_elem() * sizeof(float));
    cudaMalloc(&d_mlp1_w[order[i]], mlp1_w[order[i]]->num_elem() * sizeof(float));
    cudaMalloc(&d_mlp2_b[order[i]], mlp2_b[order[i]]->num_elem() * sizeof(float));
    cudaMalloc(&d_mlp2_w[order[i]], mlp2_w[order[i]]->num_elem() * sizeof(float));
  }
  cudaMalloc(&d_ln_f_b, ln_f_b->num_elem() * sizeof(float));
  cudaMalloc(&d_ln_f_g, ln_f_g->num_elem() * sizeof(float));
  cudaMalloc(&d_wpe, wpe->num_elem() * sizeof(float));
  cudaMalloc(&d_wte, wte->num_elem() * sizeof(float));
  
  /* Copy data to device */
  for (int i = 0; i < NUM_LAYER; i++) {
    cudaMemcpy(d_attn_b[order[i]], attn_b[order[i]]->buf, attn_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_attn_w[order[i]], attn_w[order[i]]->buf, attn_w[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj_b[order[i]], proj_b[order[i]]->buf, proj_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj_w[order[i]], proj_w[order[i]]->buf, proj_w[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln_1_b[order[i]], ln_1_b[order[i]]->buf, ln_1_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln_1_g[order[i]], ln_1_g[order[i]]->buf, ln_1_g[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln_2_b[order[i]], ln_2_b[order[i]]->buf, ln_2_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln_2_g[order[i]], ln_2_g[order[i]]->buf, ln_2_g[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp1_b[order[i]], mlp1_b[order[i]]->buf, mlp1_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp1_w[order[i]], mlp1_w[order[i]]->buf, mlp1_w[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp2_b[order[i]], mlp2_b[order[i]]->buf, mlp2_b[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp2_w[order[i]], mlp2_w[order[i]]->buf, mlp2_w[order[i]]->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  }
  cudaMemcpy(d_ln_f_b, ln_f_b->buf, ln_f_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ln_f_g, ln_f_g->buf, ln_f_g->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wpe, wpe->buf, wpe->num_elem() * sizeof(float), cudaMemcpyHostToDevice); 
  cudaMemcpy(d_wte, wte->buf, wte->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
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

float *d_embd_a, *d_ffn_proj_a;
float *d_mha_qkv_proj_a, *d_mha_out_a, *d_mha_split_qkv_a,
    *d_mha_split_head_a, *d_mha_mask_a, *d_mha_merge_head_a, *d_mha_q_a,
    *d_mha_k_a, *d_mha_v_a, *d_mha_attn_out_a, *d_mha_concat_head_a;
float *d_attn_score_a, *d_k_transposed_a;
float *d_wte_transposed_a, *d_residual_a, *d_logit_a;
float *d_transformer_block_a;

void alloc_activations(size_t prompt_size) {
  /* Allocate device memory */
  // TODO check device pointers without prompt_size or has prompt_size twice 
  cudaMalloc(&d_embd_a, prompt_size * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_ffn_proj_a, prompt_size * 4 * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_mha_qkv_proj_a, prompt_size * 3 * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_mha_out_a, prompt_size * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_mha_split_qkv_a, 3 * prompt_size * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_mha_split_head_a, 3 * NUM_HEAD * prompt_size * HIDDEN_DIM / NUM_HEAD * sizeof(float));
  cudaMalloc(&d_mha_mask_a, prompt_size * prompt_size * sizeof(float));
  cudaMalloc(&d_mha_merge_head_a, NUM_HEAD * prompt_size * HIDDEN_DIM / NUM_HEAD * sizeof(float));
  cudaMalloc(&d_mha_q_a, prompt_size * HIDDEN_DIM / NUM_HEAD * sizeof(float));
  cudaMalloc(&d_mha_k_a, prompt_size * HIDDEN_DIM / NUM_HEAD * sizeof(float));
  cudaMalloc(&d_mha_v_a, prompt_size * HIDDEN_DIM / NUM_HEAD * sizeof(float));
  cudaMalloc(&d_mha_attn_out_a, prompt_size * HIDDEN_DIM / NUM_HEAD * sizeof(float));
  cudaMalloc(&d_mha_concat_head_a, prompt_size * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_attn_score_a, prompt_size * prompt_size * sizeof(float));
  cudaMalloc(&d_k_transposed_a, HIDDEN_DIM / NUM_HEAD * prompt_size * sizeof(float));
  cudaMalloc(&d_wte_transposed_a, HIDDEN_DIM * NUM_VOCAB * sizeof(float));
  cudaMalloc(&d_residual_a, prompt_size * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_logit_a, prompt_size * NUM_VOCAB * sizeof(float));
  cudaMalloc(&d_transformer_block_a, prompt_size * HIDDEN_DIM * sizeof(float));
}

void free_activations() {
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
// void ffn(Activation *in, Parameter *mlp1_w, Parameter *mlp1_b,
//          Parameter *mlp2_w, Parameter *mlp2_b, Activation *out) {
//   /* Projection Up:
//     [seq_len, HIDDEN_DIM] -> [seq_len, 4*HIDDEN_DIM] */
//   linear(in, mlp1_w, mlp1_b, ffn_proj_a);

//   /* GELU */
//   gelu(ffn_proj_a);

//   /* Projection Down:
//     [seq_len, 4*HIDDEN_DIM] -> [seq_len, HIDDEN_DIM] */
//   linear(ffn_proj_a, mlp2_w, mlp2_b, out);
// }
void ffn(float *d_in, float *d_mlp1_w, float *d_mlp1_b,
         float *d_mlp2_w, float *d_mlp2_b, float *d_out, size_t seq_len, size_t batch_size) {
    /* Projection Up: [seq_len, HIDDEN_DIM] -> [seq_len, 4*HIDDEN_DIM] */
    // linear(d_in, d_mlp1_w, d_mlp1_b, d_ffn_proj_a, seq_len, HIDDEN_DIM, 4 * HIDDEN_DIM);
    batch_linear(d_in, d_mlp1_w, d_mlp1_b, d_ffn_proj_a, batch_size, seq_len, HIDDEN_DIM, 4 * HIDDEN_DIM);

    /* GELU */
    // gelu(d_ffn_proj_a, seq_len * 4 * HIDDEN_DIM);
    batch_gelu(d_ffn_proj_a, batch_size, seq_len * 4 * HIDDEN_DIM);

    /* Projection Down: [seq_len, 4*HIDDEN_DIM] -> [seq_len, HIDDEN_DIM] */
    // linear(d_ffn_proj_a, d_mlp2_w, d_mlp2_b, d_out, seq_len, 4 * HIDDEN_DIM, HIDDEN_DIM);
    batch_linear(d_ffn_proj_a, d_mlp2_w, d_mlp2_b, d_out, batch_size, seq_len, 4 * HIDDEN_DIM, HIDDEN_DIM);
}

/* Attention
 * @param [in1]    q: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in2]    k: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in3]    v: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in4] mask: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [out]  out: [seq_len, HIDDEN_DIM/NUM_HEAD]
 */
// void attention(Activation *q, Activation *k, Activation *v, Activation *mask,
//                Activation *out) {
//   /* Get Attention score by q @ k */
//   transpose(k, k_transposed_a);
//   matmul(q, k_transposed_a, attn_score_a);

//   /* Scaling */
//   scaling(attn_score_a, (1.0 / sqrt(k->shape[1])));

//   /* Masking */
//   add(attn_score_a, mask);

//   /* Softmax */
//   softmax(attn_score_a);

//   /* Attention score @ v */
//   matmul(attn_score_a, v, out);
// }
void attention(float *d_q, float *d_k, float *d_v, float *d_mask, float *d_out, size_t seq_len, size_t head_dim, size_t batch_size) {
    /* Get Attention score by q @ k */
    batch_transpose(d_k, d_k_transposed_a, batch_size, seq_len, head_dim);
    batch_matmul(d_q, d_k_transposed_a, d_attn_score_a, batch_size, seq_len, head_dim, seq_len);
    batch_scaling(d_attn_score_a, 1.0 / sqrt(head_dim), batch_size, seq_len * seq_len);
    batch_add(d_attn_score_a, d_mask, batch_size, seq_len * seq_len);
    batch_softmax(d_attn_score_a, batch_size, seq_len, seq_len);
    batch_matmul(d_attn_score_a, d_v, d_out, batch_size, seq_len, seq_len, head_dim);
}

/* (Masked) Multi-Head Self Attention
 * @param [in1]     in: [seq_len, HIDDEN_DIM]
 * @param [in2] attn_b: [3*HIDDEN_DIM]
 * @param [in3] attn_w: [HIDDEN_DIM, 3*HIDDEN_DIM]
 * @param [in4] proj_b: [HIDDEN_DIM]
 * @param [in5] proj_w: [HIDDEN_DIM, HIDDEN_DIM]
 * @param [out]    out: [seq_len, HIDDEN_DIM]
 */
// void mha(Activation *in, Parameter *attn_b, Parameter *attn_w,
//          Parameter *proj_b, Parameter *proj_w, Activation *out) {
//   /* QKV projection:
//     [seq_len, HIDDEN_DIM] ->
//     [seq_len, 3*HIDDEN_DIM] */
//   linear(in, attn_w, attn_b, mha_qkv_proj_a);

//   /* Split into Q, K, V:
//     [seq_len, 3*HIDDEN_DIM] ->
//     [3, seq_len, HIDDEN_DIM] */
//   split_qkv(mha_qkv_proj_a, mha_split_qkv_a);

//   /* Split into multiple heads:
//     [3, seq_len, HIDDEN_DIM] ->
//     [3, NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
//   split_head(mha_split_qkv_a, NUM_HEAD, mha_split_head_a);

//   /* Generate mask to hide future inputs */
//   generate_mask(mha_mask_a);

//   /* Perform Attention over each head:
//     [NUM_HEAD, 3, seq_len, HIDDEN_DIM/NUM_HEAD] ->
//     [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
//   for (size_t idx = 0; idx < NUM_HEAD; idx++) {
//     /* Extract Q, K, V from qkv_head */
//     extract_qkv(mha_split_head_a, idx, NUM_HEAD, mha_q_a, mha_k_a, mha_v_a);

//     /* Attention */
//     attention(mha_q_a, mha_k_a, mha_v_a, mha_mask_a, mha_attn_out_a);

//     /* Merge each head's attn output
//       [seq_len, HIDDEN_DIM/NUM_HEAD] ->
//       [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
//     merge_head(mha_attn_out_a, idx, NUM_HEAD, mha_merge_head_a);
//   }

//   /* Concat each heads:
//     [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] ->
//     [seq_len, HIDDEN_DIM] */
//   concat_head(mha_merge_head_a, mha_concat_head_a);

//   /* OUT projection:
//     [seq_len, HIDDEN_DIM] -> [seq_len, HIDDEN_DIM] */
//   linear(mha_concat_head_a, proj_w, proj_b, out);
// }
void mha(float *d_in, float *d_attn_b, float *d_attn_w,
         float *d_proj_b, float *d_proj_w, float *d_out, size_t seq_len, size_t batch_size) {
    /* QKV projection: [seq_len, HIDDEN_DIM] -> [seq_len, 3*HIDDEN_DIM] */
    batch_linear(d_in, d_attn_w, d_attn_b, d_mha_qkv_proj_a, batch_size, seq_len, HIDDEN_DIM, 3 * HIDDEN_DIM);
    batch_split_qkv(d_mha_qkv_proj_a, d_mha_split_qkv_a, batch_size, seq_len, 3 * HIDDEN_DIM);
    batch_split_head(d_mha_split_qkv_a, d_mha_split_head_a, batch_size, NUM_HEAD, seq_len, HIDDEN_DIM); // / NUM_HEAD);
    batch_generate_mask(d_mha_mask_a, batch_size, seq_len);

    /* Perform Attention over each head: [NUM_HEAD, 3, seq_len, HIDDEN_DIM/NUM_HEAD] -> [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
    for (size_t idx = 0; idx < NUM_HEAD; idx++) {
        /* Extract Q, K, V from qkv_head */
        // extract_qkv(d_mha_split_head_a, d_mha_q_a, d_mha_k_a, d_mha_v_a, idx, NUM_HEAD, seq_len, HIDDEN_DIM / NUM_HEAD);
        batch_extract_qkv(d_mha_split_head_a, d_mha_q_a, d_mha_k_a, d_mha_v_a, batch_size, idx, NUM_HEAD, seq_len, HIDDEN_DIM / NUM_HEAD);

        /* Attention */
        // TODO change
        attention(d_mha_q_a, d_mha_k_a, d_mha_v_a, d_mha_mask_a, d_mha_attn_out_a, seq_len, HIDDEN_DIM / NUM_HEAD, batch_size);

        /* Merge each head's attn output [seq_len, HIDDEN_DIM/NUM_HEAD] -> [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
        // merge_head(d_mha_attn_out_a, d_mha_merge_head_a, idx, seq_len, HIDDEN_DIM / NUM_HEAD);
        batch_merge_head(d_mha_attn_out_a, d_mha_merge_head_a, batch_size, NUM_HEAD, idx, seq_len, HIDDEN_DIM / NUM_HEAD);
    }

    /* Concat each heads: [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] -> [seq_len, HIDDEN_DIM] */
    // concat_head(d_mha_merge_head_a, d_mha_concat_head_a, NUM_HEAD, seq_len, HIDDEN_DIM / NUM_HEAD);
    batch_concat_head(d_mha_merge_head_a, d_mha_concat_head_a, batch_size, NUM_HEAD, seq_len, HIDDEN_DIM / NUM_HEAD);

    /* OUT projection: [seq_len, HIDDEN_DIM] -> [seq_len, HIDDEN_DIM] */
    // linear(d_mha_concat_head_a, d_proj_w, d_proj_b, d_out, seq_len, HIDDEN_DIM, HIDDEN_DIM);
    batch_linear(d_mha_concat_head_a, d_proj_w, d_proj_b, d_out, batch_size, seq_len, HIDDEN_DIM, HIDDEN_DIM);
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
// void transformer_block(Activation *in, Parameter *attn_b, Parameter *attn_w,
//                        Parameter *proj_b, Parameter *proj_w, Parameter *ln_1_b,
//                        Parameter *ln_1_g, Parameter *ln_2_b, Parameter *ln_2_g,
//                        Parameter *mlp1_b, Parameter *mlp1_w, Parameter *mlp2_b,
//                        Parameter *mlp2_w, Activation *out) {
//   /* Copy Residual */
//   copy(in, residual_a);

//   /* Layer Normalization */
//   layer_norm(in, ln_1_g, ln_1_b);

//   /* Masked Multi-Head Self-Attention */
//   mha(in, attn_b, attn_w, proj_b, proj_w, mha_out_a);

//   /* Add Residual */
//   add(mha_out_a, residual_a);

//   /* Copy Residual */
//   copy(mha_out_a, residual_a);

//   /* Layer Normalization */
//   layer_norm(mha_out_a, ln_2_g, ln_2_b);

//   /* Position-wise Feed-Forward Network */
//   ffn(mha_out_a, mlp1_w, mlp1_b, mlp2_w, mlp2_b, out);

//   /* Add Residual */
//   add(out, residual_a);
// }
void transformer_block(float *d_in, float *d_attn_b, float *d_attn_w,
                       float *d_proj_b, float *d_proj_w, float *d_ln_1_b,
                       float *d_ln_1_g, float *d_ln_2_b, float *d_ln_2_g,
                       float *d_mlp1_b, float *d_mlp1_w, float *d_mlp2_b,
                       float *d_mlp2_w, float *d_out, size_t seq_len, size_t batch_size) { 
    /* Copy Residual */
    batch_copy(d_in, d_residual_a, batch_size, seq_len * HIDDEN_DIM);
    batch_layer_norm(d_in, d_ln_1_g, d_ln_1_b, batch_size, seq_len, HIDDEN_DIM, 1e-5);

    /* Masked Multi-Head Self-Attention */
    // mha(d_in, d_attn_b, d_attn_w, d_proj_b, d_proj_w, d_mha_out_a, seq_len);
    mha(d_in, d_attn_b, d_attn_w, d_proj_b, d_proj_w, d_mha_out_a, seq_len, batch_size);

    /* Add Residual */
    // add(d_mha_out_a, d_residual_a, seq_len * HIDDEN_DIM);
    batch_add(d_mha_out_a, d_residual_a, batch_size, seq_len * HIDDEN_DIM);

    /* Copy Residual */
    // copy(d_mha_out_a, d_residual_a, seq_len * HIDDEN_DIM);
    batch_copy(d_mha_out_a, d_residual_a, batch_size, seq_len * HIDDEN_DIM);

    /* Layer Normalization */
    // layer_norm(d_mha_out_a, d_ln_2_g, d_ln_2_b, seq_len, HIDDEN_DIM, 1e-5);
    batch_layer_norm(d_mha_out_a, d_ln_2_g, d_ln_2_b, batch_size, seq_len, HIDDEN_DIM, 1e-5);

    /* Position-wise Feed-Forward Network */
    ffn(d_mha_out_a, d_mlp1_w, d_mlp1_b, d_mlp2_w, d_mlp2_b, d_out, seq_len, batch_size);

    /* Add Residual */
    // add(d_out, d_residual_a, seq_len * HIDDEN_DIM);
    batch_add(d_out, d_residual_a, batch_size, seq_len * HIDDEN_DIM);
}

/* [Model Computation: Token Generation] */
void generate_tokens(int *input, int *output, size_t n_prompt, size_t n_token) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (mpi_rank != 0) {
        input = (int *)malloc(n_prompt * tokens_per_prompt * sizeof(int));
        output = (int *)malloc(n_prompt * n_token * sizeof(int));
    }

    // Broadcast input data to all ranks
    if (mpi_rank == 0) {
        printf("\nBroadcasting input data...\n");
    }
    MPI_Bcast(input, n_prompt * tokens_per_prompt, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(output, n_prompt * n_token, MPI_INT, 0, MPI_COMM_WORLD);

    size_t prompts_per_node = (n_prompt + mpi_size - 1) / mpi_size;
    size_t start_prompt = mpi_rank * prompts_per_node;
    size_t end_prompt = MIN(start_prompt + prompts_per_node, n_prompt);

    // printf("mpi_rank: %d | Initializing input and parameters...\n", mpi_rank);
    alloc_and_set_device_parameters();
    // printf("mpi_rank: %d | Generating tokens...\n", mpi_rank);

    /* Outer loop: generate tokens for each prompt */
    for (size_t p = start_prompt; p < end_prompt; p += BATCH_SIZE) {
        int batch_size = MIN(BATCH_SIZE, end_prompt - p);
        // if (mpi_rank == 0) {
        // printf("mpi_rank: %d | Prompts %zu-%zu\n", mpi_rank, p, p + batch_size);
        // }
        int prompt_size = tokens_per_prompt;

        /* Initialize input prompt */
        std::vector<int> input_prompt(batch_size * prompt_size);
        // printf("mpi_rank: %d | Declared input prompt\n", mpi_rank);
        // printf("mpi_rank: %d | Address of input: %p\n", mpi_rank, input);
        // printf("mpi_rank: %d | p: %zu\n", mpi_rank, p);
        // printf("mpi_rank: %d | Address of input + p * prompt_size: %p\n", mpi_rank, input + p * prompt_size);
        memcpy(input_prompt.data(), input + p * prompt_size, batch_size * prompt_size * sizeof(int));

        // printf("mpi_rank: %d | Initialized input prompt\n", mpi_rank);

        int *d_input_prompt;
        CHECK_CUDA(cudaMalloc(&d_input_prompt, batch_size * (prompt_size + n_token - 1) * sizeof(int)));

        int *d_out;
        CHECK_CUDA(cudaMalloc(&d_out, batch_size * sizeof(int)));

        // printf("mpi_rank: %d | Allocated input prompt and out\n", mpi_rank);

        /* Initialize activations */
        alloc_activations(batch_size * (prompt_size + n_token - 1));

        // printf("mpi_rank: %d | Allocated activations\n", mpi_rank);

        /* Inner loop: generate next token */
        for (size_t t = 0; t < n_token; t++) {
            CHECK_CUDA(cudaMemcpy(d_input_prompt, input_prompt.data(), batch_size * prompt_size * sizeof(int), cudaMemcpyHostToDevice));

            // printf("mpi_rank: %d | After cudaMemcpy d_input_prompt\n", mpi_rank);

            /* Token + Positional Embedding */
            batch_token_pos_embedding(d_input_prompt, d_wte, d_wpe, d_embd_a, batch_size, prompt_size, HIDDEN_DIM);

            // printf("mpi_rank: %d | After batch_token_pos_embedding\n", mpi_rank);

            /* Forward path of Transformer blocks */
            for (size_t l = 0; l < NUM_LAYER; l++) {
                transformer_block(d_embd_a, d_attn_b[l], d_attn_w[l], d_proj_b[l], d_proj_w[l],
                                  d_ln_1_b[l], d_ln_1_g[l], d_ln_2_b[l], d_ln_2_g[l],
                                  d_mlp1_b[l], d_mlp1_w[l], d_mlp2_b[l], d_mlp2_w[l],
                                  d_transformer_block_a, prompt_size, batch_size);
                batch_copy(d_transformer_block_a, d_embd_a, batch_size, prompt_size * HIDDEN_DIM);
            }

            /* Final Layer Normalization */
            batch_layer_norm(d_embd_a, d_ln_f_g, d_ln_f_b, batch_size, prompt_size, HIDDEN_DIM, 1e-5);

            /* Projection to vocab. dimension */
            transpose(d_wte, d_wte_transposed_a, wte->shape[0], wte->shape[1]);
            batch_matmul_final(d_embd_a, d_wte_transposed_a, d_logit_a, batch_size, prompt_size, HIDDEN_DIM, wte->shape[0]);

            /* Greedy sampling (only last timestep is considered) */
            batch_top1_sampling(d_logit_a, d_out, batch_size, prompt_size, NUM_VOCAB);

            // printf("mpi_rank: %d | After batch_top1_sampling\n", mpi_rank);

            std::vector<int> next_token_ids(batch_size);
            CHECK_CUDA(cudaMemcpy(next_token_ids.data(), d_out, batch_size * sizeof(int), cudaMemcpyDeviceToHost));

            // printf("mpi_rank: %d | After cudaMemcpy next_token_ids\n", mpi_rank);
            for (size_t i = 0; i < batch_size; i++) {
                size_t insert_position = (i + 1) * prompt_size + i;
                input_prompt.insert(input_prompt.begin() + insert_position, next_token_ids[i]);
                // printf("mpi_rank: %d | After insertion to input_prompt\n", mpi_rank);
                output[(p + i) * n_token + t] = next_token_ids[i];
            }
            prompt_size += 1;
            // printf("mpi_rank: %d | After writing to output\n", mpi_rank);
        }

        CHECK_CUDA(cudaFree(d_input_prompt));
        CHECK_CUDA(cudaFree(d_out));
        free_activations();
    }

    // printf("mpi_rank: %d | After generating tokens\n", mpi_rank);

    if (mpi_rank == 0) {
        std::vector<int> final_output(n_prompt * n_token);
        MPI_Gather(output, prompts_per_node * n_token, MPI_INT, final_output.data(), prompts_per_node * n_token, MPI_INT, 0, MPI_COMM_WORLD);
        memcpy(output, final_output.data(), n_prompt * n_token * sizeof(int));
    } else {
        MPI_Gather(output + start_prompt * n_token, prompts_per_node * n_token, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // printf("mpi_rank: %d | Token generation complete\n", mpi_rank);
}
