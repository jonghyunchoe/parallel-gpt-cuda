#include "layer.h"

#define TILE_SIZE 32 

// TODO move and coalesce cudaMalloc, cudaMemcpy, and cudaFree to start and end 
// TODO function only has blockDim, gridDim and kernel call 

/* Token + Positional Embedding
 * @param [in1]  in: [s]
 * @param [in2] wte: [NUM_VOCAB, H]
 * @param [in3] wpe: [MAX_SEQ_LEN, H]
 * @param [out] out: [s, H]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
// void token_pos_embedding(vector<int> in, Tensor *wte, Tensor *wpe,
//                          Tensor *out) {
//   size_t s = in.size();
//   size_t H = wte->shape[1];

//   for (size_t i = 0; i < s; i++) {
//     for (size_t j = 0; j < H; j++) {
//       out->buf[i * H + j] = wte->buf[in[i] * H + j] + wpe->buf[i * H + j];
//     }
//   }
// }

__global__ void token_pos_embedding_kernel(int *in, float *wte, float *wpe, float *out, size_t s, size_t H) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 

  if (idx < s * H) {
    size_t i = idx / H; 
    size_t j = idx % H; 
    out[idx] = wte[in[i] * H + j] + wpe[i * H + j]; 
  }
}

void token_pos_embedding(int *d_in, float *d_wte, float *d_wpe, float *d_out, size_t s, size_t H) {
  dim3 blockDim(256); 
  dim3 gridDim((s * H + blockDim.x - 1) / blockDim.x); 
  token_pos_embedding_kernel<<<gridDim, blockDim>>>(d_in, d_wte, d_wpe, d_out, s, H);
}

/* GELU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
// void gelu(Tensor *inout) {
//   size_t N = inout->num_elem();

//   for (size_t i = 0; i < N; i++) {
//     float x = inout->buf[i];
//     inout->buf[i] =
//         0.5 * x *
//         (1.f + tanh(sqrt(2.f / MATH_PI) * (x + 0.044715f * x * x * x)));
//   }
// }

__global__ void gelu_kernel(float *inout, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
  
  if (idx < N) {
    float x = inout[idx]; 
    inout[idx] = 
        0.5 * x *
        (1.f + tanh(sqrt(2.f / MATH_PI) * (x + 0.044715f * x * x * x)));
  }
}

void gelu(float *d_inout, size_t N) {
  dim3 blockDim(256); 
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x); 
  gelu_kernel<<<gridDim, blockDim>>>(d_inout, N);
}

/* Softmax (w/ Max Trick)
 * @param [in & out] inout: [s, H]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
// void softmax(Tensor *inout) {
//   size_t s = inout->shape[0];
//   size_t H = inout->shape[1];

//   for (size_t i = 0; i < s; i++) {
//     float max_val = inout->buf[i * H];
//     for (size_t j = 0; j < H; j++) {
//       if (inout->buf[i * H + j] > max_val) { max_val = inout->buf[i * H + j]; }
//     }

//     float sum = 0;
//     for (size_t j = 0; j < H; j++) {
//       inout->buf[i * H + j] = exp(inout->buf[i * H + j] - max_val);
//       sum += inout->buf[i * H + j];
//     }

//     for (size_t j = 0; j < H; j++) { inout->buf[i * H + j] /= sum; }
//   }
// }

__global__ void softmax_kernel(float *inout, size_t s, size_t V) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < s) {
    float max_val = -INFINITY;
    for (size_t j = 0; j < V; j++) {
      max_val = fmaxf(max_val, inout[idx * V + j]);
    }

    float sum_exp = 0.0f;
    for (size_t j = 0; j < V; j++) {
      sum_exp += expf(inout[idx * V + j] - max_val);
    }

    for (size_t j = 0; j < V; j++) {
      inout[idx * V + j] = expf(inout[idx * V + j] - max_val) / sum_exp;
    }
  }
}

void softmax(float *d_inout, size_t s, size_t V) {
  dim3 blockDim(256);
  dim3 gridDim((s + blockDim.x - 1) / blockDim.x);
  softmax_kernel<<<gridDim, blockDim>>>(d_inout, s, V);
}

/* Layer Normalization
 * @param [in1 & out] inout: [s, H]
 * @param [in2]       gamma: [H]
 * @param [in3]        beta: [H]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
// void layer_norm(Tensor *inout, Tensor *gamma, Tensor *beta) {
//   size_t s = inout->shape[0];
//   size_t H = inout->shape[1];

//   float eps = 1e-5;
//   for (size_t i = 0; i < s; i++) {
//     float mean = 0;
//     float var = 0;

//     for (size_t j = 0; j < H; j++) {
//       mean += inout->buf[i * H + j];
//       var += inout->buf[i * H + j] * inout->buf[i * H + j];
//     }

//     mean /= H;
//     var = var / H - mean * mean;

//     for (size_t j = 0; j < H; j++) {
//       inout->buf[i * H + j] = (inout->buf[i * H + j] - mean) *
//                                   (1.0 / sqrt(var + eps)) * gamma->buf[j] +
//                               beta->buf[j];
//     }
//   }
// }

__global__ void layer_norm_kernel(float *inout, float *gamma, float *beta, size_t s, size_t H, float eps) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < s) {
    float mean = 0;
    float var = 0;

    for (size_t j = 0; j < H; j++) {
      mean += inout[i * H + j];
      var += inout[i * H + j] * inout[i * H + j];
    }
    mean /= H;
    var = var / H - mean * mean;

    for (size_t j = 0; j < H; j++) {
      inout[i * H + j] = (inout[i * H + j] - mean) * (1.0 / sqrtf(var + eps)) * gamma[j] + beta[j];
    }
  }
}

void layer_norm(float *d_inout, float *d_gamma, float *d_beta, size_t s, size_t H) {
  float eps = 1e-5;
  dim3 blockDim(256);
  dim3 gridDim((s + blockDim.x - 1) / blockDim.x);
  layer_norm_kernel<<<gridDim, blockDim>>>(d_inout, d_gamma, d_beta, s, H, eps);
}

/* Linear
 * @param [in1]  in: [M, K]
 * @param [in2]   w: [K, N]
 * @param [in3]   b: [N]
 * @param [out] out: [M, N]
 */
// void linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
//   size_t M = in->shape[0];
//   size_t K = in->shape[1];
//   size_t N = w->shape[1];

// #pragma omp parallel for
//   for (size_t i = 0; i < M; i++) {
//     for (size_t j = 0; j < N; j++) {
//       out->buf[i * N + j] = 0;
//       for (size_t k = 0; k < K; k++) {
//         out->buf[i * N + j] += in->buf[i * K + k] * w->buf[k * N + j];
//       }
//       out->buf[i * N + j] += b->buf[j];
//     }
//   }
// }

__global__ void linear_kernel(float *in, float *w, float *b, float *out, size_t M, size_t K, size_t N) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0;
    for (size_t k = 0; k < K; k++) {
      sum += in[row * K + k] * w[k * N + col];
    }
    out[row * N + col] = sum + b[col];
  }
}

void linear(float *d_in, float *d_w, float *d_b, float *d_out, size_t M, size_t K, size_t N) {
  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
  linear_kernel<<<gridDim, blockDim>>>(d_in, d_w, d_b, d_out, M, K, N);
}

/* Matmul
 * @param [in1]  in1: [M, K]
 * @param [in2]  in2: [K, N]
 * @param [out]  out: [M, N]
 */
// void matmul(Tensor *in1, Tensor *in2, Tensor *out) {
//   size_t M = in1->shape[0];
//   size_t K = in1->shape[1];
//   size_t N = in2->shape[1];

// #pragma omp parallel for
//   for (size_t i = 0; i < M; i++) {
//     for (size_t j = 0; j < N; j++) {
//       out->buf[i * N + j] = 0;
//       for (size_t k = 0; k < K; k++) {
//         out->buf[i * N + j] += in1->buf[i * K + k] * in2->buf[k * N + j];
//       }
//     }
//   }
// }

__global__ void matmul_kernel_naive(float *in1, float *in2, float *out, size_t M, size_t K, size_t N) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y; 
  size_t col = blockIdx.x * blockDim.x + threadIdx.x; 
  
  if (row < M && col < N) {
    float sum = 0;
    for (size_t k = 0; k < K; k++) {
      sum += in1[row * K + k] * in2[k * N + col];
    }
    out[row * N + col] = sum; 
  }
}

__global__ void matmul_kernel(float *A, float *B, float *C, size_t M, size_t K, size_t N) {
  __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
  __shared__ float B_shared[TILE_SIZE][TILE_SIZE];   

  int global_row = blockIdx.y * blockDim.y + threadIdx.y;
  int global_col = blockIdx.x * blockDim.x + threadIdx.x; 
  int local_row = threadIdx.y;
  int local_col = threadIdx.x; 

  float sum = 0.0f;

  for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; k++) {
    int A_local_row = global_row * K + k * TILE_SIZE + local_col; 
    int B_local_col = (k * TILE_SIZE + local_row) * N + global_col; 
    
    if (global_row < M && k * TILE_SIZE + local_col < K) {
      A_shared[local_row][local_col] = A[A_local_row];
    } else {
      A_shared[local_row][local_col] = 0.0f; 
    }

    if (global_col < N && k * TILE_SIZE + local_row < K) {
      B_shared[local_row][local_col] = B[B_local_col];
    } else {
      B_shared[local_row][local_col] = 0.0f; 
    }

    __syncthreads();

    for (int n = 0; n < TILE_SIZE; n++) {
      sum += A_shared[local_row][n] * B_shared[n][local_col];
    }

    __syncthreads(); 
  }

  if (global_row < M && global_col < N) {
    C[global_row * N + global_col] = sum; 
  }
}

void matmul(float *d_in1, float *d_in2, float *d_out, size_t M, size_t K, size_t N) {
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  matmul_kernel<<<gridDim, blockDim>>>(d_in1, d_in2, d_out, M, K, N);
}

/* Transpose
 * @param [in1]  in: [M, N]
 * @param [out] out: [N, M]
 */
// void transpose(Tensor *in, Tensor *out) {
//   size_t M = in->shape[0];
//   size_t N = in->shape[1];

//   for (size_t i = 0; i < M; i++) {
//     for (size_t j = 0; j < N; j++) { out->buf[j * M + i] = in->buf[i * N + j]; }
//   }
// }

__global__ void transpose_kernel(float *in, float *out, size_t M, size_t N) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y; 
  size_t col = blockIdx.x * blockDim.x + threadIdx.x; 
  
  if (row < M && col < N) {
    out[col * M + row] = in[row * N + col];
  }
}

void transpose(float *d_in, float *d_out, size_t M, size_t N) {
  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim. y - 1) / blockDim.y);
  transpose_kernel<<<gridDim, blockDim>>>(d_in, d_out, M, N);
}

/* Scaling
 * @param [in1 & out] inout: [N]
 * @param [in2]       scale: [1]
 * 'N' is the number of elements in the tensor.
 */
// void scaling(Tensor *inout, float scale) {
//   size_t N = inout->num_elem();

//   for (size_t i = 0; i < N; i++) { inout->buf[i] *= scale; }
// }

__global__ void scaling_kernel(float *inout, float scale, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 

  if (idx < N) {
    inout[idx] *= scale; 
  }
}

void scaling(float *d_inout, float scale, size_t N) {
  dim3 blockDim(256);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
  scaling_kernel<<<gridDim, blockDim>>>(d_inout, scale, N);
}

/* Generate mask
 * @param [in & out] inout: [s, s]
 * 's' is the number of tokens in the prompt.
 */
// void generate_mask(Tensor *inout) {
//   size_t s = inout->shape[0];

//   for (size_t i = 0; i < s; i++) {
//     for (size_t j = 0; j < s; j++) {
//       if (i >= j) {
//         inout->buf[i * s + j] = 0;
//       } else {
//         inout->buf[i * s + j] = -1e10;
//       }
//     }
//   }
// }

__global__ void generate_mask_kernel(float *inout, size_t s) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < s && col < s) {
    if (row >= col) {
      inout[row * s + col] = 0;
    } else {
      inout[row * s + col] = -1e10;
    }
  }
}

void generate_mask(float *d_inout, size_t s) {
  dim3 blockDim(16, 16);
  dim3 gridDim((s + blockDim.x - 1) / blockDim.x, (s + blockDim.y - 1) / blockDim.y);
  generate_mask_kernel<<<gridDim, blockDim>>>(d_inout, s);
}

/* Copy
 * @param [in1]  in: [N]
 * @param [out] out: [N]
 * 'N' is the number of elements in the tensor.
 */
// void copy(Tensor *in, Tensor *out) {
//   size_t N = in->num_elem();

//   for (size_t i = 0; i < N; i++) { out->buf[i] = in->buf[i]; }
// }

__global__ void copy_kernel(float *in, float *out, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 

  if (idx < N) {
    out[idx] = in[idx];
  }
}

void copy(float *d_in, float *d_out, size_t N) {
  dim3 blockDim(256);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
  copy_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
}

/* Add
 * @param [in1 & out] inout: [N]
 * @param [in2]           x: [N]
 * 'N' is the number of elements in the tensor.
 */
// void add(Tensor *inout, Tensor *x) {
//   size_t N = inout->num_elem();

//   for (size_t i = 0; i < N; i++) { inout->buf[i] += x->buf[i]; }
// }

/* Add GPU kernel
 * @param [in1 & out] inout: [N]
 * @param [in2]           x: [N]
 * 'N' is the number of elements in the tensor.
 */
__global__ void add_kernel(float *inout, float *x, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) { inout[idx] += x[idx]; }
}

/* Add using CUDA GPU
 * @param [in1 & out] inout: [N]
 * @param [in2]           x: [N]
 * 'N' is the number of elements in the tensor.
 */
void add(float *d_inout, float *d_x, size_t N) {
  dim3 blockDim(256);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
  add_kernel<<<gridDim, blockDim>>>(d_inout, d_x, N);
}

/* Split into QKV
 * @param [in1]  in: [s, H]
 * @param [out] out: [3, s, H/3]
 */
// void split_qkv(Tensor *in, Tensor *out) {
//   size_t s = in->shape[0];
//   size_t H = in->shape[1];

//   for (size_t i = 0; i < 3; i++) {
//     for (size_t j = 0; j < s; j++) {
//       for (size_t k = 0; k < H / 3; k++) {
//         out->buf[i * s * (H / 3) + j * (H / 3) + k] =
//             in->buf[i * (H / 3) + j * 3 * (H / 3) + k];
//       }
//     }
//   }
// }

__global__ void split_qkv_kernel(float *in, float *out, size_t s, size_t H, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    size_t i = idx / (s * (H / 3));
    size_t j = (idx % (s * (H / 3))) / (H / 3);
    size_t k = idx % (H / 3);

    out[idx] = in[i * (H / 3) + j * H + k];
  }
}

void split_qkv(float *d_in, float *d_out, size_t s, size_t H) {
  size_t N = 3 * s * (H / 3);
  dim3 blockDim(256);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
  split_qkv_kernel<<<gridDim, blockDim>>>(d_in, d_out, s, H, N);
}

/* Split into heads
 * @param [in1]  in: [3, s, H]
 * @param [out] out: [3, n_head, s, H/n_head]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 * 'n_head' is the number of heads.
 */
// void split_head(Tensor *in, size_t n_head, Tensor *out) {
//   size_t s = in->shape[1];
//   size_t H = in->shape[2];

//   for (size_t i = 0; i < 3; i++) {
//     for (size_t j = 0; j < n_head; j++) {
//       for (size_t k = 0; k < s; k++) {
//         for (size_t l = 0; l < H / n_head; l++) {
//           out->buf[i * n_head * s * H / n_head + j * s * H / n_head +
//                    k * H / n_head + l] =
//               in->buf[i * s * H + k * H + j * H / n_head + l];
//         }
//       }
//     }
//   }
// }

__global__ void split_head_kernel(float *in, float *out, size_t n_head, size_t s, size_t H, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    size_t i = idx / (n_head * s * (H / n_head));
    size_t j = (idx % (n_head * s * (H / n_head))) / (s * (H / n_head));
    size_t k = (idx % (s * (H / n_head))) / (H / n_head);
    size_t l = idx % (H / n_head);

    out[idx] = in[i * s * H + k * H + j * (H / n_head) + l];
  }
}

void split_head(float *d_in, float *d_out, size_t n_head, size_t s, size_t H) {
  size_t N = 3 * n_head * s * (H / n_head);
  dim3 blockDim(256);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
  split_head_kernel<<<gridDim, blockDim>>>(d_in, d_out, n_head, s, H, N);
}

/* Extract Q, K, V from QKV head
 * @param [in1]       in: [3, n_head, s, H_]
 * @param [in2] head_idx: [1]
 * @param [in3]   n_head: [1]
 * @param [out]        q: [s, H_]
 * @param [out]        k: [s, H_]
 * @param [out]        v: [s, H_]
 * 's' is the number of tokens in the prompt.
 * 'H_' is the hidden dimension/n_head.
 * 'n_head' is the number of heads.
 */
// void extract_qkv(Tensor *in, size_t head_idx, size_t n_head, Tensor *q,
//                  Tensor *k, Tensor *v) {
//   size_t s = in->shape[2];
//   size_t H_ = in->shape[3];  // = HIDDEN_DIM/NUM_HEAD

//   for (size_t i = 0; i < s; i++) {
//     for (size_t j = 0; j < H_; j++) {
//       q->buf[i * H_ + j] =
//           in->buf[0 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
//       k->buf[i * H_ + j] =
//           in->buf[1 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
//       v->buf[i * H_ + j] =
//           in->buf[2 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
//     }
//   }
// }

__global__ void extract_qkv_kernel(float *in, size_t head_idx, size_t n_head, float *q, float *k, float *v, size_t s, size_t H_, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    size_t i = idx / H_;
    size_t j = idx % H_;

    q[i * H_ + j] = in[0 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
    k[i * H_ + j] = in[1 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
    v[i * H_ + j] = in[2 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
  }
}

void extract_qkv(float *d_in, size_t head_idx, size_t n_head, float *d_q, float *d_k, float *d_v, size_t s, size_t H_) {
  size_t N = s * H_;
  dim3 blockDim(256);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
  extract_qkv_kernel<<<gridDim, blockDim>>>(d_in, head_idx, n_head, d_q, d_k, d_v, s, H_, N);
}

/* Merge each heads
 * @param [in1]       in: [s, H_]
 * @param [in2] head_idx: [1]
 * @param [in3]   n_head: [1]
 * @param [out]      out: [n_head, s, H_]
 * 's' is the number of tokens in the prompt.
 * 'H_' is the hidden dimension/n_head.
 * 'n_head' is the number of heads.
 */
// void merge_head(Tensor *in, size_t head_idx, size_t n_head, Tensor *out) {
//   size_t s = in->shape[0];
//   size_t H_ = in->shape[1];  // = HIDDEN_DIM/NUM_HEAD

//   for (size_t i = 0; i < s; i++) {
//     for (size_t j = 0; j < H_; j++) {
//       out->buf[head_idx * s * H_ + i * H_ + j] = in->buf[i * H_ + j];
//     }
//   }
// }

__global__ void merge_head_kernel(float *in, size_t head_idx, size_t s, size_t H_, float *out) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < s * H_) {
    size_t i = idx / H_;
    size_t j = idx % H_;
    out[head_idx * s * H_ + i * H_ + j] = in[idx];
  }
}

void merge_head(float *d_in, size_t head_idx, size_t n_head, float *d_out, size_t s, size_t H_) {
  dim3 blockDim(256);
  dim3 gridDim((s * H_ + blockDim.x - 1) / blockDim.x);
  merge_head_kernel<<<gridDim, blockDim>>>(d_in, head_idx, s, H_, d_out);
}

/* Concatenate each heads
 * @param [in1]     in: [n_head, s, H_]
 * @param [out]    out: [s, H_*n_head]
 * 'n_head' is the number of heads.
 * 's' is the number of tokens in the prompt.
 * 'H_' is the hidden dimension/n_head.
 */
// void concat_head(Tensor *in, Tensor *out) {
//   size_t n_head = in->shape[0];
//   size_t s = in->shape[1];
//   size_t H_ = in->shape[2];  // = HIDDEN_DIM/NUM_HEAD

//   for (size_t i = 0; i < s; i++) {
//     for (size_t j = 0; j < n_head; j++) {
//       for (size_t k = 0; k < H_; k++) {
//         out->buf[i * n_head * H_ + j * H_ + k] =
//             in->buf[j * s * H_ + i * H_ + k];
//       }
//     }
//   }
// }

__global__ void concat_head_kernel(float *in, float *out, size_t n_head, size_t s, size_t H_) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 

  if (idx < s * H_ * n_head) {
    size_t i = (idx % (s * H_)) / H_; 
    size_t j = idx / (s * H_); 
    size_t k = idx % H_; 
    out[i * n_head * H_ + j * H_ + k] = in[j * s * H_ + i * H_ + k];
  }
}

void concat_head(float *d_in, float *d_out, size_t n_head, size_t s, size_t H_) {
  dim3 blockDim(256);
  dim3 gridDim((n_head * s * H_ + blockDim.x - 1) / blockDim.x);
  concat_head_kernel<<<gridDim, blockDim>>>(d_in, d_out, n_head, s, H_);
}

/* Greedy Max Sampling
 * @param  [in1]  in: [s, V]
 * @return [ret] out: [1]
 * 's' is the number of tokens in the prompt.
 * 'V' is the number of vocabulary.
 */
// int top1_sampling(Tensor *in) {
//   size_t s = in->shape[0];
//   size_t V = in->shape[1];

//   int out = 0;
//   float max = -INFINITY;
//   for (size_t i = 0; i < V; i++) {
//     if (in->buf[(s - 1) * V + i] > max) {
//       max = in->buf[(s - 1) * V + i];
//       out = i;
//     }
//   }

//   return out;
// }

__global__ void top1_sampling_kernel(float *in, int *out, size_t V) {
    size_t idx = threadIdx.x;

    __shared__ float max_val;
    __shared__ int max_idx;

    if (idx == 0) {
        max_val = -INFINITY;
        max_idx = 0;
    }
    __syncthreads();

    float val = in[idx];
    atomicMax((int*)&max_val, __float_as_int(val));
    __syncthreads();

    if (val == max_val) {
        atomicExch(&max_idx, idx);
    }
    __syncthreads();

    if (idx == 0) {
        *out = max_idx;
    }
}

int top1_sampling(float *d_in, size_t V) {
  int h_out;
  int *d_out;
  cudaMalloc(&d_out, sizeof(int));

  dim3 blockDim(V);
  top1_sampling_kernel<<<1, blockDim>>>(d_in, d_out, V);

  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_out);
  return h_out;
}