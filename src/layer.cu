#include "layer.h"
#include <cstdio>
#include <cmath>

#define TILE_SIZE 32

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

void linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t M = in->shape[0];
  size_t K = in->shape[1];
  size_t N = w->shape[1];

  float *d_in;
  float *d_w;
  float *d_b;
  float *d_out;

  cudaMalloc(&d_in, M * K * sizeof(float));
  cudaMalloc(&d_w, K * N * sizeof(float));
  cudaMalloc(&d_b, N * sizeof(float));
  cudaMalloc(&d_out, M * N * sizeof(float));

  cudaMemcpy(d_in, in->buf, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, w->buf, K * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b->buf, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

  linear_kernel<<<gridDim, blockDim>>>(d_in, d_w, d_b, d_out, M, K, N);

  cudaMemcpy(out->buf, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_w);
  cudaFree(d_b);
  cudaFree(d_out);
}

void linear(float *d_in, float *d_w, float *d_b, float *d_out, size_t M, size_t K, size_t N) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    linear_kernel<<<gridDim, blockDim>>>(d_in, d_w, d_b, d_out, M, K, N);
}

__global__ void batch_linear_kernel_v1(float *in, float *w, float *b, float *out, size_t batch_size, size_t M, size_t K, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * M * N) {
        size_t batch_id = idx / (M * N);
        size_t local_idx = idx % (M * N);
        size_t row = local_idx / N;
        size_t col = local_idx % N;

        if (row < M && col < N) {
            float sum = 0;
            for (size_t k = 0; k < K; k++) {
                sum += in[batch_id * M * K + row * K + k] * w[k * N + col];
            }
            out[batch_id * M * N + row * N + col] = sum + b[col];
        }
    }
}

__global__ void batch_linear_kernel_v2(float *A, float *B, float *C, float *D, size_t batch_size, size_t M, size_t K, size_t N) {
    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    size_t batch_id = blockIdx.z;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t m = 0; m < (K + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        if (row < M && m * TILE_SIZE + threadIdx.x < K) {
            A_shared[threadIdx.y][threadIdx.x] = A[batch_id * M * K + row * K + m * TILE_SIZE + threadIdx.x];
        } else {
            A_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (m * TILE_SIZE + threadIdx.y < K && col < N) {
            B_shared[threadIdx.y][threadIdx.x] = B[(m * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            B_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (size_t k = 0; k < TILE_SIZE; ++k) {
            sum += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        D[batch_id * M * N + row * N + col] = sum + C[col];
    }
}

void batch_linear(float *d_in, float *d_w, float *d_b, float *d_out, size_t batch_size, size_t M, size_t K, size_t N) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y, batch_size);
    batch_linear_kernel_v2<<<gridDim, blockDim>>>(d_in, d_w, d_b, d_out, batch_size, M, K, N);
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

__global__ void matmul_kernel_v1(float *in1, float *in2, float *out, size_t M, size_t K, size_t N) {
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

__global__ void matmul_kernel_v2(float *A, float *B, float *C, size_t M, size_t K, size_t N) {
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

void matmul(Tensor *in1, Tensor *in2, Tensor *out) {
  size_t M = in1->shape[0]; 
  size_t K = in1->shape[1]; 
  size_t N = in2->shape[1]; 

  // printf("M: %lu, K: %lu, N: %lu\n", (unsigned long)M, (unsigned long)K, (unsigned long)N);

  float *d_in1; 
  float *d_in2; 
  float *d_out; 

  cudaMalloc(&d_in1, M * K * sizeof(float)); 
  cudaMalloc(&d_in2, K * N * sizeof(float)); 
  cudaMalloc(&d_out, M * N * sizeof(float)); 

  cudaMemcpy(d_in1, in1->buf, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_in2, in2->buf, K * N * sizeof(float), cudaMemcpyHostToDevice); 

  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  
  matmul_kernel_v2<<<gridDim, blockDim>>>(d_in1, d_in2, d_out, M, K, N); 

  cudaMemcpy(out->buf, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost); 

  cudaFree(d_in1);
  cudaFree(d_in2);
  cudaFree(d_out);
}

void matmul(float *d_in1, float *d_in2, float *d_out, size_t M, size_t K, size_t N) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_kernel_v2<<<gridDim, blockDim>>>(d_in1, d_in2, d_out, M, K, N);
}

__global__ void batch_matmul_kernel(float *in1, float *in2, float *out, size_t batch_size, size_t M, size_t K, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    size_t batch_id = blockIdx.z;

    if (idx < N && idy < M) {
        float sum = 0;
        for (size_t k = 0; k < K; k++) {
            sum += in1[batch_id * M * K + idy * K + k] * in2[batch_id * K * N + k * N + idx];
        }
        out[batch_id * M * N + idy * N + idx] = sum;
    }
}

void batch_matmul(float *d_in1, float *d_in2, float *d_out, size_t batch_size, size_t M, size_t K, size_t N) {
    // printf("batch_size: %lu, M: %lu, K: %lu, N: %lu\n", (unsigned long)batch_size, (unsigned long)M, (unsigned long)K, (unsigned long)N);
    dim3 blockDim(16, 16); 
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y, batch_size); 
    batch_matmul_kernel<<<gridDim, blockDim>>>(d_in1, d_in2, d_out, batch_size, M, K, N);
}

__global__ void batch_matmul_kernel_v1(float *in1, float *in2, float *out, size_t batch_size, size_t M, size_t K, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    size_t batch_id = blockIdx.z;

    if (idx < N && idy < M) {
        float sum = 0;
        for (size_t k = 0; k < K; k++) {
            sum += in1[batch_id * M * K + idy * K + k] * in2[k * N + idx];
        }
        out[batch_id * M * N + idy * N + idx] = sum;
    }
}

__global__ void batch_matmul_kernel_v2(float *A, float *B, float *C, size_t batch_size, size_t M, size_t K, size_t N) {
    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    size_t batch_id = blockIdx.z;
    size_t global_row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t global_col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t local_row = threadIdx.y;
    size_t local_col = threadIdx.x;

    float sum = 0.0f;

    for (size_t t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int A_local_row = batch_id * M * K + global_row * K + t * TILE_SIZE + local_col;
        int B_local_col = (t * TILE_SIZE + local_row) * N + global_col;
        if (global_row < M && t * TILE_SIZE + local_col < K) {
            A_shared[local_row][local_col] = A[A_local_row];
        } else {
            A_shared[local_row][local_col] = 0.0f; 
        }

        if (global_col < N && t * TILE_SIZE + local_row < K) {
            B_shared[local_row][local_col] = B[B_local_col];
        } else {
            B_shared[local_row][local_col] = 0.0f; 
        }

        __syncthreads();

        for (size_t k = 0; k < TILE_SIZE; ++k) {
            sum += A_shared[local_row][k] * B_shared[k][local_col];
        }

        __syncthreads();
    }

    if (global_row < M && global_col < N) {
        C[batch_id * M * N + global_row * N + global_col] = sum;
    }
}


void batch_matmul_final(float *d_in1, float *d_in2, float *d_out, size_t batch_size, size_t M, size_t K, size_t N) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch_size);

    batch_matmul_kernel_v2<<<gridDim, blockDim>>>(d_in1, d_in2, d_out, batch_size, M, K, N);
}

/* Token + Positional Embedding
 * @param [in1]  in: [s]
 * @param [in2] wte: [NUM_VOCAB, H]
 * @param [in3] wpe: [MAX_SEQ_LEN, H]
 * @param [out] out: [s, H]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
__global__ void token_pos_embedding_kernel(int *in, float *wte, float *wpe, float *out, size_t s, size_t H) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
  if (idx < s * H) {
    size_t i = idx / H; 
    size_t j = idx % H; 
    out[idx] = wte[in[i] * H + j] + wpe[i * H + j]; 
  }
}

void token_pos_embedding(vector<int> in, Tensor *wte, Tensor *wpe, Tensor *out) {
  size_t s = in.size(); 
  size_t H = wte->shape[1]; 

  int *d_in; 
  float *d_wte; 
  float *d_wpe; 
  float *d_out; 

  cudaMalloc(&d_in, s * sizeof(int));
  cudaMalloc(&d_wte, wte->num_elem() * sizeof(float));
  cudaMalloc(&d_wpe, wpe->num_elem() * sizeof(float)); 
  cudaMalloc(&d_out, s * H * sizeof(float));

  cudaMemcpy(d_in, in.data(), s * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wte, wte->buf, wte->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wpe, wpe->buf, wpe->num_elem() * sizeof(float), cudaMemcpyHostToDevice); 

  dim3 blockDim(256); 
  dim3 gridDim((s * H + blockDim.x - 1) / blockDim.x); 
  token_pos_embedding_kernel<<<gridDim, blockDim>>>(d_in, d_wte, d_wpe, d_out, s, H);

  cudaMemcpy(out->buf, d_out, s * H * sizeof(float), cudaMemcpyDeviceToHost); 

  cudaFree(d_in);
  cudaFree(d_wte);
  cudaFree(d_wpe);
  cudaFree(d_out);
}

void token_pos_embedding(int *d_in, float *d_wte, float *d_wpe, float *d_out, size_t s, size_t H) {  
  dim3 blockDim(256);
  dim3 gridDim((s * H + blockDim.x - 1) / blockDim.x); 
  token_pos_embedding_kernel<<<gridDim, blockDim>>>(d_in, d_wte, d_wpe, d_out, s, H);
}

__global__ void batch_token_pos_embedding_kernel(int *in, float *wte, float *wpe, float *out, size_t batch_size, size_t s, size_t H) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < batch_size * s * H) {
        size_t batch_id = idx / (s * H);
        size_t local_idx = idx % (s * H);
        size_t i = local_idx / H; 
        size_t j = local_idx % H; 
        out[idx] = wte[in[batch_id * s + i] * H + j] + wpe[i * H + j]; 
    }
}

void batch_token_pos_embedding(int *d_in, float *d_wte, float *d_wpe, float *d_out, size_t batch_size, size_t s, size_t H) {
    dim3 blockDim(256); 
    dim3 gridDim((batch_size * s * H + blockDim.x - 1) / blockDim.x); 
    batch_token_pos_embedding_kernel<<<gridDim, blockDim>>>(d_in, d_wte, d_wpe, d_out, batch_size, s, H);
}

/* GELU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
__global__ void gelu_kernel(float *inout, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
  
  if (idx < N) {
    float x = inout[idx]; 
    inout[idx] = 
        0.5 * x *
        (1.f + tanh(sqrt(2.f / MATH_PI) * (x + 0.044715f * x * x * x)));
  }
}

void gelu(Tensor *inout) {
  size_t N = inout->num_elem(); 

  float *d_inout;

  cudaMalloc(&d_inout, N * sizeof(float));

  cudaMemcpy(d_inout, inout->buf, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(256); 
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x); 
  gelu_kernel<<<gridDim, blockDim>>>(d_inout, N);

  cudaMemcpy(inout->buf, d_inout, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_inout);
}

void gelu(float *d_inout, size_t N) {
    dim3 blockDim(256); 
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x); 
    gelu_kernel<<<gridDim, blockDim>>>(d_inout, N);
}

__global__ void batch_gelu_kernel(float *inout, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < N) {
        float x = inout[idx]; 
        inout[idx] = 
            0.5 * x *
            (1.f + tanh(sqrt(2.f / MATH_PI) * (x + 0.044715f * x * x * x)));
    }
}

void batch_gelu(float *d_inout, size_t batch_size, size_t N) {
    size_t total_N = batch_size * N;
    dim3 blockDim(256); 
    dim3 gridDim((total_N + blockDim.x - 1) / blockDim.x); 
    batch_gelu_kernel<<<gridDim, blockDim>>>(d_inout, total_N);
}

/* Softmax (w/ Max Trick)
 * @param [in & out] inout: [s, H]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
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

void softmax(Tensor *inout) {
  size_t s = inout->shape[0];
  size_t V = inout->shape[1];

  float *d_inout;

  cudaMalloc(&d_inout, s * V * sizeof(float));

  cudaMemcpy(d_inout, inout->buf, s * V * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(256);
  dim3 gridDim((s + blockDim.x - 1) / blockDim.x);
  softmax_kernel<<<gridDim, blockDim>>>(d_inout, s, V);

  cudaMemcpy(inout->buf, d_inout, s * V * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_inout);
}

void softmax(float *d_inout, size_t s, size_t V) {
    dim3 blockDim(256);
    dim3 gridDim((s + blockDim.x - 1) / blockDim.x);
    softmax_kernel<<<gridDim, blockDim>>>(d_inout, s, V);
}

__global__ void batch_softmax_kernel(float *inout, size_t batch_size, size_t s, size_t V) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * s) {
        size_t batch_id = idx / s;
        size_t token_id = idx % s;
        float max_val = -INFINITY;
        for (size_t j = 0; j < V; j++) {
            max_val = fmaxf(max_val, inout[(batch_id * s + token_id) * V + j]);
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < V; j++) {
            sum_exp += expf(inout[(batch_id * s + token_id) * V + j] - max_val);
        }

        for (size_t j = 0; j < V; j++) {
            inout[(batch_id * s + token_id) * V + j] = expf(inout[(batch_id * s + token_id) * V + j] - max_val) / sum_exp;
        }
    }
}

void batch_softmax(float *d_inout, size_t batch_size, size_t s, size_t V) {
    dim3 blockDim(256);
    dim3 gridDim((batch_size * s + blockDim.x - 1) / blockDim.x);
    batch_softmax_kernel<<<gridDim, blockDim>>>(d_inout, batch_size, s, V);
}

/* Layer Normalization
 * @param [in1 & out] inout: [s, H]
 * @param [in2]       gamma: [H]
 * @param [in3]        beta: [H]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
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

void layer_norm(Tensor *inout, Tensor *gamma, Tensor *beta) {
  size_t s = inout->shape[0];
  size_t H = inout->shape[1];

  float eps = 1e-5;
  float *d_inout;
  float *d_gamma;
  float *d_beta;

  cudaMalloc(&d_inout, s * H * sizeof(float));
  cudaMalloc(&d_gamma, H * sizeof(float));
  cudaMalloc(&d_beta, H * sizeof(float));

  cudaMemcpy(d_inout, inout->buf, s * H * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, gamma->buf, H * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, beta->buf, H * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(256);
  dim3 gridDim((s + blockDim.x - 1) / blockDim.x);
  layer_norm_kernel<<<gridDim, blockDim>>>(d_inout, d_gamma, d_beta, s, H, eps);

  cudaMemcpy(inout->buf, d_inout, s * H * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_inout);
  cudaFree(d_gamma);
  cudaFree(d_beta);
}

void layer_norm(float *d_inout, float *d_gamma, float *d_beta, size_t s, size_t H, float eps) {
    dim3 blockDim(256);
    dim3 gridDim((s + blockDim.x - 1) / blockDim.x);
    layer_norm_kernel<<<gridDim, blockDim>>>(d_inout, d_gamma, d_beta, s, H, eps);
}

__global__ void batch_layer_norm_kernel(float *inout, float *gamma, float *beta, size_t batch_size, size_t s, size_t H, float eps) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * s) {
        size_t batch_id = idx / s;
        size_t token_id = idx % s;
        float mean = 0;
        float var = 0;

        for (size_t j = 0; j < H; j++) {
            mean += inout[(batch_id * s + token_id) * H + j];
            var += inout[(batch_id * s + token_id) * H + j] * inout[(batch_id * s + token_id) * H + j];
        }
        mean /= H;
        var = var / H - mean * mean;

        for (size_t j = 0; j < H; j++) {
            inout[(batch_id * s + token_id) * H + j] = (inout[(batch_id * s + token_id) * H + j] - mean) * (1.0 / sqrtf(var + eps)) * gamma[j] + beta[j];
        }
    }
}

void batch_layer_norm(float *d_inout, float *d_gamma, float *d_beta, size_t batch_size, size_t s, size_t H, float eps) {
    dim3 blockDim(256);
    dim3 gridDim((batch_size * s + blockDim.x - 1) / blockDim.x);
    batch_layer_norm_kernel<<<gridDim, blockDim>>>(d_inout, d_gamma, d_beta, batch_size, s, H, eps);
}

/* Transpose
 * @param [in1]  in: [M, N]
 * @param [out] out: [N, M]
 */
__global__ void transpose_kernel(float *in, float *out, size_t M, size_t N) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y; 
  size_t col = blockIdx.x * blockDim.x + threadIdx.x; 
  
  if (row < M && col < N) {
    out[col * M + row] = in[row * N + col];
  }
}

void transpose(Tensor *in, Tensor *out) {
  size_t M = in->shape[0];
  size_t N = in->shape[1]; 

  float *d_in;
  float *d_out;

  cudaMalloc(&d_in, M * N * sizeof(float));
  cudaMalloc(&d_out, N * M * sizeof(float));

  cudaMemcpy(d_in, in->buf, M * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim. y - 1) / blockDim.y);
  transpose_kernel<<<gridDim, blockDim>>>(d_in, d_out, M, N); 

  cudaMemcpy(out->buf, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

void transpose(float *d_in, float *d_out, size_t M, size_t N) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    transpose_kernel<<<gridDim, blockDim>>>(d_in, d_out, M, N); 
}

__global__ void batch_transpose_kernel(float *in, float *out, size_t batch_size, size_t M, size_t N) {
    size_t batch_id = blockIdx.z;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y; 
    size_t col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < M && col < N) {
        out[batch_id * N * M + col * M + row] = in[batch_id * M * N + row * N + col];
    }
}

void batch_transpose(float *d_in, float *d_out, size_t batch_size, size_t M, size_t N) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y, batch_size);
    batch_transpose_kernel<<<gridDim, blockDim>>>(d_in, d_out, batch_size, M, N);
}

/* Scaling
 * @param [in1 & out] inout: [N]
 * @param [in2]       scale: [1]
 * 'N' is the number of elements in the tensor.
 */
__global__ void scaling_kernel(float *inout, float scale, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 

  if (idx < N) {
    inout[idx] *= scale; 
  }
}

void scaling(Tensor *inout, float scale) {
  size_t N = inout->num_elem();

  float *d_inout; 

  cudaMalloc(&d_inout, N * sizeof(float));

  cudaMemcpy(d_inout, inout->buf, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(256);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
  scaling_kernel<<<gridDim, blockDim>>>(d_inout, scale, N); 

  cudaMemcpy(inout->buf, d_inout, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_inout); 
}

void scaling(float *d_inout, float scale, size_t N) {
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    scaling_kernel<<<gridDim, blockDim>>>(d_inout, scale, N); 
}

__global__ void batch_scaling_kernel(float *inout, float scale, size_t batch_size, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx < batch_size * N) {
        inout[idx] *= scale; 
    }
}

void batch_scaling(float *d_inout, float scale, size_t batch_size, size_t N) {
    dim3 blockDim(256);
    dim3 gridDim((batch_size * N + blockDim.x - 1) / blockDim.x);
    batch_scaling_kernel<<<gridDim, blockDim>>>(d_inout, scale, batch_size, N); 
}

/* Generate mask
 * @param [in & out] inout: [s, s]
 * 's' is the number of tokens in the prompt.
 */
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

void generate_mask(Tensor *inout) {
  size_t s = inout->shape[0];

  float *d_inout;

  cudaMalloc(&d_inout, s * s * sizeof(float));

  cudaMemcpy(d_inout, inout->buf, s * s * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(16, 16);
  dim3 gridDim((s + blockDim.x - 1) / blockDim.x, (s + blockDim.y - 1) / blockDim.y);
  generate_mask_kernel<<<gridDim, blockDim>>>(d_inout, s);

  cudaMemcpy(inout->buf, d_inout, s * s * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_inout);
}

void generate_mask(float *d_inout, size_t s) {
    dim3 blockDim(16, 16);
    dim3 gridDim((s + blockDim.x - 1) / blockDim.x, (s + blockDim.y - 1) / blockDim.y);
    generate_mask_kernel<<<gridDim, blockDim>>>(d_inout, s);
}

__global__ void batch_generate_mask_kernel(float *inout, size_t batch_size, size_t s) {
    size_t batch_id = blockIdx.z;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < s && col < s) {
        if (row >= col) {
            inout[batch_id * s * s + row * s + col] = 0;
        } else {
            inout[batch_id * s * s + row * s + col] = -1e10;
        }
    }
}

void batch_generate_mask(float *d_inout, size_t batch_size, size_t s) {
    dim3 blockDim(16, 16);
    dim3 gridDim((s + blockDim.x - 1) / blockDim.x, (s + blockDim.y - 1) / blockDim.y, batch_size);
    batch_generate_mask_kernel<<<gridDim, blockDim>>>(d_inout, batch_size, s);
}

/* Copy
 * @param [in1]  in: [N]
 * @param [out] out: [N]
 * 'N' is the number of elements in the tensor.
 */
__global__ void copy_kernel(float *in, float *out, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 

  if (idx < N) {
    out[idx] = in[idx];
  }
}

void copy(Tensor *in, Tensor *out) {
  size_t N = in->num_elem(); 

  float *d_in;
  float *d_out; 

  cudaMalloc(&d_in, N * sizeof(float));
  cudaMalloc(&d_out, N * sizeof(float)); 

  cudaMemcpy(d_in, in->buf, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(256);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
  copy_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);

  cudaMemcpy(out->buf, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out); 
}

void copy(float *d_in, float *d_out, size_t N) {
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    copy_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
}

__global__ void batch_copy_kernel(float *in, float *out, size_t batch_size, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx < batch_size * N) {
        out[idx] = in[idx];
    }
}

void batch_copy(float *d_in, float *d_out, size_t batch_size, size_t N) {
    dim3 blockDim(256);
    dim3 gridDim((batch_size * N + blockDim.x - 1) / blockDim.x);
    batch_copy_kernel<<<gridDim, blockDim>>>(d_in, d_out, batch_size, N);
}

/* Add
 * @param [in1 & out] inout: [N]
 * @param [in2]           x: [N]
 * 'N' is the number of elements in the tensor.
 */
__global__ void add_kernel(float *inout, float *x, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) { inout[idx] += x[idx]; }
}

void add(Tensor *inout, Tensor *x) {
  size_t N = inout->num_elem();

  float *d_inout;
  float *d_x;

  cudaMalloc(&d_inout, N * sizeof(float));
  cudaMalloc(&d_x, N * sizeof(float));

  cudaMemcpy(d_inout, inout->buf, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x->buf, N * sizeof(float), cudaMemcpyHostToDevice);

  add_kernel<<<(N + 255) / 256, 256>>>(d_inout, d_x, N);

  cudaMemcpy(inout->buf, d_inout, N * sizeof(float), cudaMemcpyDeviceToHost);
}

void add(float *d_inout, float *d_x, size_t N) {
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    add_kernel<<<gridDim, blockDim>>>(d_inout, d_x, N);
}

__global__ void batch_add_kernel(float *inout, float *x, size_t batch_size, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * N) { 
        inout[idx] += x[idx]; 
    }
}

void batch_add(float *d_inout, float *d_x, size_t batch_size, size_t N) {
    dim3 blockDim(256);
    dim3 gridDim((batch_size * N + blockDim.x - 1) / blockDim.x);
    batch_add_kernel<<<gridDim, blockDim>>>(d_inout, d_x, batch_size, N);
}

/* Split into QKV
 * @param [in1]  in: [s, H]
 * @param [out] out: [3, s, H/3]
 */
__global__ void split_qkv_kernel(float *in, float *out, size_t s, size_t H, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    size_t i = idx / (s * (H / 3));
    size_t j = (idx % (s * (H / 3))) / (H / 3);
    size_t k = idx % (H / 3);

    out[idx] = in[i * (H / 3) + j * H + k];
  }
}

void split_qkv(Tensor *in, Tensor *out) {
  size_t s = in->shape[0];
  size_t H = in->shape[1];
  size_t N = 3 * s * (H / 3);

  float *d_in;
  float *d_out;

  cudaMalloc(&d_in, s * H * sizeof(float));
  cudaMalloc(&d_out, s * H * sizeof(float));

  cudaMemcpy(d_in, in->buf, s * H * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(256);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
  split_qkv_kernel<<<gridDim, blockDim>>>(d_in, d_out, s, H, N);

  cudaMemcpy(out->buf, d_out, s * H * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

void split_qkv(float *d_in, float *d_out, size_t s, size_t H) {
    dim3 blockDim(256);
    dim3 gridDim((s * H + blockDim.x - 1) / blockDim.x);
    split_qkv_kernel<<<gridDim, blockDim>>>(d_in, d_out, s, H, s * H);
}

__global__ void batch_split_qkv_kernel(float *in, float *out, size_t batch_size, size_t s, size_t H, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * N) {
        size_t batch_id = idx / N;
        size_t local_idx = idx % N;
        size_t i = local_idx / (s * (H / 3));
        size_t j = (local_idx % (s * (H / 3))) / (H / 3);
        size_t k = local_idx % (H / 3);

        out[idx] = in[batch_id * s * H + i * (H / 3) + j * H + k];
    }
}

void batch_split_qkv(float *d_in, float *d_out, size_t batch_size, size_t s, size_t H) {
    size_t N = 3 * s * (H / 3);
    dim3 blockDim(256);
    dim3 gridDim((batch_size * N + blockDim.x - 1) / blockDim.x);
    batch_split_qkv_kernel<<<gridDim, blockDim>>>(d_in, d_out, batch_size, s, H, N);
}

/* Split into heads
 * @param [in1]  in: [3, s, H]
 * @param [out] out: [3, n_head, s, H/n_head]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 * 'n_head' is the number of heads.
 */
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

void split_head(Tensor *in, size_t n_head, Tensor *out) {
  size_t s = in->shape[1];
  size_t H = in->shape[2];
  size_t N = 3 * n_head * s * (H / n_head);

  float *d_in;
  float *d_out;

  cudaMalloc(&d_in, 3 * s * H * sizeof(float));
  cudaMalloc(&d_out, 3 * n_head * s * (H / n_head) * sizeof(float));

  cudaMemcpy(d_in, in->buf, 3 * s * H * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(256);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
  split_head_kernel<<<gridDim, blockDim>>>(d_in, d_out, n_head, s, H, N);

  cudaMemcpy(out->buf, d_out, 3 * n_head * s * (H / n_head) * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

void split_head(float *d_in, float *d_out, size_t n_head, size_t s, size_t H) {
    dim3 blockDim(256);
    dim3 gridDim((3 * s * H + blockDim.x - 1) / blockDim.x);
    split_head_kernel<<<gridDim, blockDim>>>(d_in, d_out, n_head, s, H, 3 * s * H);
}

__global__ void batch_split_head_kernel(float *in, float *out, size_t batch_size, size_t n_head, size_t s, size_t H, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * N) {
        size_t batch_id = idx / N;
        size_t local_idx = idx % N;
        size_t i = local_idx / (n_head * s * (H / n_head));
        size_t j = (local_idx % (n_head * s * (H / n_head))) / (s * (H / n_head));
        size_t k = (local_idx % (s * (H / n_head))) / (H / n_head);
        size_t l = local_idx % (H / n_head);

        out[idx] = in[batch_id * 3 * s * H + i * s * H + k * H + j * (H / n_head) + l];
    }
}

void batch_split_head(float *d_in, float *d_out, size_t batch_size, size_t n_head, size_t s, size_t H) {
    size_t N = 3 * n_head * s * (H / n_head);
    dim3 blockDim(256);
    dim3 gridDim((batch_size * N + blockDim.x - 1) / blockDim.x);
    batch_split_head_kernel<<<gridDim, blockDim>>>(d_in, d_out, batch_size, n_head, s, H, N);
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

void extract_qkv(Tensor *in, size_t head_idx, size_t n_head, Tensor *q, Tensor *k, Tensor *v) {
  size_t s = in->shape[2];
  size_t H_ = in->shape[3];
  size_t N = s * H_;

  float *d_in;
  float *d_q;
  float *d_k;
  float *d_v;

  cudaMalloc(&d_in, 3 * n_head * s * H_ * sizeof(float));
  cudaMalloc(&d_q, s * H_ * sizeof(float));
  cudaMalloc(&d_k, s * H_ * sizeof(float));
  cudaMalloc(&d_v, s * H_ * sizeof(float));

  cudaMemcpy(d_in, in->buf, 3 * n_head * s * H_ * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(256);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
  extract_qkv_kernel<<<gridDim, blockDim>>>(d_in, head_idx, n_head, d_q, d_k, d_v, s, H_, N);

  cudaMemcpy(q->buf, d_q, s * H_ * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(k->buf, d_k, s * H_ * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(v->buf, d_v, s * H_ * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
}

void extract_qkv(float *d_in, float *d_q, float *d_k, float *d_v, size_t head_idx, size_t n_head, size_t s, size_t H_) {
    dim3 blockDim(256);
    dim3 gridDim((s * H_ + blockDim.x - 1) / blockDim.x);
    extract_qkv_kernel<<<gridDim, blockDim>>>(d_in, head_idx, n_head, d_q, d_k, d_v, s, H_, s * H_);
}

__global__ void batch_extract_qkv_kernel(float *in, size_t head_idx, size_t n_head, float *q, float *k, float *v, size_t batch_size, size_t s, size_t H_, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * N) {
        size_t batch_id = idx / N;
        size_t local_idx = idx % N;
        size_t i = local_idx / H_;
        size_t j = local_idx % H_;

        q[batch_id * s * H_ + i * H_ + j] = in[batch_id * 3 * n_head * s * H_ + 0 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
        k[batch_id * s * H_ + i * H_ + j] = in[batch_id * 3 * n_head * s * H_ + 1 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
        v[batch_id * s * H_ + i * H_ + j] = in[batch_id * 3 * n_head * s * H_ + 2 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
    }
}

void batch_extract_qkv(float *d_in, float *d_q, float *d_k, float *d_v, size_t batch_size, size_t head_idx, size_t n_head, size_t s, size_t H_) {
    size_t N = s * H_;
    dim3 blockDim(256);
    dim3 gridDim((batch_size * N + blockDim.x - 1) / blockDim.x);
    batch_extract_qkv_kernel<<<gridDim, blockDim>>>(d_in, head_idx, n_head, d_q, d_k, d_v, batch_size, s, H_, N);
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
__global__ void merge_head_kernel(float *in, size_t head_idx, size_t s, size_t H_, float *out) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < s * H_) {
    size_t i = idx / H_;
    size_t j = idx % H_;
    out[head_idx * s * H_ + i * H_ + j] = in[idx];
  }
}

void merge_head(Tensor *in, size_t head_idx, size_t n_head, Tensor *out) {
  size_t s = in->shape[0];
  size_t H_ = in->shape[1];

  float *d_in;
  float *d_out;

  cudaMalloc(&d_in, s * H_ * sizeof(float));
  cudaMalloc(&d_out, n_head * s * H_ * sizeof(float));  

  cudaMemcpy(d_in, in->buf, s * H_ * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out->buf, n_head * s * H_ * sizeof(float), cudaMemcpyHostToDevice);  

  dim3 blockDim(256);
  dim3 gridDim((s * H_ + blockDim.x - 1) / blockDim.x);
  merge_head_kernel<<<gridDim, blockDim>>>(d_in, head_idx, s, H_, d_out);

  cudaMemcpy(out->buf, d_out, n_head * s * H_ * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

void merge_head(float *d_in, float *d_out, size_t head_idx, size_t s, size_t H_) {
    dim3 blockDim(256);
    dim3 gridDim((s * H_ + blockDim.x - 1) / blockDim.x);
    merge_head_kernel<<<gridDim, blockDim>>>(d_in, head_idx, s, H_, d_out);
}

__global__ void batch_merge_head_kernel(float *in, size_t head_idx, size_t n_head, size_t batch_size, size_t s, size_t H_, float *out) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * s * H_) {
        size_t batch_id = idx / (s * H_);
        size_t local_idx = idx % (s * H_);
        size_t i = local_idx / H_;
        size_t j = local_idx % H_;

        out[batch_id * n_head * s * H_ + head_idx * s * H_ + i * H_ + j] = in[idx];
    }
}

void batch_merge_head(float *d_in, float *d_out, size_t batch_size, size_t n_head, size_t head_idx, size_t s, size_t H_) {
    dim3 blockDim(256);
    dim3 gridDim((batch_size * s * H_ + blockDim.x - 1) / blockDim.x);
    batch_merge_head_kernel<<<gridDim, blockDim>>>(d_in, head_idx, n_head, batch_size, s, H_, d_out);
}

/* Concatenate each heads
 * @param [in1]     in: [n_head, s, H_]
 * @param [out]    out: [s, H_*n_head]
 * 'n_head' is the number of heads.
 * 's' is the number of tokens in the prompt.
 * 'H_' is the hidden dimension/n_head.
 */
__global__ void concat_head_kernel(float *in, float *out, size_t n_head, size_t s, size_t H_) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 

  if (idx < s * H_ * n_head) {
    size_t i = (idx % (s * H_)) / H_; 
    size_t j = idx / (s * H_); 
    size_t k = idx % H_; 
    out[i * n_head * H_ + j * H_ + k] = in[j * s * H_ + i * H_ + k];
  }
}

void concat_head(Tensor *in, Tensor *out) {
  size_t n_head = in->shape[0];
  size_t s = in->shape[1]; 
  size_t H_ = in->shape[2]; 

  float *d_in; 
  float *d_out; 

  cudaMalloc(&d_in, n_head * s * H_ * sizeof(float));
  cudaMalloc(&d_out, s * H_ * n_head * sizeof(float));

  cudaMemcpy(d_in, in->buf, n_head * s * H_ * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(256);
  dim3 gridDim((n_head * s * H_ + blockDim.x - 1) / blockDim.x);
  concat_head_kernel<<<gridDim, blockDim>>>(d_in, d_out, n_head, s, H_);

  cudaMemcpy(out->buf, d_out, s * n_head * H_ * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

void concat_head(float *d_in, float *d_out, size_t n_head, size_t s, size_t H_) {
    dim3 blockDim(256);
    dim3 gridDim((n_head * s * H_ + blockDim.x - 1) / blockDim.x);
    concat_head_kernel<<<gridDim, blockDim>>>(d_in, d_out, n_head, s, H_);
}

__global__ void batch_concat_head_kernel(float *in, float *out, size_t batch_size, size_t n_head, size_t s, size_t H_) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx < batch_size * s * H_ * n_head) {
        size_t batch_id = idx / (s * H_ * n_head);
        size_t local_idx = idx % (s * H_ * n_head);
        size_t i = (local_idx % (s * H_)) / H_; 
        size_t j = local_idx / (s * H_); 
        size_t k = local_idx % H_; 

        out[batch_id * s * H_ * n_head + i * n_head * H_ + j * H_ + k] = in[batch_id * n_head * s * H_ + j * s * H_ + i * H_ + k];
    }
}

void batch_concat_head(float *d_in, float *d_out, size_t batch_size, size_t n_head, size_t s, size_t H_) {
    dim3 blockDim(256);
    dim3 gridDim((batch_size * n_head * s * H_ + blockDim.x - 1) / blockDim.x);
    batch_concat_head_kernel<<<gridDim, blockDim>>>(d_in, d_out, batch_size, n_head, s, H_);
}

/* Greedy Max Sampling
 * @param  [in1]  in: [s, V]
 * @return [ret] out: [1]
 * 's' is the number of tokens in the prompt.
 * 'V' is the number of vocabulary.
 */
__global__ void top1_sampling_kernel(float *in, int *out, size_t V) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float max_val = -INFINITY;
    int max_idx = 0;
    
    for (size_t i = 0; i < V; i++) {
      if (in[i] > max_val) {
        max_val = in[i];
        max_idx = i;
      }
    }
    
    *out = max_idx;
  }
}

int top1_sampling(Tensor *in) {
  size_t s = in->shape[0];
  size_t V = in->shape[1];

  float *d_in;
  int *d_out;
  int out;

  cudaMalloc(&d_in, s * V * sizeof(float));
  cudaMalloc(&d_out, sizeof(int));

  cudaMemcpy(d_in, in->buf, s * V * sizeof(float), cudaMemcpyHostToDevice);

  top1_sampling_kernel<<<1, 1>>>(d_in + (s - 1) * V, d_out, V);

  cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);

  return out;
}

void top1_sampling(float *d_in, int *d_out, size_t s, size_t V) {
  top1_sampling_kernel<<<1, 1>>>(d_in + (s - 1) * V, d_out, V);
}

__global__ void batch_top1_sampling_kernel(float *in, int *out, size_t batch_size, size_t n_token, size_t position, size_t s, size_t V) {
    size_t batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_id < batch_size) {
        float max_val = -INFINITY;
        int max_idx = 0;

        for (size_t i = 0; i < V; i++) {
            if (in[batch_id * s * V + (s - 1) * V + i] > max_val) {
                max_val = in[batch_id * s * V + (s - 1) * V + i];
                max_idx = i;
            }
        }
        
        // out[batch_id + position] = max_idx;
        out[batch_id * n_token + position] = max_idx;
        // where position is nth token 
    }
}

void batch_top1_sampling(float *d_in, int *d_out, size_t batch_size, size_t n_token, size_t position, size_t s, size_t V) {
    dim3 blockDim(256);
    dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x);
    batch_top1_sampling_kernel<<<gridDim, blockDim>>>(d_in, d_out, batch_size, n_token, position, s, V);
}
