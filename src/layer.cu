#include <cstdio>

#include "layer.h"

#define warp_size 32

/* All-Reduction Sum
 * @param  [tmpl] size: the number of threads to be reduced
 * @param  [in]    val: a value to be reduced
 * @return [ret]      : the sum of the value across threads
 */
template <int size>
__device__ inline float all_reduce_sum(float val) {
  static_assert(warp_size % size == 0);
  #pragma unroll
  for (int offset = size / 2; offset > 0; offset /= 2)
    val += __shfl_xor_sync(0xffffffff, val, offset);
  return val;
}

/* Layer Normalization
 * @param [in1]    in: [N, H]
 * @param [in2] gamma: [H]
 * @param [in3]  beta: [H]
 * @param [out]   out: [N, H]
 * @param [in4]     N: the number of tokens in the prompt
 * @param [in5]     H: the hidden dimension
 */
__global__ void layer_norm_kernel(float *in, float *gamma, float *beta, float *out, int N, int H) {
  // we could do...
  // 1) float in_reg[H/warp_size]
  // 2) gamma_shared, beta_shared
  // 3) vectorize global memory access
  int i = blockIdx.x * blockDim.y + threadIdx.y;
  
  float mean = 0.0f, var = 0.0f;
  for (int j = threadIdx.x; j < H; j += warp_size) {
    float x = in[i * H + j];
    mean += x;
    var += x * x;
  }

  mean = all_reduce_sum<warp_size>(mean);
  var = all_reduce_sum<warp_size>(var);
  mean /= H;
  var = var / H - mean * mean;

  for (int j = threadIdx.x; j < H; j += warp_size)
    out[i * H + j] = (in[i * H + j] - mean) * rsqrtf(var + 1e-5) * gamma[j] + beta[j];
}
void layer_norm(float *in, float *gamma, float *beta, float *out, int N, int H) {
  constexpr int BN = 8;
  if (H % warp_size) {
    fprintf(stderr, "Error: H must be a multiple of %d\n", warp_size);
    exit(1);
  }
  if (N % BN) {
    fprintf(stderr, "Error: N must be a multiple of %d\n", BN);
    exit(1);
  }
  layer_norm_kernel<<<N / BN, dim3(warp_size, BN)>>>(in, gamma, beta, out, N, H);
}

/* Attention
 * @param [in1]      q: [B, H]
 * @param [in2]     kv: [B, MAX_LEN, 2 * H], but only length kv_len is valid
 * @param [out]      o: [B, H]
 * @param [in3] kv_len: the number of tokens in the context
 */
__global__ void attention_kernel(float *q, float *kv, float *o, int kv_len) {
  constexpr int HEAD_DIM = 64, NUM_HEADS = 12, MAX_LEN = LEN_INPUT + LEN_OUTPUT;
  constexpr int HIDDEN_DIM = HEAD_DIM * NUM_HEADS;
  constexpr int nfloat = sizeof(float4) / sizeof(float);
  constexpr float scale = 1.0f / 8.0f; // 1.0f / sqrtf(HEAD_DIM);
  float s[MAX_LEN];

  const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * warp_size;
  q += blockIdx.x * HIDDEN_DIM + tid * nfloat;
  o += blockIdx.x * HIDDEN_DIM + tid * nfloat;

  float max_val = -INFINITY;
  float * k = kv + 0 * HIDDEN_DIM + blockIdx.x * 2 * HIDDEN_DIM * MAX_LEN + tid * nfloat;
  float * v = kv + 1 * HIDDEN_DIM + blockIdx.x * 2 * HIDDEN_DIM * MAX_LEN + tid * nfloat;

  float4 rq = *(float4 *)q;
  for (int i = 0; i < MAX_LEN; i++, k += 2 * HIDDEN_DIM) {
    if (i == kv_len) break;
    float4 rk = *(float4 *)k;
    s[i] = rq.x * rk.x + rq.y * rk.y + rq.z * rk.z + rq.w * rk.w;
    s[i] = all_reduce_sum<warp_size / 2>(s[i]);
    s[i] *= scale;
    if (s[i] > max_val) max_val = s[i];
  }

  float sum = 0.0f;
  for (int i = 0; i < MAX_LEN; i++) {
    if (i == kv_len) break;
    s[i] = exp(s[i] - max_val);
    sum += s[i];
  }

  float4 ro = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int i = 0; i < MAX_LEN; i++, v += 2 * HIDDEN_DIM) {
    if (i == kv_len) break;
    float4 rv = *(float4 *)v;
    s[i] /= sum;
    ro.x += s[i] * rv.x;
    ro.y += s[i] * rv.y;
    ro.z += s[i] * rv.z;
    ro.w += s[i] * rv.w;
  }
  *(float4 *)o = ro;
}
/* Attention
 * @param [in1]      q: [B, kv_len, H]
 * @param [in2]     kv: [B, MAX_LEN, 2 * H], but only length kv_len is valid
 * @param [out]      o: [B, kv_len, H]
 * @param [in3] kv_len: the number of tokens in the context
 */
__global__ void attention_kernel_multi(float *q, float *kv, float *o, int kv_len) {
  constexpr int HEAD_DIM = 64, NUM_HEADS = 12, MAX_LEN = LEN_INPUT + LEN_OUTPUT;
  constexpr int HIDDEN_DIM = HEAD_DIM * NUM_HEADS;
  constexpr float scale = 1.0f / 8.0f; // 1.0f / sqrtf(HEAD_DIM);
  __shared__ float buf[LEN_INPUT * LEN_INPUT];

  int h = blockIdx.x, b = blockIdx.y;

  q += b * LEN_INPUT * HIDDEN_DIM + h * HEAD_DIM;
  float * k = kv + b * MAX_LEN * 2 * HIDDEN_DIM + 0 * HIDDEN_DIM + h * HEAD_DIM;
  float * v = kv + b * MAX_LEN * 2 * HIDDEN_DIM + 1 * HIDDEN_DIM + h * HEAD_DIM;
  o += b * LEN_INPUT * HIDDEN_DIM + h * HEAD_DIM;

  for (int i = threadIdx.x / LEN_INPUT; i < LEN_INPUT; i += warp_size / LEN_INPUT) {
    float tmp = 0.0f;
    int j = threadIdx.x % LEN_INPUT;
    for (int k_ = 0; k_ < HEAD_DIM; k_++)
      tmp += q[i * HIDDEN_DIM + k_] * k[j * 2 * HIDDEN_DIM + k_];
    tmp *= scale;
    if (i < j) tmp += -1e10;
    buf[i * LEN_INPUT + j] = tmp;
  }
  __syncthreads();

  if (threadIdx.x < LEN_INPUT) {
    float max_val = -INFINITY;
    for (int i = 0; i < kv_len; i++)
      if (buf[threadIdx.x * LEN_INPUT + i] > max_val)
        max_val = buf[threadIdx.x * LEN_INPUT + i];

    float sum = 0.0f;
    for (int i = 0; i < kv_len; i++) {
      buf[threadIdx.x * LEN_INPUT + i] = exp(buf[threadIdx.x * LEN_INPUT + i] - max_val);
      sum += buf[threadIdx.x * LEN_INPUT + i];
    }

    for (int i = 0; i < kv_len; i++)
      buf[threadIdx.x * LEN_INPUT + i] /= sum;
  }
  __syncthreads();

  for (int i = 0; i < LEN_INPUT; i++) {
    for (int j = threadIdx.x; j < HEAD_DIM; j += warp_size) {
      float tmp = 0.0f;
      for (int k_ = 0; k_ < kv_len; k_++)
        tmp += buf[i * LEN_INPUT + k_] * v[k_ * 2 * HIDDEN_DIM + j];
      o[i * HIDDEN_DIM + j] = tmp;
    }
  }
}
void attention(float *q, float *kv, float *o,
               int h_dim, int n_head, int batch_size, int q_len, int kv_len) {
  if (h_dim != 768 || n_head != 12) {
    fprintf(stderr, "Error: h_dim must be 768 and n_head must be 12\n");
    exit(1);
  }

  if (q_len == 1) attention_kernel<<<batch_size, dim3(warp_size / 2, 2, 6)>>>(q, kv, o, kv_len);
  else attention_kernel_multi<<<dim3(n_head, batch_size), warp_size>>>(q, kv, o, kv_len);
}

/* GELU
 * @param  [in] x: input float value
 * @return [ret] : GELU(x)
 */
__device__ inline float gelu_device(float x) {
  constexpr float gelu_scale = 0.79788456080286535588f; // sqrtf(2.0f / MATH_PI)
  float y = x + 0.044715f * x * x * x;
  return 0.5f * x * (1.0f + tanh(gelu_scale * y));
}

/* Linear
 * @param [in1]  in: [M, K]
 * @param [in2]   w: [K, N]
 * @param [in3]   b: [N]
 * @param [out] out: [M, N]
 */
template <int BM, int BN, int BK, int RM, int RN, int splitK>
__global__ void linear_kernel(float *in, float *w, float *b, float *out,
                              int K, int lda, int ldb, int ldc, int ldab, int ldcb, bool gelu) {
  constexpr int cTM = 4, cTN = 8;
  constexpr int cWM = BM / RM / cTM, cWN = BN / RN / cTN;
  constexpr int block_size = cWM * cWN * warp_size;
  constexpr int nfloat = sizeof(float4) / sizeof(float);

  static_assert(cTM * cTN == warp_size);
  static_assert(BM % (RM * cTM) == 0 && BN % (RN * cTN) == 0);
  static_assert(block_size % BM == 0 && block_size % BN == 0 && block_size % BK == 0);
  static_assert(RM % nfloat == 0 && RN % nfloat == 0);

  __shared__ float sA1[BK * BM], sA2[BK * BM], sB1[BK * BN], sB2[BK * BN];
  float rA[RM], rB[RN], rC[RM][RN];

  const int mid = threadIdx.x / cTN + threadIdx.y * cTM;
  const int nid = threadIdx.x % cTN + threadIdx.z * cTN;
  const int tid = threadIdx.x + (threadIdx.y + threadIdx.z * cWM) * warp_size;
  const int klen = K / splitK;

  in += blockIdx.x * BM * lda;
  w += blockIdx.y * BN; b += blockIdx.y * BN;
  out += blockIdx.x * BM * ldc + blockIdx.y * BN;
  if (splitK > 1) {
    in += blockIdx.z * klen;
    w += blockIdx.z * klen * ldb;
    out += blockIdx.z * ldcb;
  } else {
    in += blockIdx.z * ldab;
    out += blockIdx.z * ldcb;
  }

  // load b to rC
  #pragma unroll
  for (int j = 0; j < RN / nfloat; j++) {
    float4 tmp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (splitK == 1 || blockIdx.z == 0)
      tmp = ((float4 *)b)[j * (BN / RN) + nid];
    #pragma unroll
    for (int i = 0; i < RM; i++)
      ((float4 *)(rC[i]))[j] = tmp;
  }

  // BK sized K-tile
  float * __restrict__ sAw = sA1, * __restrict__ sAr = sA2;
  float * __restrict__ sBw = sB1, * __restrict__ sBr = sB2;

  #pragma unroll
  for (int x = 0; x < BM * BK / block_size; x++) {
    int ii = tid / BK + x * (block_size / BK);
    int kk = tid % BK;
    sAw[ii * BK + kk] = in[ii * lda + kk];
  }
  #pragma unroll
  for (int y = 0; y < BN * BK / block_size; y++) {
    int kk = tid / BN + y * (block_size / BN);
    int jj = tid % BN;
    sBw[kk * BN + jj] = w[kk * ldb + jj];
  }
  w += BK * ldb, in += BK;

  for (int k = BK; k < klen; k += BK, w += BK * ldb, in += BK) {
    __syncthreads();
    float * tmp = nullptr;
    tmp = sAw; sAw = sAr; sAr = tmp;
    tmp = sBw; sBw = sBr; sBr = tmp;
    float sAreg[BK * BM / block_size], sBreg[BK * BN / block_size];

    #pragma unroll
    for (int x = 0; x < BM * BK / block_size; x++) {
      int ii = tid / BK + x * (block_size / BK);
      int kk = tid % BK;
      sAreg[x] = in[ii * lda + kk];
    }
    #pragma unroll
    for (int y = 0; y < BN * BK / block_size; y++) {
      int kk = tid / BN + y * (block_size / BN);
      int jj = tid % BN;
      sBreg[y] = w[kk * ldb + jj];
    }

    #pragma unroll
    for (int kk = 0; kk < BK; kk++) {
      #pragma unroll
      for (int i = 0; i < RM; i++)
        rA[i] = sAr[(i * (BM / RM) + mid) * BK + kk];
      #pragma unroll
      for (int j = 0; j < RN / nfloat; j++)
        ((float4 *)rB)[j] = ((float4 *)(sBr + kk * BN))[j * (BN / RN) + nid];
      #pragma unroll
      for (int i = 0; i < RM; i++)
        #pragma unroll
        for (int j = 0; j < RN; j++)
          rC[i][j] += rA[i] * rB[j];
    }

    #pragma unroll
    for (int x = 0; x < BM * BK / block_size; x++) {
      int ii = tid / BK + x * (block_size / BK);
      int kk = tid % BK;
      sAw[ii * BK + kk] = sAreg[x];
    }
    #pragma unroll
    for (int y = 0; y < BN * BK / block_size; y++) {
      int kk = tid / BN + y * (block_size / BN);
      int jj = tid % BN;
      sBw[kk * BN + jj] = sBreg[y];
    }
  }

  __syncthreads();
  #pragma unroll
  for (int kk = 0; kk < BK; kk++) {
    #pragma unroll
    for (int i = 0; i < RM; i++)
      rA[i] = sAw[(i * (BM / RM) + mid) * BK + kk];
    #pragma unroll
    for (int j = 0; j < RN / nfloat; j++)
      ((float4 *)rB)[j] = ((float4 *)(sBw + kk * BN))[j * (BN / RN) + nid];
    #pragma unroll
    for (int i = 0; i < RM; i++)
      #pragma unroll
      for (int j = 0; j < RN; j++)
        rC[i][j] += rA[i] * rB[j];
  }

  if (gelu) {
    #pragma unroll
    for (int i = 0; i < RM; i++)
      #pragma unroll
      for (int j = 0; j < RN; j++)
        rC[i][j] = gelu_device(rC[i][j]);
  }

  #pragma unroll
  for (int i = 0; i < RM; i++)
    #pragma unroll
    for (int j = 0; j < RN / nfloat; j++)
      ((float4 *)(out + (i * (BM / RM) + mid) * ldc))[j * (BN / RN) + nid] = ((float4 *)(rC[i]))[j];
}

/* splitK_reduce
 * @param [tmpl] splitK: the number of splits for K
 * @param [in1] sum_out: [splitK, N]
 * @param [out]     out: [N]
 */
template <int splitK>
__global__ void splitK_reduce(float *sum_out, float *out, int ldin, int ldout) {
  static_assert(splitK > 1);
  const int xid = threadIdx.x + blockIdx.x * blockDim.x;
  const int yid = blockIdx.y;
  const int tid = yid * blockDim.x * gridDim.x + xid;

  float sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < splitK; i++)
    sum += sum_out[i * ldin + tid];
  out[yid * ldout + xid] = sum;
}

void linear(float *in, float *w, float *b, float *out,
            int M, int N, int K, int lda, int ldb, int ldc,
            int B, int ldab, int ldcb, float *sum_out, bool gelu) {
  constexpr int BM = 64, BN = 128, BK = 8, RM = 8, RN = 8;
  if (M % BM || N % BN || K % BK) {
    fprintf(stderr, "Error: M, N, K must be a multiple\n");
    exit(1);
  }
  if (B * M * N > 512 * 768 * 2) {
    linear_kernel<BM, BN, BK, RM, RN, 1>
      <<<dim3(M / BM, N / BN, B), dim3(warp_size, BM / RM / 4, BN / RN / 8)>>>
      (in, w, b, out, K, lda, ldb, ldc, ldab, ldcb, gelu);
  } else {
    constexpr int splitK = 3;
    if (B > 1) {
      fprintf(stderr, "Error: B must be 1 when using splitK\n");
      exit(1);
    }
    if (K % (BK * splitK)) {
      fprintf(stderr, "Error: M, N, K must be a multiple\n");
      exit(1);
    }
    if (sum_out == nullptr) {
      fprintf(stderr, "Error: sum_out must be provided\n");
      exit(1);
    }
    if (gelu) {
      fprintf(stderr, "Error: gelu is not supported with splitK\n");
      exit(1);
    }

    linear_kernel<BM, BN, BK, RM, RN, splitK>
      <<<dim3(M / BM, N / BN, splitK), dim3(warp_size, BM / RM / 4, BN / RN / 8)>>>
      (in, w, b, sum_out, K, lda, ldb, N, 0, M * N, false);
    splitK_reduce<splitK>
      <<<dim3(N / (warp_size * 4), M), warp_size * 4>>>
      (sum_out, out, M * N, ldc);
  }  
}

void linear(float *in, float *w, float *b, float *out,
            int M, int N, int K, int lda, int ldb, int ldc, float *sum_out, bool gelu) {
  linear(in, w, b, out, M, N, K, lda, ldb, ldc, 1, 0, 0, sum_out, gelu);
}

void lm_head(float *in1, float *in2, float *out, int N, int L, int V, int H) {
  linear(in1 + (L - 1) * H, in2, in2 + H * V, out, N, V, H, L * H, V, V, nullptr, false);
}

/* Token + Positional Embedding
 * @param [in1]  in: [B, L]
 * @param [in2] wte: [V, H]
 * @param [in3] wpe: [MAX_SEQ_LEN, H]
 * @param [out] out: [B, L, H]
 * @param [in4]   V: the number of vocabulary.
 * @param [in5]   L: the number of tokens in the prompt.
 * @param [in6]   H: the hidden dimension.
 * @param [in7] pos: the position of the first token in the prompt.
 */
__global__ void token_pos_embedding_kernel(int *in, float *wte, float *wpe, float *out, int V, int L, int H, int pos) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z;

  out[(i * L + k) * H + j] = wte[j * V + in[i * L + k]] + wpe[(pos + k) * H + j];
}
void token_pos_embedding(int *in, float *wte, float *wpe, float *out, int B, int L, int H, int pos, int V) {
  if (B % 32 && H % 32) {
    fprintf(stderr, "Error: N and H must be a multiple of %d\n", 32);
    exit(1);
  }
  token_pos_embedding_kernel<<<dim3(B / 32, H / 32, L), dim3(32, 32)>>>(in, wte, wpe, out, V, L, H, pos);
}

/* Add
 * @param [in1 & out] inout: [N]
 * @param [in2]           x: [N]
 * @param [in3]           N: the number of elements in the tensor.
 */
__global__ void add_kernel(float *inout, float *x, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  inout[i] += x[i];
}
void add(float *inout, float *x, int N) {
  constexpr int BN = 128;
  if (N % BN) {
    fprintf(stderr, "Error: N must be a multiple of %d\n", BN);
    exit(1);
  }
  add_kernel<<<N / BN, BN>>>(inout, x, N);
}

/* Greedy Max Sampling
 * @param [in1]   in: [B, V]
 * @param [out]  out: [B]
 * @param [in2]    V: the number of vocabulary.
 */
__global__ void top1_sampling_kernel(float *in, int *out, int V, int ldin) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y;
  in += i * ldin; out += i;

  float max = -INFINITY;
  int tmp = 0;
  for (int j = threadIdx.x; j < V; j += warp_size) {
    float val = in[j];
    if (val > max) {
      max = val;
      tmp = j;
    }
  }

  #pragma unroll
  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
    float mv_max = __shfl_down_sync(0xffffffff, max, offset);
    int mv_tmp = __shfl_down_sync(0xffffffff, tmp, offset);
    if (mv_max > max) {
      max = mv_max;
      tmp = mv_tmp;
    }
  }

  if (threadIdx.x == 0) *out = tmp;
}
void top1_sampling(float *in, int *out, int N, int V, int ldin) {
  constexpr int BN = 4;
  if (N % 4) {
    fprintf(stderr, "Error: N must be a multiple of %d\n", 4);
    exit(1);
  }
  top1_sampling_kernel<<<N / BN, dim3(warp_size, BN)>>>(in, out, V, ldin);
}
