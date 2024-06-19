#pragma once

#include <vector>

using std::vector;

/* [Tensor Structure] */
struct Tensor {
  size_t ndim = 0;
  size_t shape[4];
  float *buf = nullptr;

  Tensor(const vector<size_t> &shape_);
  Tensor(const vector<size_t> &shape_, float *buf_);
  ~Tensor();

  size_t num_elem();
};

typedef Tensor Parameter;
typedef Tensor Activation;

struct Cache {
  Activation *kv = nullptr;
  size_t len = 0;

  Cache();
  ~Cache();

  void clear();
  void append(Activation *x, Parameter *attn_b, Parameter *attn_w);
};
