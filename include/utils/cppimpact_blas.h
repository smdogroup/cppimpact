#pragma once

#include "../utils/cppimpact_defs.h"

enum class MatOp { NoTrans, Trans };

// Mat-vec multiplication, matrix in row major
// y += alpha * op(A) * x + beta * y
template <typename T, MatOp op>
CPPIMPACT_FUNCTION void cppimpact_gemv(const int m, const int n, const T alpha,
                                       const T* a, const T* x, const T beta,
                                       T* y) {
  if constexpr (op == MatOp::NoTrans) {
    for (int i = 0; i < m; i++) {
      y[i] += beta;
      for (int j = 0; j < n; j++) {
        y[i] += alpha * a[i * n + j] * x[j];
      }
    }
  } else {
    for (int i = 0; i < n; i++) {
      y[i] += beta;
      for (int j = 0; j < m; j++) {
        y[i] += alpha * a[j * n + i] * x[j];
      }
    }
  }
}

// Mat-mat multiplication, matrix in row major
