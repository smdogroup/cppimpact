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

/**
 * @brief Matrix-Matrix Multiplication, matrix in row-major order.
 *
 * Performs the operation:
 *   C = alpha * op(A) * B + beta
 *
 * Where:
 * - op(A) is either A or A<sup>T</sup> depending on the MatOp parameter.
 * - alpha and beta are scalar coefficients.
 * - A, B, and C are matrices stored in row-major order.
 *
 * @tparam T    The data type (e.g., float, double).
 * @tparam opA  Operation on matrix A: MatOp::NoTrans for A, MatOp::Trans for
 * A<sup>T</sup>.
 *
 * @param m      Number of rows in op(A).
 * @param n      Number of columns in B.
 * @param k      Number of columns in op(A) / rows in B.
 * @param alpha  Scalar multiplier for op(A) * B.
 * @param A      Pointer to the first element of matrix A.
 * @param B      Pointer to the first element of matrix B.
 * @param beta   Scalar multiplier for matrix C.
 * @param C      Pointer to the first element of matrix C.
 */

template <typename T, MatOp opA>
CPPIMPACT_FUNCTION void cppimpact_gemm(const int m, const int n, const int k,
                                       const T alpha, const T* A, const T* B,
                                       const T beta, T* C) {
  if constexpr (opA == MatOp::NoTrans) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        T sum = 0;
        for (int p = 0; p < k; ++p) {
          sum += A[i * k + p] * B[p * n + j];
        }
        C[i * n + j] = alpha * sum + beta;
      }
    }
  } else {  // MatOp::Trans
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        T sum = 0;
        for (int p = 0; p < k; ++p) {
          sum += A[p * m + i] * B[p * n + j];
        }
        C[i * n + j] = alpha * sum + beta;
      }
    }
  }
}
