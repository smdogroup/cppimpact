#pragma once
#include <cmath>

#include "../utils/cppimpact_defs.h"

// we need 3 threads
// caution: det assumed be zero-initialized
template <typename T>
__device__ inline void det3x3_gpu(int tid, const T A[], T& det) {
  constexpr int det_indices[3][4] = {{0, 4, 3, 1}, {0, 5, 3, 2}, {1, 5, 2, 4}};
  constexpr int det_coeffs[3] = {8, 7, 6};
  constexpr int det_signs[3] = {1, -1, 1};
  atomicAdd(&det, det_signs[tid] * A[det_coeffs[tid]] *
                      (A[det_indices[tid][0]] * A[det_indices[tid][1]] -
                       A[det_indices[tid][2]] * A[det_indices[tid][3]]));
}

template <typename T>
__device__ inline T det3x3_simple(const T A[]) {
  return A[0] * (A[4] * A[8] - A[5] * A[7]) -
         A[1] * (A[3] * A[8] - A[5] * A[6]) +
         A[2] * (A[3] * A[7] - A[4] * A[6]);
}

// we need 9 threads
template <typename T>
__device__ inline void inv3x3_gpu(int tid, const T A[], T Ainv[], T det) {
  constexpr int indices[9][4] = {{4, 8, 5, 7}, {1, 8, 2, 7}, {1, 5, 2, 4},
                                 {3, 8, 5, 6}, {0, 8, 2, 6}, {0, 5, 2, 3},
                                 {3, 7, 4, 6}, {0, 7, 1, 6}, {0, 4, 1, 3}};

  Ainv[tid] = (tid % 2 ? -1 : 1) *
              (A[indices[tid][0]] * A[indices[tid][1]] -
               A[indices[tid][2]] * A[indices[tid][3]]) /
              det;
}

template <typename T>
CPPIMPACT_FUNCTION inline T inv3x3(const T A[], T Ainv[]) {
  T det =
      (A[8] * (A[0] * A[4] - A[3] * A[1]) - A[7] * (A[0] * A[5] - A[3] * A[2]) +
       A[6] * (A[1] * A[5] - A[2] * A[4]));
  T detinv = 1.0 / det;

  Ainv[0] = (A[4] * A[8] - A[5] * A[7]) * detinv;
  Ainv[1] = -(A[1] * A[8] - A[2] * A[7]) * detinv;
  Ainv[2] = (A[1] * A[5] - A[2] * A[4]) * detinv;

  Ainv[3] = -(A[3] * A[8] - A[5] * A[6]) * detinv;
  Ainv[4] = (A[0] * A[8] - A[2] * A[6]) * detinv;
  Ainv[5] = -(A[0] * A[5] - A[2] * A[3]) * detinv;

  Ainv[6] = (A[3] * A[7] - A[4] * A[6]) * detinv;
  Ainv[7] = -(A[0] * A[7] - A[1] * A[6]) * detinv;
  Ainv[8] = (A[0] * A[4] - A[1] * A[3]) * detinv;

  return det;
}

template <typename T>
CPPIMPACT_FUNCTION inline void transpose3x3(const T A[], T At[]) {
  // The diagonal elements are the same for the matrix and its transpose.
  At[0] = A[0];  // Row 1, Col 1
  At[4] = A[4];  // Row 2, Col 2
  At[8] = A[8];  // Row 3, Col 3

  // Transpose the off-diagonal elements.
  At[1] = A[3];  // Row 1, Col 2 becomes Row 2, Col 1
  At[2] = A[6];  // Row 1, Col 3 becomes Row 3, Col 1
  At[3] = A[1];  // Row 2, Col 1 becomes Row 1, Col 2
  At[5] = A[7];  // Row 2, Col 3 becomes Row 3, Col 2
  At[6] = A[2];  // Row 3, Col 1 becomes Row 1, Col 3
  At[7] = A[5];  // Row 3, Col 2 becomes Row 2, Col 3
}

template <typename T>
CPPIMPACT_FUNCTION inline void mat3x3MatMult(const T A[], const T B[], T C[]) {
  C[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
  C[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
  C[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];

  C[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
  C[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
  C[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];

  C[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
  C[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
  C[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
}

template <typename T>
CPPIMPACT_FUNCTION inline void mat3x3MatTransMult(const T A[], const T B[],
                                                  T C[]) {
  C[0] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
  C[3] = A[3] * B[0] + A[4] * B[1] + A[5] * B[2];
  C[6] = A[6] * B[0] + A[7] * B[1] + A[8] * B[2];

  C[1] = A[0] * B[3] + A[1] * B[4] + A[2] * B[5];
  C[4] = A[3] * B[3] + A[4] * B[4] + A[5] * B[5];
  C[7] = A[6] * B[3] + A[7] * B[4] + A[8] * B[5];

  C[2] = A[0] * B[6] + A[1] * B[7] + A[2] * B[8];
  C[5] = A[3] * B[6] + A[4] * B[7] + A[5] * B[8];
  C[8] = A[6] * B[6] + A[7] * B[7] + A[8] * B[8];
}

template <typename T>
CPPIMPACT_FUNCTION inline T det3x3(const T A[]) {
  return (A[8] * (A[0] * A[4] - A[3] * A[1]) -
          A[7] * (A[0] * A[5] - A[3] * A[2]) +
          A[6] * (A[1] * A[5] - A[2] * A[4]));
}

template <typename T>
CPPIMPACT_FUNCTION inline void det3x3Sens(const T A[], T Ad[]) {
  Ad[0] = A[8] * A[4] - A[7] * A[5];
  Ad[1] = A[6] * A[5] - A[8] * A[3];
  Ad[2] = A[7] * A[3] - A[6] * A[4];

  Ad[3] = A[7] * A[2] - A[8] * A[1];
  Ad[4] = A[8] * A[0] - A[6] * A[2];
  Ad[5] = A[6] * A[1] - A[7] * A[0];

  Ad[6] = A[1] * A[5] - A[2] * A[4];
  Ad[7] = A[3] * A[2] - A[0] * A[5];
  Ad[8] = A[0] * A[4] - A[3] * A[1];
}

template <typename T>
CPPIMPACT_FUNCTION inline void addDet3x3Sens(const T s, const T A[], T Ad[]) {
  Ad[0] += s * (A[8] * A[4] - A[7] * A[5]);
  Ad[1] += s * (A[6] * A[5] - A[8] * A[3]);
  Ad[2] += s * (A[7] * A[3] - A[6] * A[4]);

  Ad[3] += s * (A[7] * A[2] - A[8] * A[1]);
  Ad[4] += s * (A[8] * A[0] - A[6] * A[2]);
  Ad[5] += s * (A[6] * A[1] - A[7] * A[0]);

  Ad[6] += s * (A[1] * A[5] - A[2] * A[4]);
  Ad[7] += s * (A[3] * A[2] - A[0] * A[5]);
  Ad[8] += s * (A[0] * A[4] - A[3] * A[1]);
}

template <typename T>
CPPIMPACT_FUNCTION inline void det3x32ndSens(const T s, const T A[], T Ad[]) {
  // Ad[0] = s*(A[8]*A[4] - A[7]*A[5]);
  Ad[0] = 0.0;
  Ad[1] = 0.0;
  Ad[2] = 0.0;
  Ad[3] = 0.0;
  Ad[4] = s * A[8];
  Ad[5] = -s * A[7];
  Ad[6] = 0.0;
  Ad[7] = -s * A[5];
  Ad[8] = s * A[4];
  Ad += 9;

  // Ad[1] += s*(A[6]*A[5] - A[8]*A[3]);
  Ad[0] = 0.0;
  Ad[1] = 0.0;
  Ad[2] = 0.0;
  Ad[3] = -s * A[8];
  Ad[4] = 0.0;
  Ad[5] = s * A[6];
  Ad[6] = s * A[5];
  Ad[7] = 0.0;
  Ad[8] = -s * A[3];
  Ad += 9;

  // Ad[2] += s*(A[7]*A[3] - A[6]*A[4]);
  Ad[0] = 0.0;
  Ad[1] = 0.0;
  Ad[2] = 0.0;
  Ad[3] = s * A[7];
  Ad[4] = -s * A[6];
  Ad[5] = 0.0;
  Ad[6] = -s * A[4];
  Ad[7] = s * A[3];
  Ad[8] = 0.0;
  Ad += 9;

  // Ad[3] += s*(A[7]*A[2] - A[8]*A[1]);
  Ad[0] = 0.0;
  Ad[1] = -s * A[8];
  Ad[2] = s * A[7];
  Ad[3] = 0.0;
  Ad[4] = 0.0;
  Ad[5] = 0.0;
  Ad[6] = 0.0;
  Ad[7] = s * A[2];
  Ad[8] = -s * A[1];
  Ad += 9;

  // Ad[4] += s*(A[8]*A[0] - A[6]*A[2]);
  Ad[0] = s * A[8];
  Ad[1] = 0.0;
  Ad[2] = -s * A[6];
  Ad[3] = 0.0;
  Ad[4] = 0.0;
  Ad[5] = 0.0;
  Ad[6] = -s * A[2];
  Ad[7] = 0.0;
  Ad[8] = s * A[0];
  Ad += 9;

  // Ad[5] += s*(A[6]*A[1] - A[7]*A[0]);
  Ad[0] = -s * A[7];
  Ad[1] = s * A[6];
  Ad[2] = 0.0;
  Ad[3] = 0.0;
  Ad[4] = 0.0;
  Ad[5] = 0.0;
  Ad[6] = s * A[1];
  Ad[7] = -s * A[0];
  Ad[8] = 0.0;
  Ad += 9;

  // Ad[6] += s*(A[1]*A[5] - A[2]*A[4]);
  Ad[0] = 0.0;
  Ad[1] = s * A[5];
  Ad[2] = -s * A[4];
  Ad[3] = 0.0;
  Ad[4] = -s * A[2];
  Ad[5] = s * A[1];
  Ad[6] = 0.0;
  Ad[7] = 0.0;
  Ad[8] = 0.0;
  Ad += 9;

  // Ad[7] += s*(A[3]*A[2] - A[0]*A[5]);
  Ad[0] = -s * A[5];
  Ad[1] = 0.0;
  Ad[2] = s * A[3];
  Ad[3] = s * A[2];
  Ad[4] = 0.0;
  Ad[5] = -s * A[0];
  Ad[6] = 0.0;
  Ad[7] = 0.0;
  Ad[8] = 0.0;
  Ad += 9;

  // Ad[8] += s*(A[0]*A[4] - A[3]*A[1]);
  Ad[0] = s * A[4];
  Ad[1] = -s * A[3];
  Ad[2] = 0.0;
  Ad[3] = -s * A[1];
  Ad[4] = s * A[0];
  Ad[5] = 0.0;
  Ad[6] = 0.0;
  Ad[7] = 0.0;
  Ad[8] = 0.0;
}

template <typename T>
class NeohookeanPhysics {
 public:
  static const int spatial_dim = 3;
  static const int dof_per_node = 3;

  static T energy(T weight, const T J[], const T grad[], const T C1,
                  const T D1) {
    // Compute the inverse and determinant of the Jacobian matrix
    T Jinv[spatial_dim * spatial_dim];
    T detJ = inv3x3(J, Jinv);

    // Compute the derformation gradient
    T F[spatial_dim * spatial_dim];
    mat3x3MatMult(grad, Jinv, F);
    F[0] += 1.0;
    F[4] += 1.0;
    F[8] += 1.0;

    // Compute the invariants
    T detF = det3x3(F);

    // Compute tr(C) = tr(F^{T}*F) = sum_{ij} F_{ij}^2
    T I1 =
        (F[0] * F[0] + F[1] * F[1] + F[2] * F[2] + F[3] * F[3] + F[4] * F[4] +
         F[5] * F[5] + F[6] * F[6] + F[7] * F[7] + F[8] * F[8]);

    // Compute the energy density for the model
    T energy_density = C1 * (I1 - 3.0 - 2.0 * std::log(detF)) +
                       D1 * (detF - 1.0) * (detF - 1.0);

    return weight * detJ * energy_density;
  }

  static void residual(T weight, const T J[], const T grad[], T coef[],
                       const T C1, const T D1) {
    // Compute the inverse and determinant of the Jacobian matrix
    T Jinv[spatial_dim * spatial_dim];
    T detJ = inv3x3(J, Jinv);

    // Compute the derformation gradient
    T F[spatial_dim * spatial_dim];
    mat3x3MatMult(grad, Jinv, F);
    F[0] += 1.0;
    F[4] += 1.0;
    F[8] += 1.0;

    // Compute the invariants
    T detF = det3x3(F);

    // Compute tr(C) = tr(F^{T}*F) = sum_{ij} F_{ij}^2
    T I1 =
        (F[0] * F[0] + F[1] * F[1] + F[2] * F[2] + F[3] * F[3] + F[4] * F[4] +
         F[5] * F[5] + F[6] * F[6] + F[7] * F[7] + F[8] * F[8]);

    // Compute the derivatives of the energy density wrt I1 and detF
    T bI1 = C1;
    T bdetF = -2.0 * C1 / detF + 2.0 * D1 * (detF - 1.0);

    // Add the contributions from the quadrature
    bI1 *= weight * detJ;
    bdetF *= weight * detJ;

    // Derivative in the physical coordinates
    T cphys[spatial_dim * dof_per_node];

    // Add dU0/dI1*dI1/dUx
    cphys[0] = 2.0 * F[0] * bI1;
    cphys[1] = 2.0 * F[1] * bI1;
    cphys[2] = 2.0 * F[2] * bI1;
    cphys[3] = 2.0 * F[3] * bI1;
    cphys[4] = 2.0 * F[4] * bI1;
    cphys[5] = 2.0 * F[5] * bI1;
    cphys[6] = 2.0 * F[6] * bI1;
    cphys[7] = 2.0 * F[7] * bI1;
    cphys[8] = 2.0 * F[8] * bI1;

    // Add dU0/dJ*dJ/dUx
    addDet3x3Sens(bdetF, F, cphys);

    // Transform back to derivatives in the computational coordinates
    mat3x3MatTransMult(cphys, Jinv, coef);
  }

  static void jacobian(T weight, const T J[], const T grad[], const T direct[],
                       T coef[], const T C1, const T D1) {
    // Compute the inverse and determinant of the Jacobian matrix
    T Jinv[spatial_dim * spatial_dim];
    T detJ = inv3x3(J, Jinv);

    // Compute the derformation gradient
    T F[spatial_dim * spatial_dim];
    mat3x3MatMult(grad, Jinv, F);
    F[0] += 1.0;
    F[4] += 1.0;
    F[8] += 1.0;

    // Compute the invariants
    T detF = det3x3(F);

    // Compute tr(C) = tr(F^{T}*F) = sum_{ij} F_{ij}^2
    T I1 =
        (F[0] * F[0] + F[1] * F[1] + F[2] * F[2] + F[3] * F[3] + F[4] * F[4] +
         F[5] * F[5] + F[6] * F[6] + F[7] * F[7] + F[8] * F[8]);

    // Compute the energy density for the model
    // T energy_density = C1 * (I1 - 3.0 - 2.0 * std::log(detF)) +
    //                    D1 * (detF - 1.0) * (detF - 1.0);

    // Compute the derivatives of the energy density detF
    T bI1 = C1;
    T bdetF = -2.0 * C1 / detF + 2.0 * D1 * (detF - 1.0);
    T b2detF = 2.0 * C1 / (detF * detF) + 2.0 * D1;

    // Add the contributions from the quadrature
    bI1 *= weight * detJ;
    bdetF *= weight * detJ;
    b2detF *= weight * detJ;

    // Transform the direction from computational to physical coordinates
    T dirphys[spatial_dim * dof_per_node];
    mat3x3MatMult(direct, Jinv, dirphys);

    T Jac[9 * 9];
    det3x32ndSens(bdetF, F, Jac);
    for (int i = 0; i < 9; i++) {
      Jac[10 * i] += 2.0 * bI1;
    }

    T t[9];
    det3x3Sens(F, t);

    for (int i = 0; i < 9; i++) {
      addDet3x3Sens(b2detF * t[i], F, &Jac[9 * i]);
    }

    // Derivative in the physical coordinates
    T cphys[spatial_dim * dof_per_node];

    // Compute the Jacobian-vector product
    for (int i = 0; i < 9; i++) {
      cphys[i] = 0.0;
      for (int j = 0; j < 9; j++) {
        cphys[i] += Jac[9 * i + j] * dirphys[j];
      }
    }

    // Transform back to derivatives in the computational coordinates
    mat3x3MatTransMult(cphys, Jinv, coef);
  }

  static void test_func() { printf("Test func \n"); }
};
