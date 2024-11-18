#pragma once

#include <cblas.h>

#include "../utils/cppimpact_blas.h"
#include "../utils/cppimpact_defs.h"
#include "mesh.h"
#include "physics.h"
#include "tetrahedral.h"

template <typename T, class Basis, class Quadrature, class Physics,
          class Material>
class FEAnalysis {
 public:
  // Static data taken from the element basis
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

  // Static data from the quadrature
  static constexpr int num_quadrature_pts = Quadrature::num_quadrature_pts;

  // Static data taken from the physics
  static constexpr int dof_per_node = Physics::dof_per_node;

  // Derived static data
  static constexpr int dof_per_element = dof_per_node * nodes_per_element;

  template <int ndof>
  static CPPIMPACT_FUNCTION void get_element_dof(const int nodes[],
                                                 const T dof[],
                                                 T element_dof[]) {
    for (int j = 0; j < nodes_per_element; j++) {
      int node = nodes[j];
      for (int k = 0; k < dof_per_node; k++, element_dof++) {
        element_dof[0] = dof[ndof * node + k];
      }
    }
  }

#ifdef CPPIMPACT_CUDA_BACKEND
  template <int ndof, int dof_per_element, int dof_per_node>
  static __device__ void get_element_dof(int tid, const int nodes[],
                                         const T dof[], T element_dof[]) {
    if (tid < dof_per_element) {
      int j = tid / dof_per_node;
      int k = tid % dof_per_node;
      int node = nodes[j];

      element_dof[tid] = dof[ndof * node + k];
    }
  }
#endif

#ifdef CPPIMPACT_CUDA_BACKEND
  static __device__ void element_mass_matrix_gpu(
      int tid, const T element_density, const T *element_xloc,
      const T *element_dof, T *element_mass_matrix_diagonals,
      const int nodes_per_elem_num_quad) {
    int i = tid / num_quadrature_pts;  // node index
    int k = tid % num_quadrature_pts;  // quadrature index

    __shared__ T m_i[nodes_per_element];
    if (tid < nodes_per_element) {
      m_i[tid] = 0.0;
    }

    __shared__ T pts[num_quadrature_pts * spatial_dim];
    __shared__ T coeff[num_quadrature_pts];
    __shared__ T J[num_quadrature_pts * spatial_dim * spatial_dim];

    int pts_offset = k * spatial_dim;
    int J_offset = k * spatial_dim * spatial_dim;

    __syncthreads();

    // Compute density * weight * detJ for each quadrature point
    if (tid < num_quadrature_pts) {
      coeff[k] = Quadrature::get_quadrature_pt(k, pts + pts_offset);
    }
    __syncthreads();

    // Evaluate the derivative of the spatial dof in the computational
    // coordinates
    Basis::template eval_grad<spatial_dim>(pts + pts_offset, element_xloc,
                                           J + J_offset);
    __syncthreads();

    if (tid < num_quadrature_pts) {
      coeff[k] *= det3x3(J + J_offset) * element_density;
    }
    __syncthreads();

    if (tid < nodes_per_elem_num_quad) {
      // Compute the invariants
      T N[nodes_per_element];
      Basis::eval_basis(pts + pts_offset, N);
      atomicAdd(&m_i[i], N[i] * coeff[k]);
    }

    __syncthreads();

    if (i < nodes_per_element && k < 3)
      element_mass_matrix_diagonals[3 * i + k] = m_i[i];
    __syncthreads();
  }
#endif

  static void element_mass_matrix(const T element_density,
                                  const T *element_xloc, const T *element_dof,
                                  T *element_mass_matrix_diagonals) {
    for (int i = 0; i < nodes_per_element; i++) {
      T m_i = 0.0;
      for (int k = 0; k < num_quadrature_pts; k++) {
        T pt[spatial_dim];
        T weight = Quadrature::get_quadrature_pt(k, pt);
        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        T J[spatial_dim * spatial_dim];
        Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);

        // Compute the inverse and determinant of the Jacobian matrix
        T Jinv[spatial_dim * spatial_dim];
        T detJ = inv3x3(J, Jinv);

        // Compute the invariants
        T N[nodes_per_element];
        Basis::eval_basis(pt, N);
        m_i += N[i] * weight * detJ * element_density;
      }
      element_mass_matrix_diagonals[3 * i] = m_i;
      element_mass_matrix_diagonals[3 * i + 1] = m_i;
      element_mass_matrix_diagonals[3 * i + 2] = m_i;
    }
  }

  // TODO: remove this duplicate function
  static CPPIMPACT_FUNCTION void calculate_B_T_D_B(const T *B_matrix,
                                                   const T *D_matrix,
                                                   T *B_T_D_B) {
    // B_matrix: 6 x N matrix
    // D_matrix: 6 x 6 matrix
    // B_T_D_B: N x N matrix (initialized to zero before calling)
    // N: spatial_dim * nodes_per_element

    const int N = nodes_per_element * spatial_dim;

    for (int k = 0; k < 6; ++k) {
      const T *B_row_k = &B_matrix[k * N];
      for (int l = 0; l < 6; ++l) {
        T Dkl = D_matrix[k * 6 + l];
        const T *B_row_l = &B_matrix[l * N];

        for (int i = 0; i < N; ++i) {
          T Bik_Dkl = B_row_k[i] * Dkl;
          T *B_T_D_B_row = &B_T_D_B[i * N];

          for (int j = 0; j < N; ++j) {
            B_T_D_B_row[j] += Bik_Dkl * B_row_l[j];
          }
        }
      }
    }
  }

  static T calculate_strain_energy(const T *element_xloc, const T *element_dof,
                                   Material *material) {
    T pt[spatial_dim];
    T K_e[dof_per_element * dof_per_element];
    memset(K_e, 0, sizeof(T) * dof_per_element * dof_per_element);

    T volume = 0.0;

    for (int i = 0; i < num_quadrature_pts; i++) {
      T weight = Quadrature::template get_quadrature_pt<T>(i, pt);

      // Evaluate the derivative of the spatial dof in the computational
      // coordinates
      T J[spatial_dim * spatial_dim];
      memset(J, 0, spatial_dim * spatial_dim * sizeof(T));
      Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);

      // Compute the inverse and determinant of the Jacobian matrix
      T Jinv[spatial_dim * spatial_dim];
      memset(Jinv, 0, spatial_dim * spatial_dim * sizeof(T));
      T detJ = inv3x3(J, Jinv);

#ifdef CPPIMPACT_DEBUG_MODE
      if (detJ < 0.0) {
        printf("detJ negative\n");
      }
#endif

      T J_PU[spatial_dim * spatial_dim];
      memset(J_PU, 0, spatial_dim * spatial_dim * sizeof(T));
      Basis::template eval_grad_PU<spatial_dim>(pt, element_xloc, J_PU);

      T Jinv_PU[spatial_dim * spatial_dim];
      memset(Jinv_PU, 0, spatial_dim * spatial_dim * sizeof(T));
      T detJ_PU = inv3x3(J_PU, Jinv);

      // Compute the B matrix
      T B_matrix[6 * spatial_dim * nodes_per_element];
      memset(B_matrix, 0, 6 * spatial_dim * nodes_per_element * sizeof(T));
      Basis::calculate_B_matrix(Jinv, pt, B_matrix);

      // Compute the material stiffness matrix D
      T D_matrix[6 * 6];
      memset(D_matrix, 0, 6 * 6 * sizeof(T));
      material->calculate_D_matrix<dof_per_node>(D_matrix);

      // Compute B^T * D * B
      T B_T_D_B[dof_per_element * dof_per_element];
      memset(B_T_D_B, 0, sizeof(T) * dof_per_element * dof_per_element);
      calculate_B_T_D_B(B_matrix, D_matrix, B_T_D_B);

      // Assemble the element stiffness matrix K_e
      for (int j = 0; j < dof_per_element * dof_per_element; j++) {
        K_e[j] += weight * detJ * B_T_D_B[j];
      }
    }

    T Ku[dof_per_element];
    memset(Ku, 0, sizeof(T) * dof_per_element);
    // Multiply K_e * u
    cppimpact_gemv<T, MatOp::NoTrans>(dof_per_element, dof_per_element, 1.0,
                                      K_e, element_dof, 0.0, Ku);

    T W = 0.0;
    for (int j = 0; j < dof_per_element; j++) {
      W += 0.5 * element_dof[j] * Ku[j];
    }

    return W;
  }

  static T calculate_volume(const T *element_xloc, const T *element_dof,
                            Material *material) {
    T pt[spatial_dim];

    T volume = 0.0;

    for (int i = 0; i < num_quadrature_pts; i++) {
      T weight = Quadrature::template get_quadrature_pt<T>(i, pt);

      // Evaluate the derivative of the spatial dof in the computational
      // coordinates
      T J[spatial_dim * spatial_dim];
      Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);

      // Compute the inverse and determinant of the Jacobian matrix
      T Jinv[spatial_dim * spatial_dim];
      T detJ = inv3x3(J, Jinv);
      volume += weight * detJ;
    }
    // printf("volume = %f\n", volume);

    return volume;
  }

  static CPPIMPACT_FUNCTION void calculate_stress_strain(const T *element_xloc,
                                                         const T *element_dof,
                                                         const T *pt, T *strain,
                                                         T *stress,
                                                         Material *material) {
    T J[spatial_dim * spatial_dim];
    Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);

    // Compute the inverse and determinant of the Jacobian matrix
    T Jinv[spatial_dim * spatial_dim];
    T detJ = inv3x3(J, Jinv);

    // Compute the B matrix
    T B_matrix[6 * spatial_dim * nodes_per_element];
    memset(B_matrix, 0, 6 * spatial_dim * nodes_per_element * sizeof(T));
    Basis::calculate_B_matrix(Jinv, pt, B_matrix);

    // multiply B*u
    cppimpact_gemv<T, MatOp::NoTrans>(6, dof_per_element, 1.0, B_matrix,
                                      element_dof, 0.0, strain);

    T D_matrix[6 * 6];
    memset(D_matrix, 0, 6 * 6 * sizeof(T));
    material->calculate_D_matrix(D_matrix);

    cppimpact_gemv<T, MatOp::NoTrans>(6, 6, 1.0, D_matrix, strain, 0.0, stress);
  }
};
