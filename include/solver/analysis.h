#pragma once

#include <cblas.h>

#include "../utils/cppimpact_blas.h"
#include "../utils/cppimpact_defs.h"
#include "mesh.h"
#include "physics.h"
#include "tetrahedral.h"

// The FEAnalysis class template provides finite element (FE) analysis
// operations specific to a given element type (defined by Basis and Quadrature)
// and a given Physics and Material model. It includes functions to extract
// element DOFs, compute element mass matrices, compute strain/stress, and
// evaluate strain energy and volume.
//
// Template parameters:
//   T          : Scalar type for computations (e.g., float or double)
//   Basis      : Defines element shape functions, nodes_per_element,
//   spatial_dim Quadrature : Defines integration points (num_quadrature_pts)
//   and weights for the element Physics    : Defines the number of DOFs per
//   node (dof_per_node) Material   : Provides material properties and methods
//   to compute D-matrix, internal forces, etc.

template <typename T, class Basis, class Quadrature, class Physics,
          class Material>
class FEAnalysis {
 public:
  // Extract static constants from the classes
  static constexpr int spatial_dim = Basis::spatial_dim;  // e.g. 3 for 3D
  static constexpr int nodes_per_element =
      Basis::nodes_per_element;  // e.g. 4 for a tetrahedron
  static constexpr int num_quadrature_pts =
      Quadrature::num_quadrature_pts;  // Integration points per element
  static constexpr int dof_per_node =
      Physics::dof_per_node;  // e.g. 3 in a structural 3D problem
  static constexpr int dof_per_element =
      dof_per_node * nodes_per_element;  // total DOFs in the element

  /**
   * @brief Extract element DOFs from the global DOF vector.
   *
   * This function maps the global DOF array to an element-specific DOF array.
   * For each node in the element, we copy its associated DOFs into element_dof.
   *
   * @tparam ndof The stride corresponding to the global DOF array (often ndof =
   * number of DOFs per node).
   * @param nodes       Array of node indices for this element.
   * @param dof         Global DOF array.
   * @param element_dof Local array to store the extracted element DOFs.
   */
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
  /**
   * @brief CUDA version of get_element_dof to extract element DOFs from the
   * global DOF vector.
   *
   * This is similar to the CPU version, but uses thread indexing (tid) and
   * templates for dimension parameters. Intended for use in GPU kernels.
   *
   * @tparam ndof            Number of DOFs per node
   * @tparam dof_per_element Total DOFs in this element
   * @tparam dof_per_node    DOFs per node
   * @param tid             Thread index within a block
   * @param nodes           Element node indices
   * @param dof             Global DOF array on device
   * @param element_dof     Device array to store extracted DOFs for this
   * element
   */
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
  /**
   * @brief GPU kernel function to compute a lumped mass matrix for the element.
   *
   * This function:
   *  - Retrieves quadrature points and weights.
   *  - Computes the Jacobian for the element at each integration point.
   *  - Accumulates mass contributions into lumped mass entries for each node.
   *
   * The result is stored in element_mass_matrix_diagonals (one lumped mass
   * value per DOF).
   *
   * @param tid                    Thread index
   * @param element_density        Material density
   * @param element_xloc           Element nodal coordinates
   * @param element_dof            Element DOFs (not currently used for mass but
   * included for consistency)
   * @param element_mass_matrix_diagonals Output array for lumped mass entries
   * (size = dof_per_element)
   * @param nodes_per_elem_num_quad Product of nodes_per_element and number of
   * quadrature points, used to determine indexing for atomicAdds.
   */
  static __device__ void element_mass_matrix_gpu(
      int tid, const T element_density, const T *element_xloc,
      const T *element_dof, T *element_mass_matrix_diagonals,
      const int nodes_per_elem_num_quad) {
    int i = tid / num_quadrature_pts;  // node index in element
    int k = tid % num_quadrature_pts;  // quadrature index

    __shared__ T m_i[nodes_per_element];
    if (tid < nodes_per_element) {
      m_i[tid] = 0.0;
    }

    __shared__ T pts[num_quadrature_pts * spatial_dim];
    __shared__ T coeff[num_quadrature_pts];
    __shared__ T J[num_quadrature_pts * spatial_dim * spatial_dim];

    // pts_offset: position of this quad point in pts array
    int pts_offset = k * spatial_dim;
    int J_offset = k * spatial_dim * spatial_dim;

    __syncthreads();

    // Load quadrature points and weights
    if (tid < num_quadrature_pts) {
      coeff[k] = Quadrature::get_quadrature_pt(k, pts + pts_offset);
    }
    __syncthreads();

    // Compute Jacobian at each quad point
    Basis::template eval_grad<spatial_dim>(pts + pts_offset, element_xloc,
                                           J + J_offset);
    __syncthreads();

    // Compute coefficient = weight * detJ * density
    if (tid < num_quadrature_pts) {
      coeff[k] *= det3x3(J + J_offset) * element_density;
    }
    __syncthreads();

    // Accumulate mass contributions for each node from each quadrature point
    if (tid < nodes_per_elem_num_quad) {
      T N[nodes_per_element];
      Basis::eval_basis(pts + pts_offset, N);
      atomicAdd(&m_i[i], N[i] * coeff[k]);
    }

    __syncthreads();

    // Assign lumped mass diagonals to each node
    if (i < nodes_per_element && k < 3) {
      element_mass_matrix_diagonals[3 * i + k] = m_i[i];
    }
    __syncthreads();
  }
#endif

  /**
   * @brief CPU version of element mass matrix computation (lumped mass
   * approach).
   *
   * Integrates the element density over the volume using shape functions,
   * generating a lumped mass for each node. This function loops over quadrature
   * points, computes detJ, and accumulates mass.
   *
   * @param element_density              Material density
   * @param element_xloc                 Element nodal coordinates
   * @param element_dof                  Element DOFs (not used for mass
   * computation, but provided for consistency)
   * @param element_mass_matrix_diagonals Output array: lumped mass entries for
   * each DOF in element (size = dof_per_element)
   */
  static void element_mass_matrix(const T element_density,
                                  const T *element_xloc, const T *element_dof,
                                  T *element_mass_matrix_diagonals) {
    for (int i = 0; i < nodes_per_element; i++) {
      T m_i = 0.0;
      for (int k = 0; k < num_quadrature_pts; k++) {
        T pt[spatial_dim];
        T weight = Quadrature::get_quadrature_pt(k, pt);
        T J[spatial_dim * spatial_dim];
        Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);

        T Jinv[spatial_dim * spatial_dim];
        T detJ = inv3x3(J, Jinv);

        T N[nodes_per_element];
        Basis::eval_basis(pt, N);

        // Accumulate mass contribution from this quadrature point
        m_i += N[i] * weight * detJ * element_density;
      }
      // Assign lumped mass equally in all three directions (assuming isotropic
      // mass distribution)
      element_mass_matrix_diagonals[3 * i] = m_i;
      element_mass_matrix_diagonals[3 * i + 1] = m_i;
      element_mass_matrix_diagonals[3 * i + 2] = m_i;
    }
  }

  /**
   * @brief Helper function to compute B^T * D * B matrix multiplication.
   *
   * B_matrix: 6 x N matrix (strain-displacement)
   * D_matrix: 6 x 6 material stiffness matrix
   * B_T_D_B  : N x N result (stiffness contribution)
   *
   * This is a low-level routine used internally when computing element
   * stiffness and strain energy.
   *
   * @param B_matrix Input strain-displacement matrix (6xN)
   * @param D_matrix Input material stiffness (6x6)
   * @param B_T_D_B  Output stiffness contribution (NxN), must be
   * zero-initialized before calling.
   */
  static CPPIMPACT_FUNCTION void calculate_B_T_D_B(const T *B_matrix,
                                                   const T *D_matrix,
                                                   T *B_T_D_B) {
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

  /**
   * @brief Compute the strain energy for the given element configuration.
   *
   * Strain energy W = 0.5 * u^T * K_e * u,
   * where K_e is the element stiffness matrix, and u is the element
   * displacement vector.
   *
   * This involves integrating B^T*D*B over the element domain.
   *
   * @param element_xloc Element nodal coordinates
   * @param element_dof  Element DOFs (displacements)
   * @param material     Pointer to material object
   * @return T           Computed strain energy
   */
  static T calculate_strain_energy(const T *element_xloc, const T *element_dof,
                                   Material *material) {
    T pt[spatial_dim];
    T K_e[dof_per_element * dof_per_element];
    memset(K_e, 0, sizeof(T) * dof_per_element * dof_per_element);

    for (int i = 0; i < num_quadrature_pts; i++) {
      T weight = Quadrature::template get_quadrature_pt<T>(i, pt);

      // Compute Jacobian
      T J[spatial_dim * spatial_dim];
      Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);
      T Jinv[spatial_dim * spatial_dim];
      T detJ = inv3x3(J, Jinv);

#ifdef CPPIMPACT_DEBUG_MODE
      if (detJ < 0.0) {
        printf("detJ negative\n");
      }
#endif

      // Compute B matrix
      T B_matrix[6 * spatial_dim * nodes_per_element];
      memset(B_matrix, 0, sizeof(B_matrix));
      Basis::calculate_B_matrix(Jinv, pt, B_matrix);

      // Compute material stiffness D
      T D_matrix[6 * 6];
      memset(D_matrix, 0, sizeof(D_matrix));
      material->calculate_D_matrix<dof_per_node>(D_matrix);

      // Compute B^T * D * B
      T B_T_D_B[dof_per_element * dof_per_element];
      memset(B_T_D_B, 0, sizeof(B_T_D_B));
      calculate_B_T_D_B(B_matrix, D_matrix, B_T_D_B);

      // Integrate into K_e
      for (int j = 0; j < dof_per_element * dof_per_element; j++) {
        K_e[j] += weight * detJ * B_T_D_B[j];
      }
    }

    // Compute Ku
    T Ku[dof_per_element];
    memset(Ku, 0, sizeof(Ku));
    cppimpact_gemv<T, MatOp::NoTrans>(dof_per_element, dof_per_element, 1.0,
                                      K_e, element_dof, 0.0, Ku);

    // Compute strain energy = 0.5 * u^T * Ku
    T W = 0.0;
    for (int j = 0; j < dof_per_element; j++) {
      W += 0.5 * element_dof[j] * Ku[j];
    }

    return W;
  }

  /**
   * @brief Compute the volume of the element based on its current nodal
   * configuration.
   *
   * This integrates detJ over the element volume at all quadrature points.
   *
   * @param element_xloc Element nodal coordinates
   * @param element_dof  Element DOFs (not necessary for volume calculation, but
   * included for consistency)
   * @param material     Pointer to material (not used directly here)
   * @return T           Computed element volume
   */
  static T calculate_volume(const T *element_xloc, const T *element_dof,
                            Material *material) {
    T pt[spatial_dim];
    T volume = 0.0;

    for (int i = 0; i < num_quadrature_pts; i++) {
      T weight = Quadrature::template get_quadrature_pt<T>(i, pt);

      T J[spatial_dim * spatial_dim];
      Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);
      T Jinv[spatial_dim * spatial_dim];
      T detJ = inv3x3(J, Jinv);

      volume += weight * detJ;
    }

    return volume;
  }

  /**
   * @brief Compute strain and stress at a given point (pt) within the element.
   *
   * Uses the B-matrix, computed from the inverse Jacobian, to map nodal
   * displacements (element_dof) to strains. Then applies the material D-matrix
   * to strains to obtain stresses.
   *
   * @param element_xloc Element nodal coordinates
   * @param element_dof  Element DOFs (displacements)
   * @param pt           Parametric coordinates of the evaluation point
   * @param strain       Output array for strain components (size = 6)
   * @param stress       Output array for stress components (size = 6)
   * @param material     Material object providing D_matrix computation
   */
  static CPPIMPACT_FUNCTION void calculate_stress_strain(const T *element_xloc,
                                                         const T *element_dof,
                                                         const T *pt, T *strain,
                                                         T *stress,
                                                         Material *material) {
    // Compute Jacobian and its inverse
    T J[spatial_dim * spatial_dim];
    Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);

    T Jinv[spatial_dim * spatial_dim];
    T detJ = inv3x3(J, Jinv);

    // Compute B matrix
    T B_matrix[6 * spatial_dim * nodes_per_element];
    memset(B_matrix, 0, sizeof(B_matrix));
    Basis::calculate_B_matrix(Jinv, pt, B_matrix);

    // Strain = B * u
    cppimpact_gemv<T, MatOp::NoTrans>(6, dof_per_element, 1.0, B_matrix,
                                      element_dof, 0.0, strain);

    // Compute material stiffness D
    T D_matrix[6 * 6];
    memset(D_matrix, 0, sizeof(D_matrix));
    material->calculate_D_matrix(D_matrix);

    // Stress = D * Strain
    cppimpact_gemv<T, MatOp::NoTrans>(6, 6, 1.0, D_matrix, strain, 0.0, stress);
  }
};
