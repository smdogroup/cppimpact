// LinearElastic.h
#pragma once

#include <cstdio>   // For printf (optional)
#include <cstring>  // For memset
#include <string>   // For std::string

#include "../solver/physics.h"
#include "../utils/cppimpact_blas.h"

template <typename T, class Basis, class Quadrature>
class LinearElastic {
 public:
  // Material properties
  T E;     // Young's modulus
  T rho;   // Density
  T nu;    // Poisson's ratio
  T beta;  // Damping coefficient (if applicable)
  T H;     // Hardening parameter (if applicable)
  T Y0;    // Yield strength (if applicable)
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;
  static constexpr int num_quadrature_pts = Quadrature::num_quadrature_pts;
  // static constexpr int dof_per_node = spatial_dim;  // hardcoded for now

  // Fixed-size character array for name to avoid std::string in device code
  char name[32];

  // Constructor
  __host__ __device__ LinearElastic(T E_, T rho_, T nu_, T beta_ = 0.0,
                                    T H_ = 0.0, T Y0_ = 0.0,
                                    const char* name_input = "LinearElastic")
      : E(E_), rho(rho_), nu(nu_), beta(beta_), H(H_), Y0(Y0_) {
    // Initialize the 'name' array
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      if (i < 31 && name_input[i] != '\0') {
        name[i] = name_input[i];
      } else {
        name[i] = '\0';
      }
    }
  }

  // Destructor
  __host__ __device__ ~LinearElastic() {}

  // calculate_f_internal method accessible on both host and device
  CPPIMPACT_FUNCTION void calculate_f_internal(const T* element_xloc,
                                               const T* element_dof,
                                               T* f_internal) const {
#ifdef CPPIMPACT_CUDA_BACKEND
    // GPU-specific implementation
    calculate_f_internal_device(element_xloc, element_dof, f_internal);
#else
    // CPU-specific implementation
    calculate_f_internal_host(element_xloc, element_dof, f_internal);
#endif
  }

  CPPIMPACT_FUNCTION void calculate_D_matrix(T* D_matrix) const {
    // Fill the matrix
    D_matrix[0 * 6 + 0] = D_matrix[1 * 6 + 1] = D_matrix[2 * 6 + 2] = 1 - nu;
    D_matrix[3 * 6 + 3] = D_matrix[4 * 6 + 4] = D_matrix[5 * 6 + 5] =
        (1 - 2 * nu) / 2;

    D_matrix[0 * 6 + 1] = D_matrix[1 * 6 + 0] = nu;
    D_matrix[0 * 6 + 2] = D_matrix[2 * 6 + 0] = nu;
    D_matrix[1 * 6 + 2] = D_matrix[2 * 6 + 1] = nu;

    // Apply the scalar multiplication
    for (int i = 0; i < 36; i++) {
      D_matrix[i] *= E / ((1 + nu) * (1 - 2 * nu));
    }
  }

 private:
  //  TODO: Consildate this function to only be here, fix other references to it
  static CPPIMPACT_FUNCTION void calculate_B_T_D_B(const T* B_matrix,
                                                   const T* D_matrix,
                                                   T* B_T_D_B) {
    // B_matrix: 6 x N matrix
    // D_matrix: 6 x 6 matrix
    // B_T_D_B: N x N matrix (initialized to zero before calling)
    // N: spatial_dim * nodes_per_element

    const int N = nodes_per_element * spatial_dim;

    for (int k = 0; k < 6; ++k) {
      const T* B_row_k = &B_matrix[k * N];
      for (int l = 0; l < 6; ++l) {
        T Dkl = D_matrix[k * 6 + l];
        const T* B_row_l = &B_matrix[l * N];

        for (int i = 0; i < N; ++i) {
          T Bik_Dkl = B_row_k[i] * Dkl;
          T* B_T_D_B_row = &B_T_D_B[i * N];

          for (int j = 0; j < N; ++j) {
            B_T_D_B_row[j] += Bik_Dkl * B_row_l[j];
          }
        }
      }
    }
  }

  // CPU implementation
  __host__ void calculate_f_internal_host(const T* element_xloc,
                                          const T* element_dof,
                                          T* f_internal) const {
    // Implement the CPU-specific logic here
    // Example: Perform matrix operations, compute internal forces, etc.
  }

  // GPU implementation
  __device__ void calculate_f_internal_device(const T* element_xloc,
                                              const T* element_dof,
                                              T* f_internal) const {
    int tid = threadIdx.x;  // Corrected variable name
    int constexpr dof_per_element = spatial_dim * nodes_per_element;
    int k = tid % num_quadrature_pts;  // quadrature index

    // contiguous for each quadrature points
    __shared__ T pts[num_quadrature_pts * spatial_dim];
    __shared__ T dets[num_quadrature_pts];
    __shared__ T wts[num_quadrature_pts];
    __shared__ T J[num_quadrature_pts * spatial_dim * spatial_dim];
    __shared__ T Jinv[num_quadrature_pts * spatial_dim * spatial_dim];
    __shared__ T Nxis[num_quadrature_pts][dof_per_element];
    __shared__ T B_matrix[num_quadrature_pts][6 * dof_per_element];
    __shared__ T D_matrix[num_quadrature_pts][6 * 6];
    __shared__ T B_T_D_B[num_quadrature_pts][dof_per_element * dof_per_element];
    __shared__ T K_e[dof_per_element * dof_per_element];

    if (tid == 0) {
      memset(pts, 0, num_quadrature_pts * spatial_dim * sizeof(T));
      memset(dets, 0, num_quadrature_pts * sizeof(T));
      memset(wts, 0, num_quadrature_pts * sizeof(T));
      memset(J, 0, num_quadrature_pts * spatial_dim * spatial_dim * sizeof(T));
      memset(Jinv, 0,
             num_quadrature_pts * spatial_dim * spatial_dim * sizeof(T));
      memset(K_e, 0, sizeof(T) * dof_per_element * dof_per_element);
    }

    if (tid < num_quadrature_pts) {
      memset(Nxis[tid], 0, dof_per_element * sizeof(T));
      memset(B_matrix[tid], 0, 6 * dof_per_element * sizeof(T));
      memset(D_matrix[tid], 0, 6 * 6 * sizeof(T));
      memset(B_T_D_B[tid], 0, sizeof(T) * dof_per_element * dof_per_element);
    }
    __syncthreads();

    int constexpr spatial_dim_2 = spatial_dim * spatial_dim;
    int pts_offset = k * spatial_dim;
    int J_offset = k * spatial_dim_2;

    if (tid < num_quadrature_pts) {  // tid = quad point index
      wts[tid] =
          Quadrature::template get_quadrature_pt<T>(tid, pts + pts_offset);
    }
    __syncthreads();

    if (tid < num_quadrature_pts) {
      Basis::template eval_grad_gpu<num_quadrature_pts, spatial_dim>(
          tid, pts + pts_offset, element_xloc, J + J_offset);
    }

    __syncthreads();

    // Evaluate the derivative of the spatial dof in the computational
    // coordinates

    if (tid < num_quadrature_pts) {
      int J_offset = tid * spatial_dim_2;
      dets[tid] = det3x3_simple(J + J_offset);
    }
    __syncthreads();

    if (tid == 0) {
      // printf("J = [%f, %f, %f, %f, %f, %f, %f, %f, %f], det = %f\n",
      //        J[0 + 1 * spatial_dim_2], J[1 + 1 * spatial_dim_2],
      //        J[2 + 1 * spatial_dim_2], J[3 + 1 * spatial_dim_2],
      //        J[4 + 1 * spatial_dim_2], J[5 + 1 * spatial_dim_2],
      //        J[6 + 1 * spatial_dim_2], J[7 + J_offset], J[8 + J_offset],
      //        dets[1]);

      for (int detnum = 0; detnum < 5; detnum++) {
        if (dets[detnum] <= 0) {
          printf("det[%d] = %f\n", detnum, dets[detnum]);
        }
      }
    }

    __syncthreads();

    if (tid < num_quadrature_pts * spatial_dim_2) {
      // Compute the inverse and determinant of the Jacobian matrix
      int k = tid / spatial_dim_2;  // quad index
      int i = tid % spatial_dim_2;  // 0 ~ 8
      inv3x3_gpu(i, J + J_offset, Jinv + J_offset, dets[k]);
    }
    __syncthreads();

    // TODO: parallelize more
    if (tid < num_quadrature_pts) {
      Basis::calculate_B_matrix(Jinv + J_offset, pts + pts_offset,
                                B_matrix[tid]);
    }
    __syncthreads();

    if (tid < num_quadrature_pts) {
      calculate_D_matrix(D_matrix[tid]);
    }

    __syncthreads();
    if (tid < num_quadrature_pts) {
      calculate_B_T_D_B(B_matrix[tid], D_matrix[tid], B_T_D_B[tid]);
    }

    __syncthreads();

    if (tid < num_quadrature_pts) {
      for (int j = 0; j < dof_per_element * dof_per_element; j++) {
        atomicAdd(&K_e[j], wts[tid] * dets[tid] * B_T_D_B[tid][j]);
      }
    }
    __syncthreads();

    if (tid == 0) {
      for (int mkmk = 0; mkmk < 12; mkmk++) {
        // printf("B_T_D_B = %f\n", element_xloc[mkmk]);
      }
    }

    __syncthreads();

    // TODO: parallelize
    if (tid == 0) {
      cppimpact_gemv<T, MatOp::NoTrans>(dof_per_element, dof_per_element, 1.0,
                                        K_e, element_dof, 0.0, f_internal);
    }
    __syncthreads();
  }
};
