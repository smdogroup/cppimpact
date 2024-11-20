#pragma once

#include <cstdio>   // For printf (optional)
#include <cstring>  // For memset
#include <string>   // For std::string

#include "../solver/physics.h"
#include "../utils/cppimpact_blas.h"

template <typename T, class Basis, class Quadrature>
class JohnsonCook {
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
  static constexpr int dof_per_node = spatial_dim;  // hardcoded for now
  static constexpr int dof_per_element = nodes_per_element * dof_per_node;

  // Fixed-size character array for name to avoid std::string in device code
  char name[32];

  // Constructor
  __host__ __device__ JohnsonCook(T E_, T rho_, T nu_, T beta_ = 0.0,
                                  T H_ = 0.0, T Y0_ = 0.0,
                                  const char* name_input = "JohnsonCook")
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
  __host__ __device__ ~JohnsonCook() {}

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

  CPPIMPACT_FUNCTION void calculate_D_matrix(T* D_matrix) const {}

 private:
  // CPU implementation
  __host__ void calculate_f_internal_host(const T* element_xloc,
                                          const T* element_dof,
                                          T* f_internal) const {}

  // GPU implementation
  __device__ void calculate_f_internal_device(const T* element_xloc,
                                              const T* element_dof,
                                              T* f_internal) const {}
};
