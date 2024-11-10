// LinearElastic.h
#pragma once

#include <cstdio>   // For printf (optional)
#include <cstring>  // For memset
#include <string>   // For std::string

#include "../config/simulation_config.h"  // Adjust the path as necessary"

template <typename T>
class LinearElastic {
 public:
  // Material properties
  T E;     // Young's modulus
  T rho;   // Density
  T nu;    // Poisson's ratio
  T beta;  // Damping coefficient (if applicable)
  T H;     // Hardening parameter (if applicable)
  T Y0;    // Yield strength (if applicable)

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
  __host__ __device__ void calculate_f_internal(const T* element_xloc,
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

 private:
  // CPU implementation
  __host__ void calculate_f_internal_host(const T* element_xloc,
                                          const T* element_dof,
                                          T* f_internal) const {
    printf("Calculating f_internal on the CPU\n");
    // Implement the CPU-specific logic here
    // Example: Perform matrix operations, compute internal forces, etc.
  }

  // GPU implementation
  __device__ void calculate_f_internal_device(const T* element_xloc,
                                              const T* element_dof,
                                              T* f_internal) const {
    printf("Calculating f_internal on the GPU\n");
    int tid = threadIdx.x;  // Corrected variable name

    // Implement the GPU-specific logic here
    // Example: Parallel computation of internal forces
    // Ensure that you handle synchronization and memory access appropriately
  }
};
