#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdio>   // For printf (optional)
#include <cstring>  // For memset
#include <string>   // For std::string

#include "../solver/physics.h"
#include "../utils/cppimpact_blas.h"

using Vector = Eigen::VectorXd;

using Vector3D = Eigen::Vector3d;
using Vector2D = Eigen::Vector2d;
using Matrix = Eigen::MatrixXd;

using SymmetricTensor = Eigen::VectorXd;

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
  T K;     // Bulk modulus
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
      : E(E_), rho(rho_), nu(nu_), beta(beta_), H(H_), Y0(Y0_), K(0.0) {
    // Initialize the 'name' array
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      if (i < 31 && name_input[i] != '\0') {
        name[i] = name_input[i];
      } else {
        name[i] = '\0';
      }
    }
    K = E / (3.0 * (1.0 - 2.0 * nu));
  }

  // Destructor
  __host__ __device__ ~JohnsonCook() {}

  // #ODO : support GPU as with f_internal
  CPPIMPACT_FUNCTION void compute_stress() {
    for (int k = 0; k < 5; k++) {
    }
  }

  // calculate_f_internal method accessible on both host and device
  CPPIMPACT_FUNCTION void calculate_f_internal(const T* element_xloc,
                                               const T* element_dof,
                                               T* f_internal) {
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
  void ConvertMatrixToVoigt(Matrix& mat, SymmetricTensor& vec) {
    assert(mat.rows() == mat.cols() && "Matrix must be square.");
    int n = mat.rows();
    vec.resize(6);  // For 3D symmetric tensors

    vec(0) = mat(0, 0);
    vec(1) = mat(1, 1);
    vec(2) = mat(2, 2);

    vec(3) = mat(0, 1);
    vec(4) = mat(0, 2);
    vec(5) = mat(1, 2);
  }
  __host__ void compute_pressure(const T* HenkyStrain, const T* stress) {
    T pressureIncrement = 0.0;
    T pressure = 0.0;
    // Linear element, strain is same at each quadrature point
    pressureIncrement += HenkyStrain[0] + HenkyStrain[1] + HenkyStrain[2];
    // pressure = 1.0 / 3.0 * Trace(gpt->Stress) + K * pressureIncrement;
  }

  // Polar Decomposition
  __host__ void cppimpact_PD_JC(const T* F, T* R, T* HenkyStrain) {
    T FTF[spatial_dim * spatial_dim];
    memset(FTF, 0, sizeof(T) * spatial_dim * spatial_dim);
    cppimpact_gemm<T, MatOp::Trans>(spatial_dim, spatial_dim, spatial_dim, 1.0,
                                    F, F, 0.0, FTF);
    Matrix FTF_mat(3, 3);
    Matrix F_mat(3, 3);
    Matrix R_mat(3, 3);
    Vector U_mat(6);

    // TODO: fix hardcoding or replace with A2D
    FTF_mat << FTF[0], FTF[1], FTF[2], FTF[3], FTF[4], FTF[5], FTF[6], FTF[7],
        FTF[8];
    F_mat << F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8];
    R_mat.setZero();
    U_mat.setZero();

    printf("FTF = %f %f %f %f %f %f %f %f %f\n", FTF[0], FTF[1], FTF[2], FTF[3],
           FTF[4], FTF[5], FTF[6], FTF[7], FTF[8]);

    Eigen::SelfAdjointEigenSolver<Matrix> es(FTF_mat);
    Vector eigenValues = es.eigenvalues();
    Matrix EigenVectors = es.eigenvectors();

    Matrix N1diadN1 = EigenVectors.col(0) * EigenVectors.col(0).transpose();
    Matrix N2diadN2 = EigenVectors.col(1) * EigenVectors.col(1).transpose();
    Matrix N3diadN3;
    Matrix UStretch =
        sqrt(eigenValues(0)) * N1diadN1 + sqrt(eigenValues(1)) * N2diadN2;

    N3diadN3 = EigenVectors.col(2) * EigenVectors.col(2).transpose();
    UStretch += sqrt(eigenValues(2)) * N3diadN3;

    // Rotation matrix R = F.(Uinverse)
    R_mat = F_mat * UStretch.inverse();

    // Compute Henky strain E = ln(U)
    Matrix HenkyStrain_mat = log(eigenValues(0)) * N1diadN1 +
                             log(eigenValues(1)) * N2diadN2 +
                             log(eigenValues(2)) * N3diadN3;

    HenkyStrain_mat *= 0.5;

    ConvertMatrixToVoigt(HenkyStrain_mat, U_mat);

    // Convert back to regular arrays
    for (int i = 0; i < 6; i++) {
      HenkyStrain[i] = U_mat(i);
    }

    for (int i = 0; i < 9; i++) {
      R[i] = R_mat(i);
    }
  }

  // CPU implementation
  __host__ void calculate_f_internal_host(const T* element_xloc,
                                          const T* element_dof, T* f_internal) {
    T pt[spatial_dim];
    T K_e[dof_per_element * dof_per_element];
    memset(K_e, 0, sizeof(T) * dof_per_element * dof_per_element);

    for (int i = 0; i < num_quadrature_pts; i++) {
      T weight = Quadrature::template get_quadrature_pt<T>(i, pt);

      T J[spatial_dim * spatial_dim];
      memset(J, 0, spatial_dim * spatial_dim * sizeof(T));
      Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);

      // standard basis here
      // Compute the inverse and determinant of the Jacobian matrix
      T Jinv[spatial_dim * spatial_dim];
      memset(Jinv, 0, spatial_dim * spatial_dim * sizeof(T));
      T detJ = inv3x3(J, Jinv);

      T F[spatial_dim * spatial_dim];
      memset(F, 0, sizeof(T) * 9);
      Basis::template calculate_def_grad<Quadrature>(pt, Jinv, element_dof, F);

      T R[spatial_dim * spatial_dim];
      memset(R, 0, sizeof(T) * spatial_dim * spatial_dim);
      T HenkyStrain[6];
      memset(HenkyStrain, 0, sizeof(T) * 6);
      cppimpact_PD_JC(F, R, HenkyStrain);

      for (int i = 0; i < 6; i++) {
        printf("%f\n", HenkyStrain[i]);
      }
    }
  }

  // GPU implementation
  __device__ void calculate_f_internal_device(const T* element_xloc,
                                              const T* element_dof,
                                              T* f_internal) const {}
};
