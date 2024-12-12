#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdio>   // For printf (optional)
#include <cstring>  // For memset
#include <iostream>
#include <string>  // For std::string

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
  T E;    // Young's modulus
  T rho;  // Density
  T nu;   // Poisson's ratio
  T Y0;   // Yield strength (if applicable)
  T K;    // Bulk modulus
  T cp;   // Specific heat
  T G;    // Shear modulus

  // Material Constants
  T B;
  T C;
  T M;
  T N;
  T T0;               // Reference temperature
  T TM;               // Reference Melting temperature
  T ref_strain_rate;  // Reference strain rate
  T taylor_quinney;

  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;
  static constexpr int num_quadrature_pts = Quadrature::num_quadrature_pts;
  static constexpr int dof_per_node = spatial_dim;  // hardcoded for now
  static constexpr int dof_per_element = nodes_per_element * dof_per_node;

  // Fixed-size character array for name to avoid std::string in device code
  char name[32];

  // Constructor
  __host__ __device__ JohnsonCook(T E_, T rho_, T nu_, T Y0_, T cp_, T B_, T C_,
                                  T M_, T N_, T T0_, T TM_, T ref_strain_rate_,
                                  T taylor_quinney_,
                                  const char* name_input = "JohnsonCook")
      : E(E_),
        rho(rho_),
        nu(nu_),
        Y0(Y0_),
        cp(cp_),
        B(B_),
        C(C_),
        M(M_),
        N(N_),
        T0(T0_),
        TM(TM_),
        ref_strain_rate(ref_strain_rate_),
        taylor_quinney(taylor_quinney_) {
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
    G = E / (2.0 * (1.0 + nu));
  }

  // Destructor
  __host__ __device__ ~JohnsonCook() {}

  // TODO : support GPU as with f_internal
  CPPIMPACT_FUNCTION void compute_stress(
      const T* element_old_stress, T* element_plastic_strain_eq,
      T* element_plastic_strain_rate, T* element_yield_stress,
      T* element_old_gamma, T* element_strain_increment, const T delta_T,
      T* T_current, T* gamma_cummulate, T* internal_energy, T* inelastic_energy,
      T* rotation_matrix) {
    T heatFrac = taylor_quinney / (rho * cp);

    for (int k = 0; k < num_quadrature_pts; k++) {
      // extract stress for this quad point
      T* quad_stress = element_old_stress[k * 6];
      T* quad_strain_increment = element_strain_increment[k * 6];
      T quad_plastic_strain = element_plastic_strain_eq[k];
      T quad_plastic_strain_rate = element_plastic_strain_rate[k];
      T gamma;

      // Compute pressure
      T pressureIncrement = 0.0;
      T pressure = 0.0;
      // Linear element, strain is same at each quadrature point
      pressureIncrement += quad_strain_increment[0] + quad_strain_increment[1] +
                           quad_strain_increment[2];

      pressure =
          1.0 / 3.0 * (quad_stress[0] + quad_stress[1] + quad_stress[2]) +
          K * pressureIncrement;

      // Calculate deviatoric stress
      T deviatoric_pressure_stress =
          (1.0 / 3.0) * (quad_stress[0] + quad_stress[1] + quad_stress[2]);
      T deviatoric_pressure_strain =
          (1.0 / 3.0) * (quad_strain_increment[0] + quad_strain_increment[1] +
                         quad_strain_increment[2]);

      T* deviatoric_stress[3];
      T* deviatoric_strain_increment[3];
      T norm0 = 0.0;
      T norm = 0.0;

      for (int i = 0; i < 3; i++) {
        deviatoric_stress[i] = quad_stress[i] - deviatoric_pressure_stress;
        norm0 += deviatoric_stress[i] * deviatoric_stress[i];
        norm0 += 2 * deviatoric_stress[i + 3];

        deviatoric_strain_increment[i] =
            quad_strain_increment[i] - deviatoric_pressure_strain;
      }

      // Normalize the deviatoric stress

      norm0 = sqrt(norm0);

      for (int i = 0; i < 6; i++) {
        deviatoric_stress += 2 * G * deviatoric_strain_increment[i];
      }

      for (int i = 0; i < 3; i++) {
        norm += deviatoric_stress[i] * deviatoric_stress[i];
        norm += 2 * deviatoric_stress[i + 3];
      }

      T Strial = SQRT32 * norm;

      // Calculate yield stress
      T quad_yield_stress = element_yield_stress[k];
      T gamma_initial = 1e-8;
      if (quad_yield_stress == 0.0) {
        GetYieldStress(gamma_initial, gamma_initial / delta_T, T0,
                       &quad_yield_stress);
      }

      int iter = 0;
      int bisect_iter = 0;
      T fun;
      T dfun;
      T Dyield_stress_Dconsistency_gamma = 0.0;
      T TOLNR = 1e-8;
      int ITMAX = 250;

      if (Strial > quad_yield_stress) {
        T gamma_min = 0.0;
        T gamma_max = (Strial - quad_yield_stress) / (2 * G * SQRT32);
        gamma = element_old_gamma[k];
        if (quad_plastic_strain == 0) gamma = SQRT32 * gamma_initial;

        // Update plasticStrain, rate and T for next loop
        quad_plastic_strain = quad_plastic_strain + SQRT23 * gamma;
        quad_plastic_strain_rate = SQRT23 * gamma / delta_T;
        T_current =
            T0 + 0.5 * gamma * heatFrac * (SQRT23 * quad_yield_stress + norm0);

        // initialize loop and run Newton rhapson loop
        bool irun = true;
        while (irun) {
          // Compute yield stress and hardening parameter
          GetYieldStress(element_plastic_strain_eq[k],
                         element_plastic_strain_rate[k], T_current,
                         &quad_yield_stress);

          // Compute radial return equation for isotropic case
          fun = Strial - gamma * 2 * G * SQRT32 - quad_yield_stress;

          // Reduce range of solution depending on sign of fun
          if (fun < 0.0)
            gamma_max = gamma;
          else
            gamma_min = gamma;

          // Compute hardening coefficient
          GetDerivativeYieldStress(quad_plastic_strain,
                                   quad_plastic_strain_rate, T_current, delta_T,
                                   Dyield_stress_Dconsistency_gamma);

          // Compute derivative of radial return equation
          dfun = 2 * G * SQRT32 + SQRT23 * Dyield_stress_Dconsistency_gamma;

          // increment of gamma parameter
          T dgamma = fun / dfun;

          // increment of gamma for Newton Rhapson
          gamma += dgamma;

          // if solution is outside brackets, do bisection step
          if ((gamma_max - gamma) * (gamma - gamma_min) < 0.0) {
            dgamma = 0.5 * (gamma_max - gamma_min);
            gamma = gamma_min + gamma;
            bisect_iter += 1;
          }

          // Algorithm converged, end of computation
          if (Abs(dgamma) < TOLNR)
            irun = false;
          else {
            // update values of plasticStrain, Straint rate, and T for next loop
            quad_plastic_strain = quad_plastic_strain + SQRT23 * gamma;
            quad_plastic_strain_rate = SQRT23 * gamma / delta_T;
            T_current = T0 + 0.5 * gamma * heatFrac *
                                 (SQRT23 * quad_yield_stress + norm0);

            // Increase number of iterations
            iter += 1;
            if (iter > ITMAX) {
              std::cout << "NO CONVERGENCE IN NEWTON RHAPSON \n";
              std::cout << "After " << iter << " iterations \n";
              std::cerr
                  << ("No convergence in stress update newton rhapson step \n");
            }
          }
        }

        // Compute plastic strain increment
        quad_plastic_strain += gamma * deviatoric_stress / norm;

        // New stress corrector
        deviatoric_stress *= 1.0 - 2 * G * gamma / norm;

        // Store new values
        quad_plastic_strain += SQRT23 * gamma;
        quad_plastic_strain_rate = SQRT23 * gamma / delta_T;
        element_old_gamma[k] = gamma;
        gamma_cummulate[k] += gamma;
        element_yield_stress[k] = quad_yield_stress;
      }

      T old_quad_stress[6];
      for (int i = 0; i < 6; i++) {
        old_quad_stress[i] = quad_stress[i];
      }

      // Compute final stress of element
      for (int i = 0; i < 3; i++) {
        quad_stress[i] = deviatoric_stress[i] + pressure;
      }

      norm = 0.0;

      for (int i = 0; i < 3; i++) {
        norm += deviatoric_stress[i] * deviatoric_stress[i];
        norm += 2 * deviatoric_stress[i + 3];
      }

      T* temp[6];
      T* stress_power;
      for (int i = 0; i < 6; i++) {
        temp[i] = old_quad_stress[i] + quad_stress[i];
      }

      stress_power = 0.5 * (temp[0] * quad_strain_increment[0] +
                            temp[1] * quad_strain_increment[1] +
                            temp[2] * quad_strain_increment[2] +
                            2 * temp[3] * quad_strain_increment[3] +
                            2 * temp[4] * quad_strain_increment[4] +
                            2 * temp[5] * quad_strain_increment[5]);

      internal_energy += stress_power / rho;

      // Get back gamma value
      if (gamma != 0.0) {
        // compute plastic work increment
        T plastic_work_increment = 0.5 * gamma * (norm + norm0);

        // New dissipated inelastic specific energy
        inelastic_energy += plastic_work_increment / rho;
        T_current += heatFrac * plastic_work_increment;
      }

      // Rotate for objectivity
      RxRT(quad_stress, rotation_matrix);
      RxRT(quad_strain_increment, rotation_matrix);
      for (int i = 0; i < 6; i++) {
        element_old_stress[k * 6 + i] = quad_stress[i];
        element_strain_increment[k * 6 + i] = quad_strain_increment[i];
      }
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
  static constexpr T SQRT32 = sqrt(3.0 / 2.0);
  static constexpr T SQRT23 = 1.0 / SQRT32;

  //  Get yield stress for one quad point
  void GetYieldStress(const T* quad_plastic_strain,
                      const T* quad_plastic_strain_rate, const T T_current,
                      T* yield_stress) {
    T yield_hardening = E + B * pow(quad_plastic_strain, N);

    // The rate dependence activates if rate is more than threshold
    T yield_rate = 1.0;

    if (quad_plastic_strain_rate > ref_strain_rate) {
      yield_rate = 1.0 + C * log(quad_plastic_strain_rate / ref_strain_rate);
    }

    // The thermal softening acrivates if Temp is greater than reference
    T yield_thermal = 1.0;
    if (T_current > T0) {
      if (T_current < TM)
        yield_thermal = 1.0 - pow((T_current - T0) / (TM - T0), M);
      else
        yield_thermal = 0.0;
    }

    yield_stress = yield_hardening * yield_rate * yield_thermal;
  }

  // get yield stress deriv for one quad point
  void GetDerivativeYieldStress(const T* quad_plastic_strain,
                                const T* quad_plastic_strain_rate, T* T_current,
                                const T delta_T,
                                T* Dyield_stress_Dconsistency_gamma) {
    T DSigmaYDConstistencyGamma = 0.0;  // used in radial return
    T yield_hardening = Y0 + B * pow(quad_plastic_strain, N);

    // The rate dependence activates if rate is more than threshold
    T yield_rate = 1.0;
    if (quad_plastic_strain_rate > ref_strain_rate) {
      yield_rate = 1.0 + C * log(quad_plastic_strain_rate / ref_strain_rate);
    }

    // The thermal softening acrivates if Temp is greater than reference
    T yield_thermal = 1.0;
    T yield_temp;
    if (&T_current > T0) {
      if (&T_current < TM) {
        yield_temp = pow((&T_current - T0) / (TM - T0), M);
        yield_thermal = 1.0 - yield_temp;
      } else
        yield_thermal = 0.0;
    }

    // Derivative  of yield stress wrt strain dsigmay/dep
    T dy_dep =
        N * B * pow(quad_plastic_strain, N - 1.0) * yield_rate * yield_thermal;

    Dyield_stress_Dconsistency_gamma = dy_dep;

    // Derivative of yield stress wrt strain raite dsigmay/dDoteps
    T dy_dDotdepss = 0.0;
    if (quad_plastic_strain_rate > ref_strain_rate) {
      dy_dDotdepss = yield_rate * C * yield_thermal / quad_plastic_strain_rate;
      Dyield_stress_Dconsistency_gamma += dy_dDotdepss / delta_T;
    }

    T dy_dT = 0.0;
    // Derivative wrt Temprature T
    if ((&T_current > T0) and (&T_current < TM)) {
      dy_dT =
          -M * yield_hardening * yield_rate * yield_temp / (&T_current - T0);
      Dyield_stress_Dconsistency_gamma += dy_dT * yield_hardening * yield_rate *
                                          yield_thermal * taylor_quinney / (cp);
    }
  }

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

  void ConvertVoigtToMatrix(SymmetricTensor& vec, Matrix& mat) {
    mat.resize(3, 3);

    mat(0, 0) = vec(0);
    mat(1, 1) = vec(1);
    mat(2, 2) = vec(2);

    mat(0, 1) = vec(3);
    mat(1, 0) = vec(3);

    mat(0, 2) = vec(4);
    mat(2, 0) = vec(4);

    mat(1, 2) = vec(5);
    mat(2, 1) = vec(5);
  }

  void RxRT(T* vec, T* R, T* result) {
    Eigen::VectorXd v(6);
    v << vec[0], vec[1], vec[2], vec[3], vec[4], vec[5];

    Eigen::Matrix3d R_mat;
    R_mat << R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8];

    Eigen::Matrix3d v_tensor;
    ConvertVoigtToMatrix(v, v_tensor);

    Eigen::Matrix3d result_mat = R_mat * v_tensor * R_mat.transpose();
    ConvertMatrixToVoigt(result_mat, v);
    for (int i = 0; i < 6; i++) {
      result[i] = v(i);
    }
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
  __host__ void calculate_f_internal_host(
      const T* element_xloc, const T* element_dof, T* element_old_stress,
      T* element_plastic_strain_eq, T* element_plastic_strain_rate,
      T* element_yield_stress, T* element_old_gamma,
      T* element_strain_increment, T delta_T, T* T_current, T* gamma_cummulate,
      T* internal_energy, T* inelastic_energy, T* rotation_matrix,
      T* f_internal) {
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

      compute_stress(element_old_stress, element_plastic_strain_eq,
                     element_plastic_strain_rate, element_yield_stress,
                     element_old_gamma, element_strain_increment, delta_T,
                     T_current, gamma_cummulate, internal_energy,
                     inelastic_energy, rotation_matrix);
    }
  }

  // GPU implementation
  __device__ void calculate_f_internal_device(const T* element_xloc,
                                              const T* element_dof,
                                              T* f_internal) const {}
};
