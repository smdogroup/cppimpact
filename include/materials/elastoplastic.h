#pragma once
#include <stdexcept>

template <typename T, int dof_per_node>
class Elastoplastic : public BaseMaterial<T, dof_per_node> {
 public:
  Elastoplastic(T E, T rho, T nu, T beta, T H, T Y0,
                std::string name = "Material")
      : BaseMaterial<T, dof_per_node>("InfinitesimalElastoplastic", E, rho, nu,
                                      name),
        beta(beta),
        H(H),
        Y0_initial_yield_stress(Y0) {}

  virtual ~Elastoplastic() {}

 protected:
  T beta;
  T H;
  T Y0_initial_yield_stress;
  // Additional member variables for elastoplastic-specific properties
};