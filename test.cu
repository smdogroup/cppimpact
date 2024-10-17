#include <cblas.h>

#include <chrono>
#include <string>

#include "include/analysis.h"
#include "include/cppimpact_defs.h"
#include "include/elastoplastic.h"
#include "include/mesh.h"
#include "include/physics.h"
#include "include/tetrahedral.h"
#include "include/wall.h"

#ifdef CPPIMPACT_CUDA_BACKEND
#include "include/dynamics.cuh"
#else
#include "include/dynamics.h"
#endif

// Function to print matrix for manual verification
void print_matrix(const char *name, const double *matrix, int rows, int cols) {
  std::cout << name << ":\n";
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << matrix[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char *argv[]) {
  using Quadrature = TetrahedralQuadrature5pts;
  using Physics = NeohookeanPhysics<T>;
  using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

  constexpr int dof_per_node = 3;

  bool smoke_test = false;
  if (argc > 1) {
    if ("-h" == std::string(argv[1]) or "--help" == std::string(argv[1])) {
      std::printf("Usage: ./gpu_test.cu [--smoke]\n");
      exit(0);
    }

    if ("--smoke" == std::string(argv[1])) {
      smoke_test = true;
    }
  }

  std::vector<std::string> node_set_names;
  // Load in the mesh
  std::string filename("../input/0.25 cube calculix linear 5758 elem.inp");
  Mesh<T, Basis::nodes_per_element> tensile;

  // Material Properties
  T E = 68.9E9;  // Pa
  T rho = 2700;  // kg/m3
  T nu = 0.33;
  T beta = 0.0;
  T H = 10;
  T Y0 = 1.9 * std::sqrt(3.0);
  std::string name = "AL6061";

  Elastoplastic<T, dof_per_node> material(E, rho, nu, beta, H, Y0, name);
  tensile.load_mesh(filename);

  // Set the number of degrees of freedom

  // Position and velocity in x, y, z
  T init_position[] = {-9.99E-2, -9.99E-2, 1.501E-1};
  T init_velocity[] = {0, 0.0, -1};

  const int normal = 1;
  std::string wall_name = "Wall";
  T location = -0.00005;
  double dt = 1e-6;
  double time_end = smoke_test ? dt * 1 : 0.5;

  int export_interval = INT_MAX;
#ifdef CPPIMPACT_DEBUG_MODE
  export_interval = 5;
#endif

  Wall<T, 2, Basis> w(wall_name, location, E * 10, tensile.slave_nodes,
                      tensile.num_slave_nodes, normal);

  Dynamics<T, Basis, Analysis, Quadrature> dyna(&tensile, &material, &w);
  dyna.initialize(init_position, init_velocity);

  // Solve loop with total timer
  auto start = std::chrono::high_resolution_clock::now();
  dyna.solve(dt, time_end, export_interval);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

  return 0;
}