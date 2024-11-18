#include <cblas.h>

#include <chrono>
#include <string>

#include "include/config/common_definitions.h"
#include "include/solver/analysis.h"
#include "include/solver/mesh.h"
#include "include/solver/physics.h"
#include "include/solver/tetrahedral.h"
#include "include/solver/wall.h"
#include "include/utils/cppimpact_defs.h"

#ifdef CPPIMPACT_CUDA_BACKEND
#include "include/solver/dynamics.cuh"
#else
#include "include/solver/dynamics.h"
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
  // std::string filename("../input/0.25 cube calculix linear 5758 elem.inp");
  std::string filename("../input/fuselage 5086 elements.inp");

  Mesh<T, Basis::nodes_per_element> tensile;

  // Material Properties
  T E = 68.9E9;  // Pa
  T rho = 2700;  // kg/m3
  T nu = 0.33;
  T beta = 0.0;
  T H = 10;
  T Y0 = 1.9 * std::sqrt(3.0);
  std::string name = "AL6061";

  Material material(E, rho, nu, beta, H, Y0, name.c_str());
  printf("Material: %s\n", name.c_str());
  int load_success = tensile.load_mesh(filename);
  if (load_success != 0) {
    std::cerr << "Error loading mesh" << std::endl;
    return 1;
  }

  // Set the number of degrees of freedom

  // Position and velocity in x, y, z
  T init_position[] = {0, 0, 0};
  T init_velocity[] = {0, 0.0, -10};

  const int normal = 1;
  std::string wall_name = "Wall";
  T location = 0.15 - 0.00005;
  double dt = 1e-7;
  double time_end = smoke_test ? dt * 10 : 0.5;

  int export_interval = INT_MAX;
#ifdef CPPIMPACT_DEBUG_MODE
  export_interval = 100;
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