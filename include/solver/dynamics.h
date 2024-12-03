#pragma once
#include <cfloat>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "../utils/cppimpact_utils.h"
#include "dynamics_kernels.h"
#include "mesh.h"
#include "wall.h"

// TODO: Make wall optional
template <typename T, class Basis, class Analysis, class Quadrature>
class Dynamics {
 private:
  void probe_node_data(int node_id, std::ostringstream &stream) {
    if (node_id < 0 || node_id >= mesh->num_nodes) {
      stream << "Node ID out of range.\n";
      return;
    }
    T x = global_xloc[3 * node_id];
    T y = global_xloc[3 * node_id + 1];
    T z = global_xloc[3 * node_id + 2];
    T vx = vel[3 * node_id];
    T vy = vel[3 * node_id + 1];
    T vz = vel[3 * node_id + 2];
    T ax = global_acc[3 * node_id];
    T ay = global_acc[3 * node_id + 1];
    T az = global_acc[3 * node_id + 2];
    T mx = global_mass[3 * node_id];
    T my = global_mass[3 * node_id + 1];
    T mz = global_mass[3 * node_id + 2];
    stream << "  Position: (" << x << ", " << y << ", " << z << ")\n"
           << "  Velocity: (" << vx << ", " << vy << ", " << vz << ")\n"
           << "  Acceleration: (" << ax << ", " << ay << ", " << az << ")\n"
           << "  Mass: (" << mx << ", " << my << ", " << mz << ")\n";
  }

 public:
  int *reduced_nodes;
  int reduced_dofs_size;
  int ndof;
  static constexpr int nodes_per_element = Basis::nodes_per_element;
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int dof_per_node = spatial_dim;
  static constexpr int num_quadrature_pts = Quadrature::num_quadrature_pts;

  Mesh<T, nodes_per_element> *mesh;
  Material *material;
  Wall<T, 2, Basis> *wall;
  T *global_xloc;
  T *vel;
  T *global_strains;
  T *global_stress;
  T *global_dof;
  T *global_acc;
  T *global_mass;
  T *vel_i;

  // Vector/tensor at each quad point
  T *global_stress_quads;
  T *global_plastic_strain_quads;

  // Scalar at each quad point
  T *eqPlasticStrain;
  T *pressure;
  T *plasticStrainRate;
  T *gamma;
  T *gamma_accumulated;
  T *yieldStress;
  T *plasticWork;
  T *internalEnergy;
  T *temperature;
  T *density;

  int timestep;

  Dynamics(Mesh<T, nodes_per_element> *input_mesh, Material *input_material,
           Wall<T, 2, Basis> *input_wall = nullptr)
      : mesh(input_mesh),
        material(input_material),
        wall(input_wall),
        reduced_nodes(nullptr),
        reduced_dofs_size(0),
        vel(new T[mesh->num_nodes * dof_per_node]),
        global_xloc(
            new T[mesh->num_nodes *
                  dof_per_node]),  // Allocate memory for global_xloc here
        global_strains(new T[mesh->num_nodes * 6]),
        global_stress(new T[mesh->num_nodes * 6]),
        global_dof(new T[mesh->num_nodes * dof_per_node]),
        global_acc(new T[mesh->num_nodes * dof_per_node]),
        global_mass(new T[mesh->num_nodes * dof_per_node]),
        vel_i(new T[mesh->num_nodes * dof_per_node]),
        global_stress_quads(new T[mesh->num_elements * num_quadrature_pts * 6]),
        global_plastic_strain_quads(
            new T[mesh->num_elements * num_quadrature_pts * 6]),
        eqPlasticStrain(new T[mesh->num_elements * num_quadrature_pts]),
        pressure(new T[mesh->num_elements * num_quadrature_pts]),
        plasticStrainRate(new T[mesh->num_elements * num_quadrature_pts]),
        gamma(new T[mesh->num_elements * num_quadrature_pts]),
        gamma_accumulated(new T[mesh->num_elements * num_quadrature_pts]),
        yieldStress(new T[mesh->num_elements * num_quadrature_pts]),
        plasticWork(new T[mesh->num_elements * num_quadrature_pts]),
        internalEnergy(new T[mesh->num_elements * num_quadrature_pts]),
        temperature(new T[mesh->num_elements * num_quadrature_pts]),
        density(new T[mesh->num_elements * num_quadrature_pts])

  {
    ndof = mesh->num_nodes * dof_per_node;
  }

  ~Dynamics() {
    delete[] reduced_nodes;
    delete[] vel;
    delete[] global_xloc;
    delete[] global_strains;
    delete[] global_stress;
    delete[] global_dof;
    delete[] global_acc;
    delete[] global_mass;
    delete[] vel_i;
    delete[] global_stress_quads;
    delete[] global_plastic_strain_quads;
    delete[] eqPlasticStrain;
    delete[] pressure;
    delete[] plasticStrainRate;
    delete[] gamma;
    delete[] gamma_accumulated;
    delete[] yieldStress;
    delete[] plasticWork;
    delete[] internalEnergy;
    delete[] temperature;
    delete[] density;
  }

  // Initialize the body. Move the mesh origin to init_position and give all
  // nodes init_velocity.
  void initialize(T init_position[dof_per_node],
                  T init_velocity[dof_per_node]) {
    std::cout << "ndof: " << ndof << std::endl;
    for (int i = 0; i < mesh->num_nodes; i++) {
      vel[3 * i] = init_velocity[0];
      vel[3 * i + 1] = init_velocity[1];
      vel[3 * i + 2] = init_velocity[2];

      mesh->xloc[3 * i] = mesh->xloc[3 * i] + init_position[0];
      mesh->xloc[3 * i + 1] = mesh->xloc[3 * i + 1] + init_position[1];
      mesh->xloc[3 * i + 2] = mesh->xloc[3 * i + 2] + init_position[2];
    }
  }

  void export_to_vtk(int timestep, T *vel_i, T *acc_i, T *mass_i) {
    const std::string directory = "../cpu_output";
    const std::string filename =
        directory + "/simulation_" + std::to_string(timestep) + ".vtk";
    std::ofstream vtkFile(filename);

    if (!vtkFile.is_open()) {
      std::cerr << "Failed to open " << filename << std::endl;
      return;
    }

    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "FEA simulation data\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET UNSTRUCTURED_GRID\n";
    const double threshold = 1e15;

    vtkFile << "POINTS " << mesh->num_nodes << " float\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      T x = global_xloc[3 * i];
      T y = global_xloc[3 * i + 1];
      T z = global_xloc[3 * i + 2];

      // Check for NaN or extremely large values and set to 0 if found
      if (std::isnan(x) || std::isinf(x) || std::abs(x) > threshold) {
        printf(
            "Invalid value detected in x-coordinate at node %d: %f, setting to "
            "0.\n",
            i, x);
        x = 0.0;
      }
      if (std::isnan(y) || std::isinf(y) || std::abs(y) > threshold) {
        printf(
            "Invalid value detected in y-coordinate at node %d: %f, setting to "
            "0.\n",
            i, y);
        y = 0.0;
      }
      if (std::isnan(z) || std::isinf(z) || std::abs(z) > threshold) {
        printf(
            "Invalid value detected in z-coordinate at node %d: %f, setting to "
            "0.\n",
            i, z);
        z = 0.0;
      }

      vtkFile << std::fixed << std::setprecision(6);
      vtkFile << x << " " << y << " " << z << "\n";
    }

    vtkFile << "CELLS " << mesh->num_elements << " "
            << mesh->num_elements * (nodes_per_element + 1) << "\n";
    for (int i = 0; i < mesh->num_elements; ++i) {
      vtkFile << nodes_per_element;  // Number of points in this cell
      for (int j = 0; j < nodes_per_element; ++j) {
        vtkFile << " " << mesh->element_nodes[nodes_per_element * i + j];
      }
      vtkFile << "\n";  // Ensure newline at the end of each cell's connectivity
                        // list
    }

    vtkFile << "CELL_TYPES " << mesh->num_elements << "\n";
    for (int i = 0; i < mesh->num_elements; ++i) {
      vtkFile << "10\n";  // VTK_TETRA
    }

    vtkFile << "POINT_DATA " << mesh->num_nodes << "\n";
    vtkFile << "VECTORS velocity double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      for (int j = 0; j < 3; ++j) {  // Check each component of the velocity
        T value = vel_i[3 * i + j];
        if (std::isnan(value)) {
          std::cerr << "NaN detected in velocity at node " << i
                    << ", component " << j << std::endl;
          value = 0.0;
        }
        vtkFile << value << (j < 2 ? " " : "\n");
      }
    }

    // First part of the strain
    vtkFile << "VECTORS strain1 double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      vtkFile << global_strains[6 * i + 0] << " "    // First component (e_xx)
              << global_strains[6 * i + 1] << " "    // Second component (e_yy)
              << global_strains[6 * i + 2] << "\n";  // Third component (e_zz)
    }

    // Second part of the strain
    vtkFile << "VECTORS strain2 double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      vtkFile << global_strains[6 * i + 3] << " "    // Fourth component (e_xy)
              << global_strains[6 * i + 4] << " "    // Fifth component (e_xz)
              << global_strains[6 * i + 5] << "\n";  // Sixth component (e_yz)
    }

    // First part of the stress
    vtkFile << "VECTORS stress1 double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      vtkFile << global_stress[6 * i + 0] << " "  // First component (sigma_xx)
              << global_stress[6 * i + 1] << " "  // Second component (sigma_yy)
              << global_stress[6 * i + 2]
              << "\n";  // Third component (sigma_zz)
    }

    // Second part of the stress
    vtkFile << "VECTORS stress2 double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      vtkFile << global_stress[6 * i + 3] << " "  // Fourth component (sigma_xy)
              << global_stress[6 * i + 4] << " "  // Fifth component (sigma_xz)
              << global_stress[6 * i + 5]
              << "\n";  // Sixth component (sigma_yz)
    }

    vtkFile << "VECTORS acceleration double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      for (int j = 0; j < 3; ++j) {
        T value = acc_i[3 * i + j];
        if (std::isnan(value)) {
          std::cerr << "NaN detected in acceleration at node " << i
                    << ", component " << j << std::endl;
          value = 0.0;
        }
        vtkFile << value << (j < 2 ? " " : "\n");
      }
    }

    vtkFile << "VECTORS mass double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      for (int j = 0; j < 3; ++j) {
        T value = mass_i[3 * i + j];
        if (std::isnan(value) || value < 0.0) {
          std::cerr << "Invalid value detected in mass at node " << i
                    << ", component " << j << std::endl;
        }
        vtkFile << value << (j < 2 ? " " : "\n");
      }
    }

    vtkFile.close();
    std::cout << "Exported " << filename << std::endl;
  }

  void probe_node(int node_id) {
    std::string filename =
        "../output/nodes/node_" + std::to_string(node_id) + ".txt";
    // Check if the timestep is 0 and if the file exists, then delete it
    if (timestep == 0) {
      std::remove(filename.c_str());  // Remove the file if it exists
    }
    std::ofstream file;
    file.open(filename, std::ios::app);  // Open file in append mode
    std::ostringstream node_data;
    probe_node_data(node_id, node_data);  // Gather node data
    // Now write the timestep and current simulation time along with node data
    // to the file
    file << "Timestep " << timestep << ", Time: " << std::fixed
         << std::setprecision(2) << time << "s:\n";
    file << node_data.str() << "\n";  // Write the populated node data
    file.close();
  }

  // Function to probe details about a specific element and output to a file
  void probe_element(int element_id) {
    if (element_id < 0 || element_id >= mesh->num_elements) {
      std::cerr << "Element ID out of range.\n";
      return;
    }
    std::string filename =
        "../output/elements/element_" + std::to_string(element_id) + ".txt";
    // Check if the timestep is 0 and if the file exists, then delete it
    if (timestep == 0) {
      std::remove(filename.c_str());  // Remove the file if it exists
    }
    std::ofstream file;
    file.open(filename, std::ios::app);  // Open file in append mode
    int *nodes = &mesh->element_nodes[nodes_per_element * element_id];

    file << "Timestep " << timestep << ", Time: " << std::fixed
         << std::setprecision(2) << time << "s:\n"
         << "Element " << element_id << " consists of nodes:\n";
    for (int i = 0; i < nodes_per_element; ++i) {
      std::ostringstream node_data;
      probe_node_data(nodes[i], node_data);  // Gather node data
      file << " Node " << nodes[i] << " details:\n" << node_data.str();
    }
    file << "\n";
    file.close();
  }

  void debug_strain(const T alpha, const int def_case) {
    memcpy(global_xloc, mesh->xloc,
           ndof * sizeof(T));  // mesh->xloc will store initial positions
    T *global_dof = new T[ndof];

    memset(global_dof, 0, sizeof(T) * ndof);

    for (int i = 0; i < mesh->num_nodes; i++) {
      T x = global_xloc[i * 3 + 0];
      T y = global_xloc[i * 3 + 1];
      T z = global_xloc[i * 3 + 2];

      switch (def_case) {
        case 0:
          if (i == 0) {
            printf("Constant displacement case\n");
          }

          global_dof[i * 3 + 0] = alpha * x;
          global_dof[i * 3 + 1] = -alpha * x * material->nu;
          global_dof[i * 3 + 2] = -alpha * x * material->nu;
          break;
        case 1:
          if (i == 0) {
            printf("Linear displacement case\n");
          }
          global_dof[i * 3 + 0] = alpha * 0.5 * x * x;
          global_dof[i * 3 + 1] = -alpha * 0.5 * x * x * material->nu;
          global_dof[i * 3 + 2] = -alpha * 0.5 * x * x * material->nu;
          break;
        default:
          break;
      }
    }

    memset(vel, 0, sizeof(T) * ndof);
    memset(global_acc, 0, sizeof(T) * ndof);
    memset(global_mass, 0, sizeof(T) * ndof);
    memset(global_strains, 0, sizeof(T) * 6 * mesh->num_nodes);
    memset(global_stress, 0, sizeof(T) * 6 * mesh->num_nodes);

    constexpr int dof_per_element = spatial_dim * nodes_per_element;
    // Allocate element quantities
    std::vector<T> element_xloc(dof_per_element);
    std::vector<T> element_dof(dof_per_element);
    std::vector<int> this_element_nodes(nodes_per_element);

    T total_energy = 0.0;
    T total_volume = 0.0;
    T node_coords[spatial_dim];
    T element_strains[6];
    T element_stress[6];

    for (int i = 0; i < mesh->num_elements; i++) {
      memset(node_coords, 0, sizeof(T) * spatial_dim);

      for (int k = 0; k < dof_per_element; k++) {
        element_xloc[k] = 0.0;
        element_dof[k] = 0.0;
      }

      for (int j = 0; j < nodes_per_element; j++) {
        this_element_nodes[j] = mesh->element_nodes[nodes_per_element * i + j];
      }

      // Get the element locations
      Analysis::template get_element_dof<spatial_dim>(
          this_element_nodes.data(), global_xloc, element_xloc.data());

      // Get the element degrees of freedom
      Analysis::template get_element_dof<spatial_dim>(
          this_element_nodes.data(), global_dof, element_dof.data());

      T element_W = Analysis::calculate_strain_energy(
          element_xloc.data(), element_dof.data(), material);

      T element_volume = Analysis::calculate_volume(
          element_xloc.data(), element_dof.data(), material);

      for (int node = 0; node < nodes_per_element; node++) {
        memset(element_strains, 0, sizeof(T) * 6);
        for (int k = 0; k < spatial_dim; k++) {
          node_coords[k] = element_xloc[node * spatial_dim + k];
        }
        Analysis::calculate_stress_strain(
            element_xloc.data(), element_dof.data(), node_coords,
            element_strains, element_stress, material);
        int node_idx = this_element_nodes[node];
        for (int k = 0; k < 6; k++) {
          global_strains[node_idx * 6 + k] = element_strains[k];
          global_stress[node_idx * 6 + k] = element_stress[k];
          // #ifdef
          // printf("Node %d, Strain %d: %f\n", node_idx, k,
          // element_strains[k]); #endif
        }
      }

      total_energy += element_W;
      total_volume += element_volume;
    }

    printf("Total Strain Energy = %f\n", total_energy);
    printf("Total Volume = %f\n", total_volume);

    for (int i = 0; i < ndof; i++) {
      global_xloc[i] += global_dof[i];
    }

    export_to_vtk(0, vel, global_acc, global_mass);
  }

  void solve(double dt, double time_end, int export_interval) {
    // Perform a dynamic analysis. The algorithm is staggered as follows:
    // This assumes that the initial u, v, a and fext are already initialized
    // at nodes.

    // Given U0 and V0,
    // a. A0 = (Fext - Fint(U0))/M
    // b. Stagger V0.5 = V0 +dt/2*a0

    // Now start the loop
    // 1. Compute U1 = U +dt*V0.5
    // 2. Compute A1 = (Fext - Fint(U1)/M
    // 3. Compute V1.5 = V0.5 + A1*dt
    // 3. Compute V1 = V1.5 - dt/2 * a1
    // 4. Loop back to 1.

    // This scheme is common among various commercial solvers,
    // and hence, preferrable.

    // ------------------- Initialization -------------------
    printf("Solving dynamics\n");

    // Material and mesh information
    int *element_nodes = mesh->element_nodes;

    // Initialize global data
    memcpy(global_xloc, mesh->xloc,
           ndof * sizeof(T));  // mesh->xloc will store initial positions
    for (int i = 0; i < ndof; i++) {
      global_dof[i] = 0.0;
      global_acc[i] = 0.0;
      global_mass[i] = 0.0;
    }
    memset(global_stress_quads, 0,
           sizeof(T) * 6 * mesh->num_elements * num_quadrature_pts);
    memset(global_plastic_strain_quads, 0,
           sizeof(T) * 6 * mesh->num_elements * num_quadrature_pts);

    // Intermediate velocity for vtk export
    for (int i = 0; i < ndof; i++) {
      vel_i[i] = 0.0;
    }
    double time = 0.0;
    // Initialize states
    update<T, spatial_dim, nodes_per_element>(
        mesh->num_nodes, mesh->num_elements, ndof, dt, material, wall, mesh,
        element_nodes, vel, global_xloc, global_dof, global_acc, global_mass,
        global_strains, global_stress, global_stress_quads,
        global_plastic_strain_quads, eqPlasticStrain, pressure,
        plasticStrainRate, gamma, gamma_accumulated, yieldStress, plasticWork,
        internalEnergy, temperature, density, time);

    // b.Stagger V0 .5 = V0 + dt / 2 * a0
    // Update velocity
    for (int i = 0; i < ndof; i++) {
      vel[i] += 0.5 * dt * global_acc[i];
    }

    array_to_txt<T>("cpu_vel.txt", vel, ndof);
    array_to_txt<T>("cpu_xloc.txt", global_xloc, ndof);

    //------------------- End of Initialization -------------------
    // ------------------- Start of Time Loop -------------------

    while (time <= time_end) {
      printf("Time: %f\n", time);

      memset(global_dof, 0, sizeof(T) * ndof);
      // 1. Compute U1 = U +dt*V0.5
      // Update nodal displacements
      for (int j = 0; j < ndof; j++) {
        global_dof[j] = dt * vel[j];
      }

      update<T, spatial_dim, nodes_per_element>(
          mesh->num_nodes, mesh->num_elements, ndof, dt, material, wall, mesh,
          element_nodes, vel, global_xloc, global_dof, global_acc, global_mass,
          global_strains, global_stress, global_stress_quads,
          global_plastic_strain_quads, eqPlasticStrain, pressure,
          plasticStrainRate, gamma, gamma_accumulated, yieldStress, plasticWork,
          internalEnergy, temperature, density, time);

      // Compute total mass (useful?)
      T total_mass = 0.0;
      for (int i = 0; i < ndof; i++) {
        total_mass += global_mass[i] / 3.0;
      }
      // printf("mass: %30.15e\n", total_mass);

      // 3. Compute V1.5 = V0.5 + A1*dt
      // 3. Compute V1 = V1.5 - dt/2 * a1
      // 4. Loop back to 1.

      for (int i = 0; i < ndof; i++) {
        global_xloc[i] += global_dof[i];
        vel[i] += dt * global_acc[i];

        // TODO: only run this on export steps
        vel_i[i] = vel[i] - 0.5 * dt * global_acc[i];
      }

      if (timestep % export_interval == 0) {
        export_to_vtk(timestep, vel_i, global_acc, global_mass);
        probe_node(96);
      }
      time += dt;
      timestep += 1;
    }
  }
};
