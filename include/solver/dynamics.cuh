#pragma once
#include <cfloat>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "../config/common_definitions.h"
#include "../utils/cppimpact_defs.h"
#include "../utils/cppimpact_utils.h"
#include "dynamics_kernels.cuh"
#include "mesh.h"
#include "tetrahedral.h"
#include "wall.h"

// The GPU version of the Dynamics class sets up and runs an explicit dynamics
// finite element simulation on the GPU. It uses CUDA kernels (defined in
// dynamics_kernels.cuh) to perform element-level computations in parallel. This
// class manages device memory for nodal, element, and material data, as well as
// transfers results back to the host for output.
//
// Template parameters:
//   T           : Numeric type, e.g., float or double
//   Basis       : A struct/class defining element shape functions and spatial
//   dimension Analysis    : A class providing methods for calculating strain,
//   stress, etc. (used on GPU) Quadrature  : Integration rule for elements
//   (number and position of integration points)

template <typename T, class Basis, class Analysis, class Quadrature>
class Dynamics {
 private:
  /**
   * @brief Probe and format data about a single node (position, velocity,
   * acceleration, mass).
   *
   * @param node_id The global node index.
   * @param stream  A reference to an output stream to append node data.
   */
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
  // Arrays and parameters associated with the simulation
  int *reduced_nodes;
  int reduced_dofs_size;
  int ndof;
  static constexpr int nodes_per_element = Basis::nodes_per_element;
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int num_quadrature_pts = Quadrature::num_quadrature_pts;
  static constexpr int dof_per_node = spatial_dim;

  // Pointers to the problem definition
  Mesh<T, nodes_per_element> *mesh;
  Material *material;
  Wall<T, 2, Basis> *wall;

  // Nodal arrays on the host
  T *global_xloc;     // Nodal coordinates (updated each timestep)
  T *vel;             // Nodal velocities
  T *global_strains;  // Strain tensor components at each node (6 per node in
                      // 3D)
  T *global_stress;  // Stress tensor components at each node (6 per node in 3D)
  T *global_dof;     // Incremental displacement (U_{n+1} - U_n)
  T *global_acc;     // Nodal accelerations
  T *global_mass;    // Nodal masses
  T *vel_i;          // Intermediate velocity array for output

  int timestep;
  double time;  // Current simulation time

  /**
   * @brief Constructor that sets up the Dynamics object and allocates host
   * arrays.
   *
   * @param input_mesh    Pointer to mesh containing nodes, elements, etc.
   * @param input_material Pointer to material definition (properties, etc.).
   * @param input_wall    Optional pointer to wall object for contact
   * simulation.
   */
  Dynamics(Mesh<T, nodes_per_element> *input_mesh, Material *input_material,
           Wall<T, 2, Basis> *input_wall = nullptr)
      : mesh(input_mesh),
        material(input_material),
        wall(input_wall),
        reduced_nodes(nullptr),
        reduced_dofs_size(0),
        vel(new T[mesh->num_nodes * dof_per_node]),
        global_xloc(new T[mesh->num_nodes * dof_per_node]),
        global_strains(new T[mesh->num_nodes * 6]),
        global_stress(new T[mesh->num_nodes * 6]),
        global_dof(new T[mesh->num_nodes * dof_per_node]),
        global_acc(new T[mesh->num_nodes * dof_per_node]),
        global_mass(new T[mesh->num_nodes * dof_per_node]),
        vel_i(new T[mesh->num_nodes * dof_per_node]),
        timestep(0),
        time(0.0) {
    ndof = mesh->num_nodes * dof_per_node;
  }

  /**
   * @brief Destructor that frees all host arrays.
   */
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
  }

  /**
   * @brief Initialize the entire mesh by translating its initial position and
   * assigning initial velocities.
   *
   * @param init_position A 3-component array specifying a uniform translation.
   * @param init_velocity A 3-component array specifying the initial velocity
   * for all nodes.
   */
  void initialize(T init_position[dof_per_node],
                  T init_velocity[dof_per_node]) {
    std::cout << "ndof: " << ndof << std::endl;
    for (int i = 0; i < mesh->num_nodes; i++) {
      // Set initial velocity
      vel[3 * i] = init_velocity[0];
      vel[3 * i + 1] = init_velocity[1];
      vel[3 * i + 2] = init_velocity[2];

      // Translate node position
      mesh->xloc[3 * i] += init_position[0];
      mesh->xloc[3 * i + 1] += init_position[1];
      mesh->xloc[3 * i + 2] += init_position[2];
    }
  }

  /**
   * @brief Write simulation data at a given timestep to a VTK file for
   * visualization.
   *
   * Outputs node coordinates, element connectivity, velocity, strain, stress,
   * acceleration, and mass.
   *
   * @param timestep The current timestep number (used for file naming).
   * @param vel_i    Intermediate velocity array for output.
   * @param acc_i    Current acceleration array.
   * @param mass_i   Current mass array.
   */
  void export_to_vtk(int timestep, T *vel_i, T *acc_i, T *mass_i) {
    const std::string directory = "../gpu_output";
    const std::string filename =
        directory + "/simulation_" + std::to_string(timestep) + ".vtk";
    std::ofstream vtkFile(filename);

    if (!vtkFile.is_open()) {
      std::cerr << "Failed to open " << filename << std::endl;
      return;
    }

    // Write standard VTK header
    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "FEA simulation data\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET UNSTRUCTURED_GRID\n";

    const double threshold = 1e15;

    // Nodal coordinates
    vtkFile << "POINTS " << mesh->num_nodes << " float\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      T x = global_xloc[3 * i];
      T y = global_xloc[3 * i + 1];
      T z = global_xloc[3 * i + 2];

      // Sanitize output to avoid NaNs/Infs
      if (std::isnan(x) || std::isinf(x) || std::abs(x) > threshold) {
        printf("Invalid x-coordinate at node %d: %f, setting to 0.\n", i, x);
        x = 0.0;
      }
      if (std::isnan(y) || std::isinf(y) || std::abs(y) > threshold) {
        printf("Invalid y-coordinate at node %d: %f, setting to 0.\n", i, y);
        y = 0.0;
      }
      if (std::isnan(z) || std::isinf(z) || std::abs(z) > threshold) {
        printf("Invalid z-coordinate at node %d: %f, setting to 0.\n", i, z);
        z = 0.0;
      }

      vtkFile << std::fixed << std::setprecision(6);
      vtkFile << x << " " << y << " " << z << "\n";
    }

    // Element connectivity
    vtkFile << "CELLS " << mesh->num_elements << " "
            << mesh->num_elements * (nodes_per_element + 1) << "\n";
    for (int i = 0; i < mesh->num_elements; ++i) {
      vtkFile << nodes_per_element;
      for (int j = 0; j < nodes_per_element; ++j) {
        vtkFile << " " << mesh->element_nodes[nodes_per_element * i + j];
      }
      vtkFile << "\n";
    }

    // Specify cell types as tetrahedra
    vtkFile << "CELL_TYPES " << mesh->num_elements << "\n";
    for (int i = 0; i < mesh->num_elements; ++i) {
      vtkFile << "10\n";  // VTK_TETRA = 10
    }

    // Nodal data
    vtkFile << "POINT_DATA " << mesh->num_nodes << "\n";

    // Velocity field
    vtkFile << "VECTORS velocity double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      for (int j = 0; j < 3; ++j) {
        T value = vel_i[3 * i + j];
        if (std::isnan(value)) {
          std::cerr << "NaN in velocity at node " << i << ", component " << j
                    << std::endl;
          value = 0.0;
        }
        vtkFile << value << (j < 2 ? " " : "\n");
      }
    }

    // Strain tensor (split into two vectors for simplicity)
    vtkFile << "VECTORS strain1 double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      vtkFile << global_strains[6 * i + 0] << " " << global_strains[6 * i + 1]
              << " " << global_strains[6 * i + 2] << "\n";
    }

    vtkFile << "VECTORS strain2 double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      vtkFile << global_strains[6 * i + 3] << " " << global_strains[6 * i + 4]
              << " " << global_strains[6 * i + 5] << "\n";
    }

    // Acceleration
    vtkFile << "VECTORS acceleration double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      for (int j = 0; j < 3; ++j) {
        T value = acc_i[3 * i + j];
        if (std::isnan(value)) {
          std::cerr << "NaN in acceleration at node " << i << ", component "
                    << j << std::endl;
          value = 0.0;
        }
        vtkFile << value << (j < 2 ? " " : "\n");
      }
    }

    // Mass
    vtkFile << "VECTORS mass double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      for (int j = 0; j < 3; ++j) {
        T value = mass_i[3 * i + j];
        if (std::isnan(value) || value < 0.0) {
          std::cerr << "Invalid mass at node " << i << ", component " << j
                    << std::endl;
        }
        vtkFile << value << (j < 2 ? " " : "\n");
      }
    }

    vtkFile.close();
    std::cout << "Exported " << filename << std::endl;
  }

  /**
   * @brief Log data for a specific node to a text file for debugging or
   * post-processing.
   *
   * Creates or appends to a file named after the node ID.
   *
   * @param node_id The node index to probe.
   */
  void probe_node(int node_id) {
    std::string filename =
        "../output/nodes/node_" + std::to_string(node_id) + ".txt";
    if (timestep == 0) {
      std::remove(filename.c_str());
    }
    std::ofstream file(filename, std::ios::app);
    std::ostringstream node_data;
    probe_node_data(node_id, node_data);

    file << "Timestep " << timestep << ", Time: " << std::fixed
         << std::setprecision(2) << time << "s:\n";
    file << node_data.str() << "\n";
    file.close();
  }

  /**
   * @brief Log data for a specific element and its nodes to a text file.
   *
   * Useful for detailed inspection of localized behavior.
   *
   * @param element_id The element index to probe.
   */
  void probe_element(int element_id) {
    if (element_id < 0 || element_id >= mesh->num_elements) {
      std::cerr << "Element ID out of range.\n";
      return;
    }
    std::string filename =
        "../output/elements/element_" + std::to_string(element_id) + ".txt";
    if (timestep == 0) {
      std::remove(filename.c_str());
    }
    std::ofstream file(filename, std::ios::app);
    int *nodes = &mesh->element_nodes[nodes_per_element * element_id];

    file << "Timestep " << timestep << ", Time: " << std::fixed
         << std::setprecision(2) << time << "s:\n"
         << "Element " << element_id << " consists of nodes:\n";
    for (int i = 0; i < nodes_per_element; ++i) {
      std::ostringstream node_data;
      probe_node_data(nodes[i], node_data);
      file << " Node " << nodes[i] << " details:\n" << node_data.str();
    }
    file << "\n";
    file.close();
  }

  /**
   * @brief Allocate device (GPU) memory for all global arrays and copy data
   * from the host.
   *
   * This function prepares the simulation data for GPU kernels by setting up
   * device pointers and transferring node positions, velocities, and material
   * properties.
   */
  void allocate() {
    // Allocate main nodal arrays on the GPU
    cudaMalloc(&d_global_dof, sizeof(T) * ndof);
    cudaMalloc(&d_global_acc, sizeof(T) * ndof);
    cudaMalloc(&d_global_mass, sizeof(T) * ndof);
    cudaMalloc(&d_vel, sizeof(T) * ndof);
    cudaMalloc(&d_vel_i, sizeof(T) * ndof);
    cudaMalloc(&d_global_xloc, sizeof(T) * ndof);
    cudaMalloc(&d_element_nodes,
               sizeof(int) * nodes_per_element * mesh->num_elements);
    cudaMalloc(&d_global_strains, sizeof(T) * mesh->num_nodes * 6);
    cudaMalloc(&d_global_stress, sizeof(T) * mesh->num_nodes * 6);

    // Material and wall objects
    cudaMalloc((void **)&d_material, sizeof(decltype(*material)));
    cudaMalloc((void **)&d_wall, sizeof(decltype(*wall)));

    // Wall slave node indices (contact-related data)
    cudaMalloc((void **)&(d_wall_slave_node_indices),
               sizeof(int) * mesh->num_slave_nodes);

    // Initialize device memory with zeros
    cudaMemset(d_global_dof, T(0.0), sizeof(T) * ndof);
    cudaMemset(d_global_acc, T(0.0), sizeof(T) * ndof);
    cudaMemset(d_global_mass, T(0.0), sizeof(T) * ndof);
    cudaMemset(d_vel_i, T(0.0), sizeof(T) * ndof);
    cudaMemset(d_global_strains, T(0.0), sizeof(T) * mesh->num_nodes * 6);
    cudaMemset(d_global_stress, T(0.0), sizeof(T) * mesh->num_nodes * 6);

    // Transfer host data to device
    cudaMemcpy(d_global_xloc, mesh->xloc, ndof * sizeof(T),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_element_nodes, mesh->element_nodes,
               sizeof(int) * nodes_per_element * mesh->num_elements,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, vel, ndof * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_material, material, sizeof(decltype(*material)),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_wall, wall, sizeof(decltype(*wall)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wall_slave_node_indices, wall->slave_node_indices,
               sizeof(int) * mesh->num_slave_nodes, cudaMemcpyHostToDevice);

    // Update the wall object on the device to reference device memory
    cudaMemcpy(&(d_wall->slave_node_indices), &d_wall_slave_node_indices,
               sizeof(int *), cudaMemcpyHostToDevice);
  }

  /**
   * @brief Free all device (GPU) memory allocations.
   */
  void deallocate() {
    cudaFree(d_global_dof);
    cudaFree(d_global_acc);
    cudaFree(d_global_mass);
    cudaFree(d_vel);
    cudaFree(d_vel_i);
    cudaFree(d_global_xloc);
    cudaFree(d_element_nodes);
    cudaFree(d_material);
    cudaFree(d_wall);
    cudaFree(d_wall_slave_node_indices);
    cudaFree(d_global_strains);
    cudaFree(d_global_stress);
  }

  /**
   * @brief Solve the dynamic problem using explicit time-stepping with
   * CUDA-accelerated kernels.
   *
   * Uses a central difference-like integration scheme:
   * 1. Compute A0 from initial conditions.
   * 2. Stagger velocity: V0.5 = V0 + (dt/2)*A0
   * 3. For each timestep:
   *    a) U_{n+1} = U_n + dt * V_{n+1/2}
   *    b) A_{n+1} = (F_ext - F_int(U_{n+1}))/M
   *    c) V_{n+3/2} = V_{n+1/2} + A_{n+1} * dt
   *    d) V_{n+1} = V_{n+3/2} - (dt/2)*A_{n+1}
   *
   * Intermediate states and results are regularly transferred back to the host
   * for visualization and debugging.
   *
   * @param dt              Time step size
   * @param time_end        Final simulation time
   * @param export_interval Interval between output to VTK files
   */
  void solve(double dt, double time_end, int export_interval) {
    // Allocate device data
    allocate();

    printf("Solving dynamics\n");

    // Setup GPU kernel dimensions
    constexpr int nodes_per_elem_num_quad =
        nodes_per_element * num_quadrature_pts;
    constexpr int threads_per_block =
        ((nodes_per_elem_num_quad + 31) / 32) * 32;
    const int node_blocks = mesh->num_nodes / 32 + 1;
    const int ndof_blocks = ndof / 32 + 1;

    // Initialize CUDA streams for asynchronous operations
    cudaStream_t *streams;
    int num_c = 6;
    streams = new cudaStream_t[num_c];
    for (int c = 0; c < num_c; c++) {
      cudaStreamCreateWithFlags(&streams[c], cudaStreamNonBlocking);
    }

    // After computing initial A0 on GPU (omitted for brevity), stagger
    // velocity: V0.5 = V0 + (dt/2)*A0 Here, due to code simplification, we just
    // copy results and update on host side: (The code for initial update calls
    // and synchronization is incomplete for brevity.)

    // Example of how we update after initialization:
    // Copy velocities, accelerations back to host, etc., and apply V0.5 = V0 +
    // 0.5*dt*A0 on host (In the actual code, you'd do the initial call to
    // 'update' kernels here.)

    // Time loop
    while (time <= time_end) {
      // Clear accelerations and DOF arrays
      cudaMemsetAsync(d_global_acc, T(0.0), sizeof(T) * ndof, streams[0]);
      cudaMemsetAsync(d_global_dof, T(0.0), sizeof(T) * ndof, streams[1]);
      cudaStreamSynchronize(streams[0]);
      cudaStreamSynchronize(streams[1]);

      printf("Time: %f\n", time);

      // Compute U_{n+1} increment: U_{n+1} = U_n + dt * V_{n+1/2}
      update_dof<T>
          <<<ndof_blocks, 32, 0, streams[0]>>>(ndof, dt, d_vel, d_global_dof);
      cudaStreamSynchronize(streams[0]);

      // Compute internal forces, accelerations, etc.
      update<T, spatial_dim, nodes_per_element>
          <<<mesh->num_elements, threads_per_block, 0, streams[0]>>>(
              mesh->num_elements, dt, d_material, d_wall, d_element_nodes,
              d_vel, d_global_xloc, d_global_dof, d_global_acc, d_global_mass,
              d_global_strains, d_global_stress, nodes_per_elem_num_quad, time);
      cudaStreamSynchronize(streams[0]);

      // Apply external forces, e.g., wall contact
      external_forces<T><<<node_blocks, 32, 0, streams[0]>>>(
          mesh->num_nodes, d_wall, d_global_xloc, d_global_dof, d_global_mass,
          d_global_acc);
      cudaStreamSynchronize(streams[0]);

      // Update velocities and positions for the next step
      timeloop_update<T><<<ndof_blocks, 32, 0, streams[0]>>>(
          ndof, dt, d_global_xloc, d_vel, d_global_acc, d_vel_i, d_global_dof);
      cudaStreamSynchronize(streams[0]);

      // Periodically export data for visualization
      if (timestep % export_interval == 0) {
        cudaMemcpyAsync(vel_i, d_vel, ndof * sizeof(T), cudaMemcpyDeviceToHost,
                        streams[0]);
        cudaMemcpyAsync(global_acc, d_global_acc, ndof * sizeof(T),
                        cudaMemcpyDeviceToHost, streams[1]);
        cudaMemcpyAsync(global_mass, d_global_mass, ndof * sizeof(T),
                        cudaMemcpyDeviceToHost, streams[2]);
        cudaMemcpyAsync(global_xloc, d_global_xloc, ndof * sizeof(T),
                        cudaMemcpyDeviceToHost, streams[3]);
        cudaMemcpyAsync(global_strains, d_global_strains,
                        mesh->num_nodes * 6 * sizeof(T), cudaMemcpyDeviceToHost,
                        streams[4]);
        cudaMemcpyAsync(global_stress, d_global_stress,
                        mesh->num_nodes * 6 * sizeof(T), cudaMemcpyDeviceToHost,
                        streams[5]);

        // Wait for all data transfers to complete before writing files
        for (int c = 0; c < num_c; c++) {
          cudaStreamSynchronize(streams[c]);
        }

        // Export to VTK and probe selected nodes
        export_to_vtk(timestep, vel_i, global_acc, global_mass);
        probe_node(13648);
        probe_node(13649);
        probe_node(13650);
        probe_node(13651);
        probe_node(13652);
      }

      time += dt;
      timestep += 1;

#ifdef CPPIMPACT_DEBUG_MODE
      cuda_show_kernel_error();
#endif
    }

    // Destroy CUDA streams
    for (int c = 0; c < num_c; c++) {
      cudaStreamDestroy(streams[c]);
    }
    delete[] streams;

    // Deallocate device memory
    deallocate();
  }

 private:
  // Device pointers for simulation data
  T *d_global_dof = nullptr;
  T *d_global_acc = nullptr;
  T *d_global_mass = nullptr;
  T *d_vel = nullptr;
  T *d_vel_i = nullptr;
  T *d_global_xloc = nullptr;
  T *d_global_strains = nullptr;
  T *d_global_stress = nullptr;
  int *d_element_nodes = nullptr;

  Material *d_material = nullptr;
  Wall<T, 2, Basis> *d_wall = nullptr;
  int *d_wall_slave_node_indices = nullptr;
};
