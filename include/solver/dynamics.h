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

// The Dynamics class template performs explicit dynamics finite element
// analysis (FEA). It manages node-based state variables (e.g., displacements,
// velocities, accelerations, mass) and element-based quantities (e.g., stress,
// strain, plastic strains) through time. The simulation updates nodal
// positions, computes internal forces/stresses, applies boundary conditions
// (including optional contact with a wall), and writes results to visualization
// files.
//
// Template parameters:
//   T           : Numeric type (e.g., float or double)
//   Basis       : A struct/class defining the finite element basis (e.g., shape
//   functions, etc.) Analysis    : A struct/class providing functions for
//   computing element stiffness, strain, stress, etc. Quadrature  : A
//   struct/class defining the integration rule for elements (number and
//   positions of integration points).

template <typename T, class Basis, class Analysis, class Quadrature>
class Dynamics {
 private:
  /**
   * @brief Collects and formats data about a single node (position, velocity,
   * acceleration, mass).
   *
   * @param node_id The global node index.
   * @param stream  A reference to an output stream to append the node data.
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
  // Global indices of nodes that may be reduced (used in certain reduced-order
  // modeling)
  int *reduced_nodes;
  // Number of degrees of freedom (DOFs) in the reduced system
  int reduced_dofs_size;
  // Total number of degrees of freedom in the system
  int ndof;

  // Element and dimension constants derived from the Basis and Quadrature
  // objects
  static constexpr int nodes_per_element = Basis::nodes_per_element;
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int dof_per_node = spatial_dim;
  static constexpr int num_quadrature_pts = Quadrature::num_quadrature_pts;

  // Primary simulation data pointers
  Mesh<T, nodes_per_element> *mesh;
  Material *material;
  Wall<T, 2, Basis>
      *wall;  // Optional contact boundary (2D is a placeholder dimension)

  // Nodal state arrays
  T *global_xloc;     // Current nodal coordinates
  T *vel;             // Current nodal velocities
  T *global_strains;  // Nodal strains (6 components per node in 3D: e_xx, e_yy,
                      // e_zz, e_xy, e_xz, e_yz)
  T *global_stress;   // Nodal stresses (6 components per node in 3D)
  T *global_dof;      // Displacement increments (used each time step)
  T *global_acc;      // Nodal accelerations
  T *global_mass;     // Nodal masses (repeated in each coordinate direction)
  T *vel_i;  // Intermediate velocity array for output or integration steps

  // Element-based quadrature data arrays
  // Stress and plastic strain at quadrature points (integration points within
  // each element)
  T *global_stress_quads;
  T *global_plastic_strain_quads;

  // Additional scalar state variables at quadrature points for advanced
  // material modeling
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

  // Current simulation timestep counter
  int timestep;
  // Current simulation time (in seconds)
  double time;

  /**
   * @brief Constructs the Dynamics solver object for a given mesh, material,
   * and optional wall.
   *
   * Allocates memory for all state variables.
   *
   * @param input_mesh    Pointer to the mesh object (node coordinates, element
   * connectivity).
   * @param input_material Pointer to the material object (properties like
   * Young's modulus, Poisson ratio, etc.).
   * @param input_wall    Optional pointer to a wall object for boundary
   * contact.
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
        density(new T[mesh->num_elements * num_quadrature_pts]),
        timestep(0),
        time(0.0) {
    ndof = mesh->num_nodes * dof_per_node;
  }

  /**
   * @brief Destructor that frees all dynamically allocated arrays.
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

  /**
   * @brief Initialize the body by shifting all nodes to an initial position and
   * giving them an initial velocity.
   *
   * @param init_position A 3-component array specifying the initial translation
   * to apply to all nodes.
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

      // Shift node position by init_position
      mesh->xloc[3 * i] += init_position[0];
      mesh->xloc[3 * i + 1] += init_position[1];
      mesh->xloc[3 * i + 2] += init_position[2];
    }
  }

  /**
   * @brief Exports the current simulation state to a VTK file for
   * visualization.
   *
   * This includes nodal coordinates, element connectivity, velocities, strains,
   * stresses, accelerations, and mass.
   *
   * @param timestep The current timestep count (used in file naming).
   * @param vel_i    Intermediate velocity array for output.
   * @param acc_i    Current acceleration array.
   * @param mass_i   Current mass array.
   */
  void export_to_vtk(int timestep, T *vel_i, T *acc_i, T *mass_i) {
    const std::string directory = "../cpu_output";
    const std::string filename =
        directory + "/simulation_" + std::to_string(timestep) + ".vtk";
    std::ofstream vtkFile(filename);

    if (!vtkFile.is_open()) {
      std::cerr << "Failed to open " << filename << std::endl;
      return;
    }

    // VTK header
    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "FEA simulation data\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET UNSTRUCTURED_GRID\n";

    const double threshold = 1e15;

    // Write nodal coordinates
    vtkFile << "POINTS " << mesh->num_nodes << " float\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      T x = global_xloc[3 * i];
      T y = global_xloc[3 * i + 1];
      T z = global_xloc[3 * i + 2];

      // Validate data, replace invalid numbers with 0.0
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

    // Write element connectivity
    vtkFile << "CELLS " << mesh->num_elements << " "
            << mesh->num_elements * (nodes_per_element + 1) << "\n";
    for (int i = 0; i < mesh->num_elements; ++i) {
      vtkFile << nodes_per_element;
      for (int j = 0; j < nodes_per_element; ++j) {
        vtkFile << " " << mesh->element_nodes[nodes_per_element * i + j];
      }
      vtkFile << "\n";
    }

    // Specify the cell type (VTK_TETRA = 10)
    vtkFile << "CELL_TYPES " << mesh->num_elements << "\n";
    for (int i = 0; i < mesh->num_elements; ++i) {
      vtkFile << "10\n";
    }

    // Nodal data output
    vtkFile << "POINT_DATA " << mesh->num_nodes << "\n";
    vtkFile << "VECTORS velocity double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      for (int j = 0; j < 3; ++j) {
        T value = vel_i[3 * i + j];
        if (std::isnan(value)) {
          std::cerr << "NaN velocity at node " << i << ", component " << j
                    << std::endl;
          value = 0.0;
        }
        vtkFile << value << (j < 2 ? " " : "\n");
      }
    }

    // Strain tensor components (split into two vectors for convenience)
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

    // Stress tensor components
    vtkFile << "VECTORS stress1 double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      vtkFile << global_stress[6 * i + 0] << " " << global_stress[6 * i + 1]
              << " " << global_stress[6 * i + 2] << "\n";
    }

    vtkFile << "VECTORS stress2 double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      vtkFile << global_stress[6 * i + 3] << " " << global_stress[6 * i + 4]
              << " " << global_stress[6 * i + 5] << "\n";
    }

    // Acceleration
    vtkFile << "VECTORS acceleration double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      for (int j = 0; j < 3; ++j) {
        T value = acc_i[3 * i + j];
        if (std::isnan(value)) {
          std::cerr << "NaN acceleration at node " << i << ", component " << j
                    << std::endl;
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
   * @brief Probe and record data of a single node at the current timestep to a
   * text file.
   *
   * @param node_id The global node index to probe.
   */
  void probe_node(int node_id) {
    std::string filename =
        "../output/nodes/node_" + std::to_string(node_id) + ".txt";

    // If this is the first timestep, clear any existing file
    if (timestep == 0) {
      std::remove(filename.c_str());
    }

    std::ofstream file;
    file.open(filename, std::ios::app);  // Append mode
    std::ostringstream node_data;
    probe_node_data(node_id, node_data);

    file << "Timestep " << timestep << ", Time: " << std::fixed
         << std::setprecision(2) << time << "s:\n";
    file << node_data.str() << "\n";
    file.close();
  }

  /**
   * @brief Probe data about a specific element and its constituent nodes,
   * writing results to a text file.
   *
   * @param element_id The global element index to probe.
   */
  void probe_element(int element_id) {
    if (element_id < 0 || element_id >= mesh->num_elements) {
      std::cerr << "Element ID out of range.\n";
      return;
    }
    std::string filename =
        "../output/elements/element_" + std::to_string(element_id) + ".txt";

    // If this is the first timestep, clear any existing file
    if (timestep == 0) {
      std::remove(filename.c_str());
    }

    std::ofstream file;
    file.open(filename, std::ios::app);  // Append mode
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
   * @brief A debug function that applies a prescribed displacement field to the
   * nodes and calculates strain/stress.
   *
   * This function can be used to verify the correctness of strain and stress
   * calculations without running a full simulation.
   *
   * @param alpha    A scalar factor applied to the displacement field.
   * @param def_case An integer flag to choose between different displacement
   * patterns (e.g., constant or linear).
   */
  void debug_strain(const T alpha, const int def_case) {
    // Copy current coordinates
    memcpy(global_xloc, mesh->xloc, ndof * sizeof(T));
    T *global_dof = new T[ndof];
    memset(global_dof, 0, sizeof(T) * ndof);

    // Apply a test deformation pattern to the nodes
    for (int i = 0; i < mesh->num_nodes; i++) {
      T x = global_xloc[i * 3 + 0];

      switch (def_case) {
        case 0:
          if (i == 0) {
            printf("Constant displacement case\n");
          }
          // A simple linear displacement field in x, with Poisson contraction
          // in y,z
          global_dof[i * 3 + 0] = alpha * x;
          global_dof[i * 3 + 1] = -alpha * x * material->nu;
          global_dof[i * 3 + 2] = -alpha * x * material->nu;
          break;
        case 1:
          if (i == 0) {
            printf("Linear displacement case\n");
          }
          // Quadratic displacement pattern
          global_dof[i * 3 + 0] = alpha * 0.5 * x * x;
          global_dof[i * 3 + 1] = -alpha * 0.5 * x * x * material->nu;
          global_dof[i * 3 + 2] = -alpha * 0.5 * x * x * material->nu;
          break;
        default:
          break;
      }
    }

    // Reset dynamic variables
    memset(vel, 0, sizeof(T) * ndof);
    memset(global_acc, 0, sizeof(T) * ndof);
    memset(global_mass, 0, sizeof(T) * ndof);
    memset(global_strains, 0, sizeof(T) * 6 * mesh->num_nodes);
    memset(global_stress, 0, sizeof(T) * 6 * mesh->num_nodes);

    // Perform strain and stress calculations
    constexpr int dof_per_element = spatial_dim * nodes_per_element;
    std::vector<T> element_xloc(dof_per_element);
    std::vector<T> element_dof(dof_per_element);
    std::vector<int> this_element_nodes(nodes_per_element);

    T total_energy = 0.0;
    T total_volume = 0.0;
    T node_coords[spatial_dim];
    T element_strains[6];
    T element_stress[6];

    for (int i = 0; i < mesh->num_elements; i++) {
      for (int k = 0; k < dof_per_element; k++) {
        element_xloc[k] = 0.0;
        element_dof[k] = 0.0;
      }

      for (int j = 0; j < nodes_per_element; j++) {
        this_element_nodes[j] = mesh->element_nodes[nodes_per_element * i + j];
      }

      // Gather element node locations and DOFs
      Analysis::template get_element_dof<spatial_dim>(
          this_element_nodes.data(), global_xloc, element_xloc.data());

      Analysis::template get_element_dof<spatial_dim>(
          this_element_nodes.data(), global_dof, element_dof.data());

      // Compute strain energy and volume as a consistency check
      T element_W = Analysis::calculate_strain_energy(
          element_xloc.data(), element_dof.data(), material);
      T element_volume = Analysis::calculate_volume(
          element_xloc.data(), element_dof.data(), material);

      // Compute strain and stress at element nodes
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
        }
      }

      total_energy += element_W;
      total_volume += element_volume;
    }

    printf("Total Strain Energy = %f\n", total_energy);
    printf("Total Volume = %f\n", total_volume);

    // Update global coordinates with the applied displacements
    for (int i = 0; i < ndof; i++) {
      global_xloc[i] += global_dof[i];
    }

    // Export the resulting fields to VTK
    export_to_vtk(0, vel, global_acc, global_mass);
    delete[] global_dof;
  }

  /**
   * @brief Solve the dynamic problem from t=0 to time_end using explicit
   * time-stepping.
   *
   * Uses a central difference scheme or a similar explicit integrator to update
   * positions, velocities, and accelerations. Regularly writes results to VTK
   * files for visualization.
   *
   * @param dt              The time step size.
   * @param time_end        The final time until which the simulation runs.
   * @param export_interval The number of timesteps between VTK exports.
   */
  void solve(double dt, double time_end, int export_interval) {
    // The time integration algorithm:
    // 1. Compute initial acceleration A0 = (F_ext - F_int(U0))/M using initial
    // conditions U0, V0.
    // 2. Stagger velocities: V0.5 = V0 + (dt/2)*A0.
    // Then iterate:
    //   a) U_{n+1} = U_n + dt * V_{n+1/2}
    //   b) A_{n+1} = (F_ext - F_int(U_{n+1}))/M
    //   c) V_{n+3/2} = V_{n+1/2} + dt * A_{n+1}
    //   d) V_{n+1} = V_{n+3/2} - (dt/2)*A_{n+1} (to re-center velocity)
    // This scheme is commonly used in explicit dynamics codes.

    printf("Solving dynamics\n");

    int *element_nodes = mesh->element_nodes;

    // Initialize global data from mesh
    memcpy(global_xloc, mesh->xloc, ndof * sizeof(T));
    for (int i = 0; i < ndof; i++) {
      global_dof[i] = 0.0;
      global_acc[i] = 0.0;
      global_mass[i] = 0.0;
    }
    memset(global_stress_quads, 0,
           sizeof(T) * 6 * mesh->num_elements * num_quadrature_pts);
    memset(global_plastic_strain_quads, 0,
           sizeof(T) * 6 * mesh->num_elements * num_quadrature_pts);

    for (int i = 0; i < ndof; i++) {
      vel_i[i] = 0.0;
    }

    double time = 0.0;

    // Compute initial acceleration
    update<T, spatial_dim, nodes_per_element>(
        mesh->num_nodes, mesh->num_elements, ndof, dt, material, wall, mesh,
        element_nodes, vel, global_xloc, global_dof, global_acc, global_mass,
        global_strains, global_stress, global_stress_quads,
        global_plastic_strain_quads, eqPlasticStrain, pressure,
        plasticStrainRate, gamma, gamma_accumulated, yieldStress, plasticWork,
        internalEnergy, temperature, density, time);

    // Stagger velocity: V0.5 = V0 + (dt/2)*A0
    for (int i = 0; i < ndof; i++) {
      vel[i] += 0.5 * dt * global_acc[i];
    }

    array_to_txt<T>("cpu_vel.txt", vel, ndof);
    array_to_txt<T>("cpu_xloc.txt", global_xloc, ndof);

    // Time loop
    while (time <= time_end) {
      printf("Time: %f\n", time);

      memset(global_dof, 0, sizeof(T) * ndof);
      // U_{n+1} = U_n + dt * V_{n+1/2}
      for (int j = 0; j < ndof; j++) {
        global_dof[j] = dt * vel[j];
      }

      // Compute acceleration A_{n+1}
      update<T, spatial_dim, nodes_per_element>(
          mesh->num_nodes, mesh->num_elements, ndof, dt, material, wall, mesh,
          element_nodes, vel, global_xloc, global_dof, global_acc, global_mass,
          global_strains, global_stress, global_stress_quads,
          global_plastic_strain_quads, eqPlasticStrain, pressure,
          plasticStrainRate, gamma, gamma_accumulated, yieldStress, plasticWork,
          internalEnergy, temperature, density, time);

      // Update nodal data and velocity
      for (int i = 0; i < ndof; i++) {
        global_xloc[i] += global_dof[i];
        vel[i] += dt * global_acc[i];
        vel_i[i] = vel[i] - 0.5 * dt * global_acc[i];  // V_{n+1} from scheme
      }

      // Export results at specified intervals
      if (timestep % export_interval == 0) {
        export_to_vtk(timestep, vel_i, global_acc, global_mass);
        probe_node(96);  // Example node for output
      }

      time += dt;
      timestep += 1;
    }
  }
};
