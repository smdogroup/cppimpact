#pragma once
#include <vector>

#include "../config/common_definitions.h"
#include "wall.h"

// Update the system states via time-stepping
template <typename T, int spatial_dim, int nodes_per_element>
void update(int num_nodes, int num_elements, int ndof, T dt, Material *material,
            Wall<T, 2, Basis> *wall, Mesh<T, nodes_per_element> *mesh,
            const int *element_nodes, const T *vel, const T *global_xloc,
            const T *global_dof, T *global_acc, T *global_mass,
            T *global_strains, T *global_stress, T time) {
  int constexpr dof_per_element = spatial_dim * nodes_per_element;

  // Zero-out states
  memset(global_acc, 0, sizeof(T) * ndof);
  // memset(global_mass, 0, sizeof(T) * ndof);

  // Allocate element quantities
  std::vector<T> element_mass_matrix_diagonals(dof_per_element);
  std::vector<T> element_xloc(dof_per_element);
  std::vector<T> element_dof(dof_per_element);
  std::vector<T> element_acc(dof_per_element);
  std::vector<T> element_internal_forces(dof_per_element);
  std::vector<T> element_original_xloc(dof_per_element);
  std::vector<T> element_strains(6);  // hardcoded for 3d
  std::vector<T> element_stress(6);   // hardcoded for 3d
  std::vector<T> element_total_dof(dof_per_element);
  std::vector<int> this_element_nodes(nodes_per_element);

  // 2. Compute A1 = (Fext - Fint(U1)/M

  if (global_mass[0] == 0.0) {
    // Update global mass
    for (int i = 0; i < num_elements; i++) {
      // Per element variables
      for (int k = 0; k < dof_per_element; k++) {
        element_mass_matrix_diagonals[k] = 0.0;
        element_xloc[k] = 0.0;
        element_dof[k] = 0.0;
      }

      // Get the nodes of this element
      for (int j = 0; j < nodes_per_element; j++) {
        this_element_nodes[j] = element_nodes[nodes_per_element * i + j];
      }

      // Get the element locations
      Analysis::template get_element_dof<spatial_dim>(
          this_element_nodes.data(), global_xloc, element_xloc.data());

      // Get the element degrees of freedom
      Analysis::template get_element_dof<spatial_dim>(
          this_element_nodes.data(), global_dof, element_dof.data());

      // Calculate the element mass matrix
      Analysis::element_mass_matrix(material->rho, element_xloc.data(),
                                    element_dof.data(),
                                    element_mass_matrix_diagonals.data());

      // assemble global mass
      for (int j = 0; j < nodes_per_element; j++) {
        int node = this_element_nodes[j];

        global_mass[3 * node] += element_mass_matrix_diagonals[3 * j];
        global_mass[3 * node + 1] += element_mass_matrix_diagonals[3 * j + 1];
        global_mass[3 * node + 2] += element_mass_matrix_diagonals[3 * j + 2];
      }
    }
  }

  for (int i = 0; i < num_elements; i++) {
    memset(element_mass_matrix_diagonals.data(), 0,
           sizeof(T) * dof_per_element);
    memset(element_xloc.data(), 0, sizeof(T) * dof_per_element);
    memset(element_dof.data(), 0, sizeof(T) * dof_per_element);
    memset(element_acc.data(), 0, sizeof(T) * dof_per_element);
    memset(element_internal_forces.data(), 0, sizeof(T) * dof_per_element);
    memset(element_strains.data(), 0, sizeof(T) * 6);
    memset(element_stress.data(), 0, sizeof(T) * 6);
    memset(element_total_dof.data(), 0, sizeof(T) * dof_per_element);
    memset(element_original_xloc.data(), 0, sizeof(T) * dof_per_element);

    for (int j = 0; j < nodes_per_element; j++) {
      this_element_nodes[j] = element_nodes[nodes_per_element * i + j];
    }

    // Get the element mass matrix
    Analysis::template get_element_dof<spatial_dim>(
        this_element_nodes.data(), global_mass,
        element_mass_matrix_diagonals.data());

    // Get the element locations
    Analysis::template get_element_dof<spatial_dim>(
        this_element_nodes.data(), global_xloc, element_xloc.data());

    // Get the element degrees of freedom
    Analysis::template get_element_dof<spatial_dim>(
        this_element_nodes.data(), global_dof, element_dof.data());

    T Mr_inv[dof_per_element];

    for (int k = 0; k < dof_per_element; k++) {
      Mr_inv[k] = 1.0 / element_mass_matrix_diagonals[k];
    }

    material->calculate_f_internal(element_xloc.data(), element_dof.data(),
                                   element_internal_forces.data());

    for (int j = 0; j < dof_per_element; j++) {
      element_acc[j] = Mr_inv[j] * (-element_internal_forces[j]);
    }

    // Get the original element locations
    Analysis::template get_element_dof<spatial_dim>(
        this_element_nodes.data(), mesh->xloc, element_original_xloc.data());

    for (int j = 0; j < dof_per_element; j++) {
      element_total_dof[j] =
          element_dof[j] + element_xloc[j] - element_original_xloc[j];
    }

    // Currently set up for linear element only
    T pt[3] = {0.0, 0.0, 0.0};
    Analysis::calculate_stress_strain(
        element_xloc.data(), element_total_dof.data(), pt,
        element_strains.data(), element_stress.data(), material);

    // assemble global acceleration
    for (int j = 0; j < nodes_per_element; j++) {
      int node = this_element_nodes[j];

      global_acc[3 * node] += element_acc[3 * j];
      global_acc[3 * node + 1] += element_acc[3 * j + 1];
      global_acc[3 * node + 2] += element_acc[3 * j + 2];
    }

    // assemble global strains
    for (int j = 0; j < nodes_per_element; j++) {
      for (int k = 0; k < 6; k++) {
        global_strains[6 * this_element_nodes[j] + k] = element_strains[k];
        global_stress[6 * this_element_nodes[j] + k] = element_stress[k];
      }
    }
  }

  // Add contact forces and body forces
  for (int i = 0; i < num_nodes; i++) {
    T node_pos[3];
    node_pos[0] = global_xloc[3 * i] + global_dof[3 * i];
    node_pos[1] = global_xloc[3 * i + 1] + global_dof[3 * i + 1];
    node_pos[2] = global_xloc[3 * i + 2] + global_dof[3 * i + 2];

    T node_mass[3];
    node_mass[0] = global_mass[3 * i];
    node_mass[1] = global_mass[3 * i + 1];
    node_mass[2] = global_mass[3 * i + 2];

    // Contact Forces
    wall->detect_contact(global_acc, i, node_pos, node_mass);

    // Body Forces
    int gravity_dim = 2;
    global_acc[3 * i + gravity_dim] += -9.81;
  }
}