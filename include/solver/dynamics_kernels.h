#pragma once
#include <vector>

#include "../config/common_definitions.h"
#include "wall.h"

// The update function performs one time-step update of the dynamic system.
// It computes nodal accelerations from internal forces, assembles global
// strains and stresses, and applies external influences such as contact or body
// forces.
//
// This function is generally called at each timestep after the displacement
// increment `global_dof` is computed. It calculates new accelerations based on
// the current state of the system.
//
// Template parameters:
//   T: Numeric type (e.g. float or double)
//   spatial_dim: Spatial dimension (e.g., 3 for 3D)
//   nodes_per_element: Number of nodes per element (e.g. 4 for a tetrahedral
//   element)
//
// Arguments:
//   num_nodes        : Number of nodes in the mesh
//   num_elements     : Number of elements in the mesh
//   ndof             : Total number of degrees of freedom (num_nodes *
//   spatial_dim) dt               : Timestep size material         : Pointer to
//   the material structure containing material properties wall             :
//   Pointer to the wall/contact object (e.g. rigid contact boundary conditions)
//   mesh             : Pointer to the mesh structure (coordinates,
//   connectivity) element_nodes    : Array of node indices for each element vel
//   : Nodal velocities array global_xloc      : Current nodal coordinates
//   (updated each iteration) global_dof       : Incremental displacement for
//   this timestep (U_{n+1} - U_n) global_acc       : Nodal accelerations array
//   (to be computed/updated here) global_mass      : Nodal mass array
//   global_strains   : Nodal strain array
//   global_stress    : Nodal stress array
//   global_stress_quads             : Stress state stored at quadrature points
//   (for advanced material models) global_plastic_strain_quads     : Plastic
//   strain stored at quadrature points eqPlasticStrain, pressure, etc. :
//   Additional fields for advanced material state tracking time             :
//   Current simulation time
//
// Steps performed:
// 1. If needed, compute global mass from element mass matrices and assemble
// nodal mass.
// 2. For each element, compute internal forces, element-wise accelerations, and
// update nodal accelerations.
// 3. Assemble global strain and stress from element contributions.
// 4. Apply external forces, such as contact (wall) and gravity.
//

template <typename T, int spatial_dim, int nodes_per_element>
void update(int num_nodes, int num_elements, int ndof, T dt, Material *material,
            Wall<T, 2, Basis> *wall, Mesh<T, nodes_per_element> *mesh,
            const int *element_nodes, const T *vel, const T *global_xloc,
            const T *global_dof, T *global_acc, T *global_mass,
            T *global_strains, T *global_stress, T *global_stress_quads,
            T *global_plastic_strain_quads, T *eqPlasticStrain, T *pressure,
            T *plasticStrainRate, T *gamma, T *gamma_accumulated,
            T *yieldStress, T *plasticWork, T *internalEnergy, T *temperature,
            T *density, T time) {
  constexpr int dof_per_element = spatial_dim * nodes_per_element;

  // Zero-out accelerations at the start of each update
  memset(global_acc, 0, sizeof(T) * ndof);

  // Temporary per-element data containers
  std::vector<T> element_mass_matrix_diagonals(dof_per_element);
  std::vector<T> element_xloc(dof_per_element);
  std::vector<T> element_dof(dof_per_element);
  std::vector<T> element_acc(dof_per_element);
  std::vector<T> element_internal_forces(dof_per_element);
  std::vector<T> element_original_xloc(dof_per_element);
  std::vector<T> element_strains(
      6);  // For a 3D problem (xx, yy, zz, xy, xz, yz)
  std::vector<T> element_stress(6);  // Similarly, 6 stress components
  std::vector<T> element_stress_quads(6 * Quadrature::num_quadrature_pts);
  std::vector<T> element_plastic_strain_quads(6 *
                                              Quadrature::num_quadrature_pts);
  std::vector<T> element_total_dof(dof_per_element);
  std::vector<int> this_element_nodes(nodes_per_element);

  // Step 1: If mass is not yet computed, assemble global mass from element mass
  // matrices global_mass[0] == 0.0 is used as a heuristic to determine if
  // masses are computed yet.
  if (global_mass[0] == 0.0) {
    // Compute nodal masses once at the start
    for (int i = 0; i < num_elements; i++) {
      // Reset local arrays for this element
      for (int k = 0; k < dof_per_element; k++) {
        element_mass_matrix_diagonals[k] = 0.0;
        element_xloc[k] = 0.0;
        element_dof[k] = 0.0;
      }

      // Gather element nodes
      for (int j = 0; j < nodes_per_element; j++) {
        this_element_nodes[j] = element_nodes[nodes_per_element * i + j];
      }

      // Extract element node coordinates
      Analysis::template get_element_dof<spatial_dim>(
          this_element_nodes.data(), global_xloc, element_xloc.data());

      // Extract element nodal DOF increments
      Analysis::template get_element_dof<spatial_dim>(
          this_element_nodes.data(), global_dof, element_dof.data());

      // Compute element lumped mass matrix diagonal entries
      Analysis::element_mass_matrix(material->rho, element_xloc.data(),
                                    element_dof.data(),
                                    element_mass_matrix_diagonals.data());

      // Assemble mass contributions to global mass array
      for (int j = 0; j < nodes_per_element; j++) {
        int node = this_element_nodes[j];
        // Accumulate mass in each spatial direction (they might be identical)
        global_mass[3 * node] += element_mass_matrix_diagonals[3 * j];
        global_mass[3 * node + 1] += element_mass_matrix_diagonals[3 * j + 1];
        global_mass[3 * node + 2] += element_mass_matrix_diagonals[3 * j + 2];
      }
    }
  }

  // Step 2: Compute element internal forces, accelerations, strains, and
  // stresses for each element and assemble results into global arrays.
  for (int i = 0; i < num_elements; i++) {
    // Clear element-level arrays
    memset(element_mass_matrix_diagonals.data(), 0,
           sizeof(T) * dof_per_element);
    memset(element_xloc.data(), 0, sizeof(T) * dof_per_element);
    memset(element_dof.data(), 0, sizeof(T) * dof_per_element);
    memset(element_acc.data(), 0, sizeof(T) * dof_per_element);
    memset(element_internal_forces.data(), 0, sizeof(T) * dof_per_element);
    memset(element_strains.data(), 0, sizeof(T) * 6);
    memset(element_stress.data(), 0, sizeof(T) * 6);
    memset(element_stress_quads.data(), 0,
           sizeof(T) * 6 * Quadrature::num_quadrature_pts);
    memset(element_plastic_strain_quads.data(), 0,
           sizeof(T) * 6 * Quadrature::num_quadrature_pts);
    memset(element_total_dof.data(), 0, sizeof(T) * dof_per_element);
    memset(element_original_xloc.data(), 0, sizeof(T) * dof_per_element);

    // Gather the node indices of this element
    for (int j = 0; j < nodes_per_element; j++) {
      this_element_nodes[j] = element_nodes[nodes_per_element * i + j];
    }

    // Extract lumped mass for this element from the global arrays
    Analysis::template get_element_dof<spatial_dim>(
        this_element_nodes.data(), global_mass,
        element_mass_matrix_diagonals.data());

    // Extract element current coordinates
    Analysis::template get_element_dof<spatial_dim>(
        this_element_nodes.data(), global_xloc, element_xloc.data());

    // Extract element DOF increments
    Analysis::template get_element_dof<spatial_dim>(
        this_element_nodes.data(), global_dof, element_dof.data());

    // Load the old stress and plastic strain states at quadrature points
    // (Currently not used directly in this snippet, but typically for advanced
    // material models)
    for (int l = 0; l < Quadrature::num_quadrature_pts; l++) {
      element_stress_quads[l] =
          global_stress_quads[i * Quadrature::num_quadrature_pts * 6 + l];
      element_plastic_strain_quads[l] =
          global_plastic_strain_quads[i * Quadrature::num_quadrature_pts * 6 +
                                      l];
    }

    // Compute inverse mass for the element DOFs
    T Mr_inv[dof_per_element];
    for (int k = 0; k < dof_per_element; k++) {
      Mr_inv[k] = 1.0 / element_mass_matrix_diagonals[k];
    }

    // Compute internal (element) forces based on current configuration
    material->calculate_f_internal(element_xloc.data(), element_dof.data(),
                                   element_internal_forces.data());

    // Compute element accelerations = -M^{-1} * F_int
    for (int j = 0; j < dof_per_element; j++) {
      element_acc[j] = Mr_inv[j] * (-element_internal_forces[j]);
    }

    // Get original element positions (reference configuration)
    Analysis::template get_element_dof<spatial_dim>(
        this_element_nodes.data(), mesh->xloc, element_original_xloc.data());

    // total_dof = current displacement field = (current_x - original_x) +
    // element_dof where element_xloc is current nodal positions
    for (int j = 0; j < dof_per_element; j++) {
      element_total_dof[j] =
          element_dof[j] + element_xloc[j] - element_original_xloc[j];
    }

    // Compute strain and stress at a chosen point (here, at the centroid: pt =
    // {0,0,0})
    T pt[3] = {0.0, 0.0, 0.0};
    Analysis::calculate_stress_strain(
        element_xloc.data(), element_total_dof.data(), pt,
        element_strains.data(), element_stress.data(), material);

    // Assemble back into global acceleration
    for (int j = 0; j < nodes_per_element; j++) {
      int node = this_element_nodes[j];
      global_acc[3 * node] += element_acc[3 * j];
      global_acc[3 * node + 1] += element_acc[3 * j + 1];
      global_acc[3 * node + 2] += element_acc[3 * j + 2];
    }

    // Assemble global strains and stresses
    // Here we simply assign the element's computed strain and stress to the
    // element's nodes as a rough approximation. More sophisticated averaging
    // may be needed.
    for (int j = 0; j < nodes_per_element; j++) {
      for (int k = 0; k < 6; k++) {
        global_strains[6 * this_element_nodes[j] + k] = element_strains[k];
        global_stress[6 * this_element_nodes[j] + k] = element_stress[k];
      }
    }
  }

  // Step 3: Add external effects
  // For each node, we add contact forces (if any) and body forces like gravity.
  // global_acc has been computed from internal forces, now we correct it by
  // adding external effects.
  for (int i = 0; i < num_nodes; i++) {
    // Compute current node position (including the displacement increment)
    T node_pos[3];
    node_pos[0] = global_xloc[3 * i] + global_dof[3 * i];
    node_pos[1] = global_xloc[3 * i + 1] + global_dof[3 * i + 1];
    node_pos[2] = global_xloc[3 * i + 2] + global_dof[3 * i + 2];

    T node_mass[3];
    node_mass[0] = global_mass[3 * i];
    node_mass[1] = global_mass[3 * i + 1];
    node_mass[2] = global_mass[3 * i + 2];

    // Apply contact forces (if the node is in contact with a wall, this will
    // modify global_acc)
    wall->detect_contact(global_acc, i, node_pos, node_mass);

    // Apply gravity (assuming gravity acts in the "z" direction for example)
    int gravity_dim =
        2;  // 0:X, 1:Y, 2:Z - Adjust this depending on your coordinate system
    global_acc[3 * i + gravity_dim] +=
        -9.81;  // gravitational acceleration [m/s^2]
  }
}
