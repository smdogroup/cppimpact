#pragma once

#include "../config/common_definitions.h"
#include "analysis.h"
#include "physics.h"
#include "tetrahedral.h"
#include "wall.h"

// The update kernel performs the element-level computations needed at each
// timestep. For each element, it:
// 1. Gathers element nodal data from global arrays.
// 2. Computes element mass matrix (lumped), and on the first step, assembles
// nodal mass.
// 3. Computes internal forces, strains, and stresses for the element.
// 4. Accumulates element contributions into global acceleration, strain, and
// stress arrays.
//
// Template parameters:
//   T             : Floating-point type (e.g., float or double)
//   spatial_dim   : Number of spatial dimensions (e.g., 3 for 3D)
//   nodes_per_element : Number of nodes per element (e.g., 4 for a tetrahedron)
//
// Arguments:
//   num_elements       : Number of elements in the mesh
//   dt                 : Time step size
//   d_material         : Pointer to material properties on device
//   d_wall             : Pointer to wall (contact) data on device
//   d_element_nodes    : Device array of element connectivity (node indices per
//   element) d_vel              : Device array of nodal velocities
//   d_global_xloc      : Device array of current nodal coordinates
//   d_global_dof       : Device array of incremental nodal displacements for
//   this timestep d_global_acc       : Device array of nodal accelerations (to
//   be updated here) d_global_mass      : Device array of nodal masses
//   d_global_strains   : Device array of nodal strains
//   d_global_stress    : Device array of nodal stresses
//   nodes_per_elem_num_quad : A parameter used in mass matrix calculation
//   (e.g., #quad pts * nodes_per_element) time               : Current
//   simulation time
//
// Note: Uses shared memory to hold element-level data. One block processes one
// element.
template <typename T, int spatial_dim, int nodes_per_element>
__global__ void update(int num_elements, T dt, Material *d_material,
                       Wall<T, 2, Basis> *d_wall, const int *d_element_nodes,
                       const T *d_vel, const T *d_global_xloc,
                       const T *d_global_dof, T *d_global_acc, T *d_global_mass,
                       T *d_global_strains, T *d_global_stress,
                       const int nodes_per_elem_num_quad, T time) {
  constexpr int dof_per_element = spatial_dim * nodes_per_element;

  int elem = blockIdx.x;
  if (elem >= num_elements) return;

  // Shared memory arrays for storing element-level calculations
  __shared__ T element_mass_matrix_diagonals[dof_per_element];
  __shared__ T element_xloc[dof_per_element];
  __shared__ T element_dof[dof_per_element];
  __shared__ T element_acc[dof_per_element];
  __shared__ T element_vel[dof_per_element];
  __shared__ T element_internal_forces[dof_per_element];
  __shared__ T element_strain[nodes_per_element *
                              6];  // strain for each node: 6 comps in 3D
  __shared__ T element_stress[nodes_per_element *
                              6];  // stress for each node: 6 comps in 3D
  __shared__ int this_element_nodes[nodes_per_element];

  int tid = threadIdx.x;

  // Initialize shared memory
  if (tid < dof_per_element) {
    element_mass_matrix_diagonals[tid] = 0.0;
    element_xloc[tid] = 0.0;
    element_dof[tid] = 0.0;
    element_acc[tid] = 0.0;
    element_vel[tid] = 0.0;
    element_internal_forces[tid] = 0.0;
  }
  if (tid < nodes_per_element * 6) {
    element_strain[tid] = 0.0;
    element_stress[tid] = 0.0;
  }
  __syncthreads();

  // Gather the nodes for this element from global connectivity
  if (tid < nodes_per_element) {
    this_element_nodes[tid] = d_element_nodes[nodes_per_element * elem + tid];
  }
  __syncthreads();

  // Extract element nodal coordinates into element_xloc
  Analysis::template get_element_dof<
      Analysis::spatial_dim, Analysis::dof_per_element, Analysis::dof_per_node>(
      tid, this_element_nodes, d_global_xloc, element_xloc);

  // Extract element displacement increments into element_dof
  Analysis::template get_element_dof<
      Analysis::spatial_dim, Analysis::dof_per_element, Analysis::dof_per_node>(
      tid, this_element_nodes, d_global_dof, element_dof);

  __syncthreads();

  // Compute element mass matrix entries (lumped)
  Analysis::element_mass_matrix_gpu(tid, d_material->rho, element_xloc,
                                    element_dof, element_mass_matrix_diagonals,
                                    nodes_per_elem_num_quad);

  // Assemble nodal mass on the first timestep (time == 0.0)
  int node = INT_MAX;
  int j = tid / 3;  // node index in element: 0 <= j < nodes_per_element
  int k = tid % 3;  // component index: 0=x,1=y,2=z
  if (tid < dof_per_element) {
    node = this_element_nodes[j];
  }

  // If it's the first step, add mass contributions atomically to the global
  // mass array
  if (time == 0.0 && tid < dof_per_element) {
    atomicAdd(&d_global_mass[3 * node + k],
              element_mass_matrix_diagonals[3 * j + k]);
  }
  __syncthreads();

  // Reset element_mass_matrix_diagonals for next operation
  if (tid < dof_per_element) {
    element_mass_matrix_diagonals[tid] = 0.0;
  }
  __syncthreads();

  // Now fetch lumped mass diagonal again from global arrays
  // This time they contain the assembled nodal mass
  Analysis::template get_element_dof<
      Analysis::spatial_dim, Analysis::dof_per_element, Analysis::dof_per_node>(
      tid, this_element_nodes, d_global_mass, element_mass_matrix_diagonals);
  __syncthreads();

  // Compute inverse mass for this DOF
  T Mr_inv = 0.0;
  if (tid < dof_per_element) {
    Mr_inv = 1.0 / element_mass_matrix_diagonals[tid];
  }

  // Compute internal element forces based on current configuration
  d_material->calculate_f_internal(element_xloc, element_dof,
                                   element_internal_forces);
  __syncthreads();

  // Compute element accelerations = -M_inv * F_int
  if (tid < dof_per_element) {
    element_acc[tid] = Mr_inv * (-element_internal_forces[tid]);
  }
  __syncthreads();

  // Assemble element accelerations into global acceleration array
  if (tid < dof_per_element) {
    atomicAdd(&d_global_acc[3 * node + k], element_acc[3 * j + k]);
  }
  __syncthreads();

  // Compute strain and stress at a representative point in the element
  // origin is always used because we are using linear basis functions
  if (tid == 0) {
    T pt[3] = {0.0, 0.0, 0.0};
    Analysis::calculate_stress_strain(element_xloc, element_dof, pt,
                                      element_strain, element_stress,
                                      d_material);
  }
  __syncthreads();

  // Write computed strain and stress back to global arrays (one strain/stress
  // set per element node) Here tid < 24 corresponds to up to 4 nodes * 6
  // components = 24 entries in total (for a tetra element)
  if (tid < 24) {
    int node_idx = tid / 6;  // which node within the element
    int comp_idx = tid % 6;  // which component of strain/stress
    int global_node_idx = this_element_nodes[node_idx];

    d_global_strains[global_node_idx * 6 + comp_idx] = element_strain[comp_idx];
    d_global_stress[global_node_idx * 6 + comp_idx] = element_stress[comp_idx];
  }
  __syncthreads();
}

// external_forces kernel applies external forces (like contact and gravity) to
// nodes. Each thread processes one node and updates d_global_acc accordingly.
//
// Arguments:
//   num_nodes     : Number of nodes
//   d_wall        : Pointer to wall object (for contact forces)
//   d_global_xloc : Current node coordinates
//   d_global_dof  : Current displacement increments at each node
//   d_global_mass : Node mass array
//   d_global_acc  : Node acceleration array (updated here)
template <typename T>
__global__ void external_forces(int num_nodes, Wall<T, 2, Basis> *d_wall,
                                const T *d_global_xloc, const T *d_global_dof,
                                const T *d_global_mass, T *d_global_acc) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int node_idx = blockDim.x * bid + tid;
  if (node_idx >= num_nodes) return;

  int node3 = 3 * node_idx;
  int node3p1 = node3 + 1;
  int node3p2 = node3 + 2;

  // Compute updated node position
  T node_pos[3];
  node_pos[0] = d_global_xloc[node3] + d_global_dof[node3];
  node_pos[1] = d_global_xloc[node3p1] + d_global_dof[node3p1];
  node_pos[2] = d_global_xloc[node3p2] + d_global_dof[node3p2];

  T node_mass[3];
  node_mass[0] = d_global_mass[node3];
  node_mass[1] = d_global_mass[node3p1];
  node_mass[2] = d_global_mass[node3p2];

  // Apply contact forces via wall object
  d_wall->detect_contact(d_global_acc, node_idx, node_pos, node_mass);

  __syncthreads();

  // Apply gravity in the vertical direction (e.g., z-direction)
  constexpr int gravity_dim = 2;
  d_global_acc[node3 + gravity_dim] += -9.81;
  __syncthreads();
}

// update_velocity kernel applies the initial staggering of velocities at the
// start of the simulation. It updates velocities by adding (dt/2)*acc to
// implement a leapfrog or central difference scheme.
//
// Arguments:
//   ndof          : Total number of degrees of freedom
//   dt            : Time step size
//   d_vel         : Device array of nodal velocities (updated here)
//   d_global_acc  : Device array of nodal accelerations
template <typename T>
__global__ void update_velocity(int ndof, T dt, T *d_vel, T *d_global_acc) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int ndof_idx = blockDim.x * bid + tid;
  if (ndof_idx < ndof) {
    d_vel[ndof_idx] += 0.5 * dt * d_global_acc[ndof_idx];
  }
  __syncthreads();
}

// update_dof kernel calculates incremental displacement: d_global_dof = dt *
// vel
//
// Arguments:
//   ndof         : Number of degrees of freedom
//   dt           : Time step size
//   d_vel        : Current velocity array
//   d_global_dof : Displacement increment array to be updated
template <typename T>
__global__ void update_dof(int ndof, T dt, T *d_vel, T *d_global_dof) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int ndof_idx = blockDim.x * bid + tid;
  if (ndof_idx < ndof) {
    d_global_dof[ndof_idx] = dt * d_vel[ndof_idx];
  }
  __syncthreads();
}

// timeloop_update kernel updates nodal positions and velocities for the next
// timestep. Additionally, it computes an intermediate velocity d_vel_i for
// output if needed.
//
// Arguments:
//   ndof          : Number of degrees of freedom
//   dt            : Time step size
//   d_global_xloc : Nodal coordinates (updated here)
//   d_vel         : Nodal velocities (updated here)
//   d_global_acc  : Nodal accelerations
//   d_vel_i       : Intermediate velocity array used for exporting data
//   d_global_dof  : Displacement increments
//
// The scheme used is something akin to a central difference update:
//   U_{n+1} = U_n + dt * V_{n+1/2}
//   V_{n+1} = V_{n+1/2} + (dt/2)*A_{n+1}
//   (and vel_i = V_{n+1} - 0.5*dt*A_{n+1} for output)
template <typename T>
__global__ void timeloop_update(int ndof, T dt, T *d_global_xloc, T *d_vel,
                                T *d_global_acc, T *d_vel_i, T *d_global_dof) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int ndof_idx = blockDim.x * bid + tid;
  if (ndof_idx < ndof) {
    // Update global positions
    d_global_xloc[ndof_idx] += d_global_dof[ndof_idx];

    // Update velocity with full acceleration contribution
    d_vel[ndof_idx] += dt * d_global_acc[ndof_idx];

    // Compute intermediate velocity (vel_i) for output/visualization
    // This is typically V_{n+1/2} re-centered or similar scheme
    d_vel_i[ndof_idx] = d_vel[ndof_idx] - 0.5 * dt * d_global_acc[ndof_idx];
  }
  __syncthreads();
}
