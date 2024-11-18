#pragma once

#include "../config/common_definitions.h"
#include "analysis.h"
#include "physics.h"
#include "tetrahedral.h"
#include "wall.h"

// Update the system states via time-stepping
template <typename T, int spatial_dim, int nodes_per_element>
__global__ void update(int num_elements, T dt, Material *d_material,
                       Wall<T, 2, Basis> *d_wall, const int *d_element_nodes,
                       const T *d_vel, const T *d_global_xloc,
                       const T *d_global_dof, T *d_global_acc, T *d_global_mass,
                       T *d_global_strains, T *d_global_stress,
                       const int nodes_per_elem_num_quad, T time) {
  int constexpr dof_per_element = spatial_dim * nodes_per_element;

  int elem = blockIdx.x;
  if (elem >= num_elements) return;

  __shared__ T element_mass_matrix_diagonals[dof_per_element];
  __shared__ T element_xloc[dof_per_element];
  __shared__ T element_dof[dof_per_element];
  __shared__ T element_acc[dof_per_element];
  __shared__ T element_vel[dof_per_element];
  __shared__ T element_internal_forces[dof_per_element];
  __shared__ T element_strain[nodes_per_element * 6];
  __shared__ T element_stress[nodes_per_element * 6];
  __shared__ int this_element_nodes[nodes_per_element];

  int tid = threadIdx.x;

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

  // Get the nodes of this element
  if (tid < nodes_per_element) {
    this_element_nodes[tid] = d_element_nodes[nodes_per_element * elem + tid];
  }
  __syncthreads();

  // Get the element locations
  Analysis::template get_element_dof<
      Analysis::spatial_dim, Analysis::dof_per_element, Analysis::dof_per_node>(
      tid, this_element_nodes, d_global_xloc, element_xloc);

  // Get the element degrees of freedom
  Analysis::template get_element_dof<
      Analysis::spatial_dim, Analysis::dof_per_element, Analysis::dof_per_node>(
      tid, this_element_nodes, d_global_dof, element_dof);

  __syncthreads();
  // Calculate element mass matrix
  Analysis::element_mass_matrix_gpu(tid, d_material->rho, element_xloc,
                                    element_dof, element_mass_matrix_diagonals,
                                    nodes_per_elem_num_quad);

  int node = INT_MAX;
  int j = tid / 3;  // 0 ~ nodes_per_element - 1
  int k = tid % 3;  // 0, 1, 2
  if (tid < dof_per_element) {
    node = this_element_nodes[j];
  }
  if (time == 0.0) {
    if (tid < dof_per_element) {
      atomicAdd(&d_global_mass[3 * node + k],
                element_mass_matrix_diagonals[3 * j + k]);
    }
  }
  __syncthreads();

  if (tid < dof_per_element) {
    element_mass_matrix_diagonals[tid] = 0.0;
  }
  __syncthreads();

  // Get the element mass matrix after assembly
  Analysis::template get_element_dof<
      Analysis::spatial_dim, Analysis::dof_per_element, Analysis::dof_per_node>(
      tid, this_element_nodes, d_global_mass, element_mass_matrix_diagonals);
  __syncthreads();

  T Mr_inv = 0.0;
  if (tid < dof_per_element) {
    Mr_inv = 1.0 / element_mass_matrix_diagonals[tid];
  }

  d_material->calculate_f_internal(element_xloc, element_dof,
                                   element_internal_forces);
  __syncthreads();

  // Calculate element acceleration
  if (tid < dof_per_element) {
    element_acc[tid] = Mr_inv * (-element_internal_forces[tid]);
  }
  __syncthreads();

  // assemble global acceleration
  if (tid < dof_per_element) {
    atomicAdd(&d_global_acc[3 * node + k], element_acc[3 * j + k]);
  }
  __syncthreads();

  if (tid == 0) {
    T pt[3] = {0.0, 0.0, 0.0};
    Analysis::calculate_stress_strain(element_xloc, element_dof, pt,
                                      element_strain, element_stress,
                                      d_material);
  }
  __syncthreads();
  if (tid < 24) {
    int node_idx = tid / 6;  // Node index within the element (0 to 3)
    int comp_idx = tid % 6;  // Strain/stress component index (0 to 5)

    int global_node_idx = this_element_nodes[node_idx];

    // Assign strain component
    d_global_strains[global_node_idx * 6 + comp_idx] = element_strain[comp_idx];

    // Assign stress component
    d_global_stress[global_node_idx * 6 + comp_idx] = element_stress[comp_idx];
  }
  __syncthreads();
}

// update d_global_acc
template <typename T>
__global__ void external_forces(int num_nodes, Wall<T, 2, Basis> *d_wall,
                                const T *d_global_xloc, const T *d_global_dof,
                                const T *d_global_mass, T *d_global_acc) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int node_idx = blockDim.x * bid + tid;
  int node3 = 3 * node_idx;
  int node3p1 = 3 * node_idx + 1;
  int node3p2 = 3 * node_idx + 2;

  if (node_idx < num_nodes) {
    T node_pos[3];
    node_pos[0] = d_global_xloc[node3] + d_global_dof[node3];
    node_pos[1] = d_global_xloc[node3p1] + d_global_dof[node3p1];
    node_pos[2] = d_global_xloc[node3p2] + d_global_dof[node3p2];

    T node_mass[3];
    node_mass[0] = d_global_mass[node3];
    node_mass[1] = d_global_mass[node3p1];
    node_mass[2] = d_global_mass[node3p2];

    d_wall->detect_contact(d_global_acc, node_idx, node_pos, node_mass);
  }
  __syncthreads();

  if (node_idx < num_nodes) {
    constexpr int gravity_dim = 2;
    d_global_acc[3 * node_idx + gravity_dim] += -9.81;
  }
  __syncthreads();
}

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

template <typename T>
__global__ void timeloop_update(int ndof, T dt, T *d_global_xloc, T *d_vel,
                                T *d_global_acc, T *d_vel_i, T *d_global_dof) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int ndof_idx = blockDim.x * bid + tid;
  if (ndof_idx < ndof) {
    d_global_xloc[ndof_idx] += d_global_dof[ndof_idx];
    d_vel[ndof_idx] += dt * d_global_acc[ndof_idx];

    // TODO: only run this on export steps
    d_vel_i[ndof_idx] = d_vel[ndof_idx] - 0.5 * dt * d_global_acc[ndof_idx];
  }
  __syncthreads();
}
