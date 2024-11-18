#pragma once
#include <string>

#include "../utils/cppimpact_defs.h"

template <typename T, int dim, class Basis>
class Wall {
 public:
  // dim denotes the plane which the wall is parallel with (0, 1, 2) for (x, y,
  // z) respectively. normal denotes which way direction the surface of the wall
  // faces 1 for positive x/y/z, -1 for negative x/y/z

  std::string name;
  T location;
  T stiffness;
  int *slave_node_indices;
  int num_slave_nodes;
  int normal;
  T norm_stiffness;
  // static_assert(normal == 1 || normal == -1, "Normal must be either 1 or
  // -1.");

  Wall(std::string name, T location, T stiffness, int *slave_node_indices,
       int num_slave_nodes, int normal)
      : name(name),
        location(location),
        stiffness(stiffness),
        slave_node_indices(slave_node_indices),
        num_slave_nodes(num_slave_nodes),
        normal(normal) {
    norm_stiffness = normal * stiffness;
  }

  ~Wall() {}

  // TODO: Use slave nodes
  CPPIMPACT_FUNCTION void detect_contact(T *global_acc, int node_idx,
                                         T *node_pos, T *node_mass) {
    T wall_distance = (node_pos[dim] - location) * normal;
    if (wall_distance < 0.0) {
      global_acc[3 * (node_idx) + dim] +=
          -1 * (1 / node_mass[dim]) * wall_distance * norm_stiffness;
    }
  }
};