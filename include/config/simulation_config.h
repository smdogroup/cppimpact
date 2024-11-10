#pragma once

#include "../materials/linear_elastic.h"

using T = double;
using Material = LinearElastic<T>;

// Degrees of Freedom per Node
constexpr int dof_per_node = 3;
