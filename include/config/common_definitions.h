#pragma once

#include "../materials/linear_elastic.h"
#include "../solver/analysis.h"
#include "../solver/physics.h"
#include "../solver/tetrahedral.h"
#include "../utils/cppimpact_defs.h"

// Define the scalar type
using T = double;

// Basis Type
#if defined(USE_LINEAR_BASIS)
using Basis = TetrahedralBasisLinear<T>;

#elif defined(USE_QUADRATIC_BASIS)
using Basis = TetrahedralBasisQuadratic<T>;

#else
#warning \
    "No Basis type defined. Using default TetrahedralBasisLinear<T> as Basis."
using Basis = TetrahedralBasisLinear<T>;
#endif

// Quadrature Type
using Quadrature = TetrahedralQuadrature5pts;

// Physics Type
using Physics = NeohookeanPhysics<T>;

// Material Type
using Material = LinearElastic<T, Basis, Quadrature>;

// Analysis Type
using Analysis = FEAnalysis<T, Basis, Quadrature, Physics, Material>;
