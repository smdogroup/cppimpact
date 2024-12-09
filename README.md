# GPU Impact

This project provides an explicit dynamics finite element analysis (FEA) code designed to run on both CPUs and GPUs using CUDA. It is focused on simulating hyperelastic Neo-Hookean materials on tetrahedral meshes. The code supports both linear and quadratic tetrahedral elements and uses an explicit time integration scheme suitable for highly nonlinear problems.

## Features

- **CPU and GPU Support:**  
  The code can run on both CPUs (using standard C++ and BLAS/LAPACK libraries) and GPUs (using CUDA).
  
- **Material Models:**  
  Supports linear elastic and johnson cook elastoplastic material model (WIP) 

- **Linear and Quadratic Tetrahedral Elements:**  
  Switch between linear (4-node) and quadratic (10-node) tetrahedral elements at compile time.


## Dependencies

- **C++17 or newer**: A C++17-compatible compiler is required.
- **Eigen3**: Used for linear algebra operations.  
  Make sure to have Eigen 3.3 or newer installed.
- **LAPACK**: A BLAS/LAPACK library is required for matrix operations.
- **CUDA Toolkit**: Required for building and running the GPU version.  
  Ensure that `nvcc` is available and that your GPU supports the specified architectures.

## Installation

1. **Install Eigen**:  
   On Ubuntu/Debian, for example:
   ```bash
   sudo apt-get install libeigen3-dev
   ```
   For other systems, download from [Eigen's website](https://eigen.tuxfamily.org/) and follow their instructions.

2. **Install LAPACK**:  
   On Ubuntu/Debian:
   ```bash
   sudo apt-get install liblapack-dev libblas-dev
   ```

## Building

The provided `CMakeLists.txt` file sets up two main targets: `cpu_test` and `gpu_test`, as well as debug variants for linear and quadratic elements.

1. **Configure with CMake**:
   ```bash
   mkdir build
   cd build
   cmake -DDEFAULT_BASIS=linear -DCPPIMPACT_DEBUG_MODE=on ..
   ```
   
2. **Build the code**:
   ```bash
   make cpu_test
   make gpu_test


## Running Simulations

The code expects mesh input files in the CalculiX `.inp` format.

- **PrePoMax**:  
  Generate or convert meshes using [PrePoMax](https://prepomax.fs.um.si/) or another preprocessor that can export CalculiX meshes.
  
- **Mesh Numbering**:  
  Ensure that the node and element numbering start from 1, as the code assumes 1-based indexing for nodes and elements.

## Debug and Test Executables

- **FEA_debug_linear** and **FEA_debug_quadratic**:  
  Generated for testing and debugging with various input parameters.  
  These can be run with different input and deformation cases as defined in the `CMakeLists.txt`.## GPU Impact

This project provides an explicit dynamics finite element analysis (FEA) code designed to run on both CPUs and GPUs using CUDA. It is focused on simulating hyperelastic Neo-Hookean materials on tetrahedral meshes. The code supports both linear and quadratic tetrahedral elements and uses an explicit time integration scheme suitable for highly nonlinear problems.

## Features

- **CPU and GPU Support:**  
  The code can run on both CPUs (using standard C++ and BLAS/LAPACK libraries) and GPUs (using CUDA).
  
- **Material Models:**  
  Supports linear elastic and johnson cook elastoplastic material model (WIP) 

- **Linear and Quadratic Tetrahedral Elements:**  
  Switch between linear (4-node) and quadratic (10-node) tetrahedral elements at compile time.


## Dependencies

- **C++17 or newer**: A C++17-compatible compiler is required.
- **Eigen3**: Used for linear algebra operations.  
  Make sure to have Eigen 3.3 or newer installed.
- **LAPACK**: A BLAS/LAPACK library is required for matrix operations.
- **CUDA Toolkit**: Required for building and running the GPU version.  
  Ensure that `nvcc` is available and that your GPU supports the specified architectures.

## Installation

1. **Install Eigen**:  
   On Ubuntu/Debian, for example:
   ```bash
   sudo apt-get install libeigen3-dev
   ```
   For other systems, download from [Eigen's website](https://eigen.tuxfamily.org/) and follow their instructions.

2. **Install LAPACK**:  
   On Ubuntu/Debian:
   ```bash
   sudo apt-get install liblapack-dev libblas-dev
   ```

## Building

The provided `CMakeLists.txt` file sets up two main targets: `cpu_test` and `gpu_test`, as well as debug variants for linear and quadratic elements.

1. **Configure with CMake**:
   ```bash
   mkdir build
   cd build
   cmake -DDEFAULT_BASIS=linear -DCPPIMPACT_DEBUG_MODE=on ..
   ```
   
2. **Build the code**:
   ```bash
   make cpu_test
   make gpu_test
   ```

## Running Simulations

The code expects mesh input files in the CalculiX `.inp` format.

- **PrePoMax**:  
  Generate or convert meshes using [PrePoMax](https://prepomax.fs.um.si/) or another preprocessor that can export CalculiX meshes.
  
- **Mesh Numbering**:  
  Ensure that the node and element numbering start from 1, as the code assumes 1-based indexing for nodes and elements.

## Debug and Test Executables

- **FEA_debug_linear** and **FEA_debug_quadratic**:  
  Generated for testing and debugging with various input parameters.  
  These can be run with different input and deformation cases as defined in the `CMakeLists.txt`.