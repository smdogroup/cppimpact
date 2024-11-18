#pragma once

#ifdef CPPIMPACT_CUDA_BACKEND
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>

// #include <nvtx3/nvtx3.hpp>
#define CPPIMPACT_FUNCTION __host__ __device__

// Usage: put gpuErrchk(...) around cuda function calls
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

void cuda_show_kernel_error() {
  auto err = cudaGetLastError();
  std::cout << "error code: " << err << "\n";
  std::cout << "error string: " << cudaGetErrorString(err) << "\n";
}

#else
#define CPPIMPACT_FUNCTION
#endif