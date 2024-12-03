cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(NEOHOOKEAN_CUDA LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set GPU architectures
set(GPU_ARCHS "80")
if(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
  set(GPU_ARCHS "${GPU_ARCHS};80")
endif()


# Find LAPACK
find_package(LAPACK REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# Option for debug mode
option(CPPIMPACT_DEBUG_MODE "Enable debug mode" OFF)
if(CPPIMPACT_DEBUG_MODE)
  add_definitions(-DCPPIMPACT_DEBUG_MODE)
endif()

# Enable testing
enable_testing()

# Define basis types, input files, and deformations
set(BASIS_TYPES "linear" "quadratic")
set(INPUT_FILES 0 1)
set(DEFORMATIONS 0 1)

# Option to set default basis for cpu_test and gpu_test
option(DEFAULT_BASIS "Select the default Basis type for non-FEA_debug targets (linear/quadratic)" "linear")

# Validate DEFAULT_BASIS and set corresponding macro
if(DEFAULT_BASIS STREQUAL "linear")
  set(DEFAULT_BASIS_MACRO "USE_LINEAR_BASIS")
elseif(DEFAULT_BASIS STREQUAL "quadratic")
  set(DEFAULT_BASIS_MACRO "USE_QUADRATIC_BASIS")
else()
  message(FATAL_ERROR "Invalid DEFAULT_BASIS value: ${DEFAULT_BASIS}. Choose 'linear' or 'quadratic'.")
endif()

# Loop over BASIS_TYPES to create FEA_debug executables
foreach(BASIS ${BASIS_TYPES})
  # Set the basis macro
  if(BASIS STREQUAL "linear")
    set(BASIS_MACRO "USE_LINEAR_BASIS")
  elseif(BASIS STREQUAL "quadratic")
    set(BASIS_MACRO "USE_QUADRATIC_BASIS")
  endif()

  # Create a target for each basis type
  set(TARGET_NAME "FEA_debug_${BASIS}")
  add_executable(${TARGET_NAME} FEA_debug.cu)
  # Always enable debug mode for test targets
  target_compile_definitions(${TARGET_NAME} PRIVATE -D${BASIS_MACRO} -DCPPIMPACT_DEBUG_MODE)
  target_link_libraries(${TARGET_NAME} ${LAPACK_LIBRARIES})

  # Loop over INPUT_FILES and DEFORMATIONS
  foreach(INPUT_FILE ${INPUT_FILES})
    foreach(DEFORMATION ${DEFORMATIONS})
      # Define a unique test name
      set(TEST_NAME "Test_${BASIS}_Input${INPUT_FILE}_Def${DEFORMATION}")

      # Add test
      add_test(NAME ${TEST_NAME}
               COMMAND ${TARGET_NAME} ${INPUT_FILE} ${DEFORMATION}
               WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    endforeach()
  endforeach()
endforeach()

# C++ target (cpu_test)
add_executable(cpu_test test.cu)
target_link_libraries(cpu_test ${LAPACK_LIBRARIES} Eigen3::Eigen)
target_compile_definitions(cpu_test PRIVATE -D${DEFAULT_BASIS_MACRO} -DCPPIMPACT_DEBUG_MODE)

# GPU target (gpu_test)
add_executable(gpu_test test.cu)
set_target_properties(gpu_test PROPERTIES CUDA_ARCHITECTURES ${GPU_ARCHS})
set_target_properties(gpu_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_test PUBLIC Eigen3::Eigen)
target_compile_definitions(gpu_test PRIVATE -D${DEFAULT_BASIS_MACRO} -DCPPIMPACT_CUDA_BACKEND -DCPPIMPACT_DEBUG_MODE)
target_compile_options(gpu_test PRIVATE -G)  # device debug
