# Building
This project uses CUDA, CMake 3.20+, and BLAS/LAPACK to compile.

To build CPU and GPU tests:
```
#Load CUDA and BLAS modules or libraries as needed
cmake -B build
cd build
make -j
```

To build with debug flags:
```
cmake -DCMAKE_BUILD_TYPE=Debug -B build_debug
#Proceed with build
```

To build with added GPU debugging :
```
cmake -DCMAKE_BUILD_TYPE=Debug -DCPPIMPACT_DEBUG_MODE=1 -B build_debug_gpu
#Proceed with build
```

## Running and Validating the Code

Running the `cpu_test` or `gpu_test` with the `--smoke` flag should run the first few steps of the algorithm (typically around 2000 updates). 

Run the GPU test:
```
./gpu_test --smoke
Material: AL6061
ndof: 6162
Solving dynamics
Exported ../gpu_output/simulation_0.vtk
Elapsed time: 0.758019 seconds
```

You can change the input flag using `--input`.
```
./gpu_test --input "../input/fuselage 5086 elements.inp"
```

Using the validate.sh script, you can compare your result for any code changes to the "golden" output for a simple input file. 

```
build]$ ../validation/validate.sh gpu_vel.txt ../validation/gpu_vel_smoke.txt
The files are identical.
```