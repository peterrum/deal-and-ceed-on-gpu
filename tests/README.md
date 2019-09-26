
Loaded module files upon login (via `.bashrc`)
```
module load daint-gpu
module load CMake/3.14.5 
module switch PrgEnv-cray PrgEnv-gnu
module load cudatoolkit
```

Compile the code as follows:

```
nvcc -ccbin=CC -std=c++11 cuda_aware_mpi.cc
srun -n 2 ./a.out
```

