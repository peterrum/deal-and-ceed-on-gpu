
Loaded module files upon login (via `.bashrc`)
```
module load daint-gpu
module load CMake/3.14.5 
module switch PrgEnv-cray PrgEnv-gnu
module load cudatoolkit
module load craype-accel-nvidia60
```

Compile the code as follows:

```
nvcc -ccbin=CC -std=c++11 cuda_aware_mpi.cc

Add this environment variable to allow GPUs to communicate and send messages:
export MPICH_RDMA_ENABLED_CUDA=1
srun -n 2 ./a.out
```

