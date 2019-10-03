# gpu_ceed

## Installation on Piz Daint

Scripts for building on Piz Daint with gcc can be found in
[scripts/daint-gcc/](scripts/daint-gcc).

To build dealii and step-64, just run
```
./scripts/daint-gcc/make_dealii.sh --download --build-p4est
./scripts/daint-gcc/make_step-64.sh
```

To resume a build of dealii, or build after a changing the source in
`build/dealii/src`, just run
```
./scripts/daint-gcc/make_dealii.sh
```

Note: LAPACK on Piz Daint is missing a needed linker flag in its config.
This problem will manifest in a failure to link the dealii shared library
and programs.
Add the option `-DLAPACK_LINKER_FLAGS="${ATP_POST_LINK_OPTS}"` to the dealii
cmake command to fix it.

Load the correct modules:

```bash
module switch PrgEnv-cray PrgEnv-gnu
module load craype-accel-nvidia60
module switch gcc/5.3.0
```

Install `CMake`:

```bash
mkdir sw
cd sw
mkdir compile
cd compile
wget https://github.com/Kitware/CMake/releases/download/v3.15.3/cmake-3.15.3.tar.Z
tar xf cmake-3.15.3.tar.Z
cd cmake-3.15.3
./bootstrap --prefix=$HOME/sw
gmake
make install
```


Get and install `deal.II` and `p4est`:

```bash
git clone https://github.com/dealii/dealii.git
cd dealii

mkdir p4est
cd p4est
wget http://p4est.github.io/release/p4est-2.2.tar.gz
F77=/opt/cray/pe/craype/2.6.0/bin/ftn FC=/opt/cray/pe/craype/2.6.0/bin/ftn CC=/opt/cray/pe/craype/2.6.0/bin/cc CXX=/opt/cray/pe/craype/2.6.0/bin/CC ../doc/external-libs/p4est-setup.sh p4est-2.2.tar.gz `pwd`


mkdir build
~/sw/bin/cmake -DDEAL_II_WITH_CUDA=ON -DDEAL_II_WITH_CXX14=OFF -D DEAL_II_WITH_MPI=ON -DDEAL_II_MPI_WITH_CUDA_SUPPORT=ON -DDEAL_II_WITH_P4EST=ON -DCMAKE_CXX_COMPILER="/opt/cray/pe/craype/2.6.0/bin/CC" -D CMAKE_C_COMPILER="/opt/cray/pe/craype/2.6.0/bin/cc" -DP4EST_DIR=../p4est -DDEAL_II_CUDA_FLAGS="-arch=sm_60" ../


 ~/sw/compile/cmake-3.15.3/bin/cmake -DDEAL_II_WITH_CUDA=ON -DDEAL_II_WITH_CXX14=OFF -D DEAL_II_WITH_MPI=ON -DDEAL_II_MPI_WITH_CUDA_SUPPORT=ON -DDEAL_II_WITH_P4EST=ON -DCMAKE_CXX_COMPILER="/opt/cray/pe/craype/2.6.0/bin/CC" -D CMAKE_C_COMPILER="/opt/cray/pe/craype/2.6.0/bin/cc" -DP4EST_DIR=../p4est -DDEAL_II_CUDA_FLAGS="-arch=sm_60" -DLAPACK_LIBRARIES="/usr/lib64/liblapack.so;cupti;cuda;rca;sci_acc_gnu_49_nv60;sci_gnu_51_mpi;sci_gnu_51;mpich_gnu_51;mpichf90_gnu_51;cudart;pthread;gfortran;quadmath;m;c;gcc_s;gcc" ../
```

Build an application:

```bash
~/sw/compile/cmake-3.15.3/bin/cmake -D DEAL_II_DIR=$HOME/dealii/build  ..
make release
make VERBOSE=1
```

Step-64 (modified):
```bash
/opt/nvidia/cudatoolkit9.1/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52/bin/nvcc -ccbin=/opt/cray/pe/craype/2.6.0/bin/CC  -DBOOST_NO_AUTO_PTR -isystem=/users/hck23/dealii/build/include -isystem=/users/hck23/dealii/include -isystem=/users/hck23/dealii/bundled/tbb-2018_U2/include -isystem=/users/hck23/dealii/bundled/boost-1.70.0/include -isystem=/users/hck23/dealii/bundled/umfpack/UMFPACK/Include -isystem=/users/hck23/dealii/bundled/umfpack/AMD/Include -isystem=/users/hck23/dealii/bundled/muparser_v2_2_4/include -isystem=/users/hck23/dealii/p4est/FAST/include  -Xcompiler "-fPIC -Wall -Wextra -Woverloaded-virtual -Wpointer-arith -Wsign-compare  -Wswitch -Wsynth -Wwrite-strings  -Wno-literal-suffix -Wno-psabi -fopenmp-simd -std=c++11 -Wno-parentheses -Wno-unused-local-typedefs -O2 -funroll-loops -funroll-all-loops -fstrict-aliasing -Wno-unused-local-typedefs" -std=c++11 -arch=sm_60 -x cu -c /users/hck23/deal-and-ceed-on-gpu/bp5/step-64.cu -o CMakeFiles/step-64.dir/step-64.cu.o

/opt/cray/pe/craype/2.6.0/bin/CC  -rdynamic -fuse-ld=gold -fopenmp  CMakeFiles/step-64.dir/step-64.cu.o -o step-64 -Wl,-rpath,/users/hck23/dealii/build/lib:/users/hck23/dealii/p4est/FAST/lib /users/hck23/dealii/build/lib/libdeal_II.so.9.2.0-pre /users/hck23/dealii/p4est/FAST/lib/libp4est.so /users/hck23/dealii/p4est/FAST/lib/libsc.so /opt/nvidia/cudatoolkit9.1/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52/lib64/libcudart.so /opt/nvidia/cudatoolkit9.1/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52/lib64/libcusparse.so /opt/nvidia/cudatoolkit9.1/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52/lib64/libcusolver.so -ldl /usr/lib64/libz.so -lrt /usr/lib64/liblapack.so -lcupti -lcuda -lrca -lsci_acc_gnu_49_nv60 -lsci_gnu_51_mpi -lsci_gnu_51 -lmpich_gnu_51 -lmpichf90_gnu_51 -lcudart -lpthread -lgfortran -lquadmath -lm -lc -lgcc_s -lgcc  -L"/opt/nvidia/cudatoolkit9.1/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52/lib64/stubs" -L"/opt/nvidia/cudatoolkit9.1/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52/lib64" -lcudadevrt -lcudart_static -lrt -lpthread -ldl
```


Step-64:
```bash
/opt/nvidia/cudatoolkit9.1/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52/bin/nvcc -ccbin=/opt/cray/pe/craype/2.6.0/bin/CC  -DBOOST_NO_AUTO_PTR -isystem=/users/hck23/dealii/build/include -isystem=/users/hck23/dealii/include -isystem=/users/hck23/dealii/bundled/tbb-2018_U2/include -isystem=/users/hck23/dealii/bundled/boost-1.70.0/include -isystem=/users/hck23/dealii/bundled/umfpack/UMFPACK/Include -isystem=/users/hck23/dealii/bundled/umfpack/AMD/Include -isystem=/users/hck23/dealii/bundled/muparser_v2_2_4/include -isystem=/users/hck23/dealii/p4est/FAST/include  -Xcompiler "-fPIC -Wall -Wextra -Woverloaded-virtual -Wpointer-arith -Wsign-compare  -Wswitch -Wsynth -Wwrite-strings  -Wno-literal-suffix -Wno-psabi -fopenmp-simd -std=c++11 -Wno-parentheses -Wno-unused-local-typedefs -O2 -funroll-loops -funroll-all-loops -fstrict-aliasing -Wno-unused-local-typedefs" -std=c++11 -arch=sm_60 -x cu -c /users/hck23/step-64/step-64.cu -o CMakeFiles/step-64.dir/step-64.cu.o

/opt/cray/pe/craype/2.6.0/bin/CC   -rdynamic -fuse-ld=gold -fopenmp  CMakeFiles/step-64.dir/step-64.cu.o -o step-64 -Wl,-rpath,/users/hck23/dealii/build/lib:/users/hck23/dealii/p4est/FAST/lib /users/hck23/dealii/build/lib/libdeal_II.so.9.2.0-pre /users/hck23/dealii/p4est/FAST/lib/libp4est.so /users/hck23/dealii/p4est/FAST/lib/libsc.so /opt/nvidia/cudatoolkit9.1/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52/lib64/libcudart.so /opt/nvidia/cudatoolkit9.1/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52/lib64/libcusparse.so /opt/nvidia/cudatoolkit9.1/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52/lib64/libcusolver.so -ldl /usr/lib64/libz.so -lrt /usr/lib64/liblapack.so -lcupti -lcuda -lrca -lsci_acc_gnu_49_nv60 -lsci_gnu_51_mpi -lsci_gnu_51 -lmpich_gnu_51 -lmpichf90_gnu_51 -lcudart -lpthread -lgfortran -lquadmath -lm -lc -lgcc_s -lgcc  -L"/opt/nvidia/cudatoolkit9.1/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52/lib64/stubs" -L"/opt/nvidia/cudatoolkit9.1/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52/lib64" -lcudadevrt -lcudart_static -lrt -lpthread -ldl
```



Using Nvprof + NVVP:
------------

We compile on Daint with cudatoolkit 9.1 due to some transitive dependencies
from pre-installed modules. However, to profile P100 GPUs with nvprof, we
need nvprof from cudatoolkit 9.2.

The following module setup should set the needed environment
```
module load daint-gpu
module swap cudatoolkit/9.2.148_3.19-6.0.7.1_2.1__g3d9acc8
```

First, generate a timeline:

```
srun nvprof -f -o profile-timeline.nvvp ./step-64
```

And then generate metrics and analysis-metrics for a kernel.
To analyze the `apply_kernel_shmem` kernel, for example, we can run
```
nvprof -f -o profile-metrics-apply_kernel_shmem.metrics --kernels ::apply_kernel_shmem: --analysis-metrics --metrics all ./step-64
```
The `--kernels` syntax is `[context]:[nvtx range]:kernel_id:[invocation]`.
You can leave the optional values blank to match all instances.

From there, you can open the profiles in NVVP.
You need to "import...", and then choose the `.nvvp` file for the timeline, the
`.metrics` file for the metrics, and include the kernel syntax in the kernels
panel.

To generate source-level statistics to see stalls, memory accesses, branching etc., add the -lineinfo flag to nvcc, and the --source-level-analysis flags to nvprof e.g.
```
nvprof -f -o profile-metrics-apply_kernel_shmem.metrics --kernels ::apply_kernel_shmem: --analysis-metrics --metrics all --source-level-analysis global_access,shared_access,branch,instruction_execution,pc_sampling ./step-64
```
Note the source level analysis will *significantly* slow down the execution time!

Displaying source-level info in nvvp requires `nvdisasm` is installed, which should be available in the cuda toolkit.
