# gpu_ceed

## Installation on Piz Daint

Scripts for building on Piz Daint with gcc can be found in
[scripts/daint-gcc/](scripts/daint-gcc).

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


