#!/usr/bin/env bash

set -e

module swap PrgEnv-cray PrgEnv-gnu
module load craype-accel-nvidia60
module load cudatoolkit/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52
module swap gcc/6.2.0 gcc/5.3.0

# ATP library needed for LAPACK. Linker options not properly passed to
# CMake CUDA shared linker.
export LIBRARY_PATH=${ATP_HOME}/libApp:${LIBRARY_PATH}


# cc and CC now refer to gcc and g++
# Daint requires all programs be called with the cc and CC wrappers
# Not using cc and CC will cause failures to find MPI
export CC=$(which cc)
export CXX=$(which CC)
export F77=$(which ftn)
export FC=$(which ftn)


# Set this to the appropriate source + build root
mkdir -p ${SCRATCH}/prog
pushd ${SCRATCH}/prog

# The source / builds will be laid out

# pwd
# - dealii      (dealii source dir)
#   - build
#     - dealii  (dealii build dir)
#     - p4est   (p4est source/build/install root)

mkdir -p dealii
pushd dealii

# After the first run, comment the git command out

#git clone https://github.com/dealii/dealii.git .
git clone https://github.com/peterrum/dealii.git . --branch dealii-on-gpu


mkdir -p build
pushd build

mkdir -p p4est
pushd p4est
# After the first run, comment these lines out
wget http://p4est.github.io/release/p4est-2.2.tar.gz
../../doc/external-libs/p4est-setup.sh p4est-2.2.tar.gz $PWD
popd

mkdir -p dealii
pushd dealii

~/sw/bin/cmake \
	-DDEAL_II_WITH_CUDA=ON \
	-DDEAL_II_WITH_CXX14=OFF \
	-DDEAL_II_WITH_CXX17=OFF \
	-DDEAL_II_WITH_CXX11=ON \
	-DDEAL_II_WITH_MPI=ON \
	-DDEAL_II_MPI_WITH_CUDA_SUPPORT=ON \
	-DDEAL_II_WITH_P4EST=ON \
	-DCMAKE_CXX_COMPILER="${CXX}" \
	-DCMAKE_C_COMPILER="${CC}" \
	-DP4EST_DIR=../p4est \
	-DDEAL_II_CUDA_FLAGS="-arch=sm_60" \
	-DCMAKE_VERBOSE_MAKEFILE=ON \
	../../

make -j 20
