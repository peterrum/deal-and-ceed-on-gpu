#!/usr/bin/env bash

set -e

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#
# Process command line options
#
while [ $# -ne 0 ]; do
    case $1 in
        --download-sources)
            download_sources=--download-sources
            shift
            ;;
        --no-download-sources)
            download_sources=--no-download-sources
            shift
            ;;
        --build-p4est)
            build_p4est=--build-p4est
            shift
            ;;
        --no-build-p4est)
            build_p4est=--no-build-p4est
            shift
            ;;
        --build-root=*)
            build_root=${1#--build-root=}
            shift
            ;;
        --dealii-source-dir=*)
            dealii_source_dir=${1#--dealii-source-dir=}
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
    esac
done


#
# Default values for command line options
#
: ${build_root:=${script_dir}/../../build}
: ${dealii_source_dir:=${build_root}/dealii/src}
: ${dealii_build_dir:=${build_root}/dealii/build}
: ${p4est_dir:=${build_root}/p4est}

: ${download_sources:=--no-download-sources}
: ${build_p4est:=--no-build-p4est}


source ${script_dir}/daint-modules.sh

# ATP library needed for LAPACK. Linker options not properly passed to
# CMake CUDA shared linker.
#export LIBRARY_PATH=${ATP_HOME}/libApp:${LIBRARY_PATH}


# cc and CC now refer to gcc and g++
# Daint requires all programs be called with the cc and CC wrappers
# Not using cc and CC will cause failures to find MPI
export CC=$(which cc)
export CXX=$(which CC)
export F77=$(which ftn)
export FC=$(which ftn)


# Set this to the appropriate source + build root

# The default source / builds will be laid out

# build_root
# - dealii      (dealii directory)
#   - src       (dealii source directory)
#   - build
# - p4est       (p4est source/build/install root)

mkdir -p ${dealii_source_dir} ${dealii_build_dir} ${p4est_dir}


#
# Download dealii and p4est
#
if [ ${download_sources} = --download-sources ]
then
    cd ${dealii_source_dir}
    #git clone https://github.com/dealii/dealii.git .
    git clone https://github.com/peterrum/dealii.git --branch dealii-on-gpu .

    cd ${p4est_dir}
    wget http://p4est.github.io/release/p4est-2.2.tar.gz
fi


#
# Build p4est
#
if [ ${build_p4est} = --build-p4est ]
then
    ${dealii_source_dir}/doc/external-libs/p4est-setup.sh p4est-2.2.tar.gz $PWD
fi


#
# Configure & build dealii
#
cd ${dealii_build_dir}

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
    -DP4EST_DIR=${p4est_dir} \
    -DDEAL_II_CUDA_FLAGS="-arch=sm_60" \
    -DLAPACK_LINKER_FLAGS="${ATP_POST_LINK_OPTS}" \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    ${dealii_source_dir}

make -j 20
