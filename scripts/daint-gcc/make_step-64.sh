#!/usr/bin/env bash

set -e

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#
# Process command line options
#
while [ $# -ne 0 ]; do
    case $1 in
        --build-root=*)
            build_root=${1#--build-root=}
            shift
            ;;
    esac
done

# Set these to the appropriate source and build directories
: ${build_root:=${script_dir}/../../build}

deal_and_ceed_on_gpu_dir=${script_dir}/../../
deal_and_ceed_on_gpu_build_dir=${build_root}/deal-and-ceed-on-gpu

dealii_dir=${build_root}/dealii/src
dealii_build_dir=${build_root}/dealii/build
p4est_fast_dir=${build_root}/p4est/FAST


source ${script_dir}/daint-modules.sh


mkdir -p ${deal_and_ceed_on_gpu_build_dir}
pushd ${deal_and_ceed_on_gpu_build_dir}

function echo_and_run() {
	echo "$@"
	"$@"
}

echo_and_run nvcc \
	-ccbin=$(which CC) \
	-DBOOST_NO_AUTO_PTR \
	-isystem=${dealii_build_dir}/include \
	-isystem=${dealii_dir}/include \
	-isystem=${dealii_dir}/bundled/tbb-2018_U2/include \
	-isystem=${dealii_dir}/bundled/boost-1.70.0/include \
	-isystem=${dealii_dir}/bundled/umfpack/UMFPACK/Include \
	-isystem=${dealii_dir}/bundled/umfpack/AMD/Include \
	-isystem=${dealii_dir}/bundled/muparser_v2_2_4/include \
	-isystem=${p4est_fast_dir}/include \
	\
	-Xcompiler "-fPIC -Wall -Wextra -Woverloaded-virtual -Wpointer-arith -Wsign-compare -Wswitch -Wsynth -Wwrite-strings -Wno-parentheses -Wno-unused-local-typedefs -Wno-literal-suffix -Wno-psabi -Wno-unused-local-typedefs" \
	-Xcompiler "-fopenmp-simd -std=c++11 -O2 -funroll-loops -funroll-all-loops -fstrict-aliasing" \
	-Xlinker -rpath,${dealii_build_dir}/lib:${p4est_fast_dir}/lib \
	-Xlinker ${dealii_build_dir}/lib/libdeal_II.so.9.2.0-pre \
	\
	-std=c++11 \
	-arch=sm_60 \
	-x cu \
    -lineinfo \
	\
	${deal_and_ceed_on_gpu_dir}/bp5/step-64.cu \
	-o step-64
