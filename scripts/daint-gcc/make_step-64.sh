#!/usr/bin/env bash

set -e

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Set these to the appropriate source and build directories
deal_and_ceed_on_gpu_dir=${script_dir}/../../
deal_and_ceed_on_gpu_build_dir=${script_dir}/build/deal-and-ceed-on-gpu

dealii_dir=${SCRATCH}/prog/dealii
dealii_build_dir=${dealii_dir}/build/dealii
p4est_fast_dir=${dealii_dir}/build/p4est/FAST


module swap PrgEnv-cray PrgEnv-gnu
module load craype-accel-nvidia60
module load cudatoolkit/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52
module swap gcc/6.2.0 gcc/5.3.0


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
	\
	-std=c++11 \
	-arch=sm_60 \
	-x cu \
	-Xlinker -rpath,${dealii_build_dir}/lib:${p4est_fast_dir}/lib \
	-Xlinker ${dealii_build_dir}/lib/libdeal_II.so.9.2.0-pre \
	\
	${deal_and_ceed_on_gpu_dir}/bp5/step-64.cu \
	-o step-64
