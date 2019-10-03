#!/usr/bin/env bash

# Source this script to load the necessary modules on daint

module swap PrgEnv-cray PrgEnv-gnu
module load craype-accel-nvidia60
module load cudatoolkit/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52
module switch gcc/5.3.0
