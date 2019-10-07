# deal-on-gpu

This is the project repo of the `deal-on-gpu` team at [EuroHack19](https://www.cscs.ch/events/upcoming-events/event-detail/eurohack19-gpu-programming-hackathon/) in Lugano, Switzerland. 

The presentation of the final results can be found [here](https://github.com/vkarak/Eurohack19/wiki/presentations/deal-on-gpu.pdf).

## Team members

The team `deal-on-gpu` consisted of (in alphabetical order):

* [Momme Allalen](https://github.com/mallalen)
* [Paddy Ó Conbhuı́](https://github.com/poconbhui) (mentor)
* [Prashanth Kanduri](https://github.com/kanduri) (mentor)
* [Martin Kronbichler](https://github.com/kronbichler)
* [Peter Munch](https://github.com/peterrum)

We profited from the dedicated work by:
* [Daniel Arndt](https://github.com/masterleinad)
* [Karl Ljungkvist](https://github.com/kalj)
* [Bruno Turcksin](https://github.com/Rombur)

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

If you have the dealii source in a different directory, use the
`--dealii-source-dir=<dealii source>` option when running `make_dealii.sh`.
Change the build root with the `--build-root=<build root>` option for both
`make_dealii.sh` and `make_step-64.sh`.

Note: LAPACK on Piz Daint is missing a needed linker flag in its config.
This problem will manifest in a failure to link the dealii shared library
and programs.
Add the option `-DLAPACK_LINKER_FLAGS="${ATP_POST_LINK_OPTS}"` to the dealii
cmake command to fix it.

## Using Nvprof + NVVP:


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
