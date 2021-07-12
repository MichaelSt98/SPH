# HPC SPH/N-body

**High Performance Computing Smooth(ed) Particle Hydrodynamics/N-body simulations.**

## Aim

Overall aim is to write a Multi-CPU and/or Multi-GPU (thus HPC) SPH code including self-gravity targeting distributed memory systems.  

SPH includes short-range forces and self-gravity or generally gravity corresponds to long-range forces, wherefore solving the N-body problem is a milestone to achieve a SPH code including self-gravity. 

**Primary aim is a proof of concept.**

## Challenges

The problem is not trivially parallelizable, especially regarding

* distributed memory
* load balancing

since forces act over the whole domain.

In an abstract way the problem can be summarized to:  
reduce communication amount (performance) vs. reduce amount of (locally required) memory 

Some consequences:

* Brute-force self-gravity not applicable
* Collecting/Gathering information on one process need to be avoided at all costs


## Parallelization

* The **M**essage **P**assing **I**nterface (MPI) is used for the internode (and currently also for the intranode) communication
* **CUDA** is used for GPGPU programming targeting NVIDIA GPUs
* OpenMP could be used in the future for intranode optimizations

## Implementation

Self-gravity is implemented via the **Barnes-Hut method**, thus using an Octree, which can be used for the **Neighborhood-search** as well, enabling SPH.

### Multi-CPU

See [MultiCPU](MultiCPU/README.md) for more information.

**Current status:** Multi-CPU N-body & basic SPH working!

### Multi-GPU

See [MultiGPU](MultiGPU/README.md) for more information.

**Current status:** Multi-GPU N-body working, but **not** yet extended to SPH!
