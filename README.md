[![CI](https://github.com/jishnub/RossbyWaveSpectrum.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/jishnub/RossbyWaveSpectrum.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jishnub/RossbyWaveSpectrum.jl/graph/badge.svg?token=ohT5BvJaf8)](https://codecov.io/gh/jishnub/RossbyWaveSpectrum.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://jishnub.github.io/RossbyWaveSpectrum.jl/dev)

# RossbyWaveSpectrum.jl
Julia code to compute the spectrum of solar intertial waves

# Documentation
[Click here](https://jishnub.github.io/RossbyWaveSpectrum.jl/dev) for the documentation.

# Installation
To install the code, run `./INSTALL.sh`. This requires `bash` and `curl` to run, and will download and install `julia` using the installer `juliaup`. It will also install the requisite project dependencies. The project uses Julia v1.10.0, which may be installed independently as well, in which case one only needs to instantiate the environments as listed in `INSTALL.sh`.

Users running the code on a heterogeneous cluster (where the login and compute nodes have different CPU architectures) may want to set the environment variable `JULIA_CPU_TARGET` appropriately in their shell rc file. One may check the CPU architecture by running
```julia
julia> Sys.CPU_NAME
"skylake-avx512"
```
in the julia REPL on each node.

As an example, a valid setting may be
```
export JULIA_CPU_TARGET="generic;skylake-avx512,clone_all;icelake-server,clone_all"
```
if e.g. the login node has `skylake-avx512` and the compute node has `icelake-server`. Setting this is particularly important if one node has Intel processors whereas the other has AMD ones.

# Quick start
Check `compute_rossby_spectrum.jl` for the steps to run the code. This is typically run on an HPC cluster, although it may be run equally well on a laptop or a standalone computer. Sample jobscript files for a Slurm cluster (`jobscript.slurm`) and a PBS cluster (`jobscript.qsub`) are provided.

