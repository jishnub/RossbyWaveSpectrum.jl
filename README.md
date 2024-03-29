[![CI](https://github.com/jishnub/RossbyWaveSpectrum.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/jishnub/RossbyWaveSpectrum.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jishnub/RossbyWaveSpectrum.jl/graph/badge.svg?token=ohT5BvJaf8)](https://codecov.io/gh/jishnub/RossbyWaveSpectrum.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://jishnub.github.io/RossbyWaveSpectrum.jl/dev)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10091582.svg)](https://doi.org/10.5281/zenodo.10091582)

# RossbyWaveSpectrum.jl
A Julia code to compute the spectrum of solar inertial waves, including realistic solar-like differential rotation.

# Documentation
[Click here](https://jishnub.github.io/RossbyWaveSpectrum.jl/dev) for the documentation.

# Installation
To install the code, run `./INSTALL.sh`. This requires `bash` and `curl` to run, and will download and install `julia` using the installer `juliaup`. It will also install the requisite project dependencies. The project uses Julia v1.10.2, which may be installed independently as well, in which case one only needs to instantiate the environments as listed in `installjulia.sh`.

One may install the code for a specific Julia version using `./INSTALL.sh -v version`, e.g. for julia version `1.10.0` as `./INSTALL.sh -v 1.10.0`. This will instantiate the environments accordingly, and will also set up the jobscripts to use the specified version of Julia. If you also wish to resolve the environments (i.e. fetch those dependencies that are compatible with the Julia version), then you may pass the flag `-r` to `INSTALL.sh`. This may be necessary if the dependnecy versions that were used while developing the project are incompatible with the julia version that is to be installed. Note that changing the dependency versions may lead to bugs or to the results being non-replicable (although this should usually not be the case).

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

# References
[The paper on the code](https://arxiv.org/abs/2211.03323)

[The paper on including differential rotation](https://arxiv.org/abs/2308.12766)

## Citing this project
If you find this code useful in your research, we request you to cite this code, and the associated papers.
Please see the entries in `CITATION.bib` for the bibtex entries.
