[![CI](https://github.com/jishnub/RossbyWaveSpectrum.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/jishnub/RossbyWaveSpectrum.jl/actions/workflows/ci.yml)

# RossbyWaveSpectrum.jl
Julia code to compute the spectrum of solar intertial waves

# Installation
To install the code, run `./INSTALL.sh`. This requires `bash` and `curl` to run, and will download and install `julia` using the installer `juliaup`. It will also install the requisite project dependencies.

# Quick start
Check `compute_rossby_spectrum.jl` for the steps to run the code. This is typically run on an HPC cluster, although it may be run equally well on a laptop or a standalone computer. Sample jobscript files for a Slurm cluster (`jobscript.slurm`) and a PBS cluster (`jobscript.qsub`) are provided.

