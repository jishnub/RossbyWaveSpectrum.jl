#!/bin/bash
#SBATCH --time="2:00:00"
#SBATCH --job-name=rosdr
#SBATCH -o rossbydiffrot.out
#SBATCH -e rossbydiffrot.err
#SBATCH --ntasks=28

cd $SCRATCH/jobs
julia="$SCRATCH/julia/julia-1.7.0-rc1/bin/julia"
module purge
export MKL_NUM_THREADS=28
$julia --project="$HOME/RossbyWaveSpectrum.jl" -e 'include("$(ENV["HOME"])/RossbyWaveSpectrum.jl/compute_rossby_spectrum.jl")'
