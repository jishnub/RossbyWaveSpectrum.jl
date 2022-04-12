#!/bin/bash
#SBATCH --time="2:00:00"
#SBATCH --job-name=rosbyspec
#SBATCH --ntasks=128

cd $SCRATCH/jobs
julia="$HOME/.local/bin/julia"
module purge
project="$HOME/RossbyWaveSpectrum.jl"
$julia --project=$project -t 20 -e \
'include("$(ENV["HOME"])/RossbyWaveSpectrum.jl/compute_rossby_spectrum.jl")'
