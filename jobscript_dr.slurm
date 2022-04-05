#!/bin/bash
#SBATCH --time="40:00"
#SBATCH --job-name=rosbyspec
#SBATCH --ntasks=20

cd $SCRATCH/jobs
julia="$HOME/.local/bin/julia"
module purge
export OMP_NUM_THREADS=1
project="$HOME/RossbyWaveSpectrum.jl"
$julia --project=$project -t 20 -e \
'include("$(ENV["HOME"])/RossbyWaveSpectrum.jl/compute_rossby_spectrum.jl")'
