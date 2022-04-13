#!/bin/bash
#SBATCH --time="50:00"
#SBATCH --job-name=rosbyspec
#SBATCH --nodes=1
#SBATCH --ntasks=128

cd $SCRATCH/jobs
julia="$HOME/.local/bin/julia"
module purge
project="$HOME/RossbyWaveSpectrum.jl"
export OMP_NUM_THREADS=$SLURM_NTASKS
$julia --project=$project -t 20 -e \
'include("$(ENV["HOME"])/RossbyWaveSpectrum.jl/compute_rossby_spectrum.jl")'
