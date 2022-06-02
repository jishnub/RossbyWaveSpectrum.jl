#!/bin/bash
#SBATCH --time="8:00:00"
#SBATCH --job-name=rosbyspec
#SBATCH --nodes=1
#SBATCH -C dalma
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28

cd $SCRATCH/jobs
julia="$HOME/.local/bin/julia"
module purge
project="$HOME/RossbyWaveSpectrum.jl"
export MKL_DYNAMIC="FALSE"
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
$julia --project=$project -t 15 -e \
'include("$(ENV["HOME"])/RossbyWaveSpectrum.jl/compute_rossby_spectrum.jl")'
