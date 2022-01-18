#!/bin/bash
#SBATCH --time="3:00:00"
#SBATCH --job-name=convrg
#SBATCH --ntasks=28

cd $SCRATCH/jobs
julia="$SCRATCH/julia/julia-1.7.0-rc2/bin/julia"
module purge
$julia --project="$HOME/RossbyWaveSpectrum.jl" -t 7 -e \
'include("$(ENV["HOME"])/RossbyWaveSpectrum.jl/testconvergence.jl")'
