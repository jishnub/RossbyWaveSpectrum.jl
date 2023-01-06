#!/bin/bash
#SBATCH --time="2:00:00"
#SBATCH --job-name=rosbyspec
#SBATCH --nodes=1
#SBATCH -C dalma
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28

echo "############"
hostname
echo $JULIA_DEPOT_PATH
echo "############"
cd $SCRATCH/jobs
julia="$HOME/.local/bin/julia-1.9"
module purge
project="$SCRATCH/RossbyWaveSpectrum.jl"
export MKL_DYNAMIC="FALSE"
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun $julia --project=$project --startup-file=no -t 14 -e \
'include("$(ENV["SCRATCH"])/RossbyWaveSpectrum.jl/compute_rossby_spectrum.jl");\
ComputeRossbySpectrum.main(parse(Bool, ENV["SLURM_PROCID"]); nr=40, nℓ=20, mrange=1:14)'
