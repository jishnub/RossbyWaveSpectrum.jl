#!/bin/bash
#SBATCH --time="12:00:00"
#SBATCH --job-name=rosbyspec
#SBATCH --nodes=2
#SBATCH -C dalma
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28

echo "############"
hostname
echo $JULIA_DEPOT_PATH
echo "############"
cd $SCRATCH/jobs
julia="$HOME/.local/bin/julia-1.8"
module purge
project="$SCRATCH/RossbyWaveSpectrum.jl"
export MKL_DYNAMIC="FALSE"
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun $julia --project=$project --startup-file=no -t 15 -e \
'include("$(ENV["SCRATCH"])/RossbyWaveSpectrum.jl/compute_rossby_spectrum.jl"); ComputeRossbySpectrum.main()'
