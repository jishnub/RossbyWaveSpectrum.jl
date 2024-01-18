#!/bin/bash
#SBATCH --time="1:00:00"
#SBATCH --job-name=solar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -C dalma
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G

echo "############"
hostname
echo $JULIA_DEPOT_PATH
echo "############"
juliapath="$HOME/.juliaup/bin/julia"
module purge
project="$HOME/RossbyWaveSpectrum.jl" # set this path to wherever the code is located
export MKL_DYNAMIC="FALSE"
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun time $juliapath +1.10.0 --project=$project --startup-file=no -t 5 $project/jobscript.jl
