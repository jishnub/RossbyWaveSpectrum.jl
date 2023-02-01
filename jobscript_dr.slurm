#!/bin/bash
#SBATCH --time="1:00:00"
#SBATCH --job-name=rosbyspec
#SBATCH --nodes=1
#SBATCH -C dalma
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
##SBATCH --mem-per-cpu=6G

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
'include("$(ENV["SCRATCH"])/RossbyWaveSpectrum.jl/compute_rossby_spectrum.jl");
using RossbyWaveSpectrum
V_symmetric = !parse(Bool, ENV["SLURM_PROCID"]);
r_out_frac = 0.995;
nr=60;
nℓ=30;
mrange=1:14;
Ω0 = RossbyWaveSpectrum.equatorial_rotation_angular_velocity_surface(r_out_frac);
trackingratescaling = 1 + (2.3e-9/(Ω0/2pi));
#= trackingratescaling = 1; =#
ΔΩ_scale = 1.0;
Δl_cutoff = 20;
n_cutoff = 20;
diffrot=false
rotation_profile=:solar_latrad
modeltag = diffrot ? "rotscale$(round(ΔΩ_scale, digits=2))" : "";

ComputeRossbySpectrum.main(V_symmetric;
nr, nℓ, mrange, diffrot, rotation_profile,
r_out_frac, trackingratescaling, ΔΩ_scale, Δl_cutoff, n_cutoff,
modeltag)'
