#!/bin/bash
#SBATCH --time="2:00:00"
#SBATCH --job-name=const
#SBATCH --nodes=1
#SBATCH -C dalma
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=5G

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
srun $julia --project=$project --startup-file=no -t 19 -e \
'include("$(ENV["SCRATCH"])/RossbyWaveSpectrum.jl/compute_rossby_spectrum.jl");
using RossbyWaveSpectrum
using RossbyWaveSpectrum.Filters
V_symmetric = !parse(Bool, ENV["SLURM_PROCID"]); # 0=>true, 1=>false
r_in_frac, r_out_frac = 0.65, 0.985
nr, nℓ = 50, 25
mrange = 2:20
viscosity = 5e11
trackingrate = :hanson2020 # :carrington, :hanson2020, :cutoff or :surface
ΔΩ_scale = 1
Δl_cutoff = 20
n_cutoff = 15
diffrot=true
ΔΩ_frac = 467/453.1 - 1
rotation_profile = :solar_constant
# superadiabaticityparams = (; δrad = -1e-6)
modeltag = "nu$(viscosity)_rin$(round(r_in_frac, digits=3))_rout$(round(r_out_frac, digits=3))_track$(trackingrate)"
modeltag *= diffrot && !isone(ΔΩ_scale) ? "_rotscale$(round(ΔΩ_scale, digits=2))" : ""
filterflags = Filters.EIGEN | Filters.EIGVEC | Filters.EIGVAL | Filters.BC | Filters.SPATIAL

ComputeRossbySpectrum.main(V_symmetric;
nr, nℓ, mrange, diffrot, rotation_profile, r_in_frac,
r_out_frac, trackingrate, ΔΩ_scale, Δl_cutoff, n_cutoff,
modeltag, viscosity, filterflags, ΔΩ_frac)'
