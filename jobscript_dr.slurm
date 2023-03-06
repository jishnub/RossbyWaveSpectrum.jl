#!/bin/bash
#SBATCH --time="4:00:00"
#SBATCH --job-name=rosbyspec
#SBATCH --nodes=2
#SBATCH -C dalma
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8G

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
srun $julia --project=$project --startup-file=no -t 20 -e \
'include("$(ENV["SCRATCH"])/RossbyWaveSpectrum.jl/compute_rossby_spectrum.jl");
using RossbyWaveSpectrum
using RossbyWaveSpectrum.Filters
V_symmetric = !parse(Bool, ENV["SLURM_PROCID"]); # 0=>true, 1=>false
r_in_frac, r_out_frac = 0.65, 0.985
nr, nℓ = 60, 30
mrange = 1:20
viscosity = 5e11
trackingrate = :carrington
ΔΩ_scale = 1.0
Δl_cutoff = 20
n_cutoff = 20
diffrot=true
rotation_profile=:solar_latrad
modeltag = "nu$(viscosity)_rin$(round(r_in_frac, digits=3))_rout$(round(r_out_frac, digits=3))"
modeltag *= diffrot && !isone(ΔΩ_scale) ? "_rotscale$(round(ΔΩ_scale, digits=2))" : ""
eig_imag_to_real_ratio_cutoff = 3
eigvec_spectrum_power_cutoff = 0.4
filterflags = Filters.EIGEN | Filters.EIGVAL | Filters.EIGVEC | Filters.BC

ComputeRossbySpectrum.main(V_symmetric;
nr, nℓ, mrange, diffrot, rotation_profile, r_in_frac,
r_out_frac, trackingrate, ΔΩ_scale, Δl_cutoff, n_cutoff,
modeltag, eig_imag_to_real_ratio_cutoff, viscosity,
eigvec_spectrum_power_cutoff, filterflags)'
