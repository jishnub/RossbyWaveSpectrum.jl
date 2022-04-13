@time using RossbyWaveSpectrum
using LinearAlgebra

nr = 60;
nℓ = 30;
mrange = 1:20;

# test
# nr = 15;
# nℓ = 15;
# mrange = 1:1;

# boundary condition tolerance
atol_constraint = 1e-5

# filtering parameters
Δl_cutoff = 10;
Δl_power_cutoff = 0.9;
eigen_rtol = 0.01;

n_cutoff = 10
n_power_cutoff = 0.9

# used for filtering, zero for uniform rotation
ΔΩ_by_Ω_low = -0.5
ΔΩ_by_Ω_high = 0.5
# ΔΩ_by_Ω_low = 0
# ΔΩ_by_Ω_high = 0

r_in_frac = 0.7
r_out_frac = 1

print_timer = false
scale_eigenvectors = false

operators = RossbyWaveSpectrum.radial_operators(nr, nℓ; r_in_frac, r_out_frac);

f = (x...; kw...) -> RossbyWaveSpectrum.differential_rotation_spectrum!(x...;
    rotation_profile = :constant, kw...);
# f = RossbyWaveSpectrum.uniform_rotation_spectrum!
@show nr nℓ mrange Δl_cutoff Δl_power_cutoff eigen_rtol ΔΩ_by_Ω_low ΔΩ_by_Ω_high;
@show scale_eigenvectors;
@show Threads.nthreads() LinearAlgebra.BLAS.get_num_threads();

@time RossbyWaveSpectrum.save_eigenvalues(f, nr, nℓ, mrange;
    atol_constraint, Δl_cutoff, Δl_power_cutoff,
    eigen_rtol, n_cutoff, n_power_cutoff,
    ΔΩ_by_Ω_low, ΔΩ_by_Ω_high, operators,
    print_timer, scale_eigenvectors)
