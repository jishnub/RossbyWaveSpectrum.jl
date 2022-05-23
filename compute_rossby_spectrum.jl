@time using RossbyWaveSpectrum
using LinearAlgebra

nr = 60;
nℓ = 25;
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

n_cutoff = 20
n_power_cutoff = 0.9

# used for filtering, zero for uniform rotation
ΔΩ_frac_low = -10
ΔΩ_frac_high = 10

r_in_frac = 0.5
r_out_frac = 0.985

print_timer = false
scale_eigenvectors = false

operators = RossbyWaveSpectrum.radial_operators(nr, nℓ; r_in_frac, r_out_frac);

diffrot = false

# spectrumfn! = RossbyWaveSpectrum.diffrotspectrumfn!(:radial_linear);
spectrumfn! = RossbyWaveSpectrum.uniform_rotation_spectrum!
@show nr nℓ mrange Δl_cutoff Δl_power_cutoff eigen_rtol ΔΩ_frac_low ΔΩ_frac_high;
@show scale_eigenvectors;
@show Threads.nthreads() LinearAlgebra.BLAS.get_num_threads();

@time RossbyWaveSpectrum.save_eigenvalues(spectrumfn!, mrange;
    atol_constraint, Δl_cutoff, Δl_power_cutoff,
    eigen_rtol, n_cutoff, n_power_cutoff,
    ΔΩ_frac_low, ΔΩ_frac_high, operators,
    print_timer, scale_eigenvectors, diffrot)
