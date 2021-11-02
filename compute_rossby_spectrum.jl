using RossbyWaveSpectrum

nr = 50;
nℓ = 64;
mrange = 5:10;

# boundary condition tolerance
atol_constraint = 1e-5

# filtering parameters
Δl_cutoff = 10;
power_cutoff = 0.9;
eigen_rtol = 0.1;

subtract_doppler = false
test = true
ΔΩscale = 1

f = RossbyWaveSpectrum.uniform_rotation_spectrum
# f = RossbyWaveSpectrum.differential_rotation_spectrum
# f = RossbyWaveSpectrum.differential_rotation_spectrum_constantΩ
@show nr nℓ mrange f
@show Threads.nthreads()

# precompile
RossbyWaveSpectrum.filter_eigenvalues(f, 4, 4, 1)

@time RossbyWaveSpectrum.save_eigenvalues(f, nr, nℓ, mrange;
    atol_constraint, Δl_cutoff, power_cutoff,
    subtract_doppler, ΔΩscale, test, eigen_rtol)
