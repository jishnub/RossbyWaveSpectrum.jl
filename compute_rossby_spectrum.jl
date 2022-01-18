@time using RossbyWaveSpectrum

nr = 50;
nℓ = 25;
mrange = 1:20;

# boundary condition tolerance
atol_constraint = 1e-5

# filtering parameters
Δl_cutoff = 10;
Δl_power_cutoff = 0.9;
eigen_rtol = 0.1;

n_cutoff = 10
n_power_cutoff = 0.9

filenametag = "dr"
ΔΩ_by_Ω = 0.01 # used for filtering, zero for uniform rotation

# f = RossbyWaveSpectrum.uniform_rotation_spectrum
f = (x...; kw...) -> RossbyWaveSpectrum.differential_rotation_spectrum(x...;
    rotation_profile = :constant, kw...)
@show nr nℓ mrange Δl_cutoff Δl_power_cutoff eigen_rtol filenametag ΔΩ_by_Ω
@show Threads.nthreads()

@time RossbyWaveSpectrum.save_eigenvalues(f, nr, nℓ, mrange;
    atol_constraint, Δl_cutoff, Δl_power_cutoff,
    eigen_rtol,
    n_cutoff, n_power_cutoff, filenametag, ΔΩ_by_Ω)
