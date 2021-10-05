using RossbyWaveSpectrum

const nr = 32;
const ntheta = 256;
const mrange = 5:6;

# boundary condition tolerance
const atol_constraint = 1e-5

# filtering parameters
const Δl_cutoff = 5
const power_cutoff = 0.9;

@show nr ntheta mrange

@time RossbyWaveSpectrum.save_eigenvalues(
    RossbyWaveSpectrum.uniform_rotation_spectrum,
    nr, ntheta, mrange;
    atol_constraint, Δl_cutoff, power_cutoff)
# @time RossbyWaveSpectrum.save_eigenvalues(RossbyWaveSpectrum.differential_rotation_spectrum, nr, ntheta, mrange)
