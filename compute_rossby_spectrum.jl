module ComputeRossbySpectrum
@time using RossbyWaveSpectrum
using LinearAlgebra

nr = 60;
nℓ = 30;
mrange = 1:15;

# test
# nr = 20;
# nℓ = 10;
# mrange = 1:1;

# boundary condition tolerance
atol_constraint = 1e-5

# filtering parameters
Δl_cutoff = min(15, nℓ-2);
Δl_power_cutoff = 0.9;
eigen_rtol = 0.01;

n_cutoff = min(15, nr-2)
n_power_cutoff = 0.9

r_in_frac = 0.6
r_out_frac = 0.985

print_timer = false
scale_eigenvectors = false

@info "operators"
@time operators = RossbyWaveSpectrum.radial_operators(nr, nℓ; r_in_frac, r_out_frac, ν = 4e12);

diffrot = false
V_symmetric = false

# const spectrumfn! = RossbyWaveSpectrum.diffrotspectrum!(:radial_linear, V_symmetric)
const spectrumfn! = RossbyWaveSpectrum.uniformrotspectrumfn!(V_symmetric)
@show nr nℓ mrange Δl_cutoff Δl_power_cutoff eigen_rtol V_symmetric scale_eigenvectors;
@show Threads.nthreads() LinearAlgebra.BLAS.get_num_threads();

@time RossbyWaveSpectrum.save_eigenvalues(spectrumfn!, mrange;
    atol_constraint, Δl_cutoff, Δl_power_cutoff,
    eigen_rtol, n_cutoff, n_power_cutoff,
    operators,
    print_timer, scale_eigenvectors, diffrot, V_symmetric)

end
