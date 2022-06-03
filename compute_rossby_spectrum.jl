module ComputeRossbySpectrum
@time using RossbyWaveSpectrum
using LinearAlgebra

flush(stdout)

function main(nr, nℓ, mrange)
    # boundary condition tolerance
    bc_atol = 1e-5

    # filtering parameters
    Δl_cutoff = min(15, nℓ-2);
    n_cutoff = min(15, nr-2)
    eigvec_spectrum_power_cutoff = 0.9;

    eigen_rtol = 0.01;

    print_timer = false
    scale_eigenvectors = false

    @info "operators"
    r_in_frac = 0.6
    r_out_frac = 0.985
    @time operators = RossbyWaveSpectrum.radial_operators(nr, nℓ; r_in_frac, r_out_frac, ν = 4e12);

    diffrot = false
    V_symmetric = false

    # spectrumfn! = RossbyWaveSpectrum.diffrotspectrum!(:radial_linear, V_symmetric)
    spectrumfn! = RossbyWaveSpectrum.uniformrotspectrumfn!(V_symmetric)
    @show nr nℓ mrange Δl_cutoff eigvec_spectrum_power_cutoff eigen_rtol V_symmetric scale_eigenvectors;
    @show Threads.nthreads() LinearAlgebra.BLAS.get_num_threads();

    flush(stdout)

    @time RossbyWaveSpectrum.save_eigenvalues(spectrumfn!, mrange;
        bc_atol, Δl_cutoff, n_cutoff, eigvec_spectrum_power_cutoff, eigen_rtol,
        operators, print_timer, scale_eigenvectors, diffrot, V_symmetric)

    flush(stdout)
end

nr = 60;
nℓ = 30;
mrange = 1:15;

main(8, 6, 1:1)
main(nr, nℓ, mrange)

end
