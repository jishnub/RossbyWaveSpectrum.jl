module ComputeRossbySpectrum
@time using RossbyWaveSpectrum
using LinearAlgebra
using TimerOutputs

function computespectrum(nr, nℓ, mrange, V_symmetric, diffrot, rotation_profile;
            save = true, smoothing_param = 1e-5, print_timer = true)
    flush(stdout)
    # boundary condition tolerance
    bc_atol = 1e-5

    # filtering parameters
    Δl_cutoff = min(15, nℓ-2);
    n_cutoff = min(15, nr-2)
    eigvec_spectrum_power_cutoff = 0.9;

    eigen_rtol = 0.01;

    scale_eigenvectors = false

    @info "operators"
    r_in_frac = 0.6
    r_out_frac = 0.985

    # Ω0 = RossbyWaveSpectrum.equatorial_rotation_angular_velocity_surface(r_out_frac)
    # trackingratescaling = 1 + (7.1e-9/(Ω0/2pi))
    trackingratescaling = 1.0
    Seqglobalscaling = 1e-7
    scalings = (; Seqglobalscaling, trackingratescaling)

    @show nr nℓ mrange Δl_cutoff n_cutoff r_in_frac r_out_frac smoothing_param;
    @show eigvec_spectrum_power_cutoff eigen_rtol V_symmetric diffrot rotation_profile;
    @show Threads.nthreads() LinearAlgebra.BLAS.get_num_threads();

    timer = TimerOutput()

    operators = @timeit timer "operators" begin
        RossbyWaveSpectrum.radial_operators(nr, nℓ; r_in_frac, r_out_frac, ν = 2e12, scalings);
    end

    spectrumfn! = @timeit timer "spectrumfn" begin
        RossbyWaveSpectrum.RotMatrix(Val(:spectrum), V_symmetric, diffrot, rotation_profile;
            operators, smoothing_param)
    end

    println(timer)

    flush(stdout)

    kw = Base.pairs((; bc_atol, Δl_cutoff, n_cutoff, eigvec_spectrum_power_cutoff, eigen_rtol,
        print_timer, scale_eigenvectors, diffrot, rotation_profile, V_symmetric, smoothing_param))

    @time RossbyWaveSpectrum.save_eigenvalues(spectrumfn!, mrange;
        operators, print_timer, save, kw...)

    flush(stdout)
    return nothing
end

function main(V_symmetric = true;
    nr = 40,
    nℓ = 20,
    mrange = 1:1,
    diffrot = true,
    rotation_profile = :solar_latrad,
    save = true,
    )

    @show Libc.gethostname(), V_symmetric

    computespectrum(8, 6, 1:1, V_symmetric, diffrot, rotation_profile,
            save = false, smoothing_param = 1e-1,
            print_timer = false)
    computespectrum(nr, nℓ, mrange, V_symmetric, diffrot, rotation_profile;
            save)
end

end
