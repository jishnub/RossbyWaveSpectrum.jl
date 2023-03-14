module ComputeRossbySpectrum
@time using RossbyWaveSpectrum
using LinearAlgebra
using TimerOutputs

function computespectrum(nr, nℓ, mrange, V_symmetric, diffrot, rotation_profile;
            smoothing_param = 1e-4,
            r_in_frac = 0.7, r_out_frac = 0.995,
            trackingratescaling = 1.0, Seqglobalscaling = 1.0,
            ΔΩ_scale = 1.0,
            Δl_cutoff = 15,
            n_cutoff = 15,
            viscosity = 2e12,
            trackingrate = :cutoff,
            extrakw...
            )

    flush(stdout)
    # boundary condition tolerance

    # filtering parameters
    Δl_cutoff = min(Δl_cutoff, nℓ-2);
    n_cutoff = min(n_cutoff, nr-2)

    scalings = (; Seqglobalscaling, trackingratescaling)

    hostname = Libc.gethostname()
    @show hostname
    @show nr nℓ mrange Δl_cutoff n_cutoff r_in_frac r_out_frac smoothing_param ΔΩ_scale;
    @show V_symmetric diffrot rotation_profile viscosity;
    @show extrakw
    @show Threads.nthreads() LinearAlgebra.BLAS.get_num_threads();

    timer = TimerOutput()

    operators = @timeit timer "operators" begin
        RossbyWaveSpectrum.radial_operators(nr, nℓ; r_in_frac, r_out_frac, ν = viscosity, trackingrate, scalings);
    end

    spectrumfn! = @timeit timer "spectrumfn" begin
        RossbyWaveSpectrum.RotMatrix(Val(:spectrum), V_symmetric, diffrot, rotation_profile;
            operators, smoothing_param, ΔΩ_scale)
    end

    println(timer)

    flush(stdout)

    kw = Base.pairs((; Δl_cutoff, n_cutoff,
        diffrot, rotation_profile, V_symmetric,
        smoothing_param, ΔΩ_scale))

    @time RossbyWaveSpectrum.save_eigenvalues(spectrumfn!, mrange;
        operators, kw..., extrakw...)

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
    additional_kw...
    )

    @show Libc.gethostname(), V_symmetric

    computespectrum(8, 6, 1:1, V_symmetric, diffrot, rotation_profile,
            save = false, smoothing_param = 1e-1, print_timer = false)
    computespectrum(nr, nℓ, mrange, V_symmetric, diffrot, rotation_profile;
            save, additional_kw...)
end

end
