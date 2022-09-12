module ComputeRossbySpectrum
@time using RossbyWaveSpectrum
using LinearAlgebra

flush(stdout)

function computespectrum(nr, nℓ, mrange, V_symmetric, diffrot, diffrotprof; save = true)
    # boundary condition tolerance
    bc_atol = 1e-5

    # filtering parameters
    Δl_cutoff = min(25, nℓ-2);
    n_cutoff = min(25, nr-2)
    eigvec_spectrum_power_cutoff = 0.9;

    eigen_rtol = 0.01;

    print_timer = false
    scale_eigenvectors = false

    # bit of a hack to reduce the doppler shift in the radial differential rotation case
    # should be removed to compare with the Sun
    trackingratescaling = (diffrot && (diffrotprof = :radial_solar_equator)) ? 1.01 : 1.0
    scalings = (; trackingratescaling)

    @info "operators"
    r_in_frac = 0.6
    r_out_frac = 0.985
    @time operators = RossbyWaveSpectrum.radial_operators(nr, nℓ; r_in_frac, r_out_frac, ν = 2e12, scalings);

    spectrumfn! = if diffrot
        d = RossbyWaveSpectrum.RotMatrix(V_symmetric, diffrotprof, nothing, RossbyWaveSpectrum.differential_rotation_spectrum!)
        RossbyWaveSpectrum.updaterotatationprofile(d, operators)
    else
        RossbyWaveSpectrum.RotMatrix(V_symmetric, :uniform, nothing, RossbyWaveSpectrum.uniform_rotation_spectrum!)
    end

    @show nr nℓ mrange Δl_cutoff n_cutoff
    @show eigvec_spectrum_power_cutoff eigen_rtol V_symmetric diffrot;
    @show Threads.nthreads() LinearAlgebra.BLAS.get_num_threads();

    flush(stdout)

    kw = Base.pairs((; bc_atol, Δl_cutoff, n_cutoff, eigvec_spectrum_power_cutoff, eigen_rtol,
        print_timer, scale_eigenvectors, diffrot, diffrotprof, V_symmetric))

    @time RossbyWaveSpectrum.save_eigenvalues(spectrumfn!, mrange;
        operators, kw...)

    if !save
        fname = RossbyWaveSpectrum.rossbyeigenfilename(; operators, kw...)
        @warn "removing $fname"
        rm(fname)
    end

    flush(stdout)
    return nothing
end

function main()
    nr = 60
    nℓ = 30;
    mrange = 1:15;
    diffrot = false;
    diffrotprof = :constant

    taskno = parse(Int, ENV["SLURM_PROCID"])
    V_symmetric = (true, false)[taskno + 1]
    @show Libc.gethostname(), taskno, V_symmetric

    computespectrum(8, 6, 1:1, V_symmetric, diffrot, diffrotprof, save = false)
    computespectrum(nr, nℓ, mrange, V_symmetric, diffrot, diffrotprof)
end

end
