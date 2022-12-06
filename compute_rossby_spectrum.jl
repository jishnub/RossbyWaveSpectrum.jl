module ComputeRossbySpectrum
@time using RossbyWaveSpectrum
using LinearAlgebra

function computespectrum(nr, nℓ, mrange, V_symmetric, diffrot, rotation_profile;
            save = true, smoothing_param = 1e-5)
    flush(stdout)
    # boundary condition tolerance
    bc_atol = 1e-5

    # filtering parameters
    Δl_cutoff = min(15, nℓ-2);
    n_cutoff = min(15, nr-2)
    eigvec_spectrum_power_cutoff = 0.9;

    eigen_rtol = 0.01;

    print_timer = false
    scale_eigenvectors = false

    Seqglobalscaling = 1e-7
    scalings = (; Seqglobalscaling)

    @info "operators"
    r_in_frac = 0.6
    r_out_frac = 0.985

    @show nr nℓ mrange Δl_cutoff n_cutoff r_in_frac r_out_frac smoothing_param
    @show eigvec_spectrum_power_cutoff eigen_rtol V_symmetric diffrot;
    @show Threads.nthreads() LinearAlgebra.BLAS.get_num_threads();

    @time operators = RossbyWaveSpectrum.radial_operators(nr, nℓ; r_in_frac, r_out_frac, ν = 2e12, scalings);

    spectrumfn! = if diffrot
        d = RossbyWaveSpectrum.RotMatrix(V_symmetric, rotation_profile, nothing, RossbyWaveSpectrum.differential_rotation_spectrum!)
        RossbyWaveSpectrum.updaterotatationprofile(d, operators; smoothing_param)
    else
        RossbyWaveSpectrum.RotMatrix(V_symmetric, :uniform, nothing, RossbyWaveSpectrum.uniform_rotation_spectrum!)
    end

    flush(stdout)

    kw = Base.pairs((; bc_atol, Δl_cutoff, n_cutoff, eigvec_spectrum_power_cutoff, eigen_rtol,
        print_timer, scale_eigenvectors, diffrot, rotation_profile, V_symmetric, smoothing_param))

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

function main(taskno = parse(Int, ENV["SLURM_PROCID"]))
    nr = 40;
    nℓ = 20;
    mrange = 1:15;
    diffrot = true;
    rotation_profile = :solar_latrad

    V_symmetric = (true, false)[taskno + 1]
    @show Libc.gethostname(), taskno, V_symmetric

    computespectrum(8, 6, 1:1, V_symmetric, diffrot, rotation_profile, save = false, smoothing_param = 1e-1)
    computespectrum(nr, nℓ, mrange, V_symmetric, diffrot, rotation_profile)
end

end
