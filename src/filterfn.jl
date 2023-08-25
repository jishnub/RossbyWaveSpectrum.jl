function filterfn(λ::Number, v::AbstractVector, m::Integer, M, filterparams;
        operators,
        constraints = constraintmatrix(operators),
        filtercache = allocate_filter_caches(m; operators, constraints),
        filterflags = DefaultFilter)

    @unpack BC = constraints
    @unpack nℓ = operators.radial_params;

    filterparams_all = merge(DefaultFilterParams, filterparams);

    eig_imag_unstable_cutoff = Float64(filterparams_all[:eig_imag_unstable_cutoff])::Float64
    eig_imag_to_real_ratio_cutoff = Float64(filterparams_all[:eig_imag_to_real_ratio_cutoff])::Float64
    eig_imag_stable_cutoff = Float64(filterparams_all[:eig_imag_stable_cutoff])::Float64
    eigvec_spectrum_power_cutoff = Float64(filterparams_all[:eigvec_spectrum_power_cutoff])::Float64
    bc_atol = Float64(filterparams_all[:bc_atol])::Float64
    Δl_cutoff = filterparams_all[:Δl_cutoff]::Int
    n_cutoff = filterparams_all[:n_cutoff]::Int
    θ_cutoff = Float64(filterparams_all[:θ_cutoff])::Float64
    equator_power_cutoff_frac = Float64(filterparams_all[:equator_power_cutoff_frac])::Float64
    eigen_rtol = Float64(filterparams_all[:eigen_rtol])::Float64
    filterfieldpowercutoff = Float64(filterparams_all[:filterfieldpowercutoff])::Float64
    nnodesmax = filterparams_all[:nnodesmax]::Int
    V_symmetric = filterparams[:V_symmetric]::Bool
    radial_topbotpower_cutoff = Float64(filterparams_all[:radial_topbotpower_cutoff])::Float64

    (; MVcache, BCVcache, VWSinv, VWSinvsh,
        Plcosθ, F, radproftempreal, radproftempcomplex) = filtercache;

    allfilters = Filters.FilterFlag(filterflags)
    compute_invtransform = true

    if Filters.EIGVAL in allfilters
        f = eigenvalue_filter(λ, m;
        eig_imag_unstable_cutoff, eig_imag_to_real_ratio_cutoff, eig_imag_stable_cutoff)
        if !f
            @debug "EIGVAL" λ, eig_imag_unstable_cutoff, eig_imag_to_real_ratio_cutoff, eig_imag_stable_cutoff
            return false
        end
    end

    if Filters.EIGVEC in allfilters
        f = eigvec_spectrum_filter!(F, v, m; operators,
            n_cutoff, Δl_cutoff, eigvec_spectrum_power_cutoff,
            filterfieldpowercutoff)
        if !f
            @debug "EIGVEC" n_cutoff, Δl_cutoff, eigvec_spectrum_power_cutoff, filterfieldpowercutoff
            return false
        end
    end

    if Filters.BC in allfilters
        f = boundary_condition_filter(v, BC, BCVcache, bc_atol)
        if !f
            @debug "BC" bc_atol
            return false
        end
    end

    if Filters.SPATIAL in allfilters
        f = spatial_filter!(filtercache, v, m;
            θ_cutoff, equator_power_cutoff_frac, operators, nℓ, Plcosθ,
            filterfieldpowercutoff, V_symmetric,
            angular_filter_equator = Filters.SPATIAL_EQUATOR in allfilters,
            angular_filter_highlat = Filters.SPATIAL_HIGHLAT in allfilters,
            radial_filter = Filters.SPATIAL_RADIAL in allfilters,
            compute_invtransform,
            radial_topbotpower_cutoff,
            )
        if !f
            @debug "SPATIAL" θ_cutoff, equator_power_cutoff_frac, filterfieldpowercutoff
            return false
        end
        compute_invtransform = false
    end

    if Filters.NODES in allfilters
        f = nodes_filter!(filtercache, v, m, operators;
                nℓ, Plcosθ, filterfieldpowercutoff, nnodesmax, V_symmetric,
                compute_invtransform, radproftempreal, radproftempcomplex)
        if !f
            @debug "NODES" filterfieldpowercutoff, nnodesmax
            return false
        end
        compute_invtransform = false
    end

    if Filters.EIGEN in allfilters
        f = eigensystem_satisfy_filter(λ, v, M, MVcache; rtol = eigen_rtol)
        if !f
            @debug "EIGEN" λ eigen_rtol
            return false
        end
    end

    return true
end

function filterfn(Feig::FilteredEigen, m, ind, Ms = operator_matrices(Feig, m),
        filterparams = pairs((;)); kw...)
    filterfn(Feig[m][ind]..., m, Ms, merge(Feig.kw, filterparams);
        Feig.operators, Feig.constraints,
        filterflags = Feig.kw[:filterflags], kw...)
end
