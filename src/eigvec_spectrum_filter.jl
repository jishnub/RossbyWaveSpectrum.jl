function eigvec_spectrum_filter!(F, v, m;
    operators,
    n_cutoff = DefaultFilterParams[:n_cutoff],
    Δl_cutoff = DefaultFilterParams[:Δl_cutoff],
    eigvec_spectrum_power_cutoff = DefaultFilterParams[:eigvec_spectrum_power_cutoff],
    filterfieldpowercutoff = DefaultFilterParams[:filterfieldpowercutoff],
    low_n_power_lowercutoff = DefaultFilterParams[:eigvec_spectrum_low_n_power_fraction_cutoff],
    kw...)

    VW = eigenfunction_spectrum_2D!(F, v; operators, kw...)
    Δl_inds = Δl_cutoff ÷ 2

    @unpack nparams = operators.radial_params
    @unpack nvariables = operators

    flag = true
    fields = filterfields(VW, v, nparams, nvariables; filterfieldpowercutoff)

    @views for _X in fields
        f, X = first(_X), last(_X)

        PV_frac_real = sum(abs2∘real, X[1:n_cutoff, 1:Δl_inds]) / sum(abs2∘real, X)
        PV_frac_imag = sum(abs2∘imag, X[1:n_cutoff, 1:Δl_inds]) / sum(abs2∘imag, X)
        flag &= (PV_frac_real > eigvec_spectrum_power_cutoff) & (PV_frac_imag > eigvec_spectrum_power_cutoff)

        @debug("$f PV_frac_real $PV_frac_real PV_frac_imag $PV_frac_imag "*
                "cutoff $eigvec_spectrum_power_cutoff flag $flag")

        flag || break

        real_low_n_pow_ratio = sum(abs2∘real, X[(n_cutoff÷2+1):n_cutoff, 1:Δl_inds])/sum(abs2∘real, X[1:n_cutoff÷2, 1:Δl_inds])
        real_low_n_flag = real_low_n_pow_ratio <= low_n_power_lowercutoff

        @debug "$f real_low_n_flag $real_low_n_flag"

        flag &= real_low_n_flag
        flag || break

        imag_low_n_pow_ratio = sum(abs2∘imag, X[(n_cutoff÷2+1):n_cutoff, 1:Δl_inds])/sum(abs2∘imag, X[1:n_cutoff÷2, 1:Δl_inds])
        imag_low_n_flag = imag_low_n_pow_ratio <= low_n_power_lowercutoff

        @debug "$f imag_low_n_flag $imag_low_n_flag"

        flag &= imag_low_n_flag
        flag || break
    end

    return flag
end

"""
    eigvec_spectrum_filter(v::AbstractVector{<:Number}, m::Integer;
        operators,
        n_cutoff = RossbyWaveSpectrum.DefaultFilterParams[:n_cutoff],
        Δl_cutoff = RossbyWaveSpectrum.DefaultFilterParams[:Δl_cutoff],
        eigvec_spectrum_power_cutoff = RossbyWaveSpectrum.DefaultFilterParams[:eigvec_spectrum_power_cutoff],
        filterfieldpowercutoff = RossbyWaveSpectrum.DefaultFilterParams[:filterfieldpowercutoff],
        low_n_power_lowercutoff = RossbyWaveSpectrum.DefaultFilterParams[:eigvec_spectrum_low_n_power_fraction_cutoff],
        kw...)

Return if the eigenvector `v` is smooth enough, according to the specified criteria.
"""
function eigvec_spectrum_filter(v::AbstractVector{<:Number}, m::Integer; operators, kw...)
    @unpack nr, nℓ = operators.radial_params
    F = allocate_field_vectors(nr, nℓ)
    eigvec_spectrum_filter!(F, v, m; operators, kw...)
end

"""
    eigvec_spectrum_filter(Feig::FilteredEigen, m::Integer, ind::Integer; kw...)

Return if the eigenvector `ind`-th eigenvector of `Feig[m]` is smooth enough, according to the specified criteria.
By default, the filter parameters are taken to be identical to those used to compute the solutions in `Feig`,
but each of these parameters may be overridden.
"""
function eigvec_spectrum_filter(Feig::FilteredEigen, m, ind; kw...)
    @unpack nr, nℓ = Feig.operators.radial_params
    F = allocate_field_vectors(nr, nℓ)
    eigvec_spectrum_filter!(F, Feig[m][ind].v, m; Feig.operators, Feig.kw..., kw...)
end
