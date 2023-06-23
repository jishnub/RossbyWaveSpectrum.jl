function spatial_filter!(filtercache, v, m;
    operators,
    V_symmetric,
    θ_cutoff = DefaultFilterParams[:θ_cutoff],
    equator_power_cutoff_frac = DefaultFilterParams[:equator_power_cutoff_frac],
    pole_cutoff_angle = DefaultFilterParams[:pole_cutoff_angle],
    pole_power_cutoff_frac = DefaultFilterParams[:pole_power_cutoff_frac],
    filterfieldpowercutoff = DefaultFilterParams[:filterfieldpowercutoff],
    radial_topbotpower_cutoff = DefaultFilterParams[:radial_topbotpower_cutoff],
    nℓ = operators.radial_params.nℓ,
    Plcosθ = allocate_Pl(m, nℓ),
    angular_filter_equator = true,
    angular_filter_highlat = false,
    radial_filter = true,
    compute_invtransform = true,
    kw...
    )

    (; θ) = spharm_θ_grid_uniform(m, nℓ)
    eqind = indexof_equator(θ)

    (; VWSinv, radproftempreal, fieldtempreal) = filtercache;

    if compute_invtransform
        eigenfunction_realspace!(filtercache, v, m;
            operators, nℓ, Plcosθ, V_symmetric)
    end

    eqfilter = true

    radfilter = true

    @unpack nparams, r_out, r_in, nr = operators.radial_params
    @unpack nvariables = operators
    @unpack rpts, rptsrev = operators
    fields = filterfields(VWSinv, v, nparams, nvariables; filterfieldpowercutoff)

    for _X in fields
        f, X = first(_X), last(_X)
        f == :S && return false
        fieldtempreal .= abs2.(X);
        tot_power = -trapz((rpts, θ), fieldtempreal); # negative sign as rpts is decreasing
        if angular_filter_equator
            θlowind = indexof_colatitude(θ, θ_cutoff)
            θhighind = indexof_colatitude(θ, pi - θ_cutoff)
            θinds = θlowind:θhighind
            pow_within_cutoff = -trapz((rpts, θ[θinds]), @view fieldtempreal[:, θinds])
            powfrac = pow_within_cutoff / tot_power
            powflag = powfrac > equator_power_cutoff_frac
            @debug "$f equatorial powfrac $powfrac cutoff $equator_power_cutoff_frac powflag $powflag"

            # ensure that there isn't too much power at the poles
            θpolecutoff = searchsortedfirst(θ, pole_cutoff_angle)
            θinds = 1:θpolecutoff
            pow_poles = -2trapz((rpts, θ[θinds]), @view fieldtempreal[:, θinds])
            powfrac = pow_poles  / tot_power
            powflag &= powfrac < pole_power_cutoff_frac
            @debug "$f polar powfrac $powfrac cutoff $pole_power_cutoff_frac powflag $powflag"

            # Ensure that power isn't localized entirely at the equator
            θlowind = indexof_colatitude(θ, deg2rad(80))
            θhighind = indexof_colatitude(θ, deg2rad(100))
            θinds = θlowind:θhighind
            pow_within_cutoff = -trapz((rpts, θ[θinds]), @view fieldtempreal[:, θinds])
            powfrac = pow_within_cutoff / tot_power
            powflag &= powfrac < 0.9
            @debug "$f sharp equatorial powfrac $powfrac cutoff 0.9 powflag $powflag"

            θlowind = indexof_colatitude(θ, θ_cutoff)
            θhighind = indexof_colatitude(θ, pi - θ_cutoff)
            r_ind_peak = peakindabs1(X)
            peak_latprofile = @view X[r_ind_peak, :]
            peak_latprofile_max = maximum(abs2, peak_latprofile)
            peak_latprofile_max_inrange = maximum(abs2, @view peak_latprofile[θlowind:θhighind])
            peakflag = peak_latprofile_max_inrange == peak_latprofile_max
            @debug "$f peakflag $peakflag"
            eqfilter &= powflag & peakflag
        elseif angular_filter_highlat
            θlowind = searchsortedfirst(θ, θ_cutoff)
            θhighind = searchsortedlast(θ, pi - θ_cutoff)
            powfrac = 1 - sum(abs2, @view X[:, θlowind:θhighind]) / tot_power
            eqfilter = powfrac > equator_power_cutoff_frac
        end
        eqfilter || break

        @debug "$f eqfilter $eqfilter"

        if radial_filter
            # ensure that the radial peak isn't at the bottom of the domain
            for (ri, row) in enumerate(eachrow(fieldtempreal))
                radproftempreal[ri] = trapz(θ, row)
            end

            indmax = findmax(radproftempreal)[2]
            maxbotflag = indmax != nr
            @debug "maximum at r_in $maxbotflag"
            radfilter &= maxbotflag
            radfilter || break

            # ensure that most of the power isn't concentrated in the top and bottom surface layers
            r_top_10pc_cutoff = r_out - (r_out - r_in)*10/100
            r_top_10pc_cutoff_ind = findlast(>(r_top_10pc_cutoff), rpts)

            r_bot_10pc_cutoff = r_in + (r_out - r_in)*10/100
            r_bot_10pc_cutoff_ind = findlast(>(r_bot_10pc_cutoff), rpts)

            # check that the power is smoothly varying, and not sharply concentrated at the top
            rinds = 1:r_top_10pc_cutoff_ind
            top_10pc_power = -trapz(view(rpts, rinds), view(radproftempreal, rinds))
            top_10pc_power_fraction = top_10pc_power/tot_power
            @debug "$f tot_power $tot_power top_10pc_power_fraction $top_10pc_power_fraction"

            rinds = r_bot_10pc_cutoff_ind:nr
            bot_10pc_power = -trapz(view(rpts, rinds), view(radproftempreal, rinds))
            bot_10pc_power_fraction = bot_10pc_power/tot_power
            @debug "$f bot_10pc_power $bot_10pc_power bot_10pc_power_fraction $bot_10pc_power_fraction"

            top_bot_power_frac = top_10pc_power_fraction + bot_10pc_power_fraction
            pow_frac_flag = top_bot_power_frac < radial_topbotpower_cutoff
            @debug "top+bot power frac $top_bot_power_frac cutoff $radial_topbotpower_cutoff"
            radfilter &= pow_frac_flag
        end
        radfilter || break
    end

    return eqfilter & radfilter
end

function spatial_filter(v, m;
    operators,
    filtercache = allocate_filter_caches(m; operators),
    kw...
    )

    spatial_filter!(filtercache, v, m; operators, kw...)
end

function spatial_filter(Feig::FilteredEigen, m, ind; kw...)
    spatial_filter(Feig[m][ind].v, m; Feig.operators,
        filtercache = allocate_filter_caches(m; Feig.operators, Feig.constraints),
        Feig.kw..., kw...)
end
