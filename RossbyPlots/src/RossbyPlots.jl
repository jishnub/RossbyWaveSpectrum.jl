module RossbyPlots

__precompile__(false)
Base.Experimental.@optlevel 1

using PyCall
using PyPlot

using RossbyWaveSpectrum

import ApproxFun: ncoefficients, itransform
using LaTeXStrings
using LinearAlgebra
using OrderedCollections
using Printf
using StructArrays
using UnPack
using Roots

const StructMatrix{T} = StructArray{T,2}

export spectrum
export uniform_rotation_spectrum
export differential_rotation_spectrum
export diffrot_rossby_ridge
export plot_diffrot_radial
export plot_diffrot_radial_derivatives
export eigenfunction
export eigenfunction_spectrum
export eigenfunction_rossbyridge
export eigenfunction_rossbyridge_allstreamfn
export eigenfunctions_allstreamfn
export multiple_eigenfunctions_surface_m
export compare_terms
export plot_scale_heights
export plot_constraint_basis
export damping_rossbyridge
export damping_highfreqridge
export plot_matrix
export plot_matrix_block

const plotdir = dirname(@__DIR__)
const ticker = PyNULL()
const axes_grid1 = PyNULL()

function __init__()
	copy!(ticker, pyimport("matplotlib.ticker"))
	copy!(axes_grid1, pyimport("mpl_toolkits.axes_grid1"))
end

_real(M::StructArray{<:Complex}) = M.re
_real(M::AbstractArray) = real(M)

_imag(M::StructArray{<:Complex}) = M.im
_imag(M::AbstractArray) = imag(M)

function realview(M::AbstractArray{Complex{T}}) where {T}
    @view reinterpret(reshape, T, M)[1, ntuple(_->:, ndims(M))...]
end
realview(M::StructArray{<:Complex}) = M.re
realview(M::AbstractArray{<:Real}) = M

function imagview(M::AbstractArray{Complex{T}}) where {T}
    @view reinterpret(reshape, T, M)[2, ntuple(_->:, ndims(M))...]
end
imagview(M::StructArray{<:Complex}) = M.im
imagview(M::AbstractArray{<:Real}) = M

function savefiginfo(f, filename; kw...)
    fpath = joinpath(plotdir, basename(filename))
    @info "saving to $fpath"
    f.savefig(fpath; kw...)
end

axlistvector(_axlist) = reshape([_axlist;], Val(1))

function exppow10fmt(x, _)
    a, b = split((@sprintf "%.1e" x), 'e')
    c = parse(Int, b)
    a * L"\times10^{%$c}"
end

function rossby_ridge_lines(mr; νnHzunit = 1, ax = gca(), ΔΩ_frac = 0, ridgescalefactor = nothing, kw...)
    if get(kw, :sectoral_rossby_ridge, true)
        ax.plot(mr, -rossby_ridge.(mr; ΔΩ_frac) .* νnHzunit,
            label = ΔΩ_frac == 0 ? L"\frac{2(Ω/2π)}{m+1}" : "Doppler\nshifted",
            lw = 1,
            color = get(kw, :sectoral_rossby_ridge_color, "grey"),
            zorder = 0,
            ls = get(kw, :sectoral_rossby_ridge_ls, "-.")
        )
    end

    if !isnothing(ridgescalefactor)
        ax.plot(mr, -ridgescalefactor .* rossby_ridge.(mr; ΔΩ_frac) .* νnHzunit ,
            label = ΔΩ_frac == 0 ? "high-frequency" :
                    L"\Delta\Omega/\Omega_0 = " * string(round(ΔΩ_frac, sigdigits = 1)),
            lw = 1,
            color = get(kw, :sectoral_rossby_ridge_color, "black"),
            zorder = 0,
            ls = get(kw, :sectoral_rossby_ridge_scaled_ls, "dotted")
        )
    end

    if get(kw, :uniform_rotation_ridge, false)
        ax.plot(mr, -rossby_ridge.(mr) .* νnHzunit,
            label = "2/(m+1)",
            lw = 0.5,
            color = "black",
            ls = "dashed",
            dashes = (6, 3),
            zorder = 0,
        )
    end
end

struct Measurement
    value :: Float64
    higherr :: Float64
    lowerr :: Float64
end
Measurement(a, b) = Measurement(a, b, b)
value(m::Measurement) = m.value
value(x) = x
uncertainty(m::Measurement) = (lowerr, higherr)
lowerr(m::Measurement) = m.lowerr
higherr(m::Measurement) = m.higherr

const ProxaufFit = OrderedDict(
    # m => (nu, γ)
    3 => (; ν = Measurement(-230, 5, 4), γ = Measurement(40, 13, 11)),
    4 => (; ν = Measurement(-195, 3), γ = Measurement(16, 7, 5)),
    5 => (; ν = Measurement(-159, 3, 2), γ = Measurement(12, 6, 5)),
    6 => (; ν = Measurement(-119, 6), γ = Measurement(84, 22, 19)),
    7 => (; ν = Measurement(-111, 6), γ = Measurement(20, 7, 5)),
    8 => (; ν = Measurement(-89, 3), γ = Measurement(19, 7, 6)),
    9 => (; ν = Measurement(-77, 4), γ = Measurement(40, 11)),
    10 => (; ν = Measurement(-77, 4, 3), γ = Measurement(29, 10, 7)),
    11 => (; ν = Measurement(-64, 4, 5), γ = Measurement(47, 13, 12)),
    12 => (; ν = Measurement(-59, 4), γ = Measurement(35, 11, 9)),
    13 => (; ν = Measurement(-45, 6), γ = Measurement(76, 22, 20)),
    14 => (; ν = Measurement(-47, 5), γ = Measurement(40, 13, 11)),
    15 => (; ν = Measurement(-39, 5, 4), γ = Measurement(41, 12, 11)),
)

const HansonHFFit = OrderedDict(
    8 => (; ν = Measurement(-278.9, 17.7, 16.0), γ = Measurement(81.6, 76.6, 54.7)),
    9 => (; ν = Measurement(-257.2, 15.8, 9.8), γ = Measurement(35.3, 62.9, 22.3)),
    10 => (; ν = Measurement(-234.2, 21.6, 13.2), γ = Measurement(38.9, 70.1, 25.1)),
    11 => (; ν = Measurement(-199.2, 6.1, 7.0), γ = Measurement(18.5, 37.2, 8.9)),
    12 => (; ν = Measurement(-198.7, 19.9, 12.9), γ = Measurement(29.9, 63.5, 18.2)),
    13 => (; ν = Measurement(-182.8, 6.7, 7.7), γ = Measurement(26.7, 24.6, 12.5)),
    14 => (; ν = Measurement(-181.4, 23.5, 22.8), γ = Measurement(42.7, 76.1, 28.4)),
)

const HansonGONGfit = OrderedDict(
    3 => (; ν = Measurement(-242.8, 3.1), γ = Measurement(30.4, 7.7, 6.2)),
    4 => (; ν = Measurement(-197.7, 2.7), γ = Measurement(24.8, 6.8, 5.3)),
    5 => (; ν = Measurement(-153.2, 3.7), γ = Measurement(29.2, 9.9, 7.4)),
    6 => (; ν = Measurement(-122.1, 4.4), γ = Measurement(51.2, 11.2, 9.2)),
    7 => (; ν = Measurement(-110.4, 2.7), γ = Measurement(28.6, 6.4, 5.3)),
    8 => (; ν = Measurement(-91.7, 2.7), γ = Measurement(30.6, 6.3, 5.2)),
    9 => (; ν = Measurement(-84.0, 3.2), γ = Measurement(35.7, 7.7, 6.3)),
    10 => (; ν = Measurement(-74.8, 3.0), γ = Measurement(28.1, 7.4, 5.9)),
    11 => (; ν = Measurement(-55.6, 4.0), γ = Measurement(36.7, 10.3, 8.1)),
    12 => (; ν = Measurement(-60.0, 3.2), γ = Measurement(26.4, 8.5, 6.5)),
    13 => (; ν = Measurement(-54.9, 4.8), γ = Measurement(46.7, 12.6, 9.9)),
    14 => (; ν = Measurement(-51.6, 4.9), γ = Measurement(38.1, 13.8, 10.1)),
    15 => (; ν = Measurement(-43.0, 4.5), γ = Measurement(44.5, 11.8, 9.3)),
)

const Liang2019MDIHMI = OrderedDict(
    3 => (; ν = Measurement(-253, 2), γ = Measurement(7, 4, 3)),
    4 => (; ν = Measurement(-198, 5), γ = Measurement(34, 15, 13)),
    5 => (; ν = Measurement(-156, 2), γ = Measurement(8, 5, 4)),
    6 => (; ν = Measurement(-135, 4, 5), γ = Measurement(20, 12, 11)),
    7 => (; ν = Measurement(-110, 4), γ = Measurement(40, 12, 10)),
    8 => (; ν = Measurement(-91, 3), γ = Measurement(19, 7, 6)),
    9 => (; ν = Measurement(-82, 5), γ = Measurement(33, 13, 12)),
    10 => (; ν = Measurement(-60, 5, 6), γ = Measurement(54, 18, 16)),
    11 => (; ν = Measurement(-48, 7), γ = Measurement(84, 27, 24)),
    12 => (; ν = Measurement(-36, 8), γ = Measurement(75, 31, 28)),
    13 => (; ν = Measurement(-26, 7), γ = Measurement(82, 29, 25)),
    14 => (; ν = Measurement(-35, 5), γ = Measurement(27, 13, 12)),
    15 => (; ν = Measurement(-22, 2, 3), γ = Measurement(11, 7, 5)),
)

const HansonHighmfit = OrderedDict(
    .=>(3:25, (x -> (; ν = x)).(
        Measurement.([-239.85023787, -194.74218881, -158.17165486, -118.97498962,
       -111.33150168,  -88.82183249,  -76.4812802 ,  -77.46255087,
        -66.15595674,  -60.27040155,  -44.4007832 ,  -49.13988037,
        -42.74126055,  -17.87075571,  -23.86233201,   -9.27675778,
        -16.82402438,   21.42364466,   -4.56253251,   -5.57888729,
         51.12102461,   41.23779238,   74.8054728 ],
    [ 3.71177019,  2.71950549,  2.74778436,  7.68047884,  3.8421045 ,
        3.06962887,  4.17586493,  3.85660104,  5.18002071,  4.7879219 ,
        7.34926256,  5.25456844,  5.48006636,  6.39776562,  9.36107323,
        8.96203759, 11.56557341, 14.44780523,  8.04412307, 54.9300657 ,
       38.8350157 , 82.12514585, 23.96035167],
    [ 3.65679971,  2.74835319,  2.73552219,  7.54232931,  3.52837351,
        3.20261367,  3.96356207,  3.78393831,  5.10844758,  4.61691889,
        7.53241866,  5.31386927,  5.46743624,  6.54103895,  9.72079996,
        9.41561522, 11.79312097, 15.06342711,  7.35867335, 30.98882865,
       30.8959761 , 81.29153888, 62.20129617])))
    )

function errorbars_pyplot(mmnt::AbstractVector{Measurement})
    hcat(lowerr.(mmnt), higherr.(mmnt))'
end

const ScatterParams = Dict(
        :edgecolors => "k",
        :s => 30,
        :marker => "o",
        :ls => "None",
        :cmap => "Greys",
        :lw => 0.5,
    )

# ν0 at r = r_out in nHz, effectively an unit
freqnHzunit(Ω0) = Ω0 * 1e9/2pi

function Vreal_peak_l(v, m; operators, V_symmetric,
        Vrf = begin
            nr = operators.radial_params[:nr]
            nℓ = operators.radial_params[:nℓ]
            Vℓ = RossbyWaveSpectrum.ℓrange(m, nℓ, V_symmetric)
            Vℓ_full = first(Vℓ):last(Vℓ)
            zeros(nr, Vℓ_full)
        end,
        pow_ℓ = similar(Vrf, axes(Vrf,2)),
    )

    compute_eigenfunction_spectra(v, m; operators, V_symmetric, fields="Vr", Vrf)
    for (i, col) in pairs(eachcol(Vrf))
        pow_ℓ[i] = sum(abs2, col)
    end
    pow = sum(pow_ℓ)
    findfirst(p -> p/pow > 0.25, pow_ℓ)
end

function spectrum_peak_ℓ_mask(vecs, mr, Δl_filter; operators, V_symmetric, kw...)
    nr = operators.radial_params[:nr]
    nℓ = operators.radial_params[:nℓ]
    map(zip(vecs, mr)) do (vsm, m)
        Vℓ = RossbyWaveSpectrum.ℓrange(m, nℓ, V_symmetric)
        Vℓ_full = first(Vℓ):last(Vℓ)
        Vrf = zeros(nr, Vℓ_full)
        peak_ls = map(eachcol(vsm)) do v
            Vreal_peak_l(v, m; operators, V_symmetric, Vrf)
        end
        peak_ls .== m + Δl_filter
    end
end

function nodes_cutoff_mask(vecs, mr, nodes_cutoff; operators, V_symmetric, kw...)
    nr = operators.radial_params[:nr]
    nℓ = operators.radial_params[:nℓ]
    ns = map(zip(vecs, mr)) do (vm, m)
        θ = RossbyWaveSpectrum.spharm_θ_grid_uniform(m, nℓ).θ
        fieldcaches = RossbyWaveSpectrum.allocate_field_caches(nr, nℓ, length(θ))
        realcache = similar(fieldcaches.VWSinv.V, real(eltype(fieldcaches.VWSinv.V)))
        map(eachcol(vm)) do v
            reimnodes = RossbyWaveSpectrum.count_radial_nodes(v, m; operators, fieldcaches, realcache, V_symmetric)
            maximum(reimnodes)
        end
    end
    ns_mask = map(ns) do ns_m
        ns_m .<= nodes_cutoff
    end
end

function update_cutoffs_nodes(lams, vecs, mr; operators,
        V_symmetric::Bool, nodes_cutoff::Int)

    ns_mask = nodes_cutoff_mask(vecs, mr, nodes_cutoff; operators, V_symmetric)
    x = map(zip(lams, vecs, ns_mask)) do (lam_m, v_m, nodes_mask_m)
        lam_m[nodes_mask_m], v_m[:, nodes_mask_m]
    end
    first.(x), last.(x)
end

function update_cutoffs_ℓ(lams, vecs, mr; operators,
        V_symmetric::Bool, Δl_filter::Int)

    Δl_mask = spectrum_peak_ℓ_mask(vecs, mr, Δl_filter; operators, V_symmetric)
    x = map(zip(lams, vecs, Δl_mask)) do (lam_m, v_m, lmask_m)
        lam_m[lmask_m], v_m[:, lmask_m]
    end
    first.(x), last.(x)
end

function update_cutoffs(lams, vecs, mr; operators,
        V_symmetric::Bool,
        nodes_cutoff::Union{Nothing,Int} = nothing,
        Δl_filter::Union{Nothing,Int} = nothing,
        )

    if !isnothing(Δl_filter)
        lams, vecs = update_cutoffs_ℓ(lams, vecs, mr; operators, V_symmetric, Δl_filter)
    end
    if !isnothing(nodes_cutoff)
        lams, vecs = update_cutoffs_nodes(lams, vecs, mr; operators, V_symmetric, nodes_cutoff)
    end

    lams, vecs
end

function update_cutoffs(F::RossbyWaveSpectrum.FilteredEigen;
    nodes_cutoff::Union{Nothing,Int} = nothing,
    Δl_filter::Union{Nothing,Int} = nothing)

    isnothing(nodes_cutoff) && isnothing(Δl_filter) && return F

    @unpack lams, vs, mr, operators, kw = F
    @unpack V_symmetric = kw
    lams, vs = update_cutoffs(lams, vs, mr; operators, V_symmetric, Δl_filter, nodes_cutoff)
    RossbyWaveSpectrum.FilteredEigen(lams, vs, mr, kw, operators)
end

struct RectPatchCoords
    xy::NTuple{2,Float64}
    width::Float64
    height::Float64
end

function spectrum(lams::AbstractArray, mr;
    operators,
    f = figure(),
    ax = subplot(),
    m_zoom = mr[max(begin, end - 6):end],
    subtract_sectoral = false,
    rossbyridges = !subtract_sectoral,
    highlight_nodes = false,
    nodes_cutoff = nothing,
    vecs = nothing,
    scale_freq = true,
    Δl_filter = nothing,
    ylim = nothing,
    encircle_m = nothing,
    rectpatchcoords = nothing,
    kw...)

    ax.set_xlabel("m", fontsize = 12)
    ax.set_ylabel(L"\Re[\nu]" * (subtract_sectoral ? L"\,-\,\frac{2Ω_0}{m+1}" : "") * " [nHz]", fontsize = 12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer = true))

    @unpack Ω0 = operators.constants

    V_symmetric = kw[:V_symmetric]

    νnHzunit = scale_freq ? freqnHzunit(Ω0)  #= around 453 =# : 1.0
    ylim2 = isnothing(ylim) ? (-400, 300) : ylim
    ax.set_ylim((ylim2[1]/453) * νnHzunit, (ylim2[2]/453) * νnHzunit)

    lams, vecs = update_cutoffs(lams, vecs, mr; operators, V_symmetric,
        Δl_filter, nodes_cutoff)

    if !isnothing(nodes_cutoff) || highlight_nodes
        ns = map(zip(vecs, mr)) do (vm, m)
            map(eachcol(vm)) do v
                RossbyWaveSpectrum.count_radial_nodes(v, m; operators, V_symmetric)[1]
            end
        end
    end

    if highlight_nodes || !isnothing(nodes_cutoff)
        ns_cat = reduce(vcat, ns)
    end

    lamscat = mapreduce(real, vcat, [λs_m .- subtract_sectoral * rossby_ridge(m) for (λs_m,m) in zip(lams,mr)]) .* νnHzunit
    mcat = reduce(vcat, [range(m, m, length(λi)) for (m, λi) in zip(mr, lams)])
    fillmarker = get(kw, :fillmarker, true) | highlight_nodes
    c = if highlight_nodes
            cbaxtitle = "n"
            cbticks = [i for i in 0:(!isnothing(nodes_cutoff) ? min(nodes_cutoff,3) : 3)]
            binedges = [-0.5; cbticks .+ 0.5]
            norm = matplotlib.colors.BoundaryNorm(binedges, length(binedges), extend="max")
            marker_colors = ["navy", "crimson", "limegreen", "gold"][
                1:(!isnothing(nodes_cutoff) ? min(nodes_cutoff +1, end) : end)]
            cmap = matplotlib.colors.ListedColormap(marker_colors)
            ns_cat
        elseif fillmarker
            lamsimcat = mapreduce(imag, vcat, lams) .* νnHzunit
            vmin, vmax = extrema(lamsimcat)
            vmin = min(0, vmin)
            norm = matplotlib.colors.Normalize(vmin, vmax)
            cbaxtitle = L"\Im[\nu]" * "[nHz]"
            cmap = ScatterParams[:cmap]
            cbticks = nothing
            lamsimcat
        else
            cmap = nothing
            norm = nothing
            "white"
        end

    s = ax.scatter(mcat, -lamscat; c, ScatterParams..., cmap, norm,
        zorder=get(kw, :zorder, 1), s=get(kw, :s, 25), linewidth = 0.7,
        marker = get(kw, :marker, "o"), picker=true, pickradius=5)

    if fillmarker
        divider = axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes("right", size = "3%", pad = 0.05)
        cb_extend = !isnothing(nodes_cutoff) && nodes_cutoff < 3 ? "neither" : "max"
        cb = colorbar(mappable = s, cax = cax, extend = cb_extend, ticks = cbticks)
        if !highlight_nodes
            cb.ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
        end
        cb.ax.set_title(cbaxtitle)
    end

    if rossbyridges
        rossby_ridge_lines(mr; νnHzunit, ax, kw...)
    end

    zoom = V_symmetric && get(kw, :zoom, !get(kw, :diffrot, false))

    if zoom
        zoom_inds = in(m_zoom).(mr)
        mr_zoom = mr[zoom_inds]
        lamscat_inset = mapreduce(real, vcat,
            [λs_m .- subtract_sectoral * rossby_ridge(m) for (λs_m) in zip(lams[zoom_inds], mr_zoom)]) .* νnHzunit
        mcat_inset = reduce(vcat, [range(m, m, length(λi)) for (m, λi) in zip(mr_zoom, lams[zoom_inds])])

        c = if fillmarker && highlight_nodes
            ns_cat_inset = reduce(vcat, ns[zoom_inds])
        elseif fillmarker
            lamsimcat_inset = mapreduce(imag, vcat, lams[zoom_inds]) .* νnHzunit
        else
            "white"
        end

        axins = ax.inset_axes([0.3, 0.1, 0.4, 0.4])
        ymax = (subtract_sectoral * RossbyWaveSpectrum.rossby_ridge(minimum(m_zoom)) + 0.005) * νnHzunit
        ymin = (subtract_sectoral * RossbyWaveSpectrum.rossby_ridge(maximum(m_zoom)) - 0.02) * νnHzunit
        axins.set_ylim(minmax(-ymin, -ymax)...)
        axins.xaxis.set_major_locator(ticker.NullLocator())
        axins.yaxis.set_major_locator(ticker.NullLocator())
        plt.setp(axins.spines.values(), color = "0.2", lw = "0.5")
        axins.scatter(mcat_inset, -lamscat_inset; c, ScatterParams..., cmap, norm,
            s=get(kw, :s, 25))

        if rossbyridges
            rossby_ridge_lines(m_zoom; νnHzunit, ax = axins, kw...)
        end

        ax.indicate_inset_zoom(axins, edgecolor = "grey", lw = 0.5)
    end

    showlegend = get(kw, :legend, true)
    if rossbyridges && showlegend
        ax.legend(loc = "best", fontsize = 12)
    end

    if get(kw, :tight_layout, true)
        f.tight_layout()
    end

    ax.axhline(0, ls="dotted", color="0.2", zorder = 0, lw=0.5)

    if !isnothing(encircle_m)
        m_encircle, eigvalinds = encircle_m
        if isnothing(rectpatchcoords)
            m_encircle_ind = findfirst(==(m_encircle), mr)
            λm = lams[m_encircle_ind]
            λenc = .- real.(λm[eigvalinds]).* νnHzunit
            λencmin, λencmax = extrema(λenc)
            Δλ = λencmax - λencmin
            minind, maxind = extrema(eigvalinds)
            if minind == 1
                λbelow = minimum(.- real.(λm[minind:min(end, maxind+1)])).* νnHzunit
                λabove = Inf
            elseif maxind == length(λm)
                λbelow = -Inf
                λabove = maximum(.- real.(λm[max(1, minind-1):maxind])).* νnHzunit
            else
                λbelow, λabove = extrema(.- real.(λm[max(1,minind-1):min(maxind+1,end)])).* νnHzunit
            end
            pad_below = min(Δλ/4, (λencmin - λbelow)/2)
            pad_below = pad_below == 0 ? max(0.05 * νnHzunit) : pad_below
            pad_above = min(Δλ/4, (λabove - λencmax)/2)
            pad_above = pad_above == 0 ? max(0.05 * νnHzunit) : pad_above
            pad_below = min(pad_below, pad_above)
            pad_above = pad_below
            pad_height = pad_below + pad_above
            height = λencmax - λencmin + pad_height
            width = 0.8
            rectpatchxy = (m_encircle-width/2, λencmin - pad_below)
            rectpatchcoords = RectPatchCoords(rectpatchxy, width, height)
        end
        (; xy, width, height) = rectpatchcoords
        xy = (xy[1], xy[2] + subtract_sectoral * rossby_ridge(m_encircle) * νnHzunit)
        rectpatch = matplotlib.patches.Rectangle(xy, width, height,
            ec="darkorange", fc="None" ,ls="dashed", lw=1.5,
            )
        ax.add_artist(rectpatch)
    end

    ΔΩ_scale = get(kw, :ΔΩ_scale, 1.0)

    titlestr = ""
    if get(kw, :showtitle, true)
        titlestr *= "Spectrum : " * (V_symmetric ? "" : "anti") * "symmetric modes"
    end
    if ΔΩ_scale != 1
        if !isempty(titlestr)
            titlestr *= "\n"
        end
        titlestr *= "$(round(ΔΩ_scale, digits=2)) ΔΩ"
    end
    ax.set_title(titlestr, fontsize = 12)

    cid = f.canvas.mpl_connect("pick_event", onpick)

    if get(kw, :save, false)
        V_symmetric = kw[:V_symmetric]
        V_symmetric_str = V_symmetric ? "sym" : "asym"
        filenametag = get(kw, :filenametag, V_symmetric_str)
        rotation = get(kw, :rotation, "uniform")
        filename = "$(rotation)_rotation_spectrum_$(filenametag).eps"
        savefiginfo(f, filename)
    end
    f, ax, s, rectpatchcoords
end

function onpick(event)
    ind = event.ind[] + 1 # zero-based in python to 1-based in Julia
    data = event.artist.get_offsets()
    m = round(Int, data[ind, 1])
    λ = data[ind, 2]
    mλ = m => round(λ, sigdigits=4)
    println(mλ)
end

function uniform_rotation_spectrum(fsym::FilteredEigen, fasym::FilteredEigen;
        plot_linewidths = true,
        kw...)
    f, axlist = subplots(2, 1 + plot_linewidths, sharex = true)
    m_min = get(kw, :m_min, nothing)
    @unpack mr = fsym
    if !isnothing(m_min)
        mr = max(m_min, minimum(mr)):maximum(mr)
        fsym = fsym[mr]
        fasym = fasym[mr]
    end
    fsymfilt = update_cutoffs(fsym; kw..., Δl_filter = get(kw, :Δl_filter_sym, nothing))
    fasymfilt = update_cutoffs(fasym; kw..., Δl_filter = get(kw, :Δl_filter_asym, nothing))
    commonkw = Base.pairs((; save = false, nodes_cutoff = nothing, Δl_filter = nothing))
    spectrum(fsymfilt; kw..., f, ax = axlist[1,1], commonkw...)
    rossby_ridge_data(fsymfilt; f, ax = axlist[1,1], kw..., commonkw...)
    spectrum(fasymfilt; kw..., f, ax = axlist[2,1], commonkw...)
    high_frequency_ridge_data(fasymfilt; f, ax = axlist[2,1], kw..., commonkw...)

    if plot_linewidths
        damping_rossbyridge(fsymfilt; f, ax = axlist[1,2], kw..., commonkw...)
        damping_highfreqridge(fasymfilt; f, ax = axlist[2,2], kw..., commonkw...)
    end

    f.set_size_inches(4.5 .* (1 + plot_linewidths), 6)
    f.tight_layout()
    if get(kw, :save, false)
        fpath = "spectrum_sym_asym.eps"
        savefiginfo(f, fpath)
    end
    return f, axlist
end

function mantissa_exponent_format(a)
    a_exponent = floor(Int, log10(a))
    a_mantissa = round(a / 10^a_exponent, sigdigits = 1)
    (a_mantissa == 1 ? "" : L"%$a_mantissa\times") * L"10^{%$a_exponent}"
end

const RidgeFrequencies = OrderedDict(
    1 => OrderedDict(
            8 => 0.0441767058623623,
            9 => -0.012902736776068121,
            10 => -0.06326721504551565,
            11 => -0.1099981491497498,
            12 => -0.1543384463727554,
            13 => -0.19691518663508156,
            14 => -0.23812653480986173,
            15 => -0.27825410209910145,
            16 => -0.317508129341382,
            17 => -0.3560509521867041,
            18 => -0.39401110035287984,
            19 => -0.43149190258758113,
            20 => -0.468576755153908,
        ),

    2 => OrderedDict(
            4 => 0.3575,
            5 => 0.2998,
            7 => 0.1917,
            8 => 0.1559,
            9 => 0.1253,
            10 => 0.09778,
            11 => 0.07234,
            12 => 0.04837,
            13 => 0.02546,
            14 => 0.003339,
            11 => 0.07234,
            12 => 0.04837,
            13 => 0.02546,
            14 => 0.003339,
            15 => -0.01818,
            16 => -0.03921,
            17 => -0.05984,
            18 => -0.08013,
            19 => -0.1001,
            20 => -0.1199,
        ),

    3 => OrderedDict(
            2 => 0.6968,
            3 => 0.557,
            4 => 0.489,
            5 => 0.468,
            6 => 0.4609,
            7 => 0.469,
            8 => 0.489,
            9 => 0.504,
            10 => 0.5444,
        ),

    4 => OrderedDict(
            12 => -0.1089,
            13 => -0.1488,
            14 => -0.1869,
            15 => -0.2239,
            16 => -0.2602,
            17 => -0.2958,
            18 => -0.331,
            20 => -0.4004,
        ),
)

function rossbyridge_mode_indices(lams, mr; operators, kw...)
    @unpack Ω0, ν = operators.constants
    νnHzunit = -freqnHzunit(Ω0)
    νtarget = map(mr) do m
        if haskey(RidgeFrequencies[2], m)
            RidgeFrequencies[2][m]
        else
            RossbyWaveSpectrum.rossby_ridge(m)
        end
    end
    map(zip(mr, lams, νtarget)) do (m, λ, λ0)
        isempty(λ) && return eltype(λ)[NaN + im*NaN]
        argmin(abs.(real.(λ) .- λ0))
    end
end
function rossbyridge_mode_frequencies(lams, mr; operators, kw...)
    inds = rossbyridge_mode_indices(lams, mr; operators, kw...)
    map(zip(lams, inds)) do (λ, i)
        λ[i]
    end
end

function damping_ridge(λs_ridge, mr; operators, f = figure(), ax = subplot(), kw...)
    @unpack Ω0, ν = operators.constants
    ν *= Ω0 * Rsun^2

    scale_freq = get(kw, :scale_freq, true)
    νnHzunit = scale_freq ? freqnHzunit(Ω0) : 1.0

    FWHM_ridge = imag.(λs_ridge) .* 2 #= HWHM to FWHM =# * νnHzunit

    ax.plot(mr, FWHM_ridge,
            ls="dotted",
            color=get(kw, :color, "grey"),
            marker=get(kw, :marker, "o"),
            mfc=get(kw, :mfc, "white"),
            mec="black",
            ms=5,
            label=get(kw, :label, ""),
            zorder = 4)

    # observations
    if get(kw, :plot_dispersion, true)
        plot_dispersion(ProxaufFit; operators, var=:γ, ax,
            color= "grey", label="Proxauf [2020]", zorder = 1,
            scale_freq)
    end

    ax.set_ylabel("FWHM [nHz]", fontsize = 12)
    ax.set_xlabel("m", fontsize = 12)
    ax.legend(loc="best")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer = true))
    f.set_size_inches(4.8,3.5)
    f.tight_layout()
    f, ax
end

function damping_ridge(Feig::FilteredEigen, ridgefreqs::OrderedDict; kw...)
    mr = collect(keys(ridgefreqs))
    λs = closest_eigenvalue_to_ridge(Feig, ridgefreqs; kw...)
    damping_ridge(λs, mr; Feig.operators, kw...)
end

function damping_multipleridges(Feig::FilteredEigen; kw...)
    f, ax = figure(), subplot()
    damping_ridge(Feig, RidgeFrequencies[1];
        f, ax, label="Ridge1", kw...)
    for (i,marker) in enumerate(["d", "s", "."])
        damping_ridge(Feig, RidgeFrequencies[i+1];
            f, ax, label="Ridge$(i+1)",
            marker, mfc="$(0.3 * i)",
            plot_dispersion = false,
            kw...)
    end
    f.set_size_inches(7,4.5)
    if get(kw, :save, false)
        filename = "damping_multiridges.eps"
        savefiginfo(f, filename)
    end
    f, ax
end

function damping_rossbyridge(lams, mr; operators, f = figure(), ax = subplot(), kw...)
    V_symmetric = kw[:V_symmetric]
    @assert V_symmetric "V must be symmetric for rossby ridge plots"
    @unpack Ω0, ν = operators.constants
    ν *= Ω0 * Rsun^2

    λs_rossbyridge = rossbyridge_mode_frequencies(lams, mr; operators)

    damping_ridge(λs_rossbyridge, mr; operators, f, ax,
        label="model, ν = $(@sprintf "%.0f" ν/10^10) " * "km" * L"^2/" * "s",
        kw...)
    ax.set_title("Rossby-ridge linewidths", fontsize = 12)
    if get(kw, :save, false)
        filename = "damping_rossby.eps"
        savefiginfo(f, filename)
    end
    return f, ax
end

function HFR_mode_indices(lams, mr; operators, kw...)
    @unpack Ω0 = operators.constants
    νnHzunit = freqnHzunit(Ω0)
    map(zip(mr, lams)) do (m, λ)
        (isempty(λ) || m < 5) && return missing
        if haskey(HansonHFFit, m)
            λ0 = value(HansonHFFit[m].ν)/(-νnHzunit)
            findmin(x -> abs(real(x) - λ0), λ)[2]
        else
            lower_cutoff = RossbyWaveSpectrum.rossby_ridge(m)
            upper_cutoff = RossbyWaveSpectrum.rossby_ridge(m) + 180/νnHzunit
            λ_inrange = λ[lower_cutoff .< real.(λ) .< upper_cutoff]
            isempty(λ_inrange) ? missing : findmin(imag, λ_inrange)[2]
        end
    end
end

function HFR_mode_frequencies(lams, mr; operators, kw...)
    @unpack Ω0 = operators.constants
    νnHzunit = freqnHzunit(Ω0)
    inds = HFR_mode_indices(lams, mr; operators, kw...)
    map(zip(mr, lams, inds)) do (m, λ, ind)
        ismissing(ind) && return eltype(λ)(NaN + im*NaN)
        return λ[argmin(abs.(real.(λ) .- λ0))]
    end
end

function damping_highfreqridge(lams, mr; operators, f = figure(), ax = subplot(), kw...)
    V_symmetric = kw[:V_symmetric]
    @assert !V_symmetric "V must be antisymmetric for high-frequency rossby ridge plots"
    @unpack Ω0, ν = operators.constants
    ν *= Ω0 * Rsun^2

    νnHzunit = freqnHzunit(Ω0)

    # observations
    plot_dispersion(HansonHFFit; operators, var=:γ, ax, color= "grey", label="H22", zorder = 1)

    # model
    λs_HFRridge = HFR_mode_frequencies(lams, mr; operators)

    ax.plot(mr, imag.(λs_HFRridge) .* 2 #= HWHM to FWHM =# * νnHzunit,
            ls="dotted", color="grey", marker="o", mfc="white",
            mec="black", ms=5, label="this work", zorder = 4)

    ax.set_ylabel("linewidth [nHz]", fontsize = 12)
    ax.set_xlabel("m", fontsize = 12)
    ax.set_title("High-frequency modes", fontsize = 12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer = true))

    ax.legend(loc="best")

    f.tight_layout()
    if get(kw, :save, false)
        filename =  "damping_highfreqridge.eps"
        savefiginfo(f, filename)
    end
    return f, ax
end

function rossby_ridge_data(lams, mr; operators, f = figure(), ax = subplot(), kw...)
    # observations
    plot_dispersion(HansonHighmfit; operators, ax, color= "0.3", ls="dashed", label="Hanson")
    ax.legend(loc="best")
    ax.set_xlim(extrema(mr) .+ (-0.5, 0.5))

    if get(kw, :save, false)
        filename = "rossbyridge.eps"
        savefiginfo(f, filename)
    end
    return f, ax
end

function high_frequency_ridge_data(lams, mr; operators, f = figure(), ax = subplot(), kw...)
    # observations
    plot_dispersion(HansonHFFit; operators, ax, color= "0.3", ls="dashed", label="H22")

    ax.legend(loc="best")
    ax.set_xlim(extrema(mr) .+ (-0.5, 0.5))

    if get(kw, :save, false)
        filename = "highfreqridge.eps"
        savefiginfo(f, filename)
    end
    return f, ax
end

for f in [:spectrum, :damping_highfreqridge, :damping_rossbyridge,
        :high_frequency_ridge_data, :rossby_ridge_data,
        :rossbyridge_mode_indices, :rossbyridge_mode_frequencies,
        :HFR_mode_frequencies, :HFR_mode_indices]
    @eval function $f(Feig::FilteredEigen; kw...)
        $f(Feig.lams, Feig.mr; Feig.operators, vecs = Feig.vs, Feig.kw..., kw...)
    end
end

function diffrot_rossby_ridge(Fsym,
        mr=min(8, maximum(axes(Fsym.lams,1))):maximum(axes(Fsym.lams,1));
        ΔΩ_frac_low = -0.01, ΔΩ_frac_high = 0.05,
        plot_spectrum = false, plot_lower_cutoff = plot_spectrum,
        plot_upper_cutoff = plot_spectrum, kw...)

    @assert Fsym.kw[:V_symmetric] "only symmetric solutions supported"
    (; lams, vs) = Fsym[mr];
    @unpack operators = Fsym;
    @unpack rpts = operators;
    @unpack Ω0 = operators.constants;
    ν0 = freqnHzunit(Ω0)
    ν0_Sun = 453.1
    λRossby_high = RossbyWaveSpectrum.rossby_ridge.(mr, ΔΩ_frac = ΔΩ_frac_low)
    λRossby_low = RossbyWaveSpectrum.rossby_ridge.(mr, ΔΩ_frac = ΔΩ_frac_high)
    lams_realfilt = map(zip(mr, lams, vs)) do (m, lam_m, vs_m)
        # select the mode that looks like l == m
        lam_m, _ = argmin(zip(lam_m, eachcol(vs_m))) do (lam, v)
            # Vr,Wr = RossbyPlots.compute_eigenfunction_spectra(v, m; operators, V_symmetric = true);
            (; VWSinv, θ) = RossbyWaveSpectrum.eigenfunction_realspace(v, m; operators, V_symmetric = true);
            Vr_realspace = real(VWSinv.V);
            r_ind_peak = RossbyWaveSpectrum.peakindabs1(Vr_realspace)
            peak_latprofile = @view Vr_realspace[r_ind_peak, :]
            _, θind = findmax(abs, peak_latprofile)
            t1 = abs(pi/2 - θ[θind])
            minval, maxval = extrema(Vr_realspace);
            maxsign = abs(minval) > abs(maxval) ? -1 : 1
            maxvalabs = maximum(abs, (minval,maxval))
            Vr_realspace_signed = Vr_realspace .* maxsign;
            target = normalize((rpts .* sin.(θ)').^m, Inf) .* maxvalabs;
            t2 = sum(abs2, Vr_realspace_signed .- target)
            t1 * t2
        end
        real(lam_m)
    end
    lams_realfilt_plot = lams_realfilt.*(-ν0)
    plot_spectrum && begin
        f, ax = spectrum(Fsym, zorder=2)
        ax.plot(mr, lams_realfilt_plot, marker="o", ls="dotted",
            mec="black", mfc="None", ms=6, color="grey", zorder=1)
        plot_upper_cutoff && ax.plot(mr, λRossby_low.*(-ν0), ".--", color="orange", zorder=0)
        plot_lower_cutoff && ax.plot(mr, λRossby_high.*(-ν0), ".--", color="orange", zorder=0)
        plot_lower_cutoff && plot_lower_cutoff && ax.fill_between(mr,
                λRossby_high.*(-ν0), λRossby_low.*(-ν0), color="navajowhite", zorder=0)
        ax.set_ylim(top = minimum(λRossby_low)*(-ν0)*1.2)
    end

    f, ax = subplots(1,1)
    mr_Hanson = collect(keys(HansonGONGfit))
    ν0_uniform_solar_tracking_rate = RossbyWaveSpectrum.rossby_ridge.(mr_Hanson).*(-ν0_Sun)
    ν_H = [HansonGONGfit[m].ν for m in mr_Hanson]
    ν_H_val = value.(ν_H)

    δν = ν_H_val .- ν0_uniform_solar_tracking_rate
    ax.errorbar(mr_Hanson, δν,
        yerr = errorbars_pyplot(ν_H),
        color= "brown", ls="None", capsize = 3, label="Hanson 2020 [GONG]",
        zorder = 3, marker = ".", ms = 5, mfc = "k")

    fit_m_min = 8
    fit_inds = mr_Hanson .>= fit_m_min
    mr_fit = mr_Hanson[fit_inds]
    fit_x = (m -> (m - 2/(m+1))).(mr_fit)
    yerr = max.(higherr.(ν_H), lowerr.(ν_H))[fit_inds]
    fit_y = δν[fit_inds]
    ΔΩ_fit_data = sum(fit_y.*fit_x./yerr.^2)/sum(fit_x.^2 ./yerr.^2)
    ax.plot(mr_fit, ΔΩ_fit_data.*fit_x, "--",
        label=L"δν_\mathrm{Ro}"*" [Doppler ΔΩ/2π = $(round(ΔΩ_fit_data, sigdigits=2)) nHz]",
        zorder=1, color="blue")

    ν0_uniform_model_tracking_rate = RossbyWaveSpectrum.rossby_ridge.(mr).*(-ν0)

    δν = lams_realfilt_plot - ν0_uniform_model_tracking_rate
    fit_x = (m -> (m - 2/(m+1))).(mr)
    fit_y = δν
    ΔΩ_fit_model = sum(fit_y.*fit_x)/sum(fit_x.^2)
    ax.plot(mr, δν,
        "o", color="grey", ms=4, label="model, radial rotation", zorder=5)
    ax.plot(mr, ΔΩ_fit_model .* fit_x, "--",
        label=L"δν_\mathrm{Ro}"*" [Doppler ΔΩ/2π = $(round(ΔΩ_fit_model, digits=1)) nHz]",
        zorder=1, color="orange")
    low_m = 1:minimum(mr)
    fit_x = (m -> (m - 2/(m+1))).(low_m)
    ax.plot(low_m, ΔΩ_fit_model .* fit_x, ls="dotted",
        zorder=1, color="orange")

    ax.axhline(0, ls="dotted", color="black")
    ax.set_xlabel("m", fontsize=12)
    ax.set_ylabel(L"δ\nu" * " [nHz]", fontsize=12)
    ax.legend(loc="best")
    f.set_size_inches(6,4)
    f.tight_layout()

    if get(kw, :save, false)
        rotation_profile = Fsym.kw[:rotation_profile]
        filename = "freqdiff_diffrot_$rotation_profile.eps"
        savefiginfo(f, filename)
    end
end

function plot_dispersion(dataset; operators, var = :ν, ax, scale_freq = true, kw...)
    @unpack Ω0 = operators.constants
    νnHzunit = scale_freq ? freqnHzunit(Ω0) : 1.0
    invνnHzunit = scale_freq ? 1.0 : 1/freqnHzunit(Ω0)

    mr = get(kw, :m_range, nothing)

    mr_Hanson = collect(keys(dataset))
    if !isnothing(mr)
        mr_Hanson = intersect(mr, mr_Hanson)
    end

    subtract_sectoral = get(kw, :subtract_sectoral, false)
    ν_H = [getproperty(dataset[m], var) for m in mr_Hanson]
    ax.errorbar(mr_Hanson, value.(ν_H) .* invνnHzunit .+ subtract_sectoral .* rossby_ridge.(mr_Hanson) .* νnHzunit;
        yerr = errorbars_pyplot(ν_H) .* invνnHzunit,
        xerr=nothing,
        ls="None",
        zorder = 2, marker = ".", ms = 5,
        mfc = "k",
        color=get(kw, :color, nothing),
        fmt=get(kw, :fmt, ""),
        ecolor=get(kw, :ecolor, nothing),
        elinewidth=get(kw, :elinewidth, nothing),
        capsize=get(kw, :capsize, 3),
        barsabove=get(kw, :barsabove, false),
        lolims=get(kw, :lolims, false),
        uplims=get(kw, :uplims, false),
        xlolims=get(kw, :xlolims, false),
        xuplims=get(kw, :xuplims, false),
        errorevery=get(kw, :errorevery, 1),
        capthick=get(kw, :capthick, nothing),
        label=get(kw, :label, nothing),
        )
end

function spectrum_with_datadispersion(Feig::FilteredEigen; kw...)
    spectrum_with_datadispersion((Feig, Feig); kw...)
end
function closest_eigenvalue_to_ridge(Feig, ridgefreqs, component = identity; kw...)
    mr_line = collect(keys(ridgefreqs))
    snaptopoint = get(kw, :snaptopoint, true)
    map(mr_line) do m
        λtarget = ridgefreqs[m]
        !snaptopoint && return λtarget
        component(argmin(x->abs(real(x) - λtarget), Feig[m].lams))
    end
end
function highlight_ridge(Feig, ridgefreqs::OrderedDict; ax = subplot(), kw...)
    @unpack Ω0 = Feig.operators.constants
    νnHzunit = get(kw, :scale_freq, true) ? freqnHzunit(Ω0)  #= around 453 =# : 1.0
    mr_line = collect(keys(ridgefreqs))
    λs = closest_eigenvalue_to_ridge(Feig, ridgefreqs, real; kw...)
    subtract_sectoral = get(kw, :subtract_sectoral, false)

    ax.plot(mr_line,
        (λs .- subtract_sectoral * rossby_ridge.(mr_line)) .* (-νnHzunit);
        marker=get(kw, :marker, "None"),
        ls=get(kw, :ls, "dashed"),
        color=get(kw, :color, "purple"),
        label=get(kw, :label, ""),
        zorder=get(kw, :zorder, 0),
    )
end
function spectrum_with_datadispersion((Feig, Feig_filt)::NTuple{2,FilteredEigen}; kw...)
    V_symmetric = Feig.kw[:V_symmetric]
    nodes_cutoff = get(kw, :nodes_cutoff, nothing)
    Δl_filter = get(kw, :Δl_filter, Int(!V_symmetric))
    plotfull = get(kw, :plotfull, true)
    plotzoom = get(kw, :plotzoom, true)
    legend_subplot = get(kw, :legend_subplot, true)
    nsubplots = 1 + legend_subplot + (plotfull & plotzoom)
    f, axlist = subplots(1, nsubplots, sharex=true,
        gridspec_kw = Dict("width_ratios" => [fill(1 + legend_subplot, nsubplots-1); 1]),
        squeeze=false)

    @unpack operators = Feig
    @unpack Ω0 = operators.constants
    νnHzunit = get(kw, :scale_freq, true) ? freqnHzunit(Ω0)  #= around 453 =# : 1.0

    highlight_ridges = get(kw, :highlight_ridge, true)
    ridgecolors = ["purple", "teal", "chocolate", "olive"]

    if plotfull
        ax = axlist[1]
        spectrum(Feig; zorder=3, fillmarker = get(kw, :fillmarker, false),
            Δl_filter, nodes_cutoff, Feig.kw..., kw..., save=false, f, ax,
            showtitle = false, legend=false)

        commonkw = (; operators, ax, m_range = Feig.mr)
        if V_symmetric
            colorederrorbars = get(kw, :colorederrorbars, false)
            hansoncolor, liangcolor, proxaufcolor = if colorederrorbars
                "red", "blue", "green"
            else
                "0.6", "0.4", "0.7"
            end
            plot_dispersion(HansonGONGfit; commonkw...,
                color= hansoncolor, label="Hanson 2020 [GONG]", kw...)
            plot_dispersion(Liang2019MDIHMI; commonkw...,
                color= liangcolor, label="Liang 2019 [MDI & HMI]", kw...)
            plot_dispersion(ProxaufFit; commonkw...,
                color= proxaufcolor, ls="dashed", label="Proxauf 2020 [HMI]", kw...)
            plot_dispersion(HansonHighmfit; commonkw...,
                color= "black", label="Hanson", kw...)

            if highlight_ridges
                for (ridge, color) in enumerate(ridgecolors)
                    highlight_ridge(Feig, RidgeFrequencies[ridge];
                        color, label="Ridge $ridge", ax)
                end
            end
        else
            color = colorederrorbars ? "brown" : "0.6"
            plot_dispersion(HansonHFFit; commonkw...,
                color, label="Hanson 2022 [HMI]", kw...)
        end
    end

    if plotzoom
        ax = axlist[plotfull+1]
        subtract_sectoral = true
        spectrum(Feig_filt; zorder=3, fillmarker = get(kw, :fillmarker, false),
            Δl_filter, nodes_cutoff, Feig_filt.kw..., kw..., save=false, f, ax,
            showtitle = false, subtract_sectoral,
            encircle_m = nothing)

        commonkw = (; operators, ax, m_range = Feig_filt.mr)
        if V_symmetric
            colorederrorbars = get(kw, :colorederrorbars, false)
            hansoncolor, liangcolor, proxaufcolor = if colorederrorbars
                "red", "blue", "green"
            else
                "0.6", "0.4", "0.7"
            end
            plot_dispersion(HansonGONGfit; commonkw...,
                color= hansoncolor, label="Hanson 2020 [GONG]", subtract_sectoral, kw...)
            plot_dispersion(Liang2019MDIHMI; commonkw...,
                color= liangcolor, label="Liang 2019 [MDI & HMI]", subtract_sectoral, kw...)
            plot_dispersion(ProxaufFit; commonkw...,
                color= proxaufcolor, ls="dashed", label="Proxauf 2020 [HMI]", subtract_sectoral, kw...)
            plot_dispersion(HansonHighmfit; commonkw...,
                color="black", label="Hanson [HMI]", subtract_sectoral, kw...)

            if highlight_ridges
                for (ridge, color) in enumerate(ridgecolors[1:3])
                    highlight_ridge(Feig_filt, RidgeFrequencies[ridge];
                        color, label="Ridge $ridge", ax, subtract_sectoral)
                end
            end

            ax.set_ylim(-300, 300)
        else
            color = colorederrorbars ? "brown" : "0.6"
            plot_dispersion(HansonHFFit; commonkw...,
                color, label="Hanson 2022 [HMI]", subtract_sectoral, kw...)
        end
    end

    leghandles, leglabels = axlist[1].get_legend_handles_labels()

    if haskey(kw, :xlim)
        ax.set_xlim(kw[:xlim])
    end

    axlist[end].legend(leghandles, leglabels, loc="best")

    if legend_subplot
        axlist[end].axis("off")
    end

    f.set_size_inches(length(axlist)*4, 3.5)
    f.tight_layout()
    if get(kw, :save, false)
        tag = get(kw, :filenametag, "")
        filename = "spectrum_$(!V_symmetric ? "a" : "")sym$tag.eps"
        savefiginfo(f, filename)
    end
end

function piformatter(x, _)
    n = round(Int, 4x / pi)
    prestr = (n == 4 || n == 1) ? "" : iseven(n) ?
             (n == 2 ? "" : string(div(n, 2))) : string(n)
    poststr = n == 4 ? "" : iseven(n) ? "/2" : "/4"
    prestr * "π" * poststr
end

function differential_rotation_spectrum(fconstsym::FilteredEigen,
    fconstasym::FilteredEigen, fradsym::FilteredEigen, fradasym::FilteredEigen; kw...)

    f, axlist = subplots(2, 3, sharex = "col")
    @unpack rpts = fconstsym.operators.coordinates;
    r_frac = rpts ./ Rsun
    Ω0_const = fconstsym.operators.constants[:Ω0];
    Ω0_radial = fradsym.operators.constants[:Ω0];

    lam_constant_sym, lam_constant_asym = fconstsym.lams, fconstasym.lams
    lam_radial_sym, lam_radial_asym = fradsym.lams, fradasym.lams

    ΔΩ_constant = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives_Fun(;
            operators = fconstsym.operators, rotation_profile = :constant).ΔΩ

    axlist[1,1].plot(r_frac, ΔΩ_constant.(rpts) .* freqnHzunit(Ω0_const), color="0.2")
    axlist[1,1].axhline(0, color="black", ls="dotted", label="tracking rate")
    axlist[1,1].set_title("Constant ΔΩ", fontsize=12)
    axlist[1,1].yaxis.set_major_locator(ticker.MaxNLocator(4))
    axlist[1,1].set_ylabel(L"\Delta\Omega/2\pi" * " [nHz]", fontsize=12)
    axlist[1,1].legend(loc="best")

    kwextra = (; uniform_rotation_ridge = true, ΔΩ_frac = ΔΩ_constant(0))
    spectrum(fconstsym; f = f, ax = axlist[1,2], kwextra...)
    spectrum(fconstasym; f = f, ax = axlist[1,3], kwextra..., sectoral_rossby_ridge = false)
    axlist[1,2].set_ylim(-200, 700)
    axlist[1,3].set_ylim(-200, 700)

    ΔΩ_solar_radial = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives_Fun(;
            operators = fradsym.operators, rotation_profile = :solar_equator).ΔΩ

    axlist[2,1].plot(r_frac, ΔΩ_solar_radial.(rpts) .* freqnHzunit(Ω0_radial), color="0.2")
    axlist[2,1].axhline(0, color="black", ls="dotted", label="tracking rate")
    axlist[2,1].set_title("Solar equatorial ΔΩ", fontsize=12)
    axlist[2,1].yaxis.set_major_locator(ticker.MaxNLocator(4))
    axlist[2,1].set_xlabel(L"r/R_\odot", fontsize=12)
    axlist[2,1].set_ylabel(L"\Delta\Omega/2\pi" * " [nHz]", fontsize=12)
    axlist[2,1].legend(loc="best")

    kwextra = (; sectoral_rossby_ridge = false, uniform_rotation_ridge = true)
    spectrum(fradsym; f = f, ax = axlist[2,2], kwextra...)
    spectrum(fradasym; f = f, ax = axlist[2,3], kwextra...)
    axlist[2,2].set_ylim(-200, 700)
    axlist[2,3].set_ylim(-200, 700)

    f.set_size_inches(12, 6)
    # for some reason, tight_layout needs to be called twice
    f.tight_layout()
    f.tight_layout()
    if get(kw, :save, false)
        filename = "diff_rot_spectrum.eps"
        savefiginfo(f, filename)
    end
    return nothing
end

function eigenfunction(VWSinv::NamedTuple, θ::AbstractVector, m;
        operators, field = :V, f = figure(), component = real,
        angular_profile = :surface,
        radial_profile = :equator,
        λ = nothing,
        kw...)

    V = getproperty(VWSinv, field)::Matrix{ComplexF64}
    Vr = copy(component(V))
    scale = get(kw, :scale) do
        eignorm(Vr)
    end
    if scale != 0
        Vr ./= scale
    end
    @unpack rpts = operators
    @unpack r_in, r_out = operators.radial_params
    r_frac = rpts ./ Rsun
    equator_ind = argmin(abs.(θ .- pi/2))
    ind_max = if radial_profile == :max
        RossbyWaveSpectrum.angularindex_maximum(Vr, θ)
    elseif radial_profile == :equator
        RossbyWaveSpectrum.indexof_equator(θ)
    else
        throw(ArgumentError("radial_profile must be one of :max or :equator"))
    end
    V_peak_depthprofile = @view Vr[:, ind_max]
    r_out_ind = argmin(abs.(rpts .- r_out))

    cmap = get(kw, :cmap, "Greys")

    polar = get(kw, :polar, false)
    crossections = get(kw, :crossections, !polar)

    if !polar && crossections
        spec = f.add_gridspec(3, 3)
        axprofile = get(kw, :axprofile) do
            f.add_subplot(py"$(spec)[1:, 1:]")
        end
        axsurf = let axprofile=axprofile
            get(kw, :axsurf) do
                f.add_subplot(py"$(spec)[0, 1:]", sharex = axprofile)
            end
        end
        axdepth = let axprofile=axprofile
            get(kw, :axdepth) do
                f.add_subplot(py"$(spec)[1:, 0]", sharey = axprofile)
            end
        end
        axlist = [axprofile, axsurf, axdepth]
    else
        axprofile = get(kw, :axprofile) do
            f.add_subplot()
        end
        axlist = [axprofile]
    end

    norm0 = matplotlib.colors.CenteredNorm()

    if !polar
        r_plot_ind = if angular_profile == :max
            argmax(abs.(V_peak_depthprofile))
        elseif angular_profile == :surface
            r_out_ind
        end
        latidudes = 90 .- rad2deg.(θ)
        axsurf.plot(latidudes, (@view Vr[r_plot_ind, :]), color = "black")
        if get(kw, :setylabel, true)
            axsurf.set_ylabel("Angular\nprofile", fontsize = 11)
        end
        # axsurf.set_xticks(pi * (1/4:1/4:1))
        axsurf.axhline(0, ls="dotted", color="black")
        # axsurf.xaxis.set_major_formatter(ticker.FuncFormatter(piformatter))
        yfmt = ticker.ScalarFormatter(useMathText = true)
        yfmt.set_powerlimits((-1,1))
        axsurf.yaxis.set_major_formatter(yfmt)

        axdepth.axvline(0, ls="dotted", color="0.3")
        axdepth.plot(V_peak_depthprofile, r_frac,
            color = "black",
        )
        axdepth.set_xlabel("Depth profile", fontsize = 11)
        axdepth.set_ylabel(L"r/R_\odot", fontsize = 11)
        axdepth.yaxis.set_major_locator(ticker.MaxNLocator(4))
        axdepth.xaxis.set_major_locator(ticker.MaxNLocator(2))
        xfmt = ticker.ScalarFormatter(useMathText = true)
        xfmt.set_powerlimits((-1,1))
        axdepth.xaxis.set_major_formatter(xfmt)
        axdepth.xaxis.tick_top()

        axprofile.pcolormesh(latidudes, r_frac, Vr; cmap, norm=norm0, rasterized = true, shading = "auto")
        xlabel = get(kw, :longxlabel, true) ? "Latitude [degrees]" : "Lat [deg]"
        axprofile.set_xlabel(xlabel, fontsize = 11)
        axprofile.invert_xaxis()
    else
        x = r_frac .* sin.(θ)'
        z = r_frac .* cos.(θ)'
        p = axprofile.pcolormesh(x, z, Vr; cmap, norm=norm0, rasterized = true,
            shading = "auto")
        axprofile.yaxis.set_major_locator(ticker.MaxNLocator(3))
        axprofile.xaxis.set_major_locator(ticker.MaxNLocator(3))
        if get(kw, :showxlabel, true)
            axprofile.set_xlabel(L"x/R_\odot", fontsize=12)
        end
        if get(kw, :showylabel, true)
            axprofile.set_ylabel(L"z/R_\odot", fontsize=12)
        end
        if get(kw, :set_figsize, true)
            f.set_size_inches(3, 3)
        end
        x = r_in/Rsun .* sin.(θ)
        z = r_in/Rsun .* cos.(θ)
        axprofile.plot(x, z, color="grey", lw="0.4")
        x = r_out/Rsun .* sin.(θ)
        z = r_out/Rsun .* cos.(θ)
        axprofile.plot(x, z, color="grey", lw="0.4")
        if get(kw, :colorbar, true)
            cb = colorbar(mappable=p, ax=axprofile)
            cb.ax.get_yaxis().set_major_locator(ticker.MaxNLocator(4))
        end
    end

    if get(kw, :title, true)
        if get(kw, :title_fieldtype, !(get(kw, :title_eigval, false)))
            fieldname = field == :V ? "Toroidal" :
                        field == :W ? "Poloidal" :
                        field == :S ? "Entropy" : nothing
            axprofile.set_title("$fieldname streamfunction m = $m")
        elseif get(kw, :title_eigval, false)
            @unpack Ω0 = operators.constants
            νnHzunit = get(kw, :scale_freq, true) ? freqnHzunit(Ω0)  #= around 453 =# : 1.0
            title = ""
            if get(kw, :titlem, false)
                title *= "m = $m"
            end
            if get(kw, :titlefreq, true)
                if !isempty(title)
                    title *= ", "
                end
                title *= L"\Re[\nu]" * " = " * string(round(-real(λ)*νnHzunit, sigdigits=3)) * " nHz"
            end
            axprofile.set_title(title)
        end
    end

    if !get(kw, :constrained_layout, false)
        f.tight_layout()
    end
    if get(kw, :save, false)
        filename = "eigenfunction.eps"
        savefiginfo(f, filename)
    end
    f, axlist
end

function eigenfunction(Feig::FilteredEigen, m::Integer, ind::Integer; kw...)
    @unpack operators = Feig
    λ, v = Feig[m][ind]
    eigenfunction(v, m; operators, λ, Feig.kw..., kw...)
end

function eigenfunction(v::AbstractVector{<:Number}, m::Integer; operators, kw...)
    (; θ, VWSinv) = RossbyWaveSpectrum.eigenfunction_realspace(v, m; operators, kw...)
    eigenfunction(VWSinv, θ, m; operators, kw...)
end

function eigenfunctions(Feig::FilteredEigen, ms, inds; layout=(2,length(inds)÷2), kw...)
    fig = plt.figure(constrained_layout = true, figsize = (3*layout[2], 2.5*layout[1]))
    subfigs = permutedims(fig.subfigures(layout...))
    CIaxes = CartesianIndices(size(subfigs))
    kwd = Dict(kw)
    if haskey(kw, :save)
        delete!(kwd, :save)
    end
    ms2 = ms isa Integer ? fill(ms, length(inds)) : ms
    titlem = !(ms isa Integer)
    for (m, ind, sf, CI) in zip(ms2, inds, subfigs, CIaxes)
        eigenfunction(Feig, m, ind; f = sf,
            constrained_layout=true, title_eigval=true,
            set_figsize = false,
            colorbar = true,
            showxlabel = Tuple(CI)[2] == size(CIaxes,1), # last row for transposed axes
            showylabel = Tuple(CI)[1] == 1, # first col for transposed axes
            titlem,
            titlefreq = get(kw, :titlefreq, false),
            kwd...)
    end
    fig.set_size_inches(layout[2]*4, layout[1]*4)
    if get(kw, :save, false)
        tag = get(kw, :filenametag, join(ms, "_"))
        filename = joinpath(plotdir, "eigenfn_m$tag.eps")
        savefiginfo(fig, filename)
    end
    fig, subfigs
end

function eigenfunctions_ridge(Feig::FilteredEigen, ms, ridgeno=2; kw...)
    ΔΩ_scale = get(kw, :ΔΩ_scale, get(Feig.kw, :ΔΩ_scale, 1.0))
    ridgedata = rescale_ridge(RidgeFrequencies[ridgeno], ΔΩ_scale)
    inds = map(ms) do m
        λtarget = ridgedata[m]
        findmin(x->abs(real(x) - λtarget), Feig[m].lams)[2]
    end
    eigenfunctions(Feig, ms, inds; filenametag = "_ridge$ridgeno", kw...)
end

function eigenfunctions_allstreamfn(Feig::FilteredEigen, m::Integer, vind::Integer; kw...)
    (; θ, VWSinv) = RossbyWaveSpectrum.eigenfunction_realspace(Feig, m, vind; kw...)
    if get(kw, :scale_eigenvectors, false)
        RossbyWaveSpectrum.scale_eigenvectors!(VWSinv; Feig.operators)
    end
    eigenfunctions_allstreamfn(VWSinv, θ, m; Feig.operators, Feig.kw..., kw...)
end
function eigenfunctions_allstreamfn(v::AbstractVector{<:Number}, m::Integer; operators, kw...)
    (; θ, VWSinv) = RossbyWaveSpectrum.eigenfunction_realspace(v, m; operators, kw...)
    if get(kw, :scale_eigenvectors, false)
        RossbyWaveSpectrum.scale_eigenvectors!(VWSinv; operators)
    end
    eigenfunctions_allstreamfn(VWSinv, θ, m; operators, kw...)
end
function eigenfunctions_allstreamfn(VWSinv::NamedTuple, θ, m; operators, kw...)
    f = plt.figure(constrained_layout = true, figsize = (12, 8))
    subfigs = f.subfigures(2, 3, wspace = 0.15, width_ratios = [1, 1, 1])

    kw2 = Dict{Symbol, Any}(kw);
    kw2[:constrained_layout] = true
    kw2[:suptitle] = false
    kw2[:longxlabel] = false
    scale = eignorm(realview(VWSinv.V))
    kw2[:scale] = scale
    itr = Iterators.product((real, imag), (:V, :W, :S))
    title_str = get(kw, :scale_eigenvectors, false) ?
        Dict(:V => "V", :W => "W", :S => "S/cp") :
        Dict(:V => "V"*L"/R_\odot", :W => "W"*L"/R_\odot^2", :S => L"\Omega_0 R_\odot"*"S/cp")

    for (ind, (component, field)) in zip(CartesianIndices(axes(itr)), itr)
        eigenfunction(VWSinv, θ, m;
            operators,
            field, f = subfigs[ind],
            constrained_layout = true,
            setylabel = ind == 1 ? true : false,
            component,
            kw2...,
            save = false)
        subfigs[ind].suptitle(string(component)*"("*title_str[field]*")", x = 0.8)
    end
    if get(kw, :save, false)
        fname = "eigenfunctions_allstreamfn_m$(m).eps"
        savefiginfo(f, fname)
    end
end

function eigenfunction_rossbyridge(Feig::FilteredEigen, m; kw...)
    eigenfunction_rossbyridge(Feig[m].lams, Feig[m].vs, m; Feig.operators, Feig.kw..., kw...)
end

function eignorm(v)
    minval, maxval = extrema(v)
    abs(minval) > abs(maxval) ? minval : maxval
end

function multiple_eigenfunctions_surface_m(Feig::FilteredEigen, m; kw...)
    multiple_eigenfunctions_surface_m(Feig[m].lams, Feig[m].vs, m; Feig.operators, Feig.kw..., kw...)
end
function multiple_eigenfunctions_surface_m(λs::AbstractVector, vecs::AbstractMatrix, m;
        operators, f = figure(), V_symmetric,  kw...)

    @unpack rpts = operators;
    rsurfind = argmax(rpts)

    ax = f.add_subplot()
    ax.set_xlabel("colatitude (θ) [radians]", fontsize = 12)
    ax.set_ylabel("Angular profile", fontsize = 12)
    ax.set_xticks(pi * (1/4:1/4:1))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(piformatter))

    ax.set_title("Surface profiles", fontsize = 12)

    lscm = Iterators.product(("solid", "dashed", "dotted"), ("black", "0.5", "0.3"), ("None", "."))

    λ0 = RossbyWaveSpectrum.rossby_ridge(m)
    rossbyindex = argmin(abs.(real.(λs) .- λ0))

    nmodes = get(kw, :nmodes, 3)
    vm = vecs[:, range(rossbyindex, step=-1, length=nmodes)]

    for (ind, (v, (ls, c, marker))) in enumerate(zip(eachcol(vm), lscm))
        (; VWSinv, θ) = RossbyWaveSpectrum.eigenfunction_realspace(v, m; operators, V_symmetric)
        @unpack V = VWSinv
        Vr = realview(V)
        Vr_surf = Vr[rsurfind, :]

        Vrmax_sign = sign(eignorm(Vr_surf))
        Vr_surf .*= Vrmax_sign
        normalize!(Vr_surf)

        ax.plot(θ, Vr_surf; ls, color = c,
            label = "n = "*string(ind-1),
            marker, markevery = 10)
    end

    legend = ax.legend()
    legend.get_title().set_fontsize("12")
    if !get(kw, :constrained_layout, false)
        f.tight_layout()
    end
end

function multiple_eigenfunctions_radial_m(Feig::FilteredEigen, m; kw...)
    multiple_eigenfunctions_radial_m(Feig[m].lams, Feig[m].vs, m; Feig.operators, Feig.kw..., kw...)
end
function multiple_eigenfunctions_radial_m(λs::AbstractVector, vecs::AbstractMatrix, m;
        operators, f = figure(), V_symmetric, kw...)

    @unpack rpts = operators;
    rfrac = rpts./Rsun
    rsurfind = argmax(rpts)

    ax = f.add_subplot()
    ax.set_xlabel("Radial profile", fontsize = 12)
    ax.set_ylabel(L"r/R_\odot", fontsize = 12)
    # ax.set_xticks(pi * (1/4:1/4:1))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

    ax.set_title("Equatorial section", fontsize = 12)

    lscm = Iterators.product(("solid", "dashed", "dotted"), ("black", "0.5", "0.3"), ("None", "."))

    λ0 = RossbyWaveSpectrum.rossby_ridge(m)
    rossbyindex = argmin(abs.(real.(λs) .- λ0))

    nmodes = get(kw, :nmodes, 3)
    vm = vecs[:, range(rossbyindex, step=-1, length=nmodes)]

    for (ind, (v, (ls, c, marker))) in enumerate(zip(eachcol(vm), lscm))
        (; VWSinv, θ) = RossbyWaveSpectrum.eigenfunction_realspace(v, m; operators, V_symmetric)
        @unpack V = VWSinv
        Vr = realview(V)
        eqind = argmin(abs.(θ .- pi/2))
        Vr_eq = Vr[:, eqind]

        Vrmax_sign = sign(eignorm(Vr_eq))
        Vr_eq .*= Vrmax_sign
        normalize!(Vr_eq)

        ax.plot(Vr_eq, rfrac; ls, color = c,
            label = "n = "*string(ind-1),
            marker, markevery = 10)
    end
    ax.axvline(0, ls="dotted", color="black")

    legend = ax.legend()
    legend.get_title().set_fontsize("12")
    if !get(kw, :constrained_layout, false)
        f.tight_layout()
    end
end

function eigenfunction_rossbyridge_allstreamfn(Feig::FilteredEigen, m; kw...)
    fig = plt.figure(constrained_layout = true, figsize = (10, 4))
    subfigs = fig.subfigures(1, 3, wspace = 0.08, width_ratios = [1, 0.8, 0.8])
    eigenfunction_rossbyridge(Feig, m; f = subfigs[1],
        kw..., constrained_layout = true, save=false)
    multiple_eigenfunctions_surface_m(Feig, m; f = subfigs[2],
        kw..., constrained_layout = true, save=false)
    multiple_eigenfunctions_radial_m(Feig, m; f = subfigs[3],
        kw..., constrained_layout = true, save=false)
    if get(kw, :save, false)
        savefiginfo(fig, "eigenfunction_rossby_all.eps")
    end
end

function eigenfunctions_polar(Feig::FilteredEigen, m; mode=nothing,
        cmap=mode == :asym ? ("RdYlBu", "YlOrBr") : ("YlOrBr", "RdYlBu"), kw...)

    f, axlist = get(kw, :faxlist) do
        subplots(1,2, sharey=true)
    end
    λs_m = Feig[m].lams
    ind = if mode==:asym
        argmin(abs.(λs_m .- 2.5*RossbyWaveSpectrum.rossby_ridge(m)))
    elseif mode==:symsectoral
        argmin(abs.(λs_m .- RossbyWaveSpectrum.rossby_ridge(m)))
    elseif mode==:symhighest
        lastindex(λs_m)
    else
        kw[:ind]
    end

    eigenfunction(Feig, m, ind, f=f, field=:V,
        cmap=cmap[1], polar=true, axprofile = axlist[1])
    eigenfunction(Feig, m, ind, f=f, field=:W,
        cmap=cmap[2], polar=true, axprofile = axlist[2])
    f.set_size_inches(8,4)
    f.tight_layout()
end

function eigenfunctions_polar(Feigsym::FilteredEigen, Feigasym::FilteredEigen, m; kw...)
    f, axlist = subplots(2,2, sharex=true, sharey=true)
    eigenfunctions_polar(Feigasym, m; mode=:asym, faxlist=(f, axlist[1, 1:2]), kw...)
    eigenfunctions_polar(Feigsym, m; mode=:symhighest, faxlist=(f, axlist[2, 1:2]), kw...)
    f.set_size_inches(8,8)
    f.tight_layout()
end

function compute_eigenfunction_spectra(v, m; operators, V_symmetric,
        realv = _real(v),
        imagv = _imag(v),
        fields = "VrViWrWiSrSi",
        nr = operators.radial_params[:nr],
        nℓ = operators.radial_params[:nℓ],
        Vℓ = RossbyWaveSpectrum.ℓrange(m, nℓ, V_symmetric),
        Vℓ_full = first(Vℓ):last(Vℓ),
        Wℓ = RossbyWaveSpectrum.ℓrange(m, nℓ, !V_symmetric),
        Wℓ_full = first(Wℓ):last(Wℓ),
        Sℓ = Wℓ,
        Sℓ_full = Wℓ_full,
        Vrf = occursin("Vr", fields) ? zeros(nr, Vℓ_full) : nothing,
        Vif = occursin("Vi", fields) ? zeros(nr, Vℓ_full) : nothing,
        Wrf = occursin("Wr", fields) ? zeros(nr, Wℓ_full) : nothing,
        Wif = occursin("Wi", fields) ? zeros(nr, Wℓ_full) : nothing,
        Srf = occursin("Sr", fields) ? zeros(nr, Sℓ_full) : nothing,
        Sif = occursin("Si", fields) ? zeros(nr, Sℓ_full) : nothing,
        )


    if occursin("Vr", fields)
        Vr = reshape(@view(realv[1:nr*nℓ]), nr, nℓ);
        @views @. Vrf[:, Vℓ] = Vr
    end
    if occursin("Vi", fields)
        Vi = reshape(@view(imagv[1:nr*nℓ]), nr, nℓ);
        @views @. Vif[:, Vℓ] = Vi
    end

    if occursin("Wr", fields)
        Wr = reshape(@view(realv[nr*nℓ .+ (1:nr*nℓ)]), nr, nℓ);
        @views @. Wrf[:, Wℓ] = Wr
    end
    if occursin("Wi", fields)
        Wi = reshape(@view(imagv[nr*nℓ .+ (1:nr*nℓ)]), nr, nℓ);
        @views @. Wif[:, Wℓ] = Wi
    end

    if occursin("Sr", fields)
        Sr = reshape(@view(realv[2nr*nℓ .+ (1:nr*nℓ)]), nr, nℓ)
        @views @. Srf[:, Sℓ] = Sr
    end
    if occursin("Si", fields)
        Si = reshape(@view(imagv[2nr*nℓ .+ (1:nr*nℓ)]), nr, nℓ)
        @views @. Sif[:, Sℓ] = Si
    end

    [Vrf, Wrf, Srf, Vif, Wif, Sif]
end

function eigenfunction_spectrum(v, m; operators, V_symmetric, kw...)
    @unpack nr = operators.radial_params
    terms = compute_eigenfunction_spectra(v, m; operators, V_symmetric)

    x = axes.(terms, 2)
    titles = ["Vr", "Wr", "Sr", "Vi", "Wi", "Si"]

    compare_terms(terms; nrows = 2, titles,
        xlabel = "spharm ℓ", ylabel = "chebyshev order", x, y = 0:nr-1)
end

function eigenfunction_spectrum(Feig::FilteredEigen, m::Integer, ind::Integer; kw...)
    eigenfunction_spectrum(Feig[m][ind].v, m; Feig.operators, Feig.kw..., kw...)
end

function plot_matrix(M, nvariables = 3)
    f, axlist = subplots(3, 3)
    for colind in 1:3, rowind in 1:3
        Mv = abs.(RossbyWaveSpectrum.matrix_block(M, rowind, colind, nvariables))
        vmax = max(maximum(Mv), 1e-200)
        ax = axlist[rowind, colind]
        p = ax.imshow(Mv, vmax = vmax, vmin = -vmax, cmap = "RdBu_r")
        colorbar(mappable = p, ax = ax)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    end
    f.tight_layout()
end

function plot_matrix_block(M, rowind, colind, nvariables = 3; reim = :reim)
    f, _axlist = subplots(1, reim == :reim ? 2 : 1)
    axlist = Iterators.Stateful(axlistvector(_axlist))

    Mblock = RossbyWaveSpectrum.matrix_block(M, rowind, colind, nvariables)
    if reim ∈ (:re, :reim)
        A = realview(Mblock)
        Amax = maximum(abs, A)
        ax = popfirst!(axlist)
        p1 = ax.pcolormesh(A, cmap = "RdBu", vmax = Amax, vmin = -Amax)
        cb1 = colorbar(mappable = p1, ax = ax)
    end
    if reim ∈ (:im, :reim)
        A = imagview(Mblock)
        Amax = maximum(abs, A)
        ax = popfirst!(axlist)
        p2 = ax.pcolormesh(A, cmap = "RdBu", vmax = Amax, vmin = -Amax)
        cb2 = colorbar(mappable = p2, ax = ax)
    end
    f.tight_layout()
end

function plot_matrix_block(M, rowind, colind, nℓ, ℓind, ℓ′ind, nvariables = 3; reim = :reim)
    f, _axlist = subplots(1, reim == :reim ? 2 : 1)
    axlist = Iterators.Stateful(axlistvector(_axlist))

    Mblock = RossbyWaveSpectrum.matrix_block(M, rowind, colind, nvariables)

    if reim ∈ (:re, :reim)
        Mblockℓℓ′ = RossbyWaveSpectrum.matrix_block(realview(Mblock), ℓind, ℓ′ind, nℓ)
        Amax = maximum(abs, Mblockℓℓ′)
        ax = popfirst!(axlist)
        p1 = ax.pcolormesh(Mblockℓℓ′, cmap = "RdBu", vmax = Amax, vmin = -Amax)
        cb1 = colorbar(mappable = p1, ax = ax)
    end
    if reim ∈ (:im, :reim)
        Mblockℓℓ′ = RossbyWaveSpectrum.matrix_block(imagview(Mblock), ℓind, ℓ′ind, nℓ)
        Amax = maximum(abs, Mblockℓℓ′)
        ax = popfirst!(axlist)
        p2 = ax.pcolormesh(Mblockℓℓ′, cmap = "RdBu", vmax = Amax, vmin = -Amax)
        cb2 = colorbar(mappable = p2, ax = ax)
    end
    f.tight_layout()
end

function plot_diffrot_solar_equatorial_data(; operators, f = figure(), ax = f.add_subplot(), kw...)
    r_raw = SolarModel.solar_rotation_profile_radii()
    Ω_raw = SolarModel.solar_rotation_profile_raw()
    θinds = axes(Ω_raw,2)
    eqind = first(θinds) + (first(θinds) + last(θinds)) ÷ 2
    Ω_raw_eq = Ω_raw[:, eqind]
    @unpack Ω0 = operators.constants
    ΔΩ_raw_eq = Ω_raw_eq .- Ω0

    squished = get(kw, :squished, false)
    @unpack r_in, r_out = operators.radial_params
    r_in_frac = SolarModel.maybe_stretched_radius(r_in; operators, squished) / Rsun
    r_out_frac = SolarModel.maybe_stretched_radius(r_out; operators, squished) / Rsun
    r_inds = r_out_frac .>= r_raw .>= r_in_frac

    rpts_plot = SolarModel.maybe_squeeze_radius(r_raw[r_inds]*Rsun; operators, squished)

    ax.plot(rpts_plot/Rsun, ΔΩ_raw_eq[r_inds]*1e9/2pi,
        color=get(kw, :color, "grey"), label="solar", ls="dashed")
end

function plot_diffrot_radial_data(; operators, kw...)
    f, ax = subplots(1,1)
    @unpack Ω0 = operators.constants
    @unpack rpts = operators
    νnHzunit = freqnHzunit(Ω0)

    plot_diffrot_solar_equatorial_data(; operators, f, ax)

    if get(kw, :plot_smoothed, true)
        (; ΔΩ,) = SolarModel.radial_differential_rotation_profile_derivatives_Fun(;
            operators, rotation_profile = get(kw, :rotation_profile, :solar_equator), kw...);
        ax.plot(rpts/Rsun, ΔΩ.(rpts).*νnHzunit, label="smoothed model")
    end

    xmin, xmax = extrema(rpts)./Rsun
    Δx = xmax - xmin
    pad = Δx*0.1
    xlim(xmin - pad, xmax + pad)
    if get(kw, :plot_guides, false)
        ax.axhline(0, ls="dotted", color="black")
        ax.axvline(r_out_frac, ls="dotted", color="black")
    end
    fontsize = get(kw, :fontsize, 12)
    ax.set_xlabel(L"r/R_\odot"; fontsize)
    ax.set_ylabel(L"\Delta\Omega/2\pi" * " [nHz]"; fontsize)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.legend(loc="best"; fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    f.tight_layout()
end

function _plot_diffrot_radial_derivatives(axlist, rpts, ΔΩ, ddrΔΩ, d2dr2ΔΩ; ncoeffs = true, labelpre = "", kw...)
    r_frac = rpts/Rsun
    axlist[1].plot(r_frac, ΔΩ.(rpts);
        label = labelpre * (ncoeffs ? "ncoeff = $(ncoefficients(ΔΩ))" : ""), kw...)
    axlist[1].axhline(0, ls="dotted", color="black")
    axlist[1].set_ylabel(L"\Delta\Omega")
    axlist[2].plot(r_frac, ddrΔΩ.(rpts).* Rsun;
        label = labelpre * (ncoeffs ? "ncoeff = $(ncoefficients(ddrΔΩ))" : ""), kw...)
    axlist[2].set_ylabel(L"R_\odot \frac{d\Delta\Omega}{dr}")
    axlist[3].plot(r_frac, d2dr2ΔΩ.(rpts).* Rsun^2;
        label = labelpre * (ncoeffs ? "ncoeff = $(ncoefficients(d2dr2ΔΩ))" : ""), kw...)
    axlist[3].set_ylabel(L"R_\odot^2 \frac{d^2\Delta\Omega}{dr^2}")
    return nothing
end
function plot_diffrot_radial_derivatives(; operators, kw...)
    @unpack rpts = operators

    @unpack Ω0 = operators.constants
    (; ΔΩ, ddrΔΩ, d2dr2ΔΩ) = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives_Fun(;
        operators, kw...);

    f, axlist = subplots(3, 1, sharex = true)
    _plot_diffrot_radial_derivatives(axlist, rpts, ΔΩ, ddrΔΩ, d2dr2ΔΩ, zorder = 2, color="0.5",
        labelpre = "model, ", marker = ".", ls = "solid")
    if get(kw, :rotation_profile, :solar_equator) !== :solar_equator
        (; ΔΩ, ddrΔΩ, d2dr2ΔΩ) = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives_Fun(;
        operators, kw..., rotation_profile = :solar_equator);
        _plot_diffrot_radial_derivatives(axlist, rpts, ΔΩ, ddrΔΩ, d2dr2ΔΩ, labelpre = "solar",
            ncoeffs = false, color="0.6", zorder = 1, ls = "dashed")
    end

    for ax in axlist
        ax.legend(loc="best")
    end

    f.set_size_inches(8, 4)
    f.tight_layout()
end

function plot_diffrot_solar(; operators,
        rotation_profile = :solar_latrad_squished,
        ΔΩprofile_deriv = SolarModel.solar_differential_rotation_profile_derivatives_Fun(;
            operators,
            rotation_profile = RossbyWaveSpectrum.rotationtag(rotation_profile)),
        f = figure(),
        ax = f.add_subplot(projection="polar"), kw...)

    @unpack rpts = operators
    @unpack Ω0 = operators.constants
    @unpack r_in, r_out = operators.radial_params
    (; ΔΩ) = ΔΩprofile_deriv
    Ω = (1 + ΔΩ) * Ω0

    Nθ = 100
    θ = range(0, pi, Nθ)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_axis_off()
    Ωgrid = Ω.(rpts, cos.(θ)')./2pi .* 1e9
    p = ax.pcolormesh(θ, rpts, Ωgrid, shading="auto", cmap="GnBu", rasterized=true)
    cb = colorbar(mappable=p, ax=ax, shrink=0.6)
    cb.ax.set_title("Ω/2π [nHz]")
    ax.contour(θ, rpts, Ωgrid, 10, colors="black")
    ax.set_rmin(0)
    ax.set_rmax(r_out)

    if get(kw, :save, false)
        filename = "rotprof.eps"
        savefiginfo(f, filename, dpi=get(kw, :dpi, 200))
    end
end

function plot_diffrot_solar_and_radialsection(; operators,
        rotation_profile = :solar_latrad_squished,
        ΔΩprofile_deriv = SolarModel.solar_differential_rotation_profile_derivatives_Fun(;
            operators,
            rotation_profile = RossbyWaveSpectrum.rotationtag(rotation_profile)),
        kw...)

    f = figure()
    ax1 = f.add_subplot(121, projection="polar")
    ax2 = f.add_subplot(122)
    plot_diffrot_solar(; operators, ΔΩprofile_deriv, f, ax = ax1, kw..., save=false)

    @unpack rpts = operators
    @unpack Ω0 = operators.constants
    (; ΔΩ) = ΔΩprofile_deriv

    νnHzunit = freqnHzunit(Ω0)

    plot_diffrot_solar_equatorial_data(; operators, f, ax=ax2,
        squished = endswith(string(rotation_profile), "squished"))

    ax2.axhline(0, ls="dotted", color="black")
    ax2.plot(rpts/Rsun, ΔΩ.(rpts, 0) .* νnHzunit, color="black", label="smoothed")
    ax2.set_xlabel(L"r/R_\odot", fontsize=12)
    ax2.set_ylabel(L"(\Omega_\mathrm{eq} - \Omega_\mathrm{ref})\;/2\pi" * "  [nHz]", fontsize=12)
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax2.set_box_aspect(1)
    ax2.legend(loc="lower right")

    f.set_size_inches(7,4)
    f.tight_layout()
    f.subplots_adjust(wspace=0.5)
    if get(kw, :save, false)
        filename = "solarrotprof_radsection.eps"
        savefiginfo(f, filename, dpi=get(kw, :dpi, 200))
    end
end

function plot_diffrot_solar_and_radialsection(Feig::FilteredEigen; kw...)
    plot_diffrot_solar_and_radialsection(; Feig.operators, Feig.kw..., kw...)
end

function plot_diffrot_solar_derivatives(; operators, ΔΩprofile_deriv, ΔΩ_ongrid_splines = nothing, kw...)
    @unpack rpts = operators
    @unpack nℓ = operators.radial_params
    @unpack Ω0 = operators.constants
    compare_splines = !isnothing(ΔΩ_ongrid_splines)
    f, axlist = subplots(1 + compare_splines, 3, sharey = true, sharex=true, squeeze=false)
    cosθ = points(Legendre(), nℓ)
    θ = acos.(cosθ)
    titles = ("ΔΩ", L"r\,∂_r(ΔΩ)", L"r^2\,∂^2_r(ΔΩ)")
    mulrpow_terms = (0, 1, 2)

    for (term, title, ind, mulr_pow) in zip(ΔΩprofile_deriv, titles, CartesianIndices((1:1, axes(axlist,2))), mulrpow_terms)
        ax = axlist[ind]
        term_plot = term.(rpts, cosθ') .* rpts.^mulr_pow
        vmax = maximum(abs, term_plot)
        norm0 = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        p = ax.pcolormesh(θ, rpts/Rsun, term_plot, shading="auto", cmap="RdBu", norm=norm0);
        ax.set_title(title, fontsize=12)
        colorbar(mappable = p, ax = ax)
    end

    if compare_splines
        for (term, title, ind, mulr_pow) in zip(ΔΩ_ongrid_splines, titles,
                CartesianIndices((2:2, axes(axlist,2))), mulrpow_terms)
            ax = axlist[ind]
            term_plot = term .* rpts.^mulr_pow
            vmax = maximum(abs, term_plot)
            norm0 = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
            p = ax.pcolormesh(θ, rpts/Rsun, term_plot, shading="auto", cmap="RdBu", norm=norm0);
            ax.set_title(title * " (spline)", fontsize=12)
            colorbar(mappable = p, ax = ax)
        end
    end

    for ax in @view axlist[end, :]
        ax.set_xlabel("θ [radian]", fontsize=12);
    end
    for ax in @view axlist[:, 1]
        ax.set_xlabel(L"r/R_\odot", fontsize=12);
    end

    f.set_size_inches(3*size(axlist,2), 3*size(axlist,1))
    f.tight_layout()
end

function plot_diffrot_solar_vorticity(; operators, ωΩ_deriv, kw...)
    (; raw, coriolis) = ωΩ_deriv
    raw_vorticity = get(kw, :coriolis, false)
    _plot_diffrot_solar_vorticity(raw_vorticity ? raw : coriolis; operators, ωΩ_deriv, kw...)
end

function _plot_diffrot_solar_vorticity(ωΩ_deriv; operators, kw...)
    (; ωΩr, ∂rωΩr, inv_sinθ_∂θωΩr, inv_rsinθ_ωΩθ,
        inv_sinθ_∂r∂θωΩr, ∂r_inv_rsinθ_ωΩθ) = ωΩ_deriv

    mulrpow_terms = (0, 1, 0, 1, 1, 2)
    titles = (L"ω_{Ωr}", L"r ∂_r\left(ω_{Ωr}\right)", L"∂_θ\left(ω_{Ωr}\right)/sinθ",
        L"ω_{Ωθ}/sinθ", L"r ∂_r ∂_θ(ω_{Ωr})/sinθ", L"r^2 ∂_r\left(ω_{Ωθ}/rsinθ\right)")

    @unpack rpts = operators
    @unpack nℓ = operators.radial_params
    @unpack Ω0 = operators.constants
    cosθ = points(Legendre(), nℓ)
    θ = acos.(cosθ)

    f, axlist = subplots(2,3, sharex = true, sharey = true, squeeze=false)

    for (term, title, ind, mulr_pow) in zip(ωΩ_deriv, titles, CartesianIndices(axlist), mulrpow_terms)
        ax = axlist[ind]
        term_plot = term.(rpts, cosθ') .* rpts.^mulr_pow
        vmax = maximum(abs, term_plot)
        norm0 = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        p = ax.pcolormesh(θ, rpts/Rsun, term_plot, shading="auto", cmap="RdBu", norm=norm0);
        ax.set_title(title, fontsize=12)
        colorbar(mappable = p, ax = ax)
    end

    for ax in @view axlist[:, 1]
        ax.set_ylabel(L"r/R_\odot", fontsize=12);
    end

    for ax in @view axlist[2, :]
        ax.set_xlabel("θ [radian]", fontsize=12);
    end

    f.set_size_inches(3*size(axlist,2), 3*size(axlist,1))
    f.tight_layout()
end

function critical_latitude_singularity_smearing(Feig::FilteredEigen, m)
    critical_latitude_singularity_smearing(m; Feig.operators)
end
function critical_latitude_singularity_smearing(m; operators)
    @unpack ν = operators.constants
    rad2deg(cbrt(ν/m))
end

function plot_diffrot_latitudinalsection(Feig::FilteredEigen; kw...)
    plot_diffrot_latitudinalsection(; Feig.operators, Feig.kw..., kw...)
end

function plot_diffrot_latitudinalsection(; operators,
        rotation_profile = :solar_latrad_squished,
        ΔΩ_scale = 1.0,
        ΔΩprofile_deriv = SolarModel.solar_differential_rotation_profile_derivatives_Fun(;
            operators,
            ΔΩ_scale,
            rotation_profile = RossbyWaveSpectrum.rotationtag(rotation_profile)),
        f = figure(),
        ax = subplot(),
        kw...)

    @unpack r_out = operators.radial_params
    (; ΔΩ) = ΔΩprofile_deriv

    cosθ = reverse(points(Legendre(), 200))
    for r_frac in [1.0, 0.95, 0.9]
        ΔΩ_θ = ΔΩ.(r_frac * r_out, cosθ)
        θ = acos.(cosθ)
        ax.plot(90 .- rad2deg.(θ), ΔΩ_θ, zorder=2, label="$r_frac")
    end
    ax.axhline(0, color="black", ls="dotted", zorder=1)
    ax.legend(loc="best")

    ax.set_xlim(-90, 90)

    for m in 2:2:10
        ym = -2/(m*(m+1))
        ax.axhline(ym, ls="dashed", color="black")
        plt.text(-20, ym, "$m", fontsize=10, va="center", ha="center", backgroundcolor="w")
    end
    f, ax
end

function lowest_m_criticallat(Feig::FilteredEigen; kw...)
    lowest_m_criticallat(Feig.mr; Feig.operators, Feig.kw..., kw...)
end

function lowest_m_criticallat(mr; operators,
        rotation_profile = :solar_latrad_squished,
        ΔΩ_scale = 1.0,
        ΔΩprofile_deriv = SolarModel.solar_differential_rotation_profile_derivatives_Fun(;
            operators,
            ΔΩ_scale,
            rotation_profile = RossbyWaveSpectrum.rotationtag(rotation_profile)),
        kw...)

    @unpack r_out = operators.radial_params
    (; ΔΩ) = ΔΩprofile_deriv
    cosθ = reverse(points(Legendre(), 200))
    θ = acos.(cosθ)
    ΔΩ_θ = ΔΩ.(r_out, cosθ)

    findfirst(mr) do m
        phasespeed = -2/(m*(m+1))
        RossbyWaveSpectrum.count_num_nodes!(ΔΩ_θ .- phasespeed, θ) > 0
    end
end

function critical_latitude_with_radius(Feig::FilteredEigen, m; kw...)
    critical_latitude_with_radius(m; Feig.operators,
        ΔΩprofile_deriv =
            SolarModel.solar_differential_rotation_profile_derivatives_Fun(Feig),
        kw...)
end

function critical_latitude_with_radius(m; operators,
        rotation_profile = :solar_latrad_squished,
        ΔΩ_scale = 1.0,
        ΔΩprofile_deriv = SolarModel.solar_differential_rotation_profile_derivatives_Fun(;
            operators,
            ΔΩ_scale,
            rotation_profile = RossbyWaveSpectrum.rotationtag(rotation_profile)),
        kw...)

    @unpack rpts = operators
    (; ΔΩ) = ΔΩprofile_deriv
    dcosθ_ΔΩ = (I ⊗ Derivative()) * ΔΩ
    map(rpts) do ri
        ΔΩ_ri = cosθ -> (ΔΩ(ri, cosθ) + 2/(m*(m+1)))
        if sign(ΔΩ_ri(cosd(90))) == sign(ΔΩ_ri(cosd(0)))
            return NaN
        else
            dcosθ_ΔΩ_ri = cosθ -> dcosθ_ΔΩ(ri, cosθ)
            cosθroot = find_zero((ΔΩ_ri, dcosθ_ΔΩ_ri), cosd(70))
            rad2deg(acos(cosθroot))
        end
    end
end

function compare_terms(@nospecialize(terms); x = nothing, y = nothing,
        titles = ["" for _ in terms], nrows = 1, cmap = "RdBu",
        xlabel = "", ylabel = "")
    ncols = ceil(Int, length(terms)/nrows)
    f = figure()
    for (ind, (term, title)) in enumerate(zip(terms, titles))
        ax = subplot(nrows, ncols, ind)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        vmax = maximum(abs, term)
        kwargs = Dict(:cmap => cmap, :vmax => vmax, :vmin => -vmax)
        p = if isnothing(x) && isnothing(y)
            ax.pcolormesh(term; kwargs...)
        elseif x isa AbstractVector{<:AbstractVector} && y isa AbstractVector{<:AbstractVector}
            ax.pcolormesh(x[ind], y[ind], term, shading = "auto"; kwargs...)
        elseif x isa AbstractVector{<:AbstractVector}
            ax.pcolormesh(x[ind], y, term, shading = "auto"; kwargs...)
        elseif y isa AbstractVector{<:AbstractVector}
            ax.pcolormesh(x, y[ind], term, shading = "auto"; kwargs...)
        else
            ax.pcolormesh(x, y, term, shading = "auto"; kwargs...)
        end
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cb = colorbar(mappable = p)
        cb.ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    end
    f.set_size_inches(4ncols, 3nrows)
    f.tight_layout()
end

function plot_constraint_basis(; operators, constraints = RossbyWaveSpectrum.constraintmatrix(operators), kw...)
    @unpack nullspacematrices = constraints
    @unpack radialspace = operators.radialspaces;
    @unpack rpts = operators

    f, axlist = subplots(1,3, sharex = true)
    n_cutoff = 4

    ls = ["solid", "dashed", "dotted"]
    color = ["black", "0.6"]
    linestyle = Iterators.product(ls, color)

    title_field = [L"V_\ell", L"W_\ell", L"S^\prime_\ell"]

    for (ind, (ax, M, fld)) in enumerate(zip(axlist, nullspacematrices, title_field))
        for (colind, (col, st)) in enumerate(zip(Iterators.take(eachcol(M), n_cutoff), linestyle))
            ls, c = st
            v = itransform(radialspace, col)
            ax.plot(rpts, v; label = "q=$(colind-1)", ls, c)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
        end
        ax.set_xlabel(L"\bar{r}", fontsize=11)
        ax.set_title("Basis for "*fld, fontsize=11)
    end
    axlist[end].legend(bbox_to_anchor = (1,1), title = "Chebyshev\ndegree")
    f.set_size_inches(6,2.5)
    f.tight_layout()

    if get(kw, :save, false)
        fname = "bc_basis.eps"
        savefiginfo(f, fname)
    end
    return nothing
end

function plot_scale_heights(; operators, kw...)
    f, axlist = subplots(1,2, sharex = true)
    @unpack rpts = operators
    @unpack ηρ, ηT = operators.rad_terms
    r_frac = rpts/Rsun
    ax = axlist[1]
    ax.plot(r_frac, .-ηρ.(rpts).*Rsun, label = L"\rho", color="black")
    ax.plot(r_frac, .-ηT.(rpts).*Rsun, label = L"T", color="black", ls="dashed")
    ax.set_xlabel(L"r/R_\odot")
    ax.set_ylabel("Inverse scale height\n(normalized)")
    ax.legend(loc="best")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))

    ax = axlist[2]
    ax.plot(r_frac, RossbyWaveSpectrum.superadiabaticity.(rpts), color="black")
    ax.set_ylabel("Super-adiabatic\ntemperature gradient")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.ticklabel_format(axis="y", style="sci", scilimits = (0,0), useMathText = true)
    ax.axhline(0, ls="dotted", color="black", lw=0.8)
    ax.set_xlabel(L"r/R_\odot")

    f.set_size_inches(5,2.5)
    f.tight_layout()
    if get(kw, :save, false)
        fname = "scale_heights.eps"
        savefiginfo(f, fname)
    end
    return nothing
end

function plot_density_scale_heights(; operators, kw...)
    @unpack splines, rpts = operators
    @unpack sηρ, ddrsηρ, d2dr2sηρ, d3dr3sηρ = splines;

    f, ax = subplots(4, 1, sharex = true)
    plotkw = pairs((; color = get(kw, :color, "grey"), marker = get(kw, :marker, ".")))
    ax[1].plot(rpts, sηρ(rpts); plotkw...)
    ax[2].plot(rpts, ddrsηρ(rpts); plotkw...)
    ax[3].plot(rpts, d2dr2sηρ(rpts); plotkw...)
    ax[4].plot(rpts, d3dr3sηρ(rpts); plotkw...)

    f.set_size_inches(4, 6)
    f.tight_layout()
end

function load_and_filter_output(nr, nℓ, rotscales = [0.0, 0.25, 0.5, 0.75];
        rottag = "dr_solar_latrad_squished", symtag = "sym",
        r_in = 0.65, r_out = 0.985, nu = 5.0e11,
        filterflags=Filters.EIGVAL|Filters.SPATIAL,
        Δl_filter = 0,
        nodes_cutoff = 2,
        filterparams...
        )

    map(rotscales) do rotscale
        Fssym = RossbyWaveSpectrum.FilteredEigen(
            rossbyeigenfilename(nr, nℓ, rottag, symtag,
            "nu$(nu)_rin$(r_in)_rout$(r_out)_trackhanson2020_rotscale$(rotscale)"))
        Fssymf = RossbyPlots.update_cutoffs(Fssym; Δl_filter, nodes_cutoff)
        RossbyWaveSpectrum.filter_eigenvalues(Fssymf; filterflags, filterparams...);
    end
end

function rescale_ridge(ridgefreqs, ΔΩ_scale)
    OrderedDict(
        m => rossby_ridge(m) + (λ - rossby_ridge(m)) * ΔΩ_scale
        for (m,λ) in ridgefreqs
        )
end

function spectra_different_rotation(Feigs::Vector{FilteredEigen}; layout = (2,2), kw...)
    f, axlist = subplots(layout..., sharex=true, sharey=true, squeeze=false)
    subtract_sectoral=true
    sps = map(zip(permutedims(axlist), Feigs)) do  (ax, Feig)
        _, _, s = spectrum(Feig; ax, f, showtitle=false, kw...,
            save=false, subtract_sectoral,
            fillmarker=false)

        ΔΩ_scale = Feig.kw[:ΔΩ_scale]

        Ridge1_modified = rescale_ridge(RidgeFrequencies[1], ΔΩ_scale)
        highlight_ridge(Feig, Ridge1_modified;
                color="purple", label="Ridge 1", ax,
                subtract_sectoral, snaptopoint = false)

        Ridge2_modified = rescale_ridge(RidgeFrequencies[2], ΔΩ_scale)
        highlight_ridge(Feig, Ridge2_modified;
                color="teal", label="Ridge 2", ax,
                subtract_sectoral, snaptopoint = false)

        Ridge3_modified = rescale_ridge(RidgeFrequencies[3], ΔΩ_scale)
        highlight_ridge(Feig, Ridge3_modified;
                color="chocolate", label="Ridge 3", ax,
                subtract_sectoral, snaptopoint = false)

        m = lowest_m_criticallat(Feig)
        ax.axvline(m, ls="dotted", color="black", zorder=0)
        s
    end
    if haskey(kw, :xlim)
        axlist[1,1].set_xlim(kw[:xlim])
    end
    f.set_size_inches(layout[1]*5, layout[2]*3.5)
    f.tight_layout()
    if get(kw, :save, false)
        filename = "spectra_rotscale.eps"
        savefiginfo(f, filename)
    end
    f, axlist, sps
end

end # module plots
