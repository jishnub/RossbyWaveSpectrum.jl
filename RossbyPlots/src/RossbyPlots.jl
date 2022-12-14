module RossbyPlots

Base.Experimental.@optlevel 1

using PyCall
using PyPlot

using RossbyWaveSpectrum

import ApproxFun: ncoefficients, itransform
using LaTeXStrings
using LinearAlgebra
using OrderedCollections
using Polynomials
using Printf
using StructArrays
using UnPack

const StructMatrix{T} = StructArray{T,2}

export spectrum
export uniform_rotation_spectrum
export differential_rotation_spectrum
export diffrot_rossby_ridge
export plot_diffrot_radial
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

realview(M::StructMatrix{<:Complex}) = M.re
function realview(M::AbstractMatrix{Complex{T}}) where {T}
    @view reinterpret(reshape, T, M)[1, :, :]
end
realview(M::AbstractMatrix{<:Real}) = M
function imagview(M::AbstractMatrix{Complex{T}}) where {T}
    @view reinterpret(reshape, T, M)[2, :, :]
end
imagview(M::StructMatrix{<:Complex}) = M.im
imagview(M::AbstractMatrix{<:Real}) = M

axlistvector(_axlist) = reshape([_axlist;], Val(1))

function cbformat(x, _)
    a, b = split((@sprintf "%.1e" x), 'e')
    c = parse(Int, b)
    a * L"\times10^{%$c}"
end

function plot_rossby_ridges(mr; νnHzunit = 1, ax = gca(), ΔΩ_frac = 0, ridgescalefactor = nothing, kw...)
    if get(kw, :sectoral_rossby_ridge, true)
        ax.plot(mr, -RossbyWaveSpectrum.rossby_ridge.(mr; ΔΩ_frac) .* νnHzunit,
            label = ΔΩ_frac == 0 ? L"\frac{-2(Ω/2π)}{m+1}" : "Doppler\nshifted",
            lw = 1,
            color = get(kw, :sectoral_rossby_ridge_color, "black"),
            zorder = 0,
            ls = get(kw, :sectoral_rossby_ridge_ls, "dotted")
        )
    end

    if !isnothing(ridgescalefactor)
        ax.plot(mr, -ridgescalefactor .* RossbyWaveSpectrum.rossby_ridge.(mr; ΔΩ_frac) .* νnHzunit ,
            label = ΔΩ_frac == 0 ? "high-frequency" :
                    L"\Delta\Omega/\Omega_0 = " * string(round(ΔΩ_frac, sigdigits = 1)),
            lw = 1,
            color = get(kw, :sectoral_rossby_ridge_color, "black"),
            zorder = 0,
            ls = get(kw, :sectoral_rossby_ridge_scaled_ls, "dotted")
        )
    end

    if get(kw, :uniform_rotation_ridge, false)
        ax.plot(mr, -RossbyWaveSpectrum.rossby_ridge.(mr) .* νnHzunit,
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
uncertainty(m::Measurement) = (lowerr, higherr)
lowerr(m::Measurement) = m.lowerr
higherr(m::Measurement) = m.higherr

const ProxaufFit = [
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
]

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

function spectrum(lam::AbstractArray, mr;
    operators,
    f = figure(),
    ax = subplot(),
    m_zoom = mr[max(begin, end - 6):end],
    rossbyridges = true,
    kw...)

    ax.set_xlabel("m", fontsize = 12)
    ax.set_ylabel(L"-\Re[\nu]" * "[nHz]", fontsize = 12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer = true))

    @unpack Ω0 = operators.constants

    νnHzunit = freqnHzunit(Ω0)
    ax.set_ylim(minmax(-0.05 * νnHzunit, -1.5 * νnHzunit)...)

    lamcat = mapreduce(real, vcat, lam) .* νnHzunit
    lamimcat = mapreduce(imag, vcat, lam) .* νnHzunit
    vmin, vmax = extrema(lamimcat)
    vmin = min(0, vmin)
    mcat = reduce(vcat, [range(m, m, length(λi)) for (m, λi) in zip(mr, lam)])
    s = ax.scatter(mcat, -lamcat; c = lamimcat, ScatterParams...,
        vmax = vmax, vmin = vmin, zorder=get(kw, :zorder, 1))
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size = "3%", pad = 0.05)
    cb = colorbar(mappable = s, cax = cax)
    cb.ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    cb.ax.set_title(L"\Im[\nu]" * "[nHz]")

    if rossbyridges
        plot_rossby_ridges(mr; νnHzunit, ax, kw...)
    end

    V_symmetric = kw[:V_symmetric]
    zoom = V_symmetric && !get(kw, :diffrot, false) && get(kw, :zoom, true)

    if zoom
        lamcat_inset = mapreduce(real, vcat, lam[m_zoom]) .* νnHzunit
        lamimcat_inset = mapreduce(imag, vcat, lam[m_zoom]) .* νnHzunit
        mcat_inset = reduce(vcat, [range(m, m, length(λi)) for (m, λi) in zip(mr[m_zoom], lam[m_zoom])])

        axins = ax.inset_axes([0.5, 0.1, 0.4, 0.4])
        ymax = (RossbyWaveSpectrum.rossby_ridge(minimum(m_zoom)) + 0.005) * νnHzunit
        ymin = (RossbyWaveSpectrum.rossby_ridge(maximum(m_zoom)) - 0.02) * νnHzunit
        axins.set_ylim(minmax(-ymin, -ymax)...)
        axins.xaxis.set_major_locator(ticker.NullLocator())
        axins.yaxis.set_major_locator(ticker.NullLocator())
        plt.setp(axins.spines.values(), color = "0.2", lw = "0.5")
        axins.scatter(mcat_inset, -lamcat_inset; c = lamimcat_inset, ScatterParams...,
            vmax = vmax, vmin = vmin)

        if rossbyridges
            plot_rossby_ridges(m_zoom; νnHzunit, ax = axins, kw...)
        end

        ax.indicate_inset_zoom(axins, edgecolor = "grey", lw = 0.5)
    end

    if rossbyridges
        ax.legend(loc = "best", fontsize = 12)
    end

    if get(kw, :tight_layout, true)
        f.tight_layout()
    end

    ax.axhline(0, ls="dotted", color="0.2", zorder = 0, lw=0.5)

    titlestr = V_symmetric ? "Symmetric" : "Antisymmetric"
    ax.set_title(titlestr, fontsize = 12)

    if get(kw, :save, false)
        V_symmetric = kw[:V_symmetric]
        V_symmetric_str = V_symmetric ? "sym" : "asym"
        filenametag = get(kw, :filenametag, V_symmetric_str)
        rotation = get(kw, :rotation, "uniform")
        f.savefig(joinpath(plotdir, "$(rotation)_rotation_spectrum_$(filenametag).eps"))
    end
    f, ax
end

function uniform_rotation_spectrum(fsym::FilteredEigen, fasym::FilteredEigen; kw...)
    f, axlist = subplots(2, 2, sharex = true)
    spectrum(fsym; kw..., f, ax = axlist[1,1], save = false)
    spectrum(fasym; kw..., f, ax = axlist[2,1], save = false)

    damping_rossbyridge(fsym; f, ax = axlist[1,2], kw..., save = false)
    plot_high_frequency_ridge(fasym; f, ax = axlist[2,1], kw..., save = false)
    damping_highfreqridge(fasym; f, ax = axlist[2,2], kw..., save = false)

    f.set_size_inches(9,6)
    f.tight_layout()
    if get(kw, :save, false)
        fpath = joinpath(plotdir, "spectrum_sym_asym.eps")
        @info "saving to $fpath"
        f.savefig(fpath)
    end
    return nothing
end

function mantissa_exponent_format(a)
    a_exponent = floor(Int, log10(a))
    a_mantissa = round(a / 10^a_exponent, sigdigits = 1)
    (a_mantissa == 1 ? "" : L"%$a_mantissa\times") * L"10^{%$a_exponent}"
end

function damping_rossbyridge(lam, mr; operators, f = figure(), ax = subplot(), kw...)
    V_symmetric = kw[:V_symmetric]
    @assert V_symmetric "V must be symmetric for rossby ridge plots"
    @unpack Ω0, ν = operators.constants
    ν *= Ω0 * Rsun^2

    νnHzunit = freqnHzunit(Ω0)

    λs_rossbyridge = [
    (isempty(λ) ? eltype(λ)[NaN + im*NaN] : λ[argmin(abs.(real.(λ) .- RossbyWaveSpectrum.rossby_ridge(m)))]) for (m,λ) in zip(mr, lam)]

    ax.plot(mr, imag.(λs_rossbyridge) .* 2 #= HWHM to FWHM =# * νnHzunit,
            ls="dotted", color="grey", marker="o", mfc="white",
            mec="black", ms=5, label="this work", zorder = 4)

    # observations
    m_P = first.(ProxaufFit)
    γ_P = last.(last.(ProxaufFit))
    γ_P_val = value.(γ_P)
    ax.errorbar(m_P, γ_P_val,
        yerr = errorbars_pyplot(γ_P),
        color= "grey", ls="None", capsize = 3, label="P20", zorder = 1, marker = ".", ms = 5, mfc = "k")
    ax.set_ylabel("linewidth [nHz]", fontsize = 12)
    ax.set_xlabel("m", fontsize = 12)
    ax.legend(loc="best")
    ax.set_title("Sectoral modes", fontsize = 12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer = true))
    f.tight_layout()
    if get(kw, :save, false)
        f.savefig(joinpath(plotdir, "damping_rossby.eps"))
    end
    return nothing
end

function damping_highfreqridge(lam, mr; operators, f = figure(), ax = subplot(), kw...)
    V_symmetric = kw[:V_symmetric]
    @assert !V_symmetric "V must be antisymmetric for high-frequency rossby ridge plots"
    @unpack Ω0, ν = operators.constants
    ν *= Ω0 * Rsun^2

    νnHzunit = freqnHzunit(Ω0)

    # observations
    ms_H22 = collect(keys(HansonHFFit))
    γ_H22 = [x.γ for x in values(HansonHFFit)]
    γ_H22_val = value.(γ_H22)

    ax.errorbar(ms_H22, γ_H22_val,
        yerr = errorbars_pyplot(γ_H22),
        color= "grey", ls="None", capsize = 3, label="H22", zorder = 1, marker = ".", ms=6, mfc="k")

    # model
    λs_HFRridge = [
    isempty(λ) || m < 5 ? eltype(λ)(NaN + im*NaN) : begin
        approx_HFR_freq = 2.5 * RossbyWaveSpectrum.rossby_ridge(m)
        λ_below = λ[real.(λ) .< approx_HFR_freq]
        λ_below[argmin(abs.(real.(λ_below) .- approx_HFR_freq))]
        end for (m,λ) in zip(mr, lam)]

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
        f.savefig(joinpath(plotdir, "damping_highfreqridge.eps"))
    end
    return nothing
end

function plot_high_frequency_ridge(lam, mr; operators, f = figure(), ax = subplot(), kw...)
    @unpack Ω0 = operators.constants

    νnHzunit = freqnHzunit(Ω0)

    # observations
    m_H = collect(keys(HansonHFFit))
    ν_H = [x.ν for x in values(HansonHFFit)]
    ax.errorbar(m_H, value.(ν_H),
        yerr = errorbars_pyplot(ν_H),
        color= "0.3", ls="dashed", capsize = 3, label="H22")

    ax.legend(loc="best")

    if get(kw, :save, false)
        f.savefig(joinpath(plotdir, "highfreqridge.eps"))
    end
    return nothing
end

for f in [:spectrum, :damping_highfreqridge, :damping_rossbyridge, :plot_high_frequency_ridge]
    @eval function $f(feig::FilteredEigen; kw...)
        $f(feig.lams, feig.mr; operators = feig.operators, feig.kw..., kw...)
    end
end

function diffrot_rossby_ridge(Frsym, mr=8:15, ΔΩ_frac_low = 0.005, ΔΩ_frac_high = 0.04; plot_spectrum = false,
        plot_lower_cutoff = plot_spectrum, plot_upper_cutoff = plot_spectrum, plot_data = true,
        plot_rotation = false, save = false)

    lams = Frsym.lams
    lams = lams[mr]
    @unpack Ω0 = Frsym.operators.constants
    ν0 = freqnHzunit(Ω0)
    ν0_Sun = 453.1
    λRossby_high = RossbyWaveSpectrum.rossby_ridge.(mr, ΔΩ_frac = ΔΩ_frac_low)
    λRossby_low = RossbyWaveSpectrum.rossby_ridge.(mr, ΔΩ_frac = ΔΩ_frac_high)
    lams_realfilt = map(zip(mr, lams, λRossby_low, λRossby_high)) do (m, lam, λRl, λRh)
        lams_realfilt = filter(lam) do λ
            λRl < real(λ) < λRh
        end
        real.(lams_realfilt)[argmin(abs.(imag.(lams_realfilt)))]
    end
    lams_realfilt_plot = lams_realfilt.*(-ν0)
    plot_spectrum && begin
        f, ax = spectrum(Frsym, zorder=2)
        ax.plot(mr, lams_realfilt_plot, marker="o", ls="dotted",
            mec="black", mfc="None", ms=6, color="grey", zorder=1)
        plot_lower_cutoff && ax.plot(mr, λRossby_low.*(-ν0), ".--", color="orange", zorder=0)
        plot_upper_cutoff && ax.plot(mr, λRossby_high.*(-ν0), ".--", color="orange", zorder=0)
        plot_lower_cutoff && plot_upper_cutoff && ax.fill_between(mr,
                λRossby_low.*(-ν0), λRossby_high.*(-ν0), color="navajowhite", zorder=0)
        ax.set_ylim(top = minimum(λRossby_low)*(-ν0)*1.2)
    end

    f, axlist = subplots(1,plot_rotation+1,squeeze=false)
    ax_spectrum = axlist[1,plot_rotation+1]
    mr_Hanson = collect(keys(HansonGONGfit))
    if plot_data
        ν0_uniform_at_tracking_rate = RossbyWaveSpectrum.rossby_ridge.(mr_Hanson).*(-ν0_Sun)
        ν_H = [HansonGONGfit[m].ν for m in mr_Hanson]
        ν_H_val = value.(ν_H)

        δν = ν_H_val .- ν0_uniform_at_tracking_rate
        ax_spectrum.errorbar(mr_Hanson, δν,
            yerr = errorbars_pyplot(ν_H),
            color= "brown", ls="None", capsize = 3, label="Hanson 2020 [GONG]",
            zorder = 3, marker = ".", ms = 5, mfc = "k")

        fit_m_min = 8
        fit_inds = mr_Hanson .>= fit_m_min
        mr_fit = mr_Hanson[fit_inds]
        fit_x = (m -> (m - 2/(m+1))).(mr_fit)
        yerr = max.(higherr.(ν_H), lowerr.(ν_H))[fit_inds]
        linear_Ro_doppler_fit = fit(fit_x, δν[fit_inds], 1, weights=1 ./ yerr.^2)
        ΔΩ_fit = Polynomials.derivative(linear_Ro_doppler_fit)(0)
        ax_spectrum.plot(mr_Hanson, linear_Ro_doppler_fit.(mr_Hanson), "--",
            label=L"ν_\mathrm{Ro}"*" [Doppler ΔΩ/2π = $(round(ΔΩ_fit, sigdigits=2)) nHz]", zorder=1)
    end

    ν0_uniform_at_tracking_rate = RossbyWaveSpectrum.rossby_ridge.(mr).*(-ν0)

    δν = lams_realfilt_plot - ν0_uniform_at_tracking_rate
    fit_x = (m -> (m - 2/(m+1))).(mr)
    linear_Ro_doppler_fit = fit(fit_x, δν, 1)
    ΔΩ_fit = Polynomials.derivative(linear_Ro_doppler_fit)(0)
    ax_spectrum.plot(mr, δν,
        "o", color="grey", ms=6, label="sectoral dispersion relation", zorder=5)
    ax_spectrum.plot(mr, linear_Ro_doppler_fit.(mr), "--",
        label=L"ν_\mathrm{uniform}"*"\n[Doppler " * L"ΔΩ_0/2π" * " = $(round(ΔΩ_fit, digits=1)) nHz]", zorder=1,
        color = "black")

    ax_spectrum.set_xlabel("m", fontsize=16)
    ax_spectrum.set_ylabel(L"\nu + \frac{2(\Omega_0\,/2\pi)}{m+1}" * " [nHz]", fontsize=16)
    ax_spectrum.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax_spectrum.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax_spectrum.legend(loc="best", fontsize=12)
    ax_spectrum.tick_params(axis="both", which="major", labelsize=12)

    if plot_rotation
        ax_rotation = axlist[1,1]
        @unpack operators = Frsym;
        @unpack rpts = operators;
        @unpack r_in, r_out = operators.radial_params
        (; ΔΩ) = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives_Fun(; operators);
        ax_rotation.plot(rpts/Rsun, ΔΩ.(rpts)*freqnHzunit(Ω0),color="black", zorder=2)
        ax_rotation.xaxis.set_major_locator(ticker.MaxNLocator(3))
        ax_rotation.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax_rotation.axhline(ΔΩ_fit, ls="dotted", zorder=1, color="grey")
        ax_rotation.set_xlabel(L"r/R_\odot", fontsize=16)
        ax_rotation.set_ylabel(L"(Ω(r,π/2) - Ω_0)/2π"*" [nHz]", fontsize=16)
        r_text = r_in + (r_out - r_in)/10
        ax_rotation.text(r_text/Rsun, ΔΩ_fit*0.6, L"ΔΩ_0/2π", fontsize=15)
        ax_rotation.tick_params(axis="both", which="major", labelsize=12)
    end
    f.set_size_inches(6,2.5)
    f.tight_layout()
    if save
        filename = joinpath(plotdir,"diffrot_rossby_ridge.eps")
        @info "saving to $filename"
        f.savefig(filename)
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
    @unpack rpts = fconstsym.operators;
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
        fname = joinpath(plotdir, "diff_rot_spectrum.eps")
        @info "saving to $fname"
        f.savefig(fname)
    end
    return nothing
end

function eigenfunction(VWSinv::NamedTuple, θ::AbstractVector, m;
        operators, field = :V, f = figure(), component = real, kw...)

    V = getproperty(VWSinv, field)::Matrix{ComplexF64}
    Vr = copy(component(V))
    scale = get(kw, :scale) do
        eignorm(Vr)
    end
    if scale != 0
        Vr ./= scale
    end
    @unpack rpts = operators
    r_frac = rpts ./ Rsun
    nθ = length(θ)
    equator_ind = nθ÷2
    Δθ_scan = div(nθ, 5)
    rangescan = intersect(equator_ind .+ (-Δθ_scan:Δθ_scan), axes(V, 2))
    ind_max = findmax(col -> maximum(real, col), eachcol(view(Vr, :, rangescan)))[2]
    ind_max += first(rangescan) - 1
    V_peak_depthprofile = @view Vr[:, ind_max]
    r_max_ind = argmax(abs.(V_peak_depthprofile))

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
    else
        axprofile = get(kw, :axprofile) do
            f.add_subplot()
        end
    end

    if !polar
        axsurf.plot(θ, (@view Vr[r_max_ind, :]), color = "black")
        if get(kw, :setylabel, true)
            axsurf.set_ylabel("Angular\nprofile", fontsize = 11)
        end
        axsurf.set_xticks(pi * (1/4:1/4:1))
        axsurf.xaxis.set_major_formatter(ticker.FuncFormatter(piformatter))
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

        axprofile.pcolormesh(θ, r_frac, Vr, cmap = cmap, rasterized = true, shading = "auto")
        xlabel = get(kw, :longxlabel, true) ? "colatitude (θ) [radians]" : "θ [radians]"
        axprofile.set_xlabel(xlabel, fontsize = 11)
    else
        x = r_frac .* sin.(θ)'
        z = r_frac .* cos.(θ)'
        p = axprofile.pcolormesh(x, z, Vr, cmap = cmap, rasterized = true,
            shading = "auto")
        axprofile.yaxis.set_major_locator(ticker.MaxNLocator(3))
        axprofile.xaxis.set_major_locator(ticker.MaxNLocator(3))
        axprofile.set_xlabel(L"x/R_\odot", fontsize=12)
        axprofile.set_ylabel(L"z/R_\odot", fontsize=12)
        f.set_size_inches(3, 3)
        cb = colorbar(mappable=p, ax=axprofile)
        cb.ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    end

    if get(kw, :suptitle, true)
        fieldname = field == :V ? "Toroidal" :
                    field == :W ? "Poloidal" :
                    field == :S ? "Entropy" : nothing
        axprofile.set_title("$fieldname streamfunction")
    end

    if !get(kw, :constrained_layout, false)
        f.tight_layout()
    end
    if get(kw, :save, false)
        f.savefig(joinpath(plotdir, "eigenfunction.eps"))
    end
end

function eigenfunction(feig::FilteredEigen, m::Integer, ind::Integer; kw...)
    @unpack operators = feig
    eigenfunction(feig.vs[m][:, ind], m; operators, feig.kw..., kw...)
end

function eigenfunction(v::AbstractVector{<:Number}, m::Integer; operators, kw...)
    (; θ, VWSinv) = RossbyWaveSpectrum.eigenfunction_realspace(v, m; operators, kw...)
    eigenfunction(VWSinv, θ, m; operators, kw...)
end

function eigenfunctions_allstreamfn(f::FilteredEigen, m::Integer, vind::Integer; kw...)
    @unpack operators = f
    (; θ, VWSinv) = RossbyWaveSpectrum.eigenfunction_realspace(f.vs[m][:, vind], m;
            operators, f.kw..., kw...)
    if get(kw, :scale_eigenvectors, false)
        RossbyWaveSpectrum.scale_eigenvectors!(VWSinv; operators)
    end
    eigenfunctions_allstreamfn(VWSinv, θ, m; operators, f.kw..., kw...)
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
        fname = joinpath(plotdir, "eigenfunctions_allstreamfn_m$(m).eps")
        @info "saving to $fname"
        f.savefig(fname)
    end
end

function eigenfunction_rossbyridge(f::FilteredEigen, m; kw...)
    eigenfunction_rossbyridge(f.lams[m], f.vs[m], m; operators = f.operators, f.kw..., kw...)
end

function eigenfunction_rossbyridge(λs::AbstractVector{<:AbstractVector},
    vs::AbstractVector{<:AbstractMatrix}, m; kw...)
    eigenfunction_rossbyridge(λs[m], vs[m], m; kw...)
end

function eigenfunction_rossbyridge(λs::AbstractVector{<:Number},
    vs::AbstractMatrix{<:Number}, m; kw...)

    ΔΩ_frac = get(kw, :ΔΩ_frac, 0)
    minind = findmin(abs, real(λs) .- RossbyWaveSpectrum.rossby_ridge(m; ΔΩ_frac))[2]
    eigenfunction(vs[:, minind], m; kw...)
end

function eignorm(v)
    minval, maxval = extrema(v)
    abs(minval) > abs(maxval) ? minval : maxval
end

function multiple_eigenfunctions_surface_m(feig::FilteredEigen, m; kw...)
    @unpack operators = feig
    multiple_eigenfunctions_surface_m(feig.lams, feig.vs, m; operators, kw...)
end
function multiple_eigenfunctions_surface_m(λs::AbstractVector, vecs::AbstractVector, m; kw...)
    multiple_eigenfunctions_surface_m(λs[m], vecs[m], m; kw...)
end
function multiple_eigenfunctions_surface_m(λs::AbstractVector, vecs::AbstractMatrix, m;
        operators, f = figure(), kw...)

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
        (; VWSinv, θ) = RossbyWaveSpectrum.eigenfunction_realspace(v, m; operators)
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

function multiple_eigenfunctions_radial_m(feig::FilteredEigen, m; kw...)
    @unpack operators = feig
    multiple_eigenfunctions_radial_m(feig.lams, feig.vs, m; operators, kw...)
end
function multiple_eigenfunctions_radial_m(λs::AbstractVector, vecs::AbstractVector, m; kw...)
    multiple_eigenfunctions_radial_m(λs[m], vecs[m], m; kw...)
end
function multiple_eigenfunctions_radial_m(λs::AbstractVector, vecs::AbstractMatrix, m;
        operators, f = figure(), kw...)

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
        (; VWSinv, θ) = RossbyWaveSpectrum.eigenfunction_realspace(v, m; operators)
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

function eigenfunction_rossbyridge_allstreamfn(f::FilteredEigen, m; kw...)
    @unpack operators = f
    eigenfunction_rossbyridge_allstreamfn(f.lams, f.vs, m; operators, kw...)
end
function eigenfunction_rossbyridge_allstreamfn(λs::AbstractVector, vs::AbstractVector, m; operators, kw...)
    fig = plt.figure(constrained_layout = true, figsize = (10, 4))
    subfigs = fig.subfigures(1, 3, wspace = 0.08, width_ratios = [1, 0.8, 0.8])
    eigenfunction_rossbyridge(λs, vs, m; operators, f = subfigs[1],
        kw..., constrained_layout = true, save=false)
    multiple_eigenfunctions_surface_m(λs, vs, m; operators, f = subfigs[2],
        kw..., constrained_layout = true, save=false)
    multiple_eigenfunctions_radial_m(λs, vs, m; operators, f = subfigs[3],
        kw..., constrained_layout = true, save=false)
    if get(kw, :save, false)
        savefig(joinpath(plotdir, "eigenfunction_rossby_all.eps"))
    end
end

function eigenfunctions_polar(feig::FilteredEigen, m; mode=nothing,
        cmap=mode == :asym ? ("RdYlBu", "YlOrBr") : ("YlOrBr", "RdYlBu"), kw...)

    f, axlist = get(kw, :faxlist) do
        subplots(1,2, sharey=true)
    end
    ind = if mode==:asym
        argmin(abs.(feig.lams[m] .- 2.5*RossbyWaveSpectrum.rossby_ridge(m)))
    elseif mode==:symsectoral
        argmin(abs.(feig.lams[m] .- RossbyWaveSpectrum.rossby_ridge(m)))
    elseif mode==:symhighest
        lastindex(feig.lams[m])
    else
        kw[:ind]
    end

    eigenfunction(feig, m, ind, f=f, field=:V,
        cmap=cmap[1], polar=true, axprofile = axlist[1])
    eigenfunction(feig, m, ind, f=f, field=:W,
        cmap=cmap[2], polar=true, axprofile = axlist[2])
    f.set_size_inches(8,4)
    f.tight_layout()
end

function eigenfunctions_polar(fsym::FilteredEigen, fasym::FilteredEigen, m; kw...)
    f, axlist = subplots(2,2, sharex=true, sharey=true)
    eigenfunctions_polar(fasym, m; mode=:asym, faxlist=(f, axlist[1, 1:2]), kw...)
    eigenfunctions_polar(fsym, m; mode=:symhighest, faxlist=(f, axlist[2, 1:2]), kw...)
    f.set_size_inches(8,8)
    f.tight_layout()
end

function eigenfunction_spectrum(v, m; operators, V_symmetric, kw...)
    @unpack nr, nℓ = operators.radial_params
    nvariables = length(v) ÷ (nr*nℓ)
    Vℓ = RossbyWaveSpectrum.ℓrange(m, nℓ, V_symmetric)
    Wℓ = RossbyWaveSpectrum.ℓrange(m, nℓ, !V_symmetric)
    Sℓ = Wℓ

    Vr = reshape(@view(v.re[1:nr*nℓ]), nr, nℓ);
    Vi = reshape(@view(v.im[1:nr*nℓ]), nr, nℓ);
    Wr = reshape(@view(v.re[nr*nℓ .+ (1:nr*nℓ)]), nr, nℓ);
    Wi = reshape(@view(v.im[nr*nℓ .+ (1:nr*nℓ)]), nr, nℓ);

    Vrf = zeros(size(Vr,1), m:maximum(Vℓ));
    @views @. Vrf[:, Vℓ] = Vr
    Vif = zeros(size(Vi,1), m:maximum(Vℓ))
    @views @. Vif[:, Vℓ] = Vi
    Wrf = zeros(size(Wr,1), m:maximum(Wℓ))
    @views @. Wrf[:, Wℓ] = Wr
    Wif = zeros(size(Wi,1), m:maximum(Wℓ))
    @views @. Wif[:, Wℓ] = Wi

    terms = [Vrf, Wrf, Vif, Wif]

    if nvariables == 3
        Sr = reshape(@view(v.re[2nr*nℓ .+ (1:nr*nℓ)]), nr, nℓ)
        Srf = zeros(size(Sr,1), m:maximum(Sℓ))
        @views @. Srf[:, Sℓ] = Sr

        Si = reshape(@view(v.im[2nr*nℓ .+ (1:nr*nℓ)]), nr, nℓ)
        Sif = zeros(size(Si,1), m:maximum(Sℓ))
        @views @. Sif[:, Sℓ] = Si

        terms = [Vrf, Wrf, Srf, Vif, Wif, Sif]
    end

    x = axes.(terms, 2)
    titles = nvariables == 3 ? ["Vr", "Wr", "Sr", "Vi", "Wi", "Si"] : ["Vr", "Wr", "Vi", "Wi"]

    compare_terms(parent.(terms); nrows = 2, titles,
        xlabel = "spharm ℓ", ylabel = "chebyshev order", x, y = 0:nr-1)
end

function eigenfunction_spectrum(f::FilteredEigen, m::Integer, ind::Integer; kw...)
    @unpack operators = f
    eigenfunction_spectrum(f.vs[m][:, ind], m; operators, f.kw..., kw...)
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

function plot_diffrot_radial(; operators, kw...)
    r_raw = RossbyWaveSpectrum.read_angular_velocity_radii()
    Ω_raw = RossbyWaveSpectrum.read_angular_velocity_raw()
    θinds = axes(Ω_raw,2)
    eqind = first(θinds) + (first(θinds) + last(θinds)) ÷ 2
    Ω_raw_eq = Ω_raw[:, eqind]
    @unpack Ω0 = operators.constants
    ΔΩ_raw_eq = Ω_raw_eq .- Ω0

    @unpack r_in = operators.radial_params
    r_in_frac = r_in / Rsun

    plot(r_raw[r_raw .>= r_in_frac], ΔΩ_raw_eq[r_raw .>= r_in_frac]*1e9/2pi, label="radial, equator")
    @unpack rpts = operators

    if get(kw, :plot_smoothed, true)
        (; ΔΩ,) = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives_Fun(;
            operators, rotation_profile = get(kw, :rotation_profile, :solar_equator), kw...);
        plot(rpts/Rsun, ΔΩ.(rpts)*Ω0*1e9/2pi, label="smoothed model")
    end

    xmin, xmax = extrema(rpts)./Rsun
    Δx = xmax - xmin
    pad = Δx*0.1
    xlim(xmin - pad, xmax + pad)
    if get(kw, :plot_guides, false)
        axhline(0, ls="dotted", color="black")
        axvline(operators.radial_params[:r_out]/Rsun, ls="dotted", color="black")
    end
    fontsize = get(kw, :fontsize, 12)
    xlabel(L"r/R_\odot"; fontsize)
    ylabel(L"\Delta\Omega/2\pi" * " [nHz]"; fontsize)
    gca().xaxis.set_major_locator(ticker.MaxNLocator(4))
    gca().yaxis.set_major_locator(ticker.MaxNLocator(4))
    legend(loc="best"; fontsize)
    gca().tick_params(axis="both", which="major", labelsize=fontsize)
    tight_layout()
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
    @unpack radialspace = operators;
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
        fname = joinpath(plotdir, "bc_basis.eps")
        @info "saving to $fname"
        f.savefig(fname)
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
        fname = joinpath(plotdir, "scale_heights.eps")
        @info "saving to $fname"
        f.savefig(fname)
    end
    return nothing
end


end # module plots
