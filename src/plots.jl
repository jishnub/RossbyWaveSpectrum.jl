using PyCall
using PyPlot
using RossbyWaveSpectrum
using RossbyWaveSpectrum: Rsun, rossbyeigenfilename
using LaTeXStrings
using SimpleDelimitedFiles
using JLD2
using Printf
using LinearAlgebra

plotdir = joinpath(dirname(dirname(@__DIR__)), "plots")
ticker = pyimport("matplotlib.ticker")
axes_grid1 = pyimport("mpl_toolkits.axes_grid1")

function cbformat(x, _)
    a, b = split((@sprintf "%.1e" x), 'e')
    c = parse(Int, b)
    a * L"\times10^{%$c}"
end

function plot_rossby_ridges(mr; ax = gca(), ΔΩ_by_Ω = 0, kw...)
    if get(kw, :sectoral_rossby_ridge, true)
        ax.plot(mr, RossbyWaveSpectrum.rossby_ridge.(mr; ΔΩ_by_Ω),
            label = ΔΩ_by_Ω == 0 ? "Sectoral" :
                    L"\Delta\Omega/\Omega_0 = " * string(round(ΔΩ_by_Ω, sigdigits = 1)),
            lw = 1,
            color = get(kw, :sectoral_rossby_ridge_color, "black"),
            zorder = 0,
            ls = get(kw, :sectoral_rossby_ridge_ls, "solid")
        )
    end

    if get(kw, :uniform_rotation_ridge, true)
        ax.plot(mr, RossbyWaveSpectrum.rossby_ridge.(mr),
            label = "Uniformly\nrotating",
            lw = 1,
            color = "black",
            ls = "dashed",
            dashes = (6, 3),
            zorder = 0,
        )
    end
end

function spectrum(fname::String; kw...)
    lam, mr = load(fname, "lam", "mr")
    spectrum(lam, mr; kw...)
end

function spectrum(lam::AbstractArray, mr;
    f = figure(),
    ax = subplot(),
    m_zoom = mr[max(begin, end - 6):end],
    kw...)

    ax.set_xlabel("m", fontsize = 12)
    ax.set_ylabel(L"\Re[\omega]/" * L"\Omega_0", fontsize = 12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer = true))

    markerkw = Dict(
        :edgecolors => "k",
        :s => 30,
        :marker => "o",
        :ls => "None",
        :cmap => "Greys",
        :lw => 0.5,
    )

    lamcat = mapreduce(real, vcat, lam)
    lamimcat = mapreduce(imag, vcat, lam)
    vmin, vmax = extrema(lamimcat)
    vmin = min(0, vmin)
    mcat = reduce(vcat, [range(m, m, length(λi)) for (m, λi) in zip(mr, lam)])
    s = ax.scatter(mcat, lamcat; c = lamimcat, markerkw..., vmax = vmax, vmin = vmin)
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size = "3%", pad = 0.05)
    cb = colorbar(mappable = s, cax = cax, format = ticker.FuncFormatter(cbformat))
    cb.ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    cb.ax.set_title(L"\Im[\omega]/\Omega_0")

    if get(kw, :rossbyridges, true)
        plot_rossby_ridges(mr; ax, kw...)
    end

    if get(kw, :zoom, false)
        lamcat_inset = mapreduce(real, vcat, lam[m_zoom])
        lamimcat_inset = mapreduce(imag, vcat, lam[m_zoom])
        mcat_inset = reduce(vcat, [range(m, m, length(λi)) for (m, λi) in zip(mr[m_zoom], lam[m_zoom])])

        axins = ax.inset_axes([0.5, 0.4, 0.4, 0.3])
        ymin = minimum(lamcat_inset) - 0.005
        ymax = maximum(lamcat_inset) + 0.005
        axins.set_ylim((ymin, ymax))
        axins.xaxis.set_major_locator(ticker.MaxNLocator(4, integer = true))
        axins.yaxis.set_major_locator(ticker.MaxNLocator(2))
        plt.setp(axins.spines.values(), color = "0.2", lw = "0.5")
        axins.scatter(mcat_inset, lamcat_inset; c = lamimcat_inset, markerkw...,
            vmax = vmax, vmin = vmin)

        if get(kw, :rossbyridges, true)
            plot_rossby_ridges(m_zoom; ax = axins, kw...)
        end

        ax.indicate_inset_zoom(axins, edgecolor = "grey", lw = 0.5)
    end

    if get(kw, :rossbyridges, true)
        ax.legend(loc = "best", fontsize = 12)
    end

    f.tight_layout()

    if get(kw, :save, false)
        filenametag = get(kw, :filenametag, "")
        rotation = get(kw, :rotation, "uniform")
        f.savefig(joinpath(plotdir, "$(rotation)_rotation_spectrum$(filenametag).eps"))
    end
end

piformatter = (x, _) -> begin
    n = round(Int, 4x / pi)
    prestr = (n == 4 || n == 1) ? "" : iseven(n) ?
             (n == 2 ? "" : string(div(n, 2))) : string(n)
    poststr = n == 4 ? "" : iseven(n) ? "/2" : "/4"
    prestr * "π" * poststr
end

function differential_rotation_spectrum(lam_rad, lam_solar, mr = axes(lam_rad, 1); operators, kw...)
    f, axlist = subplots(2, 2)
    (; r) = operators.coordinates
    r_frac = r ./ Rsun

    m = 1
    ntheta = RossbyWaveSpectrum.ntheta_ℓmax(nℓ, m)
    (; thetaGL) = RossbyWaveSpectrum.gausslegendre_theta_grid(ntheta)
    ΔΩ_r, _ = RossbyWaveSpectrum.radial_differential_rotation_profile(operators, thetaGL, :solar_equator)
    axlist[1, 1].plot(r_frac, ΔΩ_r / 2pi * 1e9, color = "black")
    axlist[1, 1].set_ylabel(L"\Delta\Omega/2\pi" * " [nHz]")
    axlist[1, 1].set_xlabel(L"r / R_\odot", fontsize = 11)
    axlist[1, 1].set_title("Radial differential rotation")
    axlist[1, 1].xaxis.set_major_locator(ticker.MaxNLocator(4))
    axlist[1, 1].yaxis.set_major_locator(ticker.MaxNLocator(4))

    spectrum(lam_rad, mr; f, ax = axlist[1, 2], kw...,
        uniform_rotation_ridge = true,
        save = false, sectoral_rossby_ridge = false)
    axlist[1, 2].set_ylabel("")

    ΔΩ, _ = RossbyWaveSpectrum.solar_differential_rotation_profile(operators, thetaGL, :solar)
    p = axlist[2, 1].pcolormesh(thetaGL, r_frac, ΔΩ / 2pi * 1e9,
        cmap = "Greys_r", rasterized = true, shading = "auto")
    axlist[2, 1].xaxis.set_major_locator(ticker.MaxNLocator(4))
    axlist[2, 1].yaxis.set_major_locator(ticker.MaxNLocator(4))
    axlist[2, 1].set_ylabel(L"r / R_\odot", fontsize = 11)
    axlist[2, 1].set_xlabel("Colatitude (θ) [radians]", fontsize = 11)
    axlist[2, 1].set_title("Solar-like differential rotation")
    cb = colorbar(mappable = p, ax = axlist[2, 1])

    spectrum(lam_solar, mr; f, ax = axlist[2, 2], kw...,
        uniform_rotation_ridge = true,
        save = false, sectoral_rossby_ridge = false)
    axlist[2, 2].set_ylabel("")
    axlist[2, 2].set_title("Solar")

    f.set_size_inches(7, 5)
    f.tight_layout()
    if get(kw, :save, false)
        f.savefig(joinpath(plotdir, "differential_rotation_spectrum.eps"))
    end
end

function eigenfunction(v, m, operators; field = :V, theory = false, f = figure(), kw...)
    (; θ, VWSinv) = RossbyWaveSpectrum.eigenfunction_realspace(v, m, operators)
    V = getproperty(VWSinv, field)::Matrix{ComplexF64}
    Vr = real(V)
    Vrmax = eignorm(Vr)
    if Vrmax != 0
        Vr ./= Vrmax
    end
    (; coordinates) = operators
    (; r) = coordinates
    r_frac = r ./ Rsun
    nθ = length(θ)
    V_equator_depthprofile = @view Vr[:, nθ÷2]
    r_max_ind = argmax(abs.(V_equator_depthprofile))

    spec = f.add_gridspec(3, 3)

    axprofile = f.add_subplot(py"$(spec)[1:, 1:]")
    axsurf = f.add_subplot(py"$(spec)[0, 1:]", sharex = axprofile)
    axdepth = f.add_subplot(py"$(spec)[1:, 0]", sharey = axprofile)

    axsurf.plot(θ, (@view Vr[r_max_ind, :]), color = "black")
    axsurf.set_ylabel("Angular\nprofile", fontsize = 11)
    axsurf.set_xticks(pi * (1/4:1/4:1))
    axsurf.xaxis.set_major_formatter(ticker.FuncFormatter(piformatter))

    if theory
        markevery_theory = get(kw, :markevery_theory, 10)
        θmarkers = @view θ[1:markevery_theory:end]
        axsurf.plot(θmarkers, sin.(θmarkers) .^ m,
            marker = "o",
            mfc = "0.7",
            mec = "0.5",
            ls = "None",
            ms = 5,
            label = "theory",
        )
        axsurf.legend(loc = "best")
    end

    axdepth.plot(V_equator_depthprofile, r_frac,
        color = "black",
    )
    axdepth.set_xlabel("Depth profile", fontsize = 11)
    axdepth.set_ylabel("Fractional radius", fontsize = 11)
    axdepth.yaxis.set_major_locator(ticker.MaxNLocator(4))

    axprofile.pcolormesh(θ, r_frac, Vr, cmap = "Greys", rasterized = true, shading = "auto")
    axprofile.set_xlabel("colatitude (θ) [radians]", fontsize = 11)

    f.suptitle("Sectoral eigenfunction for m = $m")

    if !get(kw, :constrained_layout, false)
        f.tight_layout()
    end
    if get(kw, :save, false)
        savefig(joinpath(plotdir, "eigenfunction.eps"))
    end
end

function eigenfunction_rossbyridge(λs::AbstractVector{<:AbstractVector},
    vs::AbstractVector{<:AbstractMatrix}, m, operators; kw...)
    eigenfunction_rossbyridge(λs[m], vs[m], m, operators; kw...)
end

function eigenfunction_rossbyridge(λs::AbstractVector{<:Number},
    vs::AbstractMatrix{<:Number}, m, operators; kw...)

    ΔΩ_by_Ω = get(kw, :ΔΩ_by_Ω, 0)
    minind = findmin(abs, real(λs) .- RossbyWaveSpectrum.rossby_ridge(m; ΔΩ_by_Ω))[2]
    eigenfunction(vs[:, minind], m, operators; theory = true, kw...)
end

function eignorm(v)
    minval, maxval = extrema(v)
    abs(minval) > abs(maxval) ? minval : maxval
end

function multiple_eigenfunctions_m(λs::AbstractVector, vecs::AbstractVector,
        m, operators; f = figure(), kw...)
    multiple_eigenfunctions_m(λs[m], vecs[m], m, operators; f, kw...)
end
function multiple_eigenfunctions_m(λs::AbstractVector, vecs::AbstractMatrix, m, operators; f = figure(), kw...)
    ax = f.add_subplot()
    ax.set_xlabel("colatitude (θ) [radians]", fontsize = 12)
    ax.set_ylabel("Angular profile", fontsize = 12)
    ax.set_xticks(pi * (1/4:1/4:1))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(piformatter))

    ax.set_title("Normalized eigenfunctions for m = $m", fontsize = 12)

    lscm = Iterators.product(("solid", "dashed", "dotted"), ("black", "0.5", "0.3"), ("None", "."))

    vm = reverse(vecs, dims = 2)
    λm = reverse(λs)

    λ0 = 2 / (m + 1)
    λm ./= λ0

    for (ind, (v, (ls, c, marker))) in enumerate(zip(eachcol(vm), lscm))
        (; V, θ) = RossbyWaveSpectrum.eigenfunction_realspace(v, m, operators)
        Vr = real(V)
        Vr_surf = Vr[end, :]

        Vrmax_sign = sign(eignorm(Vr_surf))
        Vr_surf .*= Vrmax_sign
        normalize!(Vr_surf)

        eigvalfrac = round(real(λm[ind]), sigdigits = 2)

        ax.plot(θ, Vr_surf; ls, color = c,
            label = string(eigvalfrac),
            marker, markevery = 10)
    end

    legend = ax.legend(title = L"\frac{\Re[\omega/\Omega]}{2/(m+1)}")
    legend.get_title().set_fontsize("12")
    if !get(kw, :constrained_layout, false)
        f.tight_layout()
    end
end

function eigenfunctions_rossbyridge_all(λs, vs, m, operators; kw...)
    fig = plt.figure(constrained_layout = true, figsize = (8, 4))
    subfigs = fig.subfigures(1, 2, wspace = 0.15, width_ratios = [1, 1])
    eigenfunction_rossbyridge(λs, vs, m, operators; f = subfigs[1], constrained_layout = true)
    multiple_eigenfunctions_m(λs, vs, m, operators; f = subfigs[2], constrained_layout = true)
    if get(kw, :save, false)
        savefig(joinpath(plotdir, "eigenfunction_rossby_all.eps"))
    end
end

function plot_matrix(M)
    f, axlist = subplots(3, 3)
    for colind in 1:3, rowind in 1:3
        Mv = abs.(RossbyWaveSpectrum.matrix_block(M, rowind, colind))
        vmax = max(maximum(Mv), 1e-200)
        ax = axlist[rowind, colind]
        p = ax.imshow(Mv, vmax = vmax, vmin = -vmax, cmap = "RdBu_r")
        colorbar(mappable = p, ax = ax)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    end
    f.tight_layout()
end

function plot_matrix_block(M, rowind, colind)
    f, axlist = subplots(1, 2)
    M = RossbyWaveSpectrum.matrix_block(M, rowind, colind)
    A = real(M)
    Amax = maximum(abs, A)
    p1 = axlist[1].pcolormesh(A, cmap = "RdBu", vmax = Amax, vmin = -Amax)
    cb1 = colorbar(mappable = p1, ax = axlist[1])
    A = imag(M)
    Amax = maximum(abs, A)
    p2 = axlist[2].pcolormesh(A, cmap = "RdBu", vmax = Amax, vmin = -Amax)
    cb2 = colorbar(mappable = p2, ax = axlist[2])
    f.tight_layout()
end

function plot_matrix_block(M, rowind, colind, nr, ℓind, ℓ′ind)
    f, axlist = subplots(1, 2)
    M = RossbyWaveSpectrum.matrix_block(M, rowind, colind)
    ℓinds = (ℓind - 1) * nr .+ (1:nr)
    ℓ′inds = (ℓ′ind - 1) * nr .+ (1:nr)
    A = real(M)[ℓinds, ℓ′inds]
    Amax = maximum(abs, A)
    p1 = axlist[1].pcolormesh(A, cmap = "RdBu", vmax = Amax, vmin = -Amax)
    cb1 = colorbar(mappable = p1, ax = axlist[1])
    A = imag(M)[ℓinds, ℓ′inds]
    Amax = maximum(abs, A)
    p2 = axlist[2].pcolormesh(A, cmap = "RdBu", vmax = Amax, vmin = -Amax)
    cb2 = colorbar(mappable = p2, ax = axlist[2])
    f.tight_layout()
end

function plot_diffrot_radial(operators, smoothing_param = 1e-5)
    (; nℓ, r_out) = operators.radial_params
    (; r) = operators.coordinates
    # arbitrary theta values for the radial profile
    m = 2
    ntheta = RossbyWaveSpectrum.ntheta_ℓmax(nℓ, m)
    (; thetaGL) = RossbyWaveSpectrum.gausslegendre_theta_grid(ntheta)
    ΔΩ_r = RossbyWaveSpectrum.radial_differential_rotation_profile(operators, thetaGL, :solar_equator; smoothing_param)
    ΔΩ_spl = Spline1D(r, ΔΩ_r)
    drΔΩ_real = derivative(ΔΩ_spl, r)
    d2rΔΩ_real = derivative(ΔΩ_spl, r, nu = 2)

    parentdir = dirname(@__DIR__)
    r_ΔΩ_raw = RossbyWaveSpectrum.read_angular_velocity_radii(parentdir)
    Ω_raw = RossbyWaveSpectrum.read_angular_velocity_raw(parentdir)
    Ω0 = RossbyWaveSpectrum.equatorial_rotation_angular_velocity(r_out / Rsun, r_ΔΩ_raw, Ω_raw)
    ΔΩ_raw = Ω_raw .- Ω0
    nθ = size(ΔΩ_raw, 2)
    lats_raw = LinRange(0, pi, nθ)
    θind_equator_raw = findmin(abs.(lats_raw .- pi / 2))[2]

    f, axlist = subplots(3, 1, sharex = true)

    r_frac = r ./ Rsun
    r_frac_min = minimum(r_frac)
    r_inds = r_ΔΩ_raw .>= r_frac_min

    axlist[1].plot(r_ΔΩ_raw[r_inds], ΔΩ_raw[r_inds, θind_equator_raw] / Ω0)
    axlist[1].plot(r_frac, ΔΩ_r / Ω0, "o-")

    axlist[2].plot(r_frac, drΔΩ_real, "o-")

    axlist[3].plot(r_frac, d2rΔΩ_real, "o-")
    f.tight_layout()
end
