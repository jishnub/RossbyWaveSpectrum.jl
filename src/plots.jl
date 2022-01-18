using PyCall
using PyPlot
using RossbyWaveSpectrum
using RossbyWaveSpectrum: Rsun
using LaTeXStrings
using SimpleDelimitedFiles
using JLD2

plotdir = joinpath(dirname(dirname(@__DIR__)), "plots")
ticker = pyimport("matplotlib.ticker")

function plot_rossby_ridges(mr; ax = gca(), ΔΩ_by_Ω = 0)
    ax.plot(mr, RossbyWaveSpectrum.rossby_ridge.(mr; ΔΩ_by_Ω),
        label = "Sectoral Rossby",
        lw = 1,
        color = "black"
    )
    if ΔΩ_by_Ω != 0
        ax.plot(mr, RossbyWaveSpectrum.rossby_ridge.(mr),
            label = "Uniformly rotating",
            lw = 0.5,
            color = "black",
            ls = "dashed",
            dashes = (6, 10),
        )
    end
end

function spectrum(f, nr, nℓ, mr::AbstractVector; kw...)
    λv = RossbyWaveSpectrum.filter_eigenvalues_mrange(f, nr, nℓ, mr; kw...)
    lam = map(first, λv)
    spectrum(lam, mr; kw...)
    λv
end

function spectrum(fname::String; kw...)
    lam, mr = load(fname, "lam", "mr")
    spectrum(lam, mr; kw...)
end

function spectrum(lam::AbstractArray, mr; ΔΩ_by_Ω = 0, kw...)
    f, ax = subplots()
    ax.set_xlabel("m", fontsize = 12)
    ax.set_ylabel(L"\omega/" * L"\Omega", fontsize = 12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer = true))

    markerkw = Dict(
        :mfc => "0.7",
        :mec => "0.4",
        :ms => 5,
        :marker => "o",
        :ls => "None",
        :lw => 0.3,
    )

    lamcat = mapreduce(real, vcat, lam)
    mcat = reduce(vcat, [range(m, m, length(λi)) for (m, λi) in zip(mr, lam)])
    plot_rossby_ridges(mr; ΔΩ_by_Ω)
    ax.plot(mcat, lamcat; markerkw...)
    ax.legend(loc = "best", fontsize = 12)

    m_zoom = mr[end-6:end]
    lamcat_inset = mapreduce(real, vcat, lam[m_zoom])
    mcat_inset = reduce(vcat, [range(m, m, length(λi)) for (m, λi) in zip(mr[m_zoom], lam[m_zoom])])

    axins = ax.inset_axes([0.5, 0.4, 0.4, 0.35])
    ymin = minimum(lamcat_inset) - 0.005
    ymax = maximum(lamcat_inset) + 0.005
    axins.set_ylim((ymin, ymax))
    axins.xaxis.set_major_locator(ticker.MaxNLocator(4, integer = true))
    plt.setp(axins.spines.values(), color = "0.2", lw = "0.5")

    plot_rossby_ridges(m_zoom, ax = axins; ΔΩ_by_Ω)
    axins.plot(mcat_inset, lamcat_inset; markerkw...)

    ax.indicate_inset_zoom(axins, edgecolor = "grey", lw = 0.5)

    f.savefig(joinpath(plotdir,
        ΔΩ_by_Ω == 0 ? "uniform_rotation_spectrum.eps" : "differential_rotation_spectrum.eps"))
end

rossbyeigenfilename(nr, nℓ, tag = "ur") = "$(tag)_nr$(nr)_nl$(nℓ).jld2"
function plot_convergence(; n_cutoff = 5, eigen_rtol = 0.01, Δl_cutoff = 7)
    nr_nℓ_list = [(30, 15), (40, 20), (50, 20), (60, 20), (60, 30),]
    λs = Dict((nr, nℓ) =>
        RossbyWaveSpectrum.filter_eigenvalues(RossbyWaveSpectrum.datadir(rossbyeigenfilename(nr, nℓ));
            n_cutoff, eigen_rtol, Δl_cutoff)[1]
              for (nr, nℓ) in nr_nℓ_list
    )
    dnr = 5
    dnℓ = 5
    nrs = range(extrema(first, nr_nℓ_list)..., step = dnr)
    nrs = range(extrema(last, nr_nℓ_list)..., step = dnℓ)

end

function eigenfunction(v, m, operators; theory = false)
    (; V, θ) = RossbyWaveSpectrum.eigenfunction_realspace(v, m, operators)
    Vr = real(V)
    Vrmax = maximum(abs, Vr)
    if Vrmax != 0
        Vr ./= Vrmax
    end
    Vrmax = maximum(abs, Vr)
    (; coordinates) = operators
    (; r) = coordinates
    r_frac = r ./ Rsun
    nθ = length(θ)

    Vr .*= sign(Vr[end, nθ÷2])

    f = figure()
    axprofile = subplot2grid((3, 3), (1, 1), colspan = 2, rowspan = 2)
    axsurf = subplot2grid((3, 3), (0, 1), colspan = 2, sharex = axprofile)
    axdepth = subplot2grid((3, 3), (1, 0), rowspan = 2, sharey = axprofile)

    piformatter = (x, _) -> begin
        n = round(Int, 4x / pi)
        prestr = (n == 4 || n == 1) ? "" : iseven(n) ?
                 (n == 2 ? "" : string(div(n, 2))) : string(n)
        poststr = n == 4 ? "" : iseven(n) ? "/2" : "/4"
        prestr * "π" * poststr
    end

    axsurf.plot(θ, (@view Vr[end, :]), color = "black")
    axsurf.set_ylabel("Angular\nprofile", fontsize = 11)
    axsurf.set_xticks(pi * (1/4:1/4:1))
    axsurf.xaxis.set_major_formatter(ticker.FuncFormatter(piformatter))

    if theory
        θmarkers = @view θ[1:10:end]
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

    axdepth.plot((@view Vr[:, nθ÷2]), r_frac,
        color = "black",
    )
    axdepth.set_xlabel("Depth profile", fontsize = 11)
    axdepth.set_ylabel("Fractional radius", fontsize = 11)
    axdepth.yaxis.set_major_locator(ticker.MaxNLocator(4))

    axprofile.pcolormesh(θ, r_frac, Vr, cmap = "Greys", rasterized = true, shading = "auto")
    axprofile.set_xlabel("colatitude (θ) [radians]", fontsize = 11)

    f.tight_layout()
    f.savefig(joinpath(plotdir, "eigenfunction.eps"))
end

function eigenfunction_rossbyridge(lam, v, m, operators)
    minind = argmin(abs.(lam .- RossbyWaveSpectrum.rossby_ridge(m)))
    eigenfunction((@view v[:, minind]), m, operators)
end
