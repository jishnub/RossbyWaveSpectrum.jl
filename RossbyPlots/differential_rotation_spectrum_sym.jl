module SpectrumSym
include("src/RossbyPlots.jl")
using .RossbyPlots
using .RossbyPlots.RossbyWaveSpectrum


function spectrum_sym(nr = 60, nℓ = 50; kw...)
	Fssym = FilteredEigen(
		datadir(
			"dr_solar_latrad_squished_nr$(nr)_nl$(nℓ)_sym_nu5.0e11_rin0.65_rout0.985_trackhanson2020.jld2"));

	Fssymf = RossbyPlots.update_cutoffs(Fssym, Δl_filter = 0, nodes_cutoff=2);

	Fssymf2 = RossbyWaveSpectrum.filter_eigenvalues(Fssymf,
				filterflags=Filters.EIGVAL|Filters.SPATIAL|Filters.EIGVEC,
				eig_imag_stable_cutoff = 0.2, radial_topbotpower_cutoff=0.7, Δl_cutoff=9);

	Fssymf3 = RossbyWaveSpectrum.filter_eigenvalues(Fssymf2,
		filterflags=Filters.EIGVAL, eig_imag_stable_cutoff = 0.1);

	RossbyPlots.spectrum_with_datadispersion((Fssymf2, Fssymf3);
		xlim=(1.5,21),
		nodes_cutoff=nothing,
		Δl_filter=nothing,
		ylim=(-350,275),
		colorederrorbars=true, fillmarker=false,
		kw...)
end

end
