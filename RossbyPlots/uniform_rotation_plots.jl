module UniformRotationPlots
include("src/RossbyPlots.jl")
import .RossbyPlots
import .RossbyPlots: rossbyeigenfilename, filteredeigen

function uniform_rotation_plots(nr, nℓ; modeltag = "", Δl_cutoff = 8, n_cutoff = 10, kw...)
	Fusym = filteredeigen(rossbyeigenfilename(nr, nℓ, "ur", "sym", modeltag);
		Δl_cutoff = 8, n_cutoff = 10, kw...);
	Fuasym = filteredeigen(rossbyeigenfilename(nr, nℓ, "ur", "asym", modeltag);
		Δl_cutoff = 8, n_cutoff = 10, kw...);
	RossbyPlots.uniform_rotation_spectrum(Fusym, Fuasym; kw...)
end

end
