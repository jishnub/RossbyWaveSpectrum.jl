using RossbyWaveSpectrum
using DelimitedFiles

nrs = 20:5:60
nℓs = 20:5:60
mr = 5:10

@time M = RossbyWaveSpectrum.spectrum_convergence_misfits(nrs, nℓs, mr)
writedlm(RossbyWaveSpectrum.datadir("convergence_mr$(mr)_nr$(nrs)_nell$(nℓs)"), M)
