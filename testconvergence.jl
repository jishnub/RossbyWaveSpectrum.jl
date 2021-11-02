using RossbyWaveSpectrum
using DelimitedFiles

nrs = 20:5:50
nℓs = 20:5:50
mr = 5:10

M = RossbyWaveSpectrum.testrossbyridgefreqconvergence(nrs, nℓs, mr)
writedlm(RossbyWaveSpectrum.datadir("convergence_mr$(mr)_nr$(nrs)_nell$(nℓs)"), M)
