include(joinpath(@__DIR__, "compute_rossby_spectrum.jl"))
using RossbyWaveSpectrum
V_symmetric = true
r_in_frac, r_out_frac = 0.65, 0.985
nr, nℓ = 40, 20
mrange = 2:3
viscosity = 5e11
trackingrate = :hanson2020 # :carrington, :hanson2020, :cutoff or :surface
ΔΩ_scale = 1
Δl_cutoff = 20
n_cutoff = 15
# ΔΩ_frac = 467/453.1 - 1
rotation_profile = :solar_latrad_squished
# rotation_profile = :solar_constant
diffrot = rotation_profile != :uniform
# superadiabaticityparams = (; δrad = -1e-6)
ΔΩ_smoothing_param = 0.01
smoothing_param=1e-4
modeltag = "nu$(viscosity)_rin$(round(r_in_frac, digits=3))_rout$(round(r_out_frac, digits=3))_track$(trackingrate)"
modeltag *= diffrot && !isone(ΔΩ_scale) ? "_rotscale$(round(ΔΩ_scale, digits=2))" : ""
filterflags = Filters.EIGEN | Filters.EIGVEC | Filters.BC | Filters.SPATIAL_EQUATOR

ComputeRossbySpectrum.main(V_symmetric;
nr, nℓ, mrange, rotation_profile, r_in_frac,
r_out_frac, trackingrate, ΔΩ_scale, Δl_cutoff, n_cutoff,
modeltag, viscosity, filterflags, ΔΩ_smoothing_param, smoothing_param)
