"""
    DefaultFilterParams

Default list of parameters that are used in the filtering process.
The full list is:
```julia
const DefaultFilterParams = Dict(
    # boundary condition filter: Filters.BC
    :bc_atol => 1e-5, # absolute tolerance to which boundary conditions must be satisfied
    # eigval filter: Filters.EIGVAL
    :eig_imag_unstable_cutoff => -1e-6, # absolute lower cutoff below which modes are considered to be exponentially growing
    :eig_imag_to_real_ratio_cutoff => 3, # relative upper cutoff on linewidths
    :eig_imag_stable_cutoff => 0.2, # absolute upper cutoff on linewidths
    # eigensystem satisfy filter: Filters.EIGEN
    :eigen_rtol => 0.01, # tolerance above which modes are rejected
    # smooth eigenvector filter: Filters.EIGVEC
    :Δl_cutoff => 7, # spherical harmonic degree
    :n_cutoff => 10, # chebyshev order
    :eigvec_spectrum_power_cutoff => 0.5, # fraction of total power required below cutoffs
    :eigvec_spectrum_low_n_power_fraction_cutoff => 1, # maximum ratio of total power above n_cutoff to total power below n_cutoff
    # spatial localization filter: Filters.SPATIAL
    :θ_cutoff => deg2rad(45), # colatitude below which a mode is considered equatorial
    :equator_power_cutoff_frac => 0.4, # minimum fraction of power that lie below θ_cutoff for equatorial modes
    :pole_cutoff_angle => deg2rad(25), # colatitude above which a mode is considered polar
    :pole_power_cutoff_frac => 0.05, # maximum fraction of power that may be near poles for equatorial modes
    :radial_topbotpower_cutoff => 0.7, # upper cutoff for power in the top and bottom 10% of the radial domain
    # radial nodes filter # Filters.NODES
    :nnodesmax => 5, # maximum number of radial nodes in either the real or imaginary component of the eigenfunction
    :nodessmallpowercutoff => 0.05, # minimum fractional power between roots beyond which a zero-crossing is considered real
    # exclude a field from filters if relative power is below a cutoff
    :filterfieldpowercutoff => 1e-2,
)
```
Each parameter may be passed as a keyword argument to [`filter_eigenvalues`](@ref).
"""
const DefaultFilterParams = Dict(
    # boundary condition filter: Filters.BC
    :bc_atol => 1e-5,
    # eigval filter: Filters.EIGVAL
    :eig_imag_unstable_cutoff => -1e-6,
    :eig_imag_to_real_ratio_cutoff => 3,
    :eig_imag_stable_cutoff => 0.2,
    # eigensystem satisfy filter: Filters.EIGEN
    :eigen_rtol => 0.01,
    # smooth eigenvector filter: Filters.EIGVEC
    :Δl_cutoff => 7,
    :n_cutoff => 10,
    :eigvec_spectrum_power_cutoff => 0.5,
    :eigvec_spectrum_low_n_power_fraction_cutoff => 1,
    # spatial localization filter: Filters.SPATIAL
    :θ_cutoff => deg2rad(45),
    :equator_power_cutoff_frac => 0.4,
    :pole_cutoff_angle => deg2rad(25),
    :pole_power_cutoff_frac => 0.05,
    :radial_topbotpower_cutoff => 0.7,
    # radial nodes filter: Filters.NODES
    :nnodesmax => 5,
    :nodessmallpowercutoff => 0.05,
    # exclude a field from a filter if relative power is below a cutoff
    :filterfieldpowercutoff => 1e-2,
)
