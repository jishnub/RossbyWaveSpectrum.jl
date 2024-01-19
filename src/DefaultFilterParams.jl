"""
    DefaultFilterParams

Default list of parameters that are used in the filtering process.
The full list is:
```julia
const DefaultFilterParams = Dict(
    # boundary condition filter: Filters.BC
    :bc_atol => 1e-5,
    # eigval filter: Filters.EIGVAL
    :eig_imag_unstable_cutoff => -1e-6, # below this, modes are considered exponentially growing
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
    :radial_topbotpower_cutoff => 0.7, # modes with higher relative power near boundaries are excluded
    # radial nodes filter # Filters.NODES
    :nnodesmax => 5,
    :nodessmallpowercutoff => 0.05, # minimum relative power above which a zero-crossing is considered
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
