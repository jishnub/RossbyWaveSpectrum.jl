const DefaultFilterParams = Dict(
    # boundary condition filter
    :bc_atol => 1e-5,
    # eigval filter
    :eig_imag_unstable_cutoff => -1e-6,
    :eig_imag_to_real_ratio_cutoff => 3,
    :eig_imag_stable_cutoff => 0.5,
    # eigensystem satisfy filter
    :eigen_rtol => 0.01,
    # smooth eigenvector filter
    :Δl_cutoff => 7,
    :n_cutoff => 10,
    :eigvec_spectrum_power_cutoff => 0.5,
    # spatial localization filter
    :θ_cutoff => deg2rad(45),
    :equator_power_cutoff_frac => 0.4,
    :pole_cutoff_angle => deg2rad(25),
    :pole_power_cutoff_frac => 0.05,
    # radial nodes filter
    :nnodesmax => 10,
    :nodessmallpowercutoff => 0.05,
    # exclude a field from a filter if relative power is below a cutoff
    :filterfieldpowercutoff => 1e-2,
    :eigvec_spectrum_low_n_power_fraction_cutoff => 1,
    :radial_topbotpower_cutoff => 0.5,
)
