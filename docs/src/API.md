# API

## Public API

```@docs
radial_operators
RotMatrix
mass_matrix
mass_matrix!
uniform_rotation_matrix
uniform_rotation_matrix!
differential_rotation_matrix
differential_rotation_matrix!
uniform_rotation_spectrum
uniform_rotation_spectrum!
differential_rotation_spectrum
differential_rotation_spectrum!
filter_eigenvalues
save_eigenvalues
FilteredEigen
datadir
```

## Post-processing Filters
### Filter flags
```@autodocs
Modules = [RossbyWaveSpectrum.Filters]
```

### Filtering functions
```@docs
RossbyWaveSpectrum.eigensystem_satisfy_filter
```

## Validation

```@docs
RossbyWaveSpectrum.constant_differential_rotation_terms!
```

## Lower level utilities

```@docs
RossbyWaveSpectrum.DefaultFilterParams
RossbyWaveSpectrum.filterfn
RossbyWaveSpectrum.solar_differential_rotation_terms!
RossbyWaveSpectrum.solar_differential_rotation_profile_derivatives_grid
RossbyWaveSpectrum.solar_differential_rotation_profile_derivatives_Fun
RossbyWaveSpectrum.solar_differential_rotation_vorticity_Fun
RossbyWaveSpectrum.constraintmatrix
RossbyWaveSpectrum.superadiabaticity
RossbyWaveSpectrum.eigenfunction_realspace
RossbyWaveSpectrum.compute_constrained_matrix
RossbyWaveSpectrum.compute_constrained_matrix!
RossbyWaveSpectrum.constrained_eigensystem
RossbyWaveSpectrum.constrained_matmul_cache
RossbyWaveSpectrum.allocate_operator_matrix
RossbyWaveSpectrum.allocate_mass_matrix
RossbyWaveSpectrum.allocate_operator_mass_matrices
RossbyWaveSpectrum.allocate_filter_caches
RossbyWaveSpectrum.allocate_projectback_temp_matrices
RossbyWaveSpectrum.operator_matrices
RossbyWaveSpectrum.rossbyeigenfilename
RossbyWaveSpectrum.FilteredEigenSingleOrder
RossbyWaveSpectrum.colatitude_grid
```

## Plotting functions
