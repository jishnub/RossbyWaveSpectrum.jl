# RossbyWaveSpectrum.jl

A Julia code to compute the spectrum of inertial waves in the Sun.

# Installation
To install the code, run `./INSTALL.sh`. This requires `bash` and `curl` to run, and will download and install `julia` using the installer `juliaup`. It will also install the requisite project dependencies. The project uses Julia v1.10.0, which may be installed independently as well, in which case one only needs to instantiate the environments as listed in `INSTALL.sh`.

Users running the code on a heterogeneous cluster (where the login and compute nodes have different CPU architectures) may want to set the environment variable `JULIA_CPU_TARGET` appropriately in their shell rc file. One may check the CPU architecture by running
```julia
julia> Sys.CPU_NAME
"skylake-avx512"
```
in the julia REPL on each node.

As an example, a valid setting may be
```
export JULIA_CPU_TARGET="generic;skylake-avx512,clone_all;icelake-server,clone_all"
```
if e.g. the login node has `skylake-avx512` and the compute node has `icelake-server`. Setting this is particularly important if one node has Intel processors whereas the other has AMD ones.

# Starting Julia

Typically the code is run multi-threaded, so one needs to specify the number of threads while launching Julia.
As an example, if we want to use `5` Julia threads, this may be done as
```
julia --project=. --startup-file=no -t 5
```
The flag `--project` should point to the path to the code.
The example above assumes that you are in the top-level code directory (`RossbyWaveSpectrum.jl`).

The flag `--startup-file=no` is optional, and may only be necessary if one has added a custom startup file that may interfere
with the code.

The flag `-t 5` indicates that `5` Julia threads are to be used.
Typically, we would want to use as many Julia threads as the number of azimuthal orders `m` that we want to solve for.
This ensures that all `m`s run in parallel.
Note that this differs from the number of BLAS threads that are used in solving the eigenvalue problem. If the code is
being run on a cluster, this is automatically inferred from the number of allocated cores and the number of Julia threads.
Alternately, this may be specified through the environment variable `MKL_NUM_THREADS`.

!!! note
	Currently, the code uses multi-threading and does not support distributed usage.
	Therefore one instance of the code must be launched on one node of a cluster.
	Distinct instances of the code that use different parameters may be run parallely on multiple nodes.

# Running the code

Before running the code, we would want to set the location where the output will be written to.
This is determined by the environment variable `SCRATCH`, which typically corresponds to a user's
scratch directory on a cluster. The output files will be written to `$SCRATCH/RossbyWaves`, and
the path will be created if it doesn't exist. Note that if `SCRATCH` is not specified, it will be set
to `homedir()` by default in the code.

The first step is to load the package
```julia
using RossbyWaveSpectrum
```

We start by computing the radial operators. We define some parameters:
```julia
nr, nℓ = 40, 20 # number of radial and latitudinal spectral coefficients
r_in_frac = 0.65 # inner boundary of the domain as a fraction of the solar radius
r_out_frac = 0.985 # outer boundary of the domain as a fraction of the solar radius
ν = 5e11 # kinematic viscosity coefficient in CGS units
trackingrate = :hanson2020 # track at 453.1 nHz
```
and compute the radial operators as
```julia
operators = radial_operators(nr, nℓ; r_in_frac, r_out_frac, ν, trackingrate)
```
Some of the common keyword arguments are demonstrated here.
There are various other keyword arguments which may be passed to specify domain and model details.
See [`radial_operators`](@ref) for details.

The second step is to obtain a function that will be used to compute the spectrum.
We need to specify whether we seek equatorially symmetric or antisymmetric solutions,
and the model of differential rotation that is used. We define some parameters
```julia
V_symmetric = true # indicates that V is symmetric about the equator, alternately set to `false` for antisymmetric
diffrot = true # indicates that differential rotation is included in computing the spectrum
rotation_profile = :latrad_squished # model of differential rotation to be used in the calculation, only needed if diffrot == true
```
and compute the spectrum function as
```julia
spectrumfn! = RotMatrix(Val(:spectrum), V_symmetric, diffrot, rotation_profile; operators)
```

Finally, we compute and save the spectrum for a range of azimuthal orders using the pre-defined parameters.
We define
```julia
mrange = 1:15 # the range of azimuthal orders for which to compute the spectra
```
and compute the spctrum using
```julia
save_eigenvalues(spectrumfn!, mrange; operators)
```

# Loading the results

The results are stored in `jld2` files that are read in using the package `JLD2.jl`, but at present these may not be loaded
independently without loading this package. We provide an interface to read these in:
```julia
```

!!! warn
	The file format and contents may not be identical across releases, and backward compatibility is not always guaranteed.
	Please use the same version of the package that was used to create the file to read it back in.

# Plotting the results

The plotting functions are provided by the module `RossbyPlots`.
For example, we may plot the spectrum as
```julia

```
