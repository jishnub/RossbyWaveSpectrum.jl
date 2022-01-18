module RossbyWaveSpectrum

using Dierckx
using FastGaussQuadrature
using FastTransforms
using FillArrays
using Folds
using ForwardDiff
using LinearAlgebra
using LinearAlgebra: BLAS
using JLD2
using MKL
using OffsetArrays
using SimpleDelimitedFiles: readdlm
using SphericalHarmonics
using WignerSymbols

export datadir

const SCRATCH = Ref("")
const DATADIR = Ref("")

# cgs units
const G = 6.6743e-8
const Msun = 1.989e+33
const Rsun = 6.959894677e+10
const Ω0 = 2pi * 453e-9

function __init__()
    SCRATCH[] = get(ENV, "SCRATCH", homedir())
    DATADIR[] = get(ENV, "DATADIR", joinpath(SCRATCH[], "RossbyWaves"))
end

datadir(f) = joinpath(DATADIR[], f)

# Legedre expansion constants
α⁺ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((ℓ - m + 1) * (ℓ + m + 1) / ((2ℓ + 1) * (2ℓ + 3)))))
α⁻ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((ℓ - m) * (ℓ + m) / ((2ℓ - 1) * (2ℓ + 1)))))

β⁻ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((2ℓ + 1) / (2ℓ - 1) * (ℓ^2 - m^2))))
γ⁺ℓm(ℓ, m) = ℓ * α⁺ℓm(ℓ, m)
γ⁻ℓm(ℓ, m) = ℓ * α⁻ℓm(ℓ, m) - β⁻ℓm(ℓ, m)

# triple integrals involving normalized legendre and associated legendre polynomials
function intPl1mPl20Pl3m(ℓ1, ℓ2, ℓ3, m)
    iseven(ℓ1 + ℓ2 + ℓ3) || return 0.0
    return oftype(0.0, √((2ℓ1 + 1) * (2ℓ2 + 1) / (2 * (2ℓ3 + 1))) *
                       clebschgordan(Float64, ℓ1, 0, ℓ2, 0, ℓ3) * clebschgordan(Float64, ℓ1, m, ℓ2, 0, ℓ3))
end

function intPl1mP′l20Pl3m(ℓ1, ℓ2, ℓ3, m)
    ℓ2 == 0 && return 0.0
    isodd(ℓ1 + ℓ2 + ℓ3) || return 0.0
    return oftype(0.0, sum(√((2ℓ2 + 1) * (2n + 1)) * intPl1mPl20Pl3m(ℓ1, n, ℓ3, m) for n = ℓ2-1:-2:0))
end

function chebyshevnodes(n, a = -1, b = 1)
    nodes = cos.(reverse(pi * ((1:n) .- 0.5) ./ n))
    nodes_scaled = nodes * (b - a) / 2 .+ (b + a) / 2
    nodes, nodes_scaled
end

function legendretransform!(v::AbstractVector, PC = plan_chebyshevtransform!(v), PC2L = plan_cheb2leg(v))
    v2 = PC * v
    v2 .= PC2L * v2
    return v2
end

function legendretransform2(A::AbstractMatrix)
    At = permutedims(A)
    for j in axes(At, 2)
        v = @view At[:, j]
        legendretransform!(v)
    end
    return permutedims(At)
end

function normalizelegendre!(v)
    vo = OffsetArray(v, OffsetArrays.Origin(0))
    for l in eachindex(vo)
        vo[l] *= √(2 / (2l + 1))
    end
    return v
end

function normalizedlegendretransform!(v::AbstractVector, PC = plan_chebyshevtransform!(v), PC2L = plan_cheb2leg(v))
    v2 = legendretransform!(v, PC, PC2L)
    normalizelegendre!(v2)
    return v2
end

function normalizedlegendretransform2(A::AbstractMatrix)
    At = permutedims(A)
    for j in axes(At, 2)
        v = @view At[:, j]
        normalizedlegendretransform!(v)
    end
    return permutedims(At)
end

function chebyshev_normalizedlegendre_transform!(A::AbstractMatrix)
    for j in axes(A, 2)
        chebyshevtransform!((@view A[:, j]))
    end
    A = permutedims(A)
    for j in axes(A, 2)
        normalizedlegendretransform!((@view A[:, j]))
    end
    return permutedims(A)
end

function gausslegendre_theta_grid(ntheta)
    costheta, w = reverse.(gausslegendre(ntheta))
    thetaGL = acos.(costheta)
    (; thetaGL, costheta, w)
end

ntheta_ℓmax(nℓ, m) = 2(nℓ + m - 1)

"""
    associatedlegendretransform_matrices(nθ)

Return matrices `(PLMfwd, PLMinv)`, where `PLMfwd` multiplies a function `f(θ)` to return `flm`,
the coefficients of an associated Legendre polynomial expansion of `f(θ)`, and `PLMinv` performs
the inverse transform and multiplies `flm` to return `f(θ)`.

The matrices satisfy `Mfwd * Minv ≈ I` and `(PLMinv * PLMfwd) * PLMinv ≈ PLMinv`. Note that
`(PLMinv * PLMfwd)` is not identity, it's only the identity in the restricted subspace spanned by
`Plm(cosθ)` for a particular `m`.
"""
function associatedlegendretransform_matrices(nℓ, m, costheta, w)
    nθ = length(costheta)
    nℓ_transform = 2nℓ # including pad
    PLMfwd = zeros(nℓ_transform, nθ)
    PLMinv = zeros(nθ, nℓ_transform)
    ℓs = range(m, length = nℓ_transform)
    ℓmax = maximum(ℓs)
    Pcache = SphericalHarmonics.allocate_p(ℓmax)
    for (θind, costhetai) in enumerate(costheta)
        computePlmx!(Pcache, costhetai, ℓmax, m, norm = SphericalHarmonics.Orthonormal())
        for (ℓind, ℓ) in enumerate(ℓs)
            Pℓm = Pcache[(ℓ, m)]
            PLMfwd[ℓind, θind] = Pℓm
            PLMinv[θind, ℓind] = Pℓm
        end
    end
    PLMfwd .*= w'
    return (; PLMfwd, PLMinv)
end
function associatedlegendretransform_matrices(nℓ, m)
    ntheta = ntheta_ℓmax(nℓ, m)
    (; costheta, w) = gausslegendre_theta_grid(ntheta)
    associatedlegendretransform_matrices(m, nℓ, costheta, w)
end

function theta_operators(nℓ, m)
    ntheta = ntheta_ℓmax(nℓ, m)
    (; thetaGL, costheta, w) = gausslegendre_theta_grid(ntheta)
    (; PLMfwd, PLMinv) = associatedlegendretransform_matrices(nℓ, m, costheta, w)
    (; ntheta, thetaGL, PLMfwd, PLMinv)
end

"""
    chebyshevpoly(n, x)

Evaluate the Chebyshev polynomials of the first kind ``T_n(x)`` for a degree ``n`` and the
argument ``x``.
"""
function chebyshevpoly(n, x)
    Tc = zeros(n, n)
    Tc[:, 1] .= 1
    Tc[:, 2] = x
    @views for k = 3:n
        @. Tc[:, k] = 2x * Tc[:, k-1] - Tc[:, k-2]
    end
    Tc
end

"""
    chebyshevderiv(n)

Evaluate the matrix that relates the Chebyshev coefficients of the derivative of a function
``df(x)/dx`` to those of the function ``f(x)``. This assumes that ``-1 <= x <= 1``.
"""
function chebyshevderiv(n)
    M = zeros(n, n)
    for col = 1:n, row = col+1:2:n
        v = (row - 1) * (2 - (col == 1))
        M[row, col] = v
    end
    UpperTriangular(transpose(M))
end

function chebyshev_integrate(y)
    n = length(y)
    sum(yi * sin((2i - 1) / 2n * pi) * pi / n for (i, yi) in enumerate(y))
end

function constraintmatrix(operators)
    (; radial_params) = operators
    (; nr, r_in, r_out, nℓ, Δr) = radial_params

    nparams = nr * nℓ
    nradconstraints = 2
    nconstraints = nradconstraints * nℓ

    # Radial constraint
    M = zeros(2nradconstraints, nr)
    MVn = @view M[1:nradconstraints, :]
    MVno = OffsetArray(MVn, :, 0:nr-1)
    MWn = @view M[nradconstraints.+(1:nradconstraints), :]
    MWno = OffsetArray(MWn, :, 0:nr-1)
    MSn = MWn

    nfields = 3
    BC = zeros(nfields * 2nℓ, nfields * nparams)

    # constraints on V, Robin
    for n = 0:nr-1
        MVno[1, n] = (-1)^n * (n^2 + Δr / r_in)
        MVno[2, n] = n^2 - Δr / r_out
    end

    # constraints on W, Dirichlet
    for n = 0:nr-1
        MWno[1, n] = (-1)^n
        MWno[2, n] = 1
    end

    indstart = 1
    indend = nr
    for ℓind = 1:nℓ
        BC[nradconstraints*(ℓind-1).+axes(MVn, 1), indstart:indend] = MVn
        BC[nradconstraints*(ℓind-1).+axes(MWn, 1).+nconstraints, nparams.+(indstart:indend)] = MWn
        if nfields == 3
            BC[nradconstraints*(ℓind-1).+axes(MSn, 1).+2nconstraints, 2nparams.+(indstart:indend)] = MSn
        end
        indstart += nr
        indend += nr
    end

    ZC = nullspace(BC)
    ZW = nullspace(MWn)

    (; BC, ZC, ZW)
end

function parameters(nr, nℓ; r_in = 0.7Rsun, r_out = 0.98Rsun)
    nchebyr = nr
    r_mid = (r_in + r_out) / 2
    Δr = r_out - r_in
    nparams = nchebyr * nℓ
    return (; nchebyr, r_in, r_out, Δr, nr, nparams, nℓ, r_mid)
end

function superadiabaticity(r; r_out = Rsun)
    δcz = 3e-6
    δtop = 3e-5
    dtrans = dtop = 0.0125Rsun
    r_sub = 0.8 * Rsun
    r_tran = 0.725 * Rsun
    δrad = -1e-5
    δconv = δtop * exp((r - r_out) / dtop) + δcz * (r - r_sub) / (r_out - r_sub)
    δconv + (δrad - δconv) * 1 / 2 * (1 - tanh((r - r_tran) / dtrans))
end

function thermal_diffusivity(ρ)
    κtop = 3e13
    κ = @. (1 / √ρ)
    κ ./ maximum(κ) .* κtop
end

function chebyshev_forward_inverse(n, boundaries...)
    nodes, r = chebyshevnodes(n, boundaries...)
    Tc = chebyshevpoly(n, nodes)
    Tcfwd = Tc' * 2 / n
    Tcfwd[1, :] ./= 2
    Tcinv = Tc
    r, Tcfwd, Tcinv
end

function sph_points(N)
    @assert N > 0
    M = 2 * N - 1
    return π / N * (0.5:(N-0.5)), 2π / M * (0:(M-1))
end

function spherical_harmonic_transform_plan(ntheta)
    # spherical harmonic transform parameters
    lmax = ntheta - 1
    θϕ_sh = sph_points(lmax)
    shtemp = zeros(length.(θϕ_sh))
    PSPH2F = plan_sph2fourier(shtemp)
    PSPHA = plan_sph_analysis(shtemp)
    (; shtemp, PSPHA, PSPH2F, θϕ_sh, lmax)
end

function costheta_operator(nℓ, m)
    dl = [α⁻ℓm(ℓ, m) for ℓ in m .+ (1:nℓ-1)]
    d = zeros(nℓ)
    SymTridiagonal(d, dl)'
end

function sintheta_dtheta_operator(nℓ, m)
    dl = [γ⁻ℓm(ℓ, m) for ℓ in m .+ (1:nℓ-1)]
    d = zeros(nℓ)
    du = [γ⁺ℓm(ℓ, m) for ℓ in m .+ (0:nℓ-2)]
    Tridiagonal(dl, d, du)'
end

function greenfn_realspace_numerical2(ℓ, operators)
    (; diff_operators, rad_terms, transforms) = operators
    (; DDr, ddr) = diff_operators
    (; onebyr2_cheby) = rad_terms
    (; Tcrinvc, Tcrfwdc) = transforms
    Bℓ = ddr * DDr - ℓ * (ℓ + 1) * onebyr2_cheby
    Bℓ_realspace = Tcrinvc * Bℓ * Tcrfwdc
    G = pinv(Bℓ_realspace[2:end-1, 2:end-1])
    A = zero(Bℓ_realspace)
    A[2:end-1, 2:end-1] .= G
    return A
end
function greenfn_cheby_numerical2(ℓ, operators)
    (; transforms) = operators
    (; Tcrinvc, Tcrfwdc) = transforms
    G = greenfn_realspace_numerical2(ℓ, operators)
    Tcrfwdc * G * Tcrinvc
end

function splderiv(r, f)
    s = Spline1D(r, f)
    derivative.((s,), r)
end

function read_solar_model(nr; r_in = 0.7Rsun, r_out = Rsun)
    ModelS = readdlm(joinpath(@__DIR__, "ModelS.detailed"))
    r_modelS = @view ModelS[:, 1]
    r_inds = r_in .<= r_modelS .<= r_out
    r_modelS = reverse(r_modelS[r_inds])
    q_modelS = exp.(reverse(ModelS[r_inds, 2]))
    T_modelS = reverse(ModelS[r_inds, 3])
    ρ_modelS = reverse(ModelS[r_inds, 5])
    # interpolate on the Chebyshev grid
    _, r = chebyshevnodes(nr, r_in, r_out)
    sρ = Spline1D(r_modelS, ρ_modelS)
    ρ = sρ.(r)
    T = Spline1D(r_modelS, T_modelS).(r)
    M = Msun * Spline1D(r_modelS, q_modelS).(r)

    g = @. G * M / r^2
    (; r, ρ, T, g)
end

function radial_operators(nr, nℓ; r_in_frac = 0.7, r_out_frac = 0.98)
    r_in = r_in_frac * Rsun
    r_out = r_out_frac * Rsun
    radial_params = parameters(nr, nℓ; r_in, r_out)
    (; Δr, nchebyr) = radial_params
    r, Tcrfwd, Tcrinv = chebyshev_forward_inverse(nr, r_in, r_out)
    Tcrfwdc, Tcrinvc = complex.(Tcrfwd), complex.(Tcrinv)
    r_cheby = Tcrfwd * Diagonal(r) * Tcrinv

    (; ρ, T, g) = read_solar_model(nr; r_in, r_out)

    ddr = Matrix(chebyshevderiv(nr) * (2 / Δr))
    d2dr2 = ddr^2
    ddr_realspace = Tcrinv * ddr * Tcrfwd
    ηρ = ddr_realspace * log.(ρ)
    ηρ_cheby = Tcrfwd * Diagonal(ηρ) * Tcrinv
    DDr = ddr + ηρ_cheby
    rDDr = r_cheby * DDr
    rddr = r_cheby * ddr
    D2Dr2 = DDr^2

    onebyr = 1 ./ r
    onebyr_cheby = Tcrfwd * Diagonal(onebyr) * Tcrinv
    onebyr2_cheby = Tcrfwd * Diagonal(onebyr)^2 * Tcrinv
    DDr_minus_2byr = DDr - 2 * onebyr_cheby

    g_cheby = Tcrfwd * Diagonal(g) * Tcrinv

    κ = thermal_diffusivity(ρ)
    κ_cheby = all(iszero, κ) ? zeros(size(g_cheby)) : Tcrfwd * Diagonal(κ) * Tcrinv
    ddr_lnκρT = all(iszero, κ) ? zero(κ_cheby) :
                Tcrfwd * Diagonal(ddr_realspace * log.(κ .* ρ .* T)) * Tcrinv

    γ = 1.64
    cp = 1.7e8
    δ_superadiabatic = superadiabaticity.(r; r_out)
    ddr_S0_by_cp = Tcrfwd * Diagonal(@. γ * δ_superadiabatic * ηρ / cp) * Tcrinv

    Ir = I(nchebyr)
    Iℓ = I(nℓ)

    constants = (; κ)
    identities = (; Ir, Iℓ)
    coordinates = (; r)
    transforms = (; Tcrfwd, Tcrinv, Tcrfwdc, Tcrinvc)
    rad_terms = (; onebyr, onebyr_cheby, ηρ, ηρ_cheby, onebyr2_cheby,
        ddr_lnκρT, ddr_S0_by_cp, g_cheby, r_cheby, κ_cheby)
    diff_operators = (; DDr, D2Dr2, DDr_minus_2byr, rDDr, rddr, ddr, d2dr2)

    (;
        constants, rad_terms,
        diff_operators,
        transforms, coordinates,
        radial_params, identities
    )
end

function blockinds((m, nr), ℓ, ℓ′ = ℓ)
    @assert ℓ >= m "ℓ must be >= m"
    @assert ℓ′ >= m "ℓ must be >= m"
    rowtopind = (ℓ - m) * nr + 1
    rowinds = range(rowtopind, length = nr)
    colleftind = (ℓ′ - m) * nr + 1
    colinds = range(colleftind, length = nr)
    CartesianIndices((rowinds, colinds))
end

function uniform_rotation_terms(nr, nℓ, m; r_in_frac = 0.7, r_out_frac = 0.98,
    operators = radial_operators(nr, nℓ; r_in_frac, r_out_frac))

    nparams = nr * nℓ
    M = zeros(ComplexF64, 3nparams, 3nparams)
    uniform_rotation_terms!(M, nr, nℓ, m; r_in_frac, r_out_frac, operators)
    return M
end

function uniform_rotation_terms!(M, nr, nℓ, m; r_in_frac = 0.7, r_out_frac = 0.98,
    operators = radial_operators(nr, nℓ; r_in_frac, r_out_frac))

    (; radial_params, rad_terms, diff_operators) = operators

    (; nchebyr) = radial_params

    (; DDr, DDr_minus_2byr, ddr, d2dr2, rddr) = diff_operators
    (; onebyr_cheby, onebyr2_cheby, ddr_lnκρT, κ_cheby,
        g_cheby, ηρ_cheby, r_cheby, ddr_S0_by_cp) = rad_terms

    Cℓ′ = similar(DDr)
    WVℓℓ′ = similar(Cℓ′)
    VWℓℓ′ = similar(Cℓ′)
    GℓC1 = similar(Cℓ′)
    GℓC1_ddr = similar(GℓC1)
    GℓCℓ′ = similar(Cℓ′)

    nparams = nchebyr * nℓ
    ℓmin = m
    ℓs = range(ℓmin, length = nℓ)

    VV = @view M[1:nparams, 1:nparams]
    WV = @view M[nparams+1:2nparams, 1:nparams]
    VW = @view M[1:nparams, nparams+1:2nparams]
    WW = @view M[nparams+1:2nparams, nparams+1:2nparams]
    SW = @view M[2nparams+1:end, nparams+1:2nparams]
    WS = @view M[nparams+1:2nparams, 2nparams+1:end]
    SS = @view M[2nparams+1:end, 2nparams+1:end]

    VV .= 2m * kron(Diagonal(@. 1 / (ℓs * (ℓs + 1))), I(nchebyr))

    C1 = DDr_minus_2byr
    C1_ddr = ddr - 2 * onebyr_cheby

    cosθ = OffsetArray(costheta_operator(nℓ, m), ℓs, ℓs)
    sinθdθ = OffsetArray(sintheta_dtheta_operator(nℓ, m), ℓs, ℓs)

    ε = (1e-6)^4

    Wscaling = Rsun

    ε_by_W = ε / Wscaling

    ddr_DDr = ddr * DDr
    onebyr2_Iplusrηρ = (I + ηρ_cheby * r_cheby) * onebyr2_cheby

    onebyr2_cheby_ddr_S0_by_cp = onebyr2_cheby * ddr_S0_by_cp
    d2dr2_plus_ddr_lnκρT_ddr = d2dr2 + ddr_lnκρT * ddr
    κ_d2dr2_plus_ddr_lnκρT_ddr = κ_cheby * d2dr2_plus_ddr_lnκρT_ddr
    κ_by_r2 = κ_cheby * onebyr2_cheby

    for ℓ in ℓs
        # numerical green function
        Gℓ = greenfn_cheby_numerical2(ℓ, operators)
        mul!(GℓC1, Gℓ, C1)
        mul!(GℓC1_ddr, Gℓ, C1_ddr)

        diaginds_ℓ = blockinds((m, nr), ℓ)
        WW[diaginds_ℓ] = 2m / (ℓ * (ℓ + 1)) * Gℓ * (ddr_DDr - onebyr2_Iplusrηρ * ℓ * (ℓ + 1))
        WS[diaginds_ℓ] = ε_by_W * (-1 / Ω0) * Gℓ * g_cheby * (2 / (ℓ * (ℓ + 1)) * rddr + I)

        @. SW[diaginds_ℓ] = (1 / ε_by_W) * (1 / Ω0) * ℓ * (ℓ + 1) * onebyr2_cheby_ddr_S0_by_cp
        @. SS[diaginds_ℓ] = -im / Ω0 * (κ_d2dr2_plus_ddr_lnκρT_ddr - ℓ * (ℓ + 1) * κ_by_r2)

        for ℓ′ = ℓ-1:2:ℓ+1
            ℓ′ in ℓs || continue
            @. Cℓ′ = DDr - ℓ′ * (ℓ′ + 1) * onebyr_cheby
            @. VWℓℓ′ = -2 / (ℓ * (ℓ + 1)) * (ℓ′ * (ℓ′ + 1) * C1 * cosθ[ℓ, ℓ′] + Cℓ′ * sinθdθ[ℓ, ℓ′]) * Wscaling

            @. Cℓ′ = ddr - ℓ′ * (ℓ′ + 1) * onebyr_cheby
            mul!(GℓCℓ′, Gℓ, Cℓ′)
            @. WVℓℓ′ = -2 / (ℓ * (ℓ + 1)) * (ℓ′ * (ℓ′ + 1) * GℓC1_ddr * cosθ[ℓ, ℓ′] + GℓCℓ′ * sinθdθ[ℓ, ℓ′]) / Wscaling

            inds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)

            VW[inds_ℓℓ′] = VWℓℓ′
            WV[inds_ℓℓ′] = WVℓℓ′
        end
    end

    viscosity_terms!(M, nr, nℓ, m; operators)

    return M
end

function viscosity_terms!(M, nr, nℓ, m; r_in_frac = 0.7, r_out_frac = 0.98,
    operators = radial_operators(nr, nℓ; r_in_frac, r_out_frac))

    (; rad_terms, diff_operators, radial_params) = operators
    (; nchebyr) = radial_params

    (; DDr, ddr, d2dr2) = diff_operators
    (; onebyr_cheby, onebyr2_cheby, ηρ_cheby, r_cheby) = rad_terms

    Ω = 2pi * 453e-9
    ν = 1e10
    ν = ν / Ω
    nparams = nchebyr * nℓ
    VV = @view M[1:nparams, 1:nparams]
    WW = @view M[nparams+1:2nparams, nparams+1:2nparams]

    ℓmin = m
    ℓs = range(ℓmin, length = nℓ)

    r2 = r_cheby^2
    ∇r2 = onebyr2_cheby * ddr * r2 * ddr
    two_by_r = 2 * onebyr_cheby
    ddr_plus_2byr = ddr + two_by_r
    ddr_minus_2byr = ddr - two_by_r
    ηρ_ddr_minus_2byr = ηρ_cheby * ddr_minus_2byr
    d2dr2_plus_2byr_ηρ = d2dr2 + 2 * onebyr_cheby * ηρ_cheby

    ℓℓp1_by_r2 = similar(onebyr2_cheby)


    for ℓ in ℓs
        ABℓ′top = (ℓ - minimum(m)) * nchebyr + 1
        ACℓ′vertinds = range(ABℓ′top, length = nr)
        diaginds_ℓ = CartesianIndices((ACℓ′vertinds, ACℓ′vertinds))

        @. ℓℓp1_by_r2 = ℓ * (ℓ + 1) * onebyr2_cheby

        @. VV[diaginds_ℓ] += -im * ν * (d2dr2 - ℓℓp1_by_r2 + ηρ_ddr_minus_2byr)

        WWop = ddr_plus_2byr * (∇r2 - ℓℓp1_by_r2) * ηρ_cheby +
               (d2dr2 - ℓℓp1_by_r2) * (d2dr2_plus_2byr_ηρ - ℓℓp1_by_r2) -
               2 / 3 * ℓℓp1_by_r2 * ηρ_cheby^2 -
               ddr_minus_2byr * (ddr * ηρ_cheby * ddr_minus_2byr +
                                 ηρ_cheby * (onebyr_cheby * DDr - ℓℓp1_by_r2)) -
               onebyr2_cheby * (ddr * ddr_minus_2byr * ηρ_cheby * ddr_minus_2byr
                                -
                                ηρ_cheby * ℓ * (ℓ + 1) * ddr_minus_2byr
                                -
                                2ηρ_cheby * (ddr + DDr - (ℓ * (ℓ + 1) + 2) * onebyr_cheby)
        )

        Gℓ = greenfn_cheby_numerical2(ℓ, operators)
        WWop = Gℓ * WWop

        @. WW[diaginds_ℓ] += -im * ν * WWop
    end

    return M
end

function interp1d(xin, z, xout; s = 0.0)
    spline = Spline1D(xin, z; s)
    spline(xout)
end

function interp2d(xin, yin, z, xout, yout; s = 0.0)
    spline = Spline2D(xin, yin, z; s)
    evalgrid(spline, xout, yout)
end

function read_angular_velocity(operators, thetaGL)
    (; coordinates) = operators
    (; r) = coordinates

    parentdir = dirname(@__DIR__)
    fname = joinpath(parentdir, "rmesh.orig")
    r_ΔΩ_raw = vec(readdlm(fname))
    r_ΔΩ_raw = r_ΔΩ_raw[1:4:end]

    Ω_raw = 2pi * 1e-9 * readdlm(joinpath(parentdir, "rot2d.hmiv72d.ave"), ' ')
    Ω_raw = [Ω_raw reverse(Ω_raw[:, 2:end], dims = 2)]
    ΔΩ_raw = Ω_raw .- Ω0

    nθ = size(ΔΩ_raw, 2)
    lats_raw = LinRange(0, pi, nθ)

    ΔΩ_r_thetaGL = permutedims(interp2d(lats_raw, r_ΔΩ_raw, ΔΩ_raw', thetaGL, r ./ Rsun))

    (; lats_raw, ΔΩ_r_thetaGL)
end

function legendre_to_associatedlegendre(vℓ, ℓ1, ℓ2, m)
    vℓo = OffsetArray(vℓ, OffsetArrays.Origin(0))
    sum(vℓi * intPl1mPl20Pl3m(ℓ1, ℓ, ℓ2, m) for (ℓ, vℓi) in pairs(vℓo))
end

function dlegendre_to_associatedlegendre(vℓ, ℓ1, ℓ2, m)
    vℓo = OffsetArray(vℓ, OffsetArrays.Origin(0))
    sum(vℓi * intPl1mP′l20Pl3m(ℓ1, ℓ, ℓ2, m) for (ℓ, vℓi) in pairs(vℓo))
end

function laplacian_operator(nℓ, m)
    ℓs = range(m, length = nℓ)
    Diagonal(@. -ℓs * (ℓs + 1))
end

function constant_differential_rotation_terms!(M, nr, nℓ, m; operators = radial_operators(nr, nℓ))
    nparams = nr * nℓ
    (; diff_operators, rad_terms) = operators
    (; ddr, DDr) = diff_operators
    (; ηρ_cheby, onebyr_cheby, onebyr2_cheby) = rad_terms

    VV = @view M[1:nparams, 1:nparams]
    VW = @view M[1:nparams, nparams.+(1:nparams)]
    WV = @view M[nparams.+(1:nparams), 1:nparams]
    WW = @view M[nparams.+(1:nparams), nparams.+(1:nparams)]

    ΔΩ_by_Ω0 = 0.01

    ℓs = range(m, length = nℓ)

    cosθ = OffsetArray(costheta_operator(nℓ, m), ℓs, ℓs)
    sinθdθ = OffsetArray(sintheta_dtheta_operator(nℓ, m), ℓs, ℓs)
    laplacian_sinθdθ = OffsetArray(Diagonal(@. -ℓs * (ℓs + 1)) * parent(sinθdθ), ℓs, ℓs)

    Wscaling = Rsun

    for ℓ in ℓs
        # numerical green function
        Gℓ = greenfn_cheby_numerical2(ℓ, operators)
        ℓℓp1 = ℓ * (ℓ + 1)
        diaginds_ℓ = blockinds((m, nr), ℓ)
        two_over_ℓℓp1 = 2 / ℓℓp1
        VV[diaginds_ℓ] += m * (two_over_ℓℓp1 - 1) * ΔΩ_by_Ω0 * I

        WW[diaginds_ℓ] += m * ΔΩ_by_Ω0 * Gℓ *
                          (-2 * onebyr_cheby * ηρ_cheby +
                           (two_over_ℓℓp1 - 1) * (ddr * DDr - ℓℓp1 * onebyr2_cheby)
                          )

        for ℓ′ in ℓ-1:2:ℓ+1
            ℓ′ in ℓs || continue
            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)
            inds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)
            @. VW[inds_ℓℓ′] -= Wscaling * ΔΩ_by_Ω0 * two_over_ℓℓp1 * (
                                   ℓ′ℓ′p1 * cosθ[ℓ, ℓ′] * (DDr - 2 * onebyr_cheby) +
                                   (DDr - ℓ′ℓ′p1 * onebyr_cheby) * sinθdθ[ℓ, ℓ′]
                               )

            WV[inds_ℓℓ′] -= 1 / Wscaling * ΔΩ_by_Ω0 / ℓℓp1 * Gℓ *
                            (4ℓ′ℓ′p1 * cosθ[ℓ, ℓ′] * ddr + (ℓ′ℓ′p1 + 2) * sinθdθ[ℓ, ℓ′] * ddr
                             + (ddr + 2 * onebyr_cheby) * laplacian_sinθdθ[ℓ, ℓ′])
        end
    end
    return M
end
function radial_differential_rotation_terms!(M, nr, nℓ, m; operators = radial_operators(nr, nℓ))
    nparams = nr * nℓ
    (; transforms, diff_operators, rad_terms, identities) = operators
    (; Iℓ, Ir) = identities
    (; ddr, DDr, d2dr2) = diff_operators
    (; Tcrfwd, Tcrinv) = transforms
    (; g_cheby, onebyr_cheby, onebyr2_cheby) = rad_terms

    VV = @view M[1:nparams, 1:nparams]
    VW = @view M[1:nparams, nparams.+(1:nparams)]
    WV = @view M[nparams.+(1:nparams), 1:nparams]
    WW = @view M[nparams.+(1:nparams), nparams.+(1:nparams)]
    SV = @view M[2nparams.+(1:nparams), 1:nparams]
    SW = @view M[2nparams.+(1:nparams), nparams.+(1:nparams)]

    (; thetaGL) = gausslegendre_theta_grid(nℓ)
    nθ = length(thetaGL)
    ΔΩ_rθ = read_angular_velocity(operators, thetaGL).ΔΩ_r_thetaGL
    ΔΩ_r = ΔΩ_rθ[:, div(nθ, 2)]
    ΔΩ = Tcrfwd * Diagonal(ΔΩ_r) * Tcrinv
    ddrΔΩ = Tcrfwd * Diagonal(Tcrinv * ddr * Tcrfwd * ΔΩ_r) * Tcrinv
    d2dr2ΔΩ = Tcrfwd * Diagonal(Tcrinv * d2dr2 * Tcrfwd * ΔΩ_r) * Tcrinv
    ddrΔΩ_over_g = ddrΔΩ * g_cheby^-1
    ddrΔΩ_over_g_DDr = ddrΔΩ_over_g * DDr
    ddr_plus_2byr = ddr + 2 * onebyr_cheby
    ddr_plus_2byr_ΔΩ = Tcrfwd * Diagonal(Tcrinv * ddr_plus_2byr * Tcrfwd * ΔΩ_r) * Tcrinv

    ℓs = range(m, length = nℓ)

    cosθ = costheta_operator(nℓ, m)
    cosθo = OffsetArray(cosθ, ℓs, ℓs)
    sinθdθ = sintheta_dtheta_operator(nℓ, m)
    sinθdθo = OffsetArray(sinθdθ, ℓs, ℓs)
    cosθsinθdθ = parent(cosθ) * parent(sinθdθ)
    cosθsinθdθo = OffsetArray(cosθsinθdθ, ℓs, ℓs)

    ωΩr = kron(2cosθ, ΔΩ)
    ωΩθ_by_rsinθ = kron(-Iℓ, ddrΔΩ + 2onebyr_cheby * ΔΩ)
    curlωΩϕ_by_rsinθ = kron(-Iℓ, d2dr2ΔΩ + 4onebyr_cheby * ddrΔΩ)

    ℓℓp1d = Diagonal(@. ℓs * (ℓs + 1))
    ωfr_V = kron(ℓℓp1d, onebyr2_cheby)
    ωfr_W = 0
    ωfθ_rsinθ_V = kron(sinθdθ, ddr)
    ωfθ_rsinθ_W = kron(m * ℓℓp1d, onebyr2_cheby) - kron(m * Iℓ, ddr * DDr)
    ufϕ_rsinθ_V = kron(sinθdθ, -Ir)
    ufϕ_rsinθ_W = kron(m * Iℓ, DDr)
    curlωfϕ_rsinθ_V = kron(sinθdθ, d2dr2) - kron(sinθdθ * ℓℓp1d, onebyr2_cheby)
    curlωfϕ_rsinθ_W = -m * (kron(Iℓ, d2dr2 * DDr) - kron(ℓℓp1d, ddr * onebyr2_cheby))

    uf_cross_ωΩ_plus_uΩ_cross_ωf_r_V = -kron(sinθdθ, ddr_plus_2byr_ΔΩ - ΔΩ * ddr)
    uf_cross_ωΩ_plus_uΩ_cross_ωf_r_W = m * (kron(Iℓ, ddr_plus_2byr_ΔΩ * DDr + ΔΩ * ddr * DDr) -
                                            kron(ℓℓp1d, ΔΩ * onebyr2_cheby))

    minusinvr2_∇²h_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_V =
        kron(-ℓℓp1d * sinθdθ, onebyr2_cheby * (ΔΩ * ddr + ddr_plus_2byr_ΔΩ))
    minusinvr2_∇²h_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_W =
        m * (kron(ℓℓp1d, -onebyr2_cheby * (-ddr_plus_2byr_ΔΩ * DDr - ΔΩ * ddr * DDr)) +
             kron(ℓℓp1d^2, -onebyr2_cheby * (ΔΩ * onebyr2_cheby)))

    div_uf_cross_ωΩ_plus_uΩ_cross_ωf_V = 2(ωΩr * ωfr_V + ωΩθ_by_rsinθ * ωfθ_rsinθ_V) -
                                         curlωΩϕ_by_rsinθ * ufϕ_rsinθ_V - kron(Iℓ, ΔΩ) * curlωfϕ_rsinθ_V
    div_uf_cross_ωΩ_plus_uΩ_cross_ωf_W = 2(ωΩr * ωfr_W + ωΩθ_by_rsinθ * ωfθ_rsinθ_W) -
                                         curlωΩϕ_by_rsinθ * ufϕ_rsinθ_W - kron(Iℓ, ΔΩ) * curlωfϕ_rsinθ_W

    ε = (1e-6)^4

    Wscaling = Rsun

    WV_rhs = (1 / Wscaling) * (1 / Ω0) *
             (kron(Iℓ, ddr_plus_2byr) * div_uf_cross_ωΩ_plus_uΩ_cross_ωf_V -
              kron(Iℓ, d2dr2 + 4onebyr_cheby * ddr + 2onebyr2_cheby) * uf_cross_ωΩ_plus_uΩ_cross_ωf_r_V +
              minusinvr2_∇²h_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_V)

    WW_rhs = (1 / Ω0) * m *
             (kron(Iℓ, ddr_plus_2byr) * div_uf_cross_ωΩ_plus_uΩ_cross_ωf_W -
              kron(Iℓ, d2dr2 + 4onebyr_cheby * ddr + 2onebyr2_cheby) * uf_cross_ωΩ_plus_uΩ_cross_ωf_r_W +
              minusinvr2_∇²h_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_W)

    for ℓ in ℓs
        # numerical green function
        Gℓ = greenfn_cheby_numerical2(ℓ, operators)
        ℓℓp1 = ℓ * (ℓ + 1)
        ℓℓp1_by_r2 = ℓℓp1 * onebyr2_cheby
        neg_r2_by_ℓℓp1 = -ℓℓp1_by_r2^-1
        diaginds_ℓ = blockinds((m, nr), ℓ)
        two_over_ℓℓp1 = 2 / ℓℓp1

        @. VV[diaginds_ℓ] += (1 / Ω0) * m * (two_over_ℓℓp1 - 1) * ΔΩ
        WW[diaginds_ℓ] .-= Gℓ * neg_r2_by_ℓℓp1 * WW_rhs[diaginds_ℓ]

        for ℓ′ in ℓ-1:2:ℓ+1
            ℓ′ in ℓs || continue
            inds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)
            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)
            VW[inds_ℓℓ′] -= (1 / Ω0) * Wscaling * two_over_ℓℓp1 *
                            (ℓ′ℓ′p1 * cosθo[ℓ, ℓ′] * ΔΩ * (DDr - 2onebyr_cheby) -
                             sinθdθo[ℓ, ℓ′] * (ΔΩ * (DDr - ℓ′ℓ′p1 * onebyr_cheby) - ℓ′ℓ′p1 / 2 * ddrΔΩ))

            @. SV[inds_ℓℓ′] -= (1 / ε) * cosθo[ℓ, ℓ′] * 2m * ddrΔΩ_over_g
            @. SW[inds_ℓℓ′] += (Wscaling / ε) * cosθsinθdθo[ℓ, ℓ′] * ddrΔΩ_over_g_DDr

            WV[inds_ℓℓ′] .-= Gℓ * neg_r2_by_ℓℓp1 * WV_rhs[inds_ℓℓ′]
        end
    end
    return M
end
function solar_differential_rotation_terms!(M, nr, nℓ, m; operators = radial_operators(nr, nℓ)) end

function differential_rotation_terms(nr, nℓ, m; rotation_profile, operators = radial_operators(nr, nℓ))
    M = uniform_rotation_terms(nr, nℓ, m; operators)
    if rotation_profile === :radial
        radial_differential_rotation_terms!(M, nr, nℓ, m; operators)
    elseif rotation_profile === :solar
        solar_differential_rotation_terms!(M, nr, nℓ, m; operators)
    elseif rotation_profile === :constant
        constant_differential_rotation_terms!(M, nr, nℓ, m; operators)
    else
        throw(ArgumentError("Invalid rotation profile"))
    end
    return M
end
function differential_rotation_terms!(M, nr, nℓ, m; rotation_profile, operators = radial_operators(nr, nℓ))
    M = uniform_rotation_terms!(M, nr, nℓ, m; operators)
    if rotation_profile === :radial
        radial_differential_rotation_terms!(M, nr, nℓ, m; operators)
    elseif rotation_profile === :solar
        solar_differential_rotation_terms!(M, nr, nℓ, m; operators)
    elseif rotation_profile === :constant
        constant_differential_rotation_terms!(M, nr, nℓ, m; operators)
    else
        throw(ArgumentError("Invalid rotation profile"))
    end
    return M
end

function constrained_eigensystem(M, operators)
    (; ZC) = constraintmatrix(operators)
    M_constrained = permutedims(ZC) * M * ZC
    λ::Vector{ComplexF64}, w::Matrix{ComplexF64} = eigen!(M_constrained)
    v = ZC * w
    λ, v, M
end

function uniform_rotation_spectrum(nr, nℓ, m;
    operators = radial_operators(nr, nℓ), kw...)

    M = uniform_rotation_terms(nr, nℓ, m; operators)
    constrained_eigensystem(M, operators)
end

function real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop)
    (; PLMfwd, PLMinv) = thetaop
    (; transforms, radial_params) = operators
    (; nℓ, nchebyr) = radial_params
    (; Tcrfwd, Tcrinv) = transforms
    pad = nchebyr * nℓ
    PaddedMatrix((PLMfwd ⊗ Tcrfwd) * Diagonal(vec(ΔΩ_r_thetaGL)) * (PLMinv ⊗ Tcrinv), pad)
end

function real_to_r_assocleg(ΔΩ_r_thetaGL, operators, thetaop)
    (; PLMfwd, PLMinv) = thetaop
    (; radial_params) = operators
    (; nℓ, nchebyr) = radial_params
    Ir = I(nchebyr)
    pad = nchebyr * nℓ
    PaddedMatrix((PLMfwd ⊗ Ir) * Diagonal(vec(ΔΩ_r_thetaGL)) * (PLMinv ⊗ Ir), pad)
end

function differential_rotation_spectrum(nr, nℓ, m;
    rotation_profile,
    operators = radial_operators(nr, nℓ),
    kw...)

    M = differential_rotation_terms(nr, nℓ, m; operators, rotation_profile)
    constrained_eigensystem(M, operators)
end

rossby_ridge(m, ℓ = m; ΔΩ_by_Ω = 0) = 2m / (ℓ * (ℓ + 1)) * (1 + ΔΩ_by_Ω) - m * ΔΩ_by_Ω

function eigenvalue_filter(x, m;
    eig_imag_unstable_cutoff = -1e-1,
    eig_imag_to_real_ratio_cutoff = 1e-2,
    ΔΩ_by_Ω = 0)

    x_real_shifted = (real(x) + m * ΔΩ_by_Ω) / (1 + ΔΩ_by_Ω)
    eig_imag_unstable_cutoff <= imag(x) < x_real_shifted * eig_imag_to_real_ratio_cutoff &&
        0 < x_real_shifted < 2rossby_ridge(m)
end
function boundary_condition_filter(v, C, BCVcache, atol = 1e-5)
    mul!(BCVcache, C, v)
    norm(BCVcache) < atol
end
function isapprox2(x, y; rtol)
    N = max(norm(x), norm(y))
    Ndiff = norm((xi - yi for (xi, yi) in zip(x, y)))
    Ndiff <= rtol * N
end
function eigensystem_satisfy_filter(λ, v, M, MVcache, rtol = 1e-1)
    mul!(MVcache, M, v, 1 / λ, false)
    isapprox2(MVcache, v; rtol)
end

function chebyshev_filter!(VWinv, F, v, m, operators, n_cutoff = 7, n_power_cutoff = 0.9;
    nℓ = operators.radial_params.nℓ,
    ℓmax = m + nℓ - 1,
    Plcosθ = SphericalHarmonics.allocate_p(ℓmax)
)

    eigenfunction_n_theta!(VWinv, F, v, m, operators; nℓ, ℓmax, Plcosθ)
    (; V) = VWinv

    n_cutoff_ind = 1 + n_cutoff
    # ensure that most of the power at the equator is below the cutoff
    V_n_equator = @view V[:, size(V, 2)÷2]
    P_frac = sum(abs2, @view V_n_equator[1:n_cutoff_ind]) / sum(abs2, V_n_equator)
    P_frac > n_power_cutoff
end

function equator_filter!(VWinv, VWinvsh, F, v, m, operators, θ_cutoff = deg2rad(75), equator_power_cutoff_frac = 0.3;
    nℓ = operators.radial_params.nℓ,
    ℓmax = m + nℓ - 1,
    Plcosθ = SphericalHarmonics.allocate_p(ℓmax)
)

    (; θ) = eigenfunction_realspace!(VWinv, VWinvsh, F, v, m, operators; nℓ, ℓmax, Plcosθ)
    (; V) = VWinv
    V_surf = @view V[end, :]
    θlowind = searchsortedfirst(θ, θ_cutoff)
    θhighind = searchsortedlast(θ, pi - θ_cutoff)
    powfrac = sum(abs2, @view V_surf[θlowind:θhighind]) / sum(abs2, V_surf)
    powflag = powfrac > equator_power_cutoff_frac
    peakflag = maximum(abs2, @view V_surf[θlowind:θhighind]) == maximum(abs2, V_surf)
    powflag & peakflag
end

function filter_eigenvalues(f, nr, nℓ, m::Integer;
    operators = radial_operators(nr, nℓ),
    BC = constraintmatrix(operators).BC, kw...)

    λ::Vector{ComplexF64}, v::Matrix{ComplexF64}, M::Matrix{ComplexF64} =
        f(nr, nℓ, m; operators, kw...)
    filter_eigenvalues(λ, v, M, nr, nℓ, m; operators, BC, kw...)
end

function filterfn(λ, v, m, M, operators, additional_params)

    (; eig_imag_unstable_cutoff, eig_imag_to_real_ratio_cutoff, ΔΩ_by_Ω,
        Δl_cutoff, Δl_power_cutoff, BC, BCVcache, atol_constraint,
        VWinv, n_cutoff, n_power_cutoff, nℓ, ℓmax, Plcosθ,
        θ_cutoff, equator_power_cutoff_frac, MVcache, eigen_rtol, VWinvsh, F) = additional_params

    eigenvalue_filter(λ, m; eig_imag_unstable_cutoff, eig_imag_to_real_ratio_cutoff, ΔΩ_by_Ω) &&
    sphericalharmonic_filter!(VWinvsh, F, v, operators, Δl_cutoff, Δl_power_cutoff) &&
    boundary_condition_filter(v, BC, BCVcache, atol_constraint) &&
    chebyshev_filter!(VWinv, F, v, m, operators, n_cutoff, n_power_cutoff; nℓ, ℓmax, Plcosθ) &&
    equator_filter!(VWinv, VWinvsh, F, v, m, operators, θ_cutoff, equator_power_cutoff_frac; nℓ, ℓmax, Plcosθ) &&
    eigensystem_satisfy_filter(λ, v, M, MVcache, eigen_rtol)
end

function filter_eigenvalues(λ::AbstractVector, v::AbstractMatrix,
    M::AbstractMatrix, nr::Integer, nℓ::Integer, m::Integer;
    atol_constraint = 1e-5,
    Δl_cutoff = 7, Δl_power_cutoff = 0.9,
    eigen_rtol = 5e-2,
    n_cutoff = 7, n_power_cutoff = 0.9,
    operators = radial_operators(nr, nℓ),
    BC = constraintmatrix(operators).BC,
    eig_imag_unstable_cutoff = -1e-1,
    eig_imag_to_real_ratio_cutoff = 1e-2,
    ΔΩ_by_Ω = 0,
    θ_cutoff = deg2rad(75),
    equator_power_cutoff_frac = 0.3,
    kw...)

    # temporary cache arrays
    MVcache = similar(v, size(M, 1))
    BCVcache = similar(v, size(BC, 1))

    nθ = length(spharm_θ_grid_uniform(m, nℓ).θ)
    VWinv = (; V = zeros(ComplexF64, nr, nθ), W = zeros(ComplexF64, nr, nθ))

    VWinvsh = (; V = zeros(ComplexF64, nr, nℓ), W = zeros(ComplexF64, nr, nℓ))

    ℓmax = m + nℓ - 1
    Plcosθ = SphericalHarmonics.allocate_p(ℓmax)

    F = (; V = zeros(ComplexF64, nr * nℓ), W = zeros(ComplexF64, nr * nℓ), S = zeros(ComplexF64, nr * nℓ))

    additional_params = (; eig_imag_unstable_cutoff, eig_imag_to_real_ratio_cutoff, ΔΩ_by_Ω,
        Δl_cutoff, Δl_power_cutoff, BC, BCVcache, atol_constraint,
        VWinv, n_cutoff, n_power_cutoff, nℓ, ℓmax, Plcosθ,
        θ_cutoff, equator_power_cutoff_frac, MVcache, eigen_rtol, VWinvsh, F)

    inds_bool = filterfn.(λ, eachcol(v), m, (M,), (operators,), (additional_params,))
    filtinds = axes(λ, 1)[inds_bool]
    λ[filtinds], v[:, filtinds]
end

macro maybe_reduce_blas_threads(ex)
    ex_esc = esc(ex)
    quote
        n = BLAS.get_num_threads()
        nt = Threads.nthreads()
        BLAS.set_num_threads(max(1, n ÷ nt))
        f = $ex_esc
        BLAS.set_num_threads(n)
        f
    end
end

function filter_eigenvalues(f, nr, nℓ, mr::AbstractVector; kw...)
    operators = radial_operators(nr, nℓ)
    (; BC) = constraintmatrix(operators)
    @maybe_reduce_blas_threads(
        Folds.map(mr) do m
            filter_eigenvalues(f, nr, nℓ, m; operators, BC, kw...)
        end
    )
end
function filter_eigenvalues(λ::AbstractVector{<:AbstractVector},
    v::AbstractVector{<:AbstractMatrix},
    nr::Integer, nℓ::Integer, mr::AbstractVector;
    Mfn = uniform_rotation_terms!,
    operators = radial_operators(nr, nℓ),
    BC = constraintmatrix(operators).BC,
    kw...)

    nparams = nr * nℓ
    Ms = [zeros(ComplexF64, 3nparams, 3nparams) for _ in 1:Threads.nthreads()]
    λv = @maybe_reduce_blas_threads(
        Folds.map(zip(λ, v, mr)) do (λm, vm, m)
            M = Ms[Threads.threadid()]
            Mfn(M, nr, nℓ, m; operators)
            filter_eigenvalues(λm, vm, M, nr, nℓ, m;
                operators, BC, kw...)
        end::Vector{Tuple{Vector{ComplexF64},Matrix{ComplexF64}}}
    )
    first.(λv), last.(λv)
end

function filter_eigenvalues(filename::String; kw...)
    λ, v, mr, nr, nℓ = load(filename, "lam", "vec", "mr", "nr", "nℓ")
    filter_eigenvalues(λ, v, nr, nℓ, mr; kw...)
end

function save_eigenvalues(f, nr, nℓ, mr; kw...)
    λv = filter_eigenvalues(f, nr, nℓ, mr; kw...)
    lam = first.(λv)
    vec = last.(λv)
    filenametag = get(kw, :filenametag, "ur")
    fname = datadir("$(filenametag)_nr$(nr)_nl$(nℓ).jld2")
    jldsave(fname; lam, vec, mr, nr, nℓ)
end

function eigenfunction_cheby_ℓm_spectrum!(F, v, operators)
    (; radial_params) = operators
    (; nparams, nr, nℓ) = radial_params
    F.V .= @view v[1:nparams]
    F.W .= @view v[nparams.+(1:nparams)]
    V = reshape(F.V, nr, nℓ)
    W = reshape(F.W, nr, nℓ)

    # F.S .= @view v[2nparams.+(1:nparams)]
    # S = reshape(F.S, nr, nℓ)
    (; V, W)
end

function eigenfunction_rad_sh!(VWinvsh, F, v, operators)
    VW = eigenfunction_cheby_ℓm_spectrum!(F, v, operators)
    (; transforms) = operators
    (; Tcrinvc) = transforms
    (; V, W) = VW

    Vinv = VWinvsh.V
    Winv = VWinvsh.W

    mul!(Vinv, Tcrinvc, V)
    mul!(Winv, Tcrinvc, W)

    return VWinvsh
end

function spharm_θ_grid_uniform(m, nℓ, ℓmax_mul = 4)
    ℓs = range(m, length = nℓ)
    ℓmax = maximum(ℓs)

    θ, _ = sph_points(ℓmax_mul * ℓmax)
    return (; ℓs, θ)
end

function invshtransform2!(VWinv, VW, m;
    nℓ = size(VW.V, 2),
    ℓmax = m + nℓ - 1,
    Plcosθ = SphericalHarmonics.allocate_p(ℓmax)
)

    V_r_lm = VW.V
    W_r_lm = VW.W

    (; ℓs, θ) = spharm_θ_grid_uniform(m, nℓ)

    V = VWinv.V; V .= 0
    W = VWinv.W; W .= 0

    for (θind, θi) in enumerate(θ)
        computePlmcostheta!(Plcosθ, θi, ℓmax, m, norm = SphericalHarmonics.Orthonormal())
        for (ℓind, ℓ) in enumerate(ℓs)
            Plmcosθ = Plcosθ[(ℓ, m)]
            for r_ind in axes(V, 1)
                V[r_ind, θind] += V_r_lm[r_ind, ℓind] * Plmcosθ
                W[r_ind, θind] += W_r_lm[r_ind, ℓind] * Plmcosθ
            end
        end
    end

    (; VWinv, θ)
end

function eigenfunction_realspace!(VWinv, VWinvsh, F, v, m, operators;
    nℓ = operators.radial_params.nℓ,
    ℓmax = m + nℓ - 1,
    Plcosθ = SphericalHarmonics.allocate_p(ℓmax)
)
    eigenfunction_rad_sh!(VWinvsh, F, v, operators)
    invshtransform2!(VWinv, VWinvsh, m; nℓ, ℓmax, Plcosθ)
end

function eigenfunction_n_theta!(VWinv, F, v, m, operators;
    nℓ = operators.radial_params.nℓ,
    ℓmax = m + nℓ - 1,
    Plcosθ = SphericalHarmonics.allocate_p(ℓmax)
)
    VW = eigenfunction_cheby_ℓm_spectrum!(F, v, operators)
    invshtransform2!(VWinv, VW, m; nℓ, ℓmax, Plcosθ)
end

function sphericalharmonic_filter!(VWinvsh, F, v, operators, Δl_cutoff = 7, power_cutoff = 0.9)
    eigenfunction_rad_sh!(VWinvsh, F, v, operators)
    (; V) = VWinvsh
    l_cutoff_ind = 1 + Δl_cutoff
    # ensure that most of the power at the surface is below the cutoff
    V_lm_surface = @view V[end, :]
    P_frac = sum(abs2, @view V_lm_surface[1:l_cutoff_ind]) / sum(abs2, V_lm_surface)
    P_frac > power_cutoff
end

function eigenfunction_rossbyridge(nr, nℓ, m; operators = radial_operators(nr, nℓ))
    λ, v = filter_eigenvalues(uniform_rotation_spectrum, nr, nℓ, m)
    eigenfunction_rossbyridge(λ, v, m, operators)
end

function eigenvalues_remove_spurious(f1::String, f2::String)
    lam1 = jldopen(f1)["lam"]::Vector{Vector{Float64}}
    lam2 = jldopen(f2)["lam"]::Vector{Vector{Float64}}
    eigenvalues_remove_spurious(lam1, lam2)
end
eigenvalues_remove_spurious(f1, f2, f3, f4...) = foldl(eigenvalues_remove_spurious, (f1, f2, f3, f4...))
function eigenvalues_remove_spurious(lam1::Vector{<:Vector}, lam2::Vector{<:Vector})
    lam_filt = similar(lam1)
    for (m_ind, (lam1_m, lam2_m)) in enumerate(zip(lam1, lam2))
        lam_filt[m_ind] = eltype(lam_filt)[]
        for λ1 in lam1_m, λ2 in lam2_m
            if abs2(λ1 - λ2) < abs2(λ1) * 1e-2
                push!(lam_filt[m_ind], λ1)
            end
        end
    end
    return lam_filt
end

function spectrum_misfit(lam1, lam2, mr)
    misfit = 0.0
    for (l1, l2) in zip(lam1, lam2)
        for λ1 in l1
            misfit += findmin(x -> abs2(1 - x / λ1), l2)[1]
        end
    end
    misfit
end

function spectrum_convergence_misfits(nrs::AbstractVector, nℓs::AbstractVector, mr)
    misfits = zeros(length(nrs), length(nℓs))
    kw = (;)
    for (nℓind, nℓ) in enumerate(nℓs), (nrind, nr) in enumerate(nrs)
        λv = filter_eigenvalues_mrange(uniform_rotation_spectrum, nr, nℓ, mr; kw...)
        lam = first.(λv)
        λ2v2 = filter_eigenvalues_mrange(uniform_rotation_spectrum, nr + 5, nℓ + 5, mr; kw...)
        lam2 = first.(λ2v2)
        misfit = spectrum_misfit(lam, lam2, mr)
        misfits[nrind, nℓind] = misfit
    end
    return misfits
end

end # module
