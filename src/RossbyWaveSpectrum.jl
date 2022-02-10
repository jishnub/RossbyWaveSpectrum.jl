module RossbyWaveSpectrum

using Dierckx
using FastGaussQuadrature
using FastTransforms
using FillArrays
using Folds
using ForwardDiff
using Kronecker
const kron2 = Kronecker.KroneckerProduct
using LinearAlgebra
using LinearAlgebra: BLAS
using JLD2
using LegendrePolynomials
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

function chebyshev_forward_inverse(n, boundaries...)
    r_chebyshev, r = chebyshevnodes(n, boundaries...)
    Tcinv = zeros(n, n)
    for (Tc, node) in zip(eachrow(Tcinv), r_chebyshev)
        chebyshevpoly!(Tc, node)
    end
    Tcfwd = Tcinv' * 2 / n # each row is one node
    Tcfwd[1, :] ./= 2
    r, Tcfwd, Tcinv
end

chebyshevtransform(A::AbstractVector) = chebyshevtransform!(similar(A), A)
function chebyshevtransform!(B::AbstractVector, A::AbstractVector, PC = plan_chebyshevtransform!(B))
    B .= @view A[end:-1:begin]
    PC * B
    return B
end

chebyshevtransform1(A::AbstractMatrix) = chebyshevtransform1!(similar(A), A)
function chebyshevtransform1!(B::AbstractMatrix, A::AbstractMatrix)
    v_temp = similar(A, size(A, 1))
    PC = plan_chebyshevtransform!(v_temp)
    for j in axes(A, 2)
        v = @view A[:, j]
        chebyshevtransform!(v_temp, v, PC)
        B[:, j] .= v_temp
    end
    return B
end

chebyshevpoly(x, n) = chebyshevpoly!(zeros(n), x, n)
function chebyshevpoly!(Tc, x, n = length(Tc))
    @assert length(Tc) >= n
    Tc[1] = 1
    if n == 1
        return Tc
    end
    Tc[2] = x
    for k = 3:n
        Tc[k] = 2x * Tc[k-1] - Tc[k-2]
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
    Matrix(UpperTriangular(transpose(M)))
end

function chebyshev_integrate(y)
    n = length(y)
    sum(yi * sin((2i - 1) / 2n * pi) * pi / n for (i, yi) in enumerate(y))
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

normalizedlegendretransform2(A::AbstractMatrix) = normalizedlegendretransform2!(similar(A), A)
function normalizedlegendretransform2!(B::AbstractMatrix, A::AbstractMatrix)
    v_temp = similar(A, size(A, 2))
    PC = plan_chebyshevtransform!(v_temp)
    PC2L = plan_cheb2leg(v_temp)
    for j in axes(A, 1)
        v = @view A[j, :]
        v_temp .= v
        normalizedlegendretransform!(v_temp, PC, PC2L)
        B[j, :] .= v_temp
    end
    return B
end


function chebyshev_normalizedlegendre_transform(A::AbstractMatrix)
    B = chebyshevtransform1(A)
    normalizedlegendretransform2!(B, B)
    return B
end

function inv_chebyshev_normalizedlegendre_transform(Fnℓ, r_chebyshev, θ;
    Pℓ = zeros(size(Fnℓ, 2), length(θ)),
    Tc = zeros(size(Fnℓ, 1), length(r_chebyshev))
)

    nr = length(r_chebyshev)
    nθ = length(θ)
    ℓmax = size(Fnℓ, 2) - 1

    for (θind, θi) in enumerate(θ)
        costhetai = cos(θi)
        Pℓ_i = @view Pℓ[:, θind]
        collectPl!(Pℓ_i, costhetai)
        # normalize
        for n in 0:ℓmax
            Pℓ_i[n+1] /= sqrt(2 / (2n + 1))
        end
    end

    for (Tc_col, ri) in zip(eachcol(Tc), r_chebyshev)
        chebyshevpoly!(Tc_col, ri)
    end

    T = kron(Pℓ, Tc)
    reshape(sum(vec(Fnℓ) .* T, dims = 1), nr, nθ)
end

function gausslegendre_theta_grid(ntheta)
    costheta, w = reverse.(gausslegendre(ntheta))
    thetaGL = acos.(costheta)
    (; thetaGL, costheta, w)
end

function ntheta_ℓmax(nℓ, m)
    ℓmax = nℓ + m - 1
    #= degree ℓmax polynomial may be represented using ℓmax + 1 points, but this
    might need to be increased as the functions are not polynomials in general
    =#
    ℓmax + 1
end

"""
    associatedlegendretransform_matrices(nθ)

Return matrices `(PLMfwd, PLMinv)`, where `PLMfwd` multiplies a function `f(θ)` to return `flm`,
the coefficients of an associated Legendre polynomial expansion of `f(θ)`, and `PLMinv` performs
the inverse transform and multiplies `flm` to return `f(θ)`.

The matrices satisfy `Mfwd * Minv ≈ I` and `(PLMinv * PLMfwd) * PLMinv ≈ PLMinv`. Note that
`(PLMinv * PLMfwd)` is not identity.
"""
function associatedlegendretransform_matrices(nℓ, m, costheta, w)
    nθ = length(costheta)
    nℓ_transform = nℓ
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
    associatedlegendretransform_matrices(nℓ, m, costheta, w)
end

function theta_operators(nℓ, m)
    ntheta = ntheta_ℓmax(nℓ, m)
    (; thetaGL, costheta, w) = gausslegendre_theta_grid(ntheta)
    (; PLMfwd, PLMinv) = associatedlegendretransform_matrices(nℓ, m, costheta, w)
    (; ntheta, thetaGL, PLMfwd, PLMinv)
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

function constrained_matmul_cache(constraints)
    (; ZC) = constraints
    MZCcache = zeros(ComplexF64, size(ZC))
    M_constrained = zeros(ComplexF64, size(ZC, 2), size(ZC, 2))
    return (; MZCcache, M_constrained)
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
    Matrix(SymTridiagonal(d, dl)')
end

function sintheta_dtheta_operator(nℓ, m)
    dl = [γ⁻ℓm(ℓ, m) for ℓ in m .+ (1:nℓ-1)]
    d = zeros(nℓ)
    du = [γ⁺ℓm(ℓ, m) for ℓ in m .+ (0:nℓ-2)]
    Matrix(Tridiagonal(dl, d, du)')
end

function invsintheta_dtheta_operator(nℓ)
    M = zeros(nℓ, nℓ)
    Mo = OffsetArray(M, OffsetArrays.Origin(0))
    for n in 0:nℓ-1, k in n-1:-2:0
        Mo[n, k] = -√((2n + 1) * (2k + 1))
    end
    return Matrix(M')
end

# defined for normalized Legendre polynomials
function cottheta_dtheta_operator(nℓ)
    M = zeros(nℓ, nℓ)
    Mo = OffsetArray(M, OffsetArrays.Origin(0))
    for n in 0:nℓ-1
        Mo[n, n] = -n
        for k in n-2:-2:0
            Mo[n, k] = -√((2n + 1) * (2k + 1))
        end
    end
    Matrix(M')
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

splderiv(f::Vector, r::Vector) = splderiv(Spline1D(r, f), r)
splderiv(spl::Spline1D, r::Vector) = derivative(spl, r)

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

function radial_operators(nr, nℓ, r_in_frac = 0.7, r_out_frac = 1)
    r_in = r_in_frac * Rsun
    r_out = r_out_frac * Rsun
    radial_params = parameters(nr, nℓ; r_in, r_out)
    (; Δr, nchebyr, r_mid) = radial_params
    r, Tcrfwd, Tcrinv = chebyshev_forward_inverse(nr, r_in, r_out)
    pseudospectralop_radial = SpectralOperatorForm(Tcrfwd, Tcrinv)
    r_chebyshev = (r .- r_mid) ./ (Δr / 2)
    Tcrfwdc, Tcrinvc = complex.(Tcrfwd), complex.(Tcrinv)
    r_cheby = Tcrfwd * Diagonal(r) * Tcrinv
    r2_cheby = r_cheby^2

    (; ρ, T, g) = read_solar_model(nr; r_in, r_out)

    ddr = Matrix(chebyshevderiv(nr) * (2 / Δr))
    rddr = r_cheby * ddr
    d2dr2 = ddr^2
    r2d2dr2 = r2_cheby * d2dr2
    ddr_realspace = Tcrinv * ddr * Tcrfwd
    ηρ = ddr_realspace * log.(ρ)
    ηρ_cheby = Tcrfwd * Diagonal(ηρ) * Tcrinv
    DDr = ddr + ηρ_cheby
    rDDr = r_cheby * DDr
    D2Dr2 = DDr^2

    onebyr = 1 ./ r
    onebyr_cheby = Tcrfwd * Diagonal(onebyr) * Tcrinv
    onebyr2_cheby = Tcrfwd * Diagonal(onebyr)^2 * Tcrinv
    DDr_minus_2byr = DDr - 2onebyr_cheby

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

    # scaling for S and W
    ε = 1e-18
    Wscaling = Rsun
    scalings = (; ε, Wscaling)

    constants = (; κ, scalings)
    identities = (; Ir, Iℓ)
    coordinates = (; r, r_chebyshev)
    transforms = (; Tcrfwd, Tcrinv, Tcrfwdc, Tcrinvc, pseudospectralop_radial)
    rad_terms = (; onebyr, onebyr_cheby, ηρ, ηρ_cheby, onebyr2_cheby,
        ddr_lnκρT, ddr_S0_by_cp, g_cheby, r_cheby, r2_cheby, κ_cheby)
    diff_operators = (; DDr, D2Dr2, DDr_minus_2byr, rDDr, rddr, ddr, d2dr2, r2d2dr2)

    (;
        constants, rad_terms,
        diff_operators,
        transforms, coordinates,
        radial_params, identities
    )
end
precompile(radial_operators, (Int, Int, Float64, Float64))

function blockinds((m, nr), ℓ, ℓ′ = ℓ)
    @assert ℓ >= m "ℓ must be >= m"
    @assert ℓ′ >= m "ℓ must be >= m"
    rowtopind = (ℓ - m) * nr + 1
    rowinds = range(rowtopind, length = nr)
    colleftind = (ℓ′ - m) * nr + 1
    colinds = range(colleftind, length = nr)
    CartesianIndices((rowinds, colinds))
end

function uniform_rotation_matrix(nr, nℓ, m; operators, kw...)
    nparams = nr * nℓ
    M = zeros(ComplexF64, 3nparams, 3nparams)
    uniform_rotation_matrix!(M, nr, nℓ, m; operators, kw...)
    return M
end

function uniform_rotation_matrix!(M, nr, nℓ, m; operators, kw...)

    (; radial_params, rad_terms, diff_operators) = operators

    (; nchebyr, r_out) = radial_params

    (; DDr, DDr_minus_2byr, ddr, d2dr2, rddr) = diff_operators
    (; onebyr_cheby, onebyr2_cheby, ddr_lnκρT, κ_cheby,
        g_cheby, ηρ_cheby, r_cheby, ddr_S0_by_cp) = rad_terms

    Ω0 = RossbyWaveSpectrum.equatorial_rotation_angular_velocity(r_out/Rsun)

    Cℓ′ = similar(DDr)
    WVℓℓ′ = similar(Cℓ′)
    VWℓℓ′ = similar(Cℓ′)
    GℓC1 = similar(Cℓ′)
    GℓC1_ddr = similar(GℓC1)
    GℓCℓ′ = similar(Cℓ′)

    nparams = nchebyr * nℓ
    ℓs = range(m, length = nℓ)

    M .= 0

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

    (; ε, Wscaling) = operators.constants.scalings

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
        WS[diaginds_ℓ] = (ε / Wscaling) * (-1 / Ω0) * Gℓ * g_cheby * (2 / (ℓ * (ℓ + 1)) * rddr + I)

        @. SW[diaginds_ℓ] = (Wscaling / ε) * (1 / Ω0) * ℓ * (ℓ + 1) * onebyr2_cheby_ddr_S0_by_cp
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

function viscosity_terms!(M, nr, nℓ, m; operators)
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
    xmin = minimum(xout)
    # rinds_inrange = xin.>=xmin
    # xin = xin[rinds_inrange]
    # z = z[rinds_inrange, :]
    evalgrid(Spline2D(xin, yin, z; s), xout, yout)
end

function read_angular_velocity_radii(dir)
    r_ΔΩ_raw = vec(readdlm(joinpath(dir, "rmesh.orig")))
    r_ΔΩ_raw[1:4:end]
end

function read_angular_velocity_raw(dir)
    ν_raw = readdlm(joinpath(dir, "rot2d.hmiv72d.ave"), ' ')
    ν_raw = [ν_raw reverse(ν_raw[:, 1:end-1], dims = 2)]
    2pi * 1e-9 * ν_raw
end

function equatorial_rotation_angular_velocity(r_frac)
    parentdir = dirname(@__DIR__)
    r_ΔΩ_raw = read_angular_velocity_radii(parentdir)
    Ω_raw = read_angular_velocity_raw(parentdir)
    equatorial_rotation_angular_velocity(r_frac, r_ΔΩ_raw, Ω_raw)
end

function equatorial_rotation_angular_velocity(r_frac, r_ΔΩ_raw, Ω_raw)
    nθ = size(Ω_raw, 2)
    lats_raw = LinRange(0, pi, nθ)
    θind_equator = findmin(abs.(lats_raw .- pi / 2))[2]
    r_frac_ind = findmin(abs.(r_ΔΩ_raw .- r_frac))[2]
    Ω_raw[r_frac_ind, θind_equator]
end

function read_angular_velocity(operators, thetaGL)
    (; coordinates, radial_params) = operators
    (; r) = coordinates
    (; r_out) = radial_params

    parentdir = dirname(@__DIR__)
    r_ΔΩ_raw = read_angular_velocity_radii(parentdir)
    Ω_raw = read_angular_velocity_raw(parentdir)
    Ω0 = equatorial_rotation_angular_velocity(r_out / Rsun, r_ΔΩ_raw, Ω_raw)
    ΔΩ_raw = Ω_raw .- Ω0

    nθ = size(ΔΩ_raw, 2)
    lats_raw = LinRange(0, pi, nθ)

    (interp2d(r_ΔΩ_raw, lats_raw, ΔΩ_raw, r ./ Rsun, thetaGL), Ω0)
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

function matrix_block(M, rowind, colind)
    nparams = div(size(M, 1), 3)
    inds1 = (rowind - 1) * nparams .+ (1:nparams)
    inds2 = (colind - 1) * nparams .+ (1:nparams)
    inds = CartesianIndices((inds1, inds2))
    @view M[inds]
end

function matrix_block_maximum(M)
    [maximum(abs, RossbyWaveSpectrum.matrix_block(M, i, j)) for i in 1:3, j in 1:3]
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

    ΔΩ_by_Ω0 = 0.02

    ℓs = range(m, length = nℓ)

    cosθo = OffsetArray(costheta_operator(nℓ, m), ℓs, ℓs)
    sinθdθo = OffsetArray(sintheta_dtheta_operator(nℓ, m), ℓs, ℓs)
    laplacian_sinθdθo = OffsetArray(Diagonal(@. -ℓs * (ℓs + 1)) * parent(sinθdθo), ℓs, ℓs)

    (; Wscaling) = operators.constants.scalings

    for ℓ in ℓs
        # numerical green function
        Gℓ = greenfn_cheby_numerical2(ℓ, operators)
        ℓℓp1 = ℓ * (ℓ + 1)
        diaginds_ℓ = blockinds((m, nr), ℓ)
        two_over_ℓℓp1 = 2 / ℓℓp1

        VVd = @view VV[diaginds_ℓ]
        @views VVd[diagind(VVd)] .+= m * (two_over_ℓℓp1 - 1) * ΔΩ_by_Ω0

        @views WW[diaginds_ℓ] .+= m * ΔΩ_by_Ω0 * Gℓ *
                                  (-2 * onebyr_cheby * ηρ_cheby +
                                   (two_over_ℓℓp1 - 1) * (ddr * DDr - ℓℓp1 * onebyr2_cheby)
                                  )

        for ℓ′ in ℓ-1:2:ℓ+1
            ℓ′ in ℓs || continue
            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)
            inds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)

            @views @. VW[inds_ℓℓ′] -= Wscaling * two_over_ℓℓp1 *
                     ΔΩ_by_Ω0 * (
                         ℓ′ℓ′p1 * cosθo[ℓ, ℓ′] * (DDr - 2 * onebyr_cheby) +
                         (DDr - ℓ′ℓ′p1 * onebyr_cheby) * sinθdθo[ℓ, ℓ′]
                     )

            @views WV[inds_ℓℓ′] .-= (1 / Wscaling) * ΔΩ_by_Ω0 / ℓℓp1 * Gℓ *
                                    ((4ℓ′ℓ′p1 * cosθo[ℓ, ℓ′] + (ℓ′ℓ′p1 + 2) * sinθdθo[ℓ, ℓ′]) * ddr
                                     +
                                     (ddr + 2 * onebyr_cheby) * laplacian_sinθdθo[ℓ, ℓ′])
        end
    end
    return M
end
function radial_differential_rotation_profile(operators, thetaGL, model = :solar_equator;
        smoothing_param = 1e-3)
    (; r) = operators.coordinates
    (; radial_params) = operators
    (; r_out, nr) = radial_params
    if model == :solar_equator
        ΔΩ_rθ, Ω0 = read_angular_velocity(operators, thetaGL)
        θind_equator = findmin(abs.(thetaGL .- pi / 2))[2]
        ΔΩ_r = ΔΩ_rθ[:, θind_equator]
        # smooth the profile in r
        s = sum(abs2, ΔΩ_r) * smoothing_param
        ΔΩ_r = Spline1D(r, ΔΩ_r, s = s)(r)
    elseif model == :linear
        Ω0 = equatorial_rotation_angular_velocity(r_out / Rsun)
        f = -0.8
        ΔΩ_r = @. Ω0 * f * (r / Rsun - 1)
    elseif model == :constant # for testing
        ΔΩ_by_Ω = 0.02
        Ω0 = equatorial_rotation_angular_velocity(r_out / Rsun)
        ΔΩ_r = fill(ΔΩ_by_Ω * Ω0, nr)
    else
        error("$model is not a valid rotation model")
    end
    return ΔΩ_r, Ω0
end
struct VWArrays{TV,TW}
    V::TV
    W::TW
end
Base.:(+)(x::VWArrays, y::VWArrays) = VWArrays(x.V + y.V, x.W + y.W)
function (+ₜ)(x::VWArrays, y::VWArrays)
    y.V .= x.V .+ y.V
    y.W .= x.W .+ y.W
    return y
end
function (+ₛ)(x::VWArrays, y::VWArrays...)
    broadcast!(+, x.V, x.V, map(x -> x.V, y)...)
    broadcast!(+, x.W, x.W, map(x -> x.W, y)...)
    return x
end
Base.:(-)(x::VWArrays) = VWArrays(-x.V, -x.W)
Base.:(-)(x::VWArrays, y::VWArrays) = VWArrays(x.V - y.V, x.W - y.W)
function (-ₛ)(x::VWArrays, y::VWArrays)
    x.V .-= y.V
    x.W .-= y.W
    return x
end
function (-ₜ)(x::VWArrays, y::VWArrays)
    y.V .= x.V .- y.V
    y.W .= x.W .- y.W
    return y
end

Base.:(*)(A::Union{Number,AbstractMatrix}, B::VWArrays) = VWArrays(A * B.V, A * B.W)
Base.:(*)(A::VWArrays, B::Union{Number,AbstractMatrix}) = VWArrays(A.V * B, A.W * B)
function LinearAlgebra.mul!(out::VWArrays, A::AbstractMatrix, B::VWArrays)
    mul!(out.V, A, B.V)
    mul!(out.W, A, B.W)
    out
end

struct SpectralOperatorForm
    fwd::Matrix{Float64}
    inv::Matrix{Float64}
    fwdcache::Matrix{Float64}
    invcache::Matrix{Float64}

    SpectralOperatorForm(fwd, inv) =
        new(fwd, inv,
            similar(fwd, size(fwd, 1), size(inv, 2)),
            similar(inv, size(fwd, 2), size(inv, 2))
        )
end

function (op::SpectralOperatorForm)(A::AbstractMatrix)
    mul!(op.invcache, A, op.inv)
    mul!(op.fwdcache, op.fwd, op.invcache)
    return copy(op.fwdcache)
end

struct RealSpace
    r_chebyshev::Vector{Float64}
    theta::Vector{Float64}
    Pℓ::Matrix{Float64}
    Tc::Matrix{Float64}
end

function (rop::RealSpace)(A)
    Diagonal(vec(
        inv_chebyshev_normalizedlegendre_transform(A, rop.r_chebyshev, rop.theta; rop.Pℓ, rop.Tc)))
end

function linear_differential_rotation_terms!(M, nr, nℓ, m;
    operators = radial_operators(nr, nℓ))

    nparams = nr * nℓ
    (; transforms, diff_operators, rad_terms, identities, coordinates) = operators
    (; r) = coordinates
    (; Iℓ, Ir) = identities
    (; ddr, DDr, d2dr2, r2d2dr2, rddr) = diff_operators
    (; Tcrfwd, Tcrinv) = transforms
    (; g_cheby, onebyr_cheby, onebyr2_cheby, r2_cheby, r_cheby) = rad_terms

    VV = @view M[1:nparams, 1:nparams]
    VW = @view M[1:nparams, nparams.+(1:nparams)]
    WV = @view M[nparams.+(1:nparams), 1:nparams]
    WW = @view M[nparams.+(1:nparams), nparams.+(1:nparams)]
    SV = @view M[2nparams.+(1:nparams), 1:nparams]
    SW = @view M[2nparams.+(1:nparams), nparams.+(1:nparams)]

    ntheta = ntheta_ℓmax(nℓ, m)
    (; thetaGL) = gausslegendre_theta_grid(ntheta)
    ΔΩ_r, Ω0 = radial_differential_rotation_profile(operators, thetaGL, :linear)
    ΔΩ1 = ΔΩ_r[1] / (r[1] - Rsun)
    pseudospectralop = SpectralOperatorForm(Tcrfwd, Tcrinv)
    ΔΩ = pseudospectralop(Diagonal(ΔΩ_r))
    drΔΩ_real = ΔΩ1 * ones(nr)
    ddrΔΩ = ΔΩ1 * I(nr)
    ddrΔΩ_over_g = ddrΔΩ * g_cheby^-1
    ddrΔΩ_over_g_DDr = ddrΔΩ_over_g * DDr
    ddr_plus_2byr_ΔΩ = pseudospectralop(Diagonal(@. drΔΩ_real + 2 / r * ΔΩ_r))

    ℓs = range(m, length = nℓ)

    cosθ = costheta_operator(nℓ, m)
    cosθo = OffsetArray(cosθ, ℓs, ℓs)
    sinθdθ = sintheta_dtheta_operator(nℓ, m)
    sinθdθo = OffsetArray(sinθdθ, ℓs, ℓs)
    cosθsinθdθ = parent(cosθ) * parent(sinθdθ)
    cosθsinθdθo = OffsetArray(cosθsinθdθ, ℓs, ℓs)

    ℓℓp1d = Diagonal(@. ℓs * (ℓs + 1))
    m_ℓℓp1d = m * ℓℓp1d

    ddr_plus_2byr_ΔΩ_plus_ΔΩ_ddr = ddr_plus_2byr_ΔΩ + ΔΩ * ddr
    ddr_plus_2byr_ΔΩ_plus_ΔΩ_ddr_DDr = ddr_plus_2byr_ΔΩ_plus_ΔΩ_ddr * DDr
    ΔΩ_by_r2 = ΔΩ * onebyr2_cheby
    r2_one_min_rminRsun_ddr_plus_2byr = r2_cheby *
                                        (I + (r_cheby - Rsun * I) * (ddr + 2onebyr_cheby))
    r2_one_min_rminRsun_ddr_plus_2byr_DDr = r2_one_min_rminRsun_ddr_plus_2byr * DDr

    r²_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_V = kron2(-sinθdθ,
        ΔΩ1 * r2_one_min_rminRsun_ddr_plus_2byr)
    r²_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_W = m * ΔΩ1 * (
                                              kron2(Iℓ, r2_one_min_rminRsun_ddr_plus_2byr_DDr) -
                                              kron2(ℓℓp1d, r_cheby - Rsun * I))

    r²_u_cross_ω_r = VWArrays(
        r²_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_V,
        r²_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_W)

    uf_cross_ωΩ_plus_uΩ_cross_ωf_r_V = kron2(-sinθdθ,
        ΔΩ1 * (I + (r_cheby - Rsun * I) * (ddr + 2onebyr_cheby)))
    uf_cross_ωΩ_plus_uΩ_cross_ωf_r_W = m * ΔΩ1 * (kron(Iℓ, (I + (r_cheby - Rsun * I) * (ddr + 2onebyr_cheby)) * DDr) -
        kron(ℓℓp1d, (r_cheby - Rsun*I) * onebyr2_cheby) )

    ∇²h_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_V =
        kron2(ℓℓp1d * sinθdθ, ddr_plus_2byr_ΔΩ_plus_ΔΩ_ddr)
    ∇²h_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_W =
        kron2(-m_ℓℓp1d, ddr_plus_2byr_ΔΩ_plus_ΔΩ_ddr_DDr) +
        kron2(m * ℓℓp1d^2, ΔΩ_by_r2)

    ∇²h_u_cross_ω_r = VWArrays(
        ∇²h_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_V,
        ∇²h_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_W
    )

    # r²_ωΩ_dot_ωf = ωΩr * r²ωfr + ωΩθ_by_rsinθ * r³sinθ_ωfθ
    r²_ωΩ_dot_ωf_V = kron2(2ΔΩ1 * cosθ * ℓℓp1d, r_cheby - Rsun * I) +
                     kron2(ΔΩ1 * sinθdθ, (2onebyr_cheby * Rsun - 3I) * r2_cheby * ddr)
    r²_ωΩ_dot_ωf_W = kron2(-m * ΔΩ1 * Iℓ, (2onebyr_cheby * Rsun - 3I) * r2_cheby * ddr * DDr) +
                     kron2(m * ΔΩ1 * ℓℓp1d, (2onebyr_cheby * Rsun - 3I))
    r²_ωΩ_dot_ωf = VWArrays(r²_ωΩ_dot_ωf_V, r²_ωΩ_dot_ωf_W)
    negr_by_sinθ_curlωΩϕ_ufϕ_rsinθ_V = kron(-4ΔΩ1 * sinθdθ, r_cheby)
    negr_by_sinθ_curlωΩϕ_ufϕ_rsinθ_W = kron(4m * ΔΩ1 * Iℓ, r_cheby * DDr)
    negr_by_sinθ_curlωΩϕ_ufϕ_rsinθ = VWArrays(negr_by_sinθ_curlωΩϕ_ufϕ_rsinθ_V,
        negr_by_sinθ_curlωΩϕ_ufϕ_rsinθ_W)
    negΔΩ_r³sinθ_curlωf_ϕ_V = kron2(-ΔΩ1 * sinθdθ, (r_cheby - Rsun * I) * r2_cheby * d2dr2) +
                              kron2(ΔΩ1 * sinθdθ * ℓℓp1d, r_cheby - Rsun * I)
    negΔΩ_r³sinθ_curlωf_ϕ_W = kron2(m * ΔΩ1 * Iℓ, (r_cheby - Rsun * I) * r2_cheby * d2dr2 * DDr) -
                              kron2(m * ΔΩ1 * ℓℓp1d, (r_cheby - Rsun * I) * (ddr - 2onebyr_cheby))
    negΔΩ_r³sinθ_curlωf_ϕ = VWArrays(negΔΩ_r³sinθ_curlωf_ϕ_V, negΔΩ_r³sinθ_curlωf_ϕ_W)
    r²_div_uf_cross_ωΩ = r²_ωΩ_dot_ωf + negr_by_sinθ_curlωΩϕ_ufϕ_rsinθ
    r²_div_uΩ_cross_ωf = r²_ωΩ_dot_ωf +ₛ negΔΩ_r³sinθ_curlωf_ϕ

    r²_div_u_cross_ω = r²_div_uf_cross_ωΩ +ₛ r²_div_uΩ_cross_ωf

    (; ε, Wscaling) = operators.constants.scalings

    negr²Ω0W_rhs = +ₛ(kron2(-Iℓ, ddr) * r²_div_u_cross_ω,
        kron2(Iℓ, d2dr2) * r²_u_cross_ω_r,
        ∇²h_u_cross_ω_r)

    DDr_min_2byr = @. DDr - 2onebyr_cheby
    ΔΩ_DDr_min_2byr = ΔΩ * DDr_min_2byr
    ΔΩ_DDr = ΔΩ * DDr
    ΔΩ_by_r = ΔΩ * onebyr_cheby

    for ℓ in ℓs
        # numerical green function
        Gℓ = greenfn_cheby_numerical2(ℓ, operators)
        ℓℓp1 = ℓ * (ℓ + 1)
        Gℓ_invℓℓp1_invΩ0 = Gℓ * ((1 / Ω0) * (1 / ℓℓp1))
        inds_ℓℓ = blockinds((m, nr), ℓ, ℓ)
        two_over_ℓℓp1 = 2 / ℓℓp1

        @. VV[inds_ℓℓ] += (1 / Ω0) * m * (two_over_ℓℓp1 - 1) * ΔΩ
        @views WW[inds_ℓℓ] .+= Gℓ_invℓℓp1_invΩ0 * negr²Ω0W_rhs.W[inds_ℓℓ]

        for ℓ′ in ℓ-1:2:ℓ+1
            ℓ′ in ℓs || continue
            inds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)
            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)
            @. VW[inds_ℓℓ′] -= Wscaling * two_over_ℓℓp1 *
                               (1 / Ω0) * (ℓ′ℓ′p1 * cosθo[ℓ, ℓ′] * ΔΩ_DDr_min_2byr +
                                           sinθdθo[ℓ, ℓ′] * (
                                   (ΔΩ_DDr - ℓ′ℓ′p1 * ΔΩ_by_r) - ℓ′ℓ′p1 / 2 * ddrΔΩ))

            @views WV[inds_ℓℓ′] .+= (1 / Wscaling) * Gℓ_invℓℓp1_invΩ0 * negr²Ω0W_rhs.V[inds_ℓℓ′]
            @. SV[inds_ℓℓ′] -= (1 / ε) * cosθo[ℓ, ℓ′] * 2m * ddrΔΩ_over_g
        end

        for ℓ′ in ℓ-2:2:ℓ+2
            ℓ′ in ℓs || continue
            inds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)

            @. SW[inds_ℓℓ′] += (Wscaling / ε) * 2cosθsinθdθo[ℓ, ℓ′] * ddrΔΩ_over_g_DDr
        end
    end
    return M
end

function radial_differential_rotation_profile_derivatives(nℓ, m, r;
    operators, rotation_profile = :constant)

    pseudospectralop = operators.transforms.pseudospectralop_radial

    ntheta = ntheta_ℓmax(nℓ, m)
    (; thetaGL) = gausslegendre_theta_grid(ntheta)
    ΔΩ_r, Ω0 = radial_differential_rotation_profile(operators, thetaGL, rotation_profile)
    ΔΩ = pseudospectralop(Diagonal(ΔΩ_r))

    ΔΩ_spl = Spline1D(r, ΔΩ_r)
    drΔΩ_real = derivative(ΔΩ_spl, r)
    ddrΔΩ = pseudospectralop(Diagonal(drΔΩ_real))
    (; ΔΩ_r, Ω0, ΔΩ, ΔΩ_spl, drΔΩ_real, ddrΔΩ)
end

function negr²Ω0W_rhs_radial(m, (ΔΩ_r, drΔΩ_real, ΔΩ_spl, ΔΩ), (cosθ, sinθdθ); operators)
    (; nℓ) = operators.radial_params
    (; r) = operators.coordinates
    (; Iℓ, Ir) = operators.identities

    (; onebyr_cheby, onebyr2_cheby, r2_cheby, r_cheby) = operators.rad_terms
    (; ddr, DDr, d2dr2, rddr, r2d2dr2) = operators.diff_operators

    pseudospectralop = operators.transforms.pseudospectralop_radial

    ddr_min_2byr = ddr - 2onebyr_cheby

    drΔΩ_real = derivative(ΔΩ_spl, r, nu = 1)
    d2rΔΩ_real = derivative(ΔΩ_spl, r, nu = 2)
    d2dr2ΔΩ = pseudospectralop(Diagonal(d2rΔΩ_real))

    ℓs = range(m, length = nℓ)

    ωΩr = kron2(2cosθ, ΔΩ)
    ddrΔΩ_by_r = pseudospectralop(Diagonal(drΔΩ_real ./ r))
    ddr_plus_2byr_ΔΩ = pseudospectralop(Diagonal(@. drΔΩ_real + 2 / r * ΔΩ_r))
    ωΩθ_by_rsinθ = kron2(-Iℓ, ddr_plus_2byr_ΔΩ)
    neg_invrsinθ_curlωΩϕ = kron2(Iℓ, d2dr2ΔΩ + 4ddrΔΩ_by_r)

    ℓℓp1d = Diagonal(@. ℓs * (ℓs + 1))
    m_ℓℓp1d = m * ℓℓp1d
    m_Iℓ = m * Iℓ

    ωfr_V = kron2(ℓℓp1d, onebyr2_cheby)
    ωfr_W = 0
    ωfr = VWArrays(ωfr_V, ωfr_W)

    rsinθ_ωfθ_V = kron2(sinθdθ, ddr)
    rsinθ_ωfθ_W = kron2(m_ℓℓp1d, onebyr2_cheby) - kron2(m * Iℓ, ddr * DDr)
    rsinθ_ωfθ = VWArrays(rsinθ_ωfθ_V, rsinθ_ωfθ_W)

    rsinθ_ufϕ_V = kron2(sinθdθ, -Ir)
    rsinθ_ufϕ_W = kron2(m * Iℓ, DDr)
    rsinθ_ufϕ = VWArrays(rsinθ_ufϕ_V, rsinθ_ufϕ_W)

    rsinθ_curlωfϕ_V = kron2(sinθdθ, d2dr2) - kron2(sinθdθ * ℓℓp1d, onebyr2_cheby)
    rsinθ_curlωfϕ_W = kron2(-m_Iℓ, d2dr2 * DDr) + kron2(m_ℓℓp1d, onebyr2_cheby * ddr_min_2byr)
    rsinθ_curlωfϕ = VWArrays(rsinθ_curlωfϕ_V, rsinθ_curlωfϕ_W)

    ddr_plus_2byr_ΔΩ_plus_ΔΩ_ddr = ddr_plus_2byr_ΔΩ + ΔΩ * ddr
    ddr_plus_2byr_ΔΩ_plus_ΔΩ_ddr_DDr = ddr_plus_2byr_ΔΩ_plus_ΔΩ_ddr * DDr
    ΔΩ_by_r2 = ΔΩ * onebyr2_cheby

    uf_cross_ωΩ_plus_uΩ_cross_ωf_r_V = kron2(-sinθdθ, ddr_plus_2byr_ΔΩ_plus_ΔΩ_ddr)
    uf_cross_ωΩ_plus_uΩ_cross_ωf_r_W = kron2(m_Iℓ, ddr_plus_2byr_ΔΩ_plus_ΔΩ_ddr_DDr) -
                                       kron2(m_ℓℓp1d, ΔΩ_by_r2)

    u_cross_ω_r = VWArrays(
        uf_cross_ωΩ_plus_uΩ_cross_ωf_r_V,
        uf_cross_ωΩ_plus_uΩ_cross_ωf_r_W)

    ∇²h_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_V =
        kron2(ℓℓp1d * sinθdθ, ddr_plus_2byr_ΔΩ_plus_ΔΩ_ddr)
    ∇²h_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_W =
        kron2(-m_ℓℓp1d, ddr_plus_2byr_ΔΩ_plus_ΔΩ_ddr_DDr) +
        kron2(m * ℓℓp1d^2, ΔΩ_by_r2)

    ∇²h_u_cross_ω_r = VWArrays(
        ∇²h_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_V,
        ∇²h_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_W
    )

    ωΩ_dot_ωf = ωΩr * ωfr + ωΩθ_by_rsinθ * rsinθ_ωfθ
    negcurlωΩ_dot_uf = neg_invrsinθ_curlωΩϕ * rsinθ_ufϕ
    div_uf_cross_ωΩ = ωΩ_dot_ωf + negcurlωΩ_dot_uf
    curlωf_dot_uΩ = kron2(Iℓ, ΔΩ) * rsinθ_curlωfϕ
    div_uΩ_cross_ωf = ωΩ_dot_ωf - curlωf_dot_uΩ

    div_u_cross_ω = div_uf_cross_ωΩ + div_uΩ_cross_ωf

    r_ddr_plus_1 = rddr + I
    d²r_r2 = r2d2dr2 + 4rddr + 2I
    negr²Ω0W_rhs = kron(-Iℓ, r_ddr_plus_1 * r_cheby) * div_u_cross_ω +
                   kron2(Iℓ, d²r_r2) * u_cross_ω_r +
                   ∇²h_u_cross_ω_r
    (; negr²Ω0W_rhs, u_cross_ω_r, div_u_cross_ω, ∇²h_u_cross_ω_r,
        ωΩ_dot_ωf, negcurlωΩ_dot_uf, curlωf_dot_uΩ,
        div_uf_cross_ωΩ, div_uΩ_cross_ωf)
end

function radial_differential_rotation_terms!(M, nr, nℓ, m;
    operators = radial_operators(nr, nℓ),
    rotation_profile = :constant)

    nparams = nr * nℓ
    (; diff_operators, rad_terms, coordinates) = operators
    (; r) = coordinates
    (; DDr) = diff_operators
    (; g_cheby, onebyr_cheby) = rad_terms

    VV = @view M[1:nparams, 1:nparams]
    VW = @view M[1:nparams, nparams.+(1:nparams)]
    WV = @view M[nparams.+(1:nparams), 1:nparams]
    WW = @view M[nparams.+(1:nparams), nparams.+(1:nparams)]
    SV = @view M[2nparams.+(1:nparams), 1:nparams]
    SW = @view M[2nparams.+(1:nparams), nparams.+(1:nparams)]

    (; ΔΩ_r, Ω0, ΔΩ, ΔΩ_spl, drΔΩ_real, ddrΔΩ) = radial_differential_rotation_profile_derivatives(nℓ, m, r;
        operators, rotation_profile)

    ddrΔΩ_over_g = ddrΔΩ * g_cheby^-1
    ddrΔΩ_over_g_DDr = ddrΔΩ_over_g * DDr

    ℓs = range(m, length = nℓ)

    cosθ = costheta_operator(nℓ, m)
    cosθo = OffsetArray(cosθ, ℓs, ℓs)
    sinθdθ = sintheta_dtheta_operator(nℓ, m)
    sinθdθo = OffsetArray(sinθdθ, ℓs, ℓs)
    cosθsinθdθ = (costheta_operator(nℓ + 1, m) * sintheta_dtheta_operator(nℓ+1, m))[1:end-1, 1:end-1]
    cosθsinθdθo = OffsetArray(cosθsinθdθ, ℓs, ℓs)

    (; negr²Ω0W_rhs) = negr²Ω0W_rhs_radial(m, (ΔΩ_r, drΔΩ_real, ΔΩ_spl, ΔΩ),
        (cosθ, sinθdθ); operators)

    (; ε, Wscaling) = operators.constants.scalings

    DDr_min_2byr = @. DDr - 2onebyr_cheby
    ΔΩ_DDr_min_2byr = ΔΩ * DDr_min_2byr
    ΔΩ_DDr = ΔΩ * DDr
    ΔΩ_by_r = ΔΩ * onebyr_cheby

    for ℓ in ℓs
        # numerical green function
        Gℓ = greenfn_cheby_numerical2(ℓ, operators)
        ℓℓp1 = ℓ * (ℓ + 1)
        Gℓ_invℓℓp1_invΩ0 = Gℓ * ((1 / Ω0) * (1 / ℓℓp1))
        inds_ℓℓ = blockinds((m, nr), ℓ, ℓ)
        two_over_ℓℓp1 = 2 / ℓℓp1

        @. VV[inds_ℓℓ] += (1 / Ω0) * m * (two_over_ℓℓp1 - 1) * ΔΩ
        @views WW[inds_ℓℓ] .+= Gℓ_invℓℓp1_invΩ0 * negr²Ω0W_rhs.W[inds_ℓℓ]

        for ℓ′ in ℓ-1:2:ℓ+1
            ℓ′ in ℓs || continue
            inds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)
            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)
            @. VW[inds_ℓℓ′] -= Wscaling * two_over_ℓℓp1 *
                               (1 / Ω0) * (ℓ′ℓ′p1 * cosθo[ℓ, ℓ′] * ΔΩ_DDr_min_2byr +
                                           sinθdθo[ℓ, ℓ′] * (
                                   (ΔΩ_DDr - ℓ′ℓ′p1 * ΔΩ_by_r) - ℓ′ℓ′p1 / 2 * ddrΔΩ))

            @views WV[inds_ℓℓ′] .+= (1 / Wscaling) * Gℓ_invℓℓp1_invΩ0 * negr²Ω0W_rhs.V[inds_ℓℓ′]
            @. SV[inds_ℓℓ′] -= (1 / ε) * cosθo[ℓ, ℓ′] * 2m * ddrΔΩ_over_g
        end

        for ℓ′ in ℓ-2:2:ℓ+2
            ℓ′ in ℓs || continue
            inds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)

            @. SW[inds_ℓℓ′] += (Wscaling / ε) * 2cosθsinθdθo[ℓ, ℓ′] * ddrΔΩ_over_g_DDr
        end
    end
    return M
end

function solar_differential_rotation_profile(operators, thetaGL, model = :solar)
    nθ = length(thetaGL)
    if model == :solar
        return read_angular_velocity(operators, thetaGL)
    elseif model == :radial
        ΔΩ_rθ, Ω0 = read_angular_velocity(operators, thetaGL)
        θind_equator = findmin(abs.(thetaGL .- pi / 2))[2]
        ΔΩ_r = ΔΩ_rθ[:, θind_equator]
        return repeat(ΔΩ_r, 1, nθ), Ω0
    elseif model == :constant # for testing
        ΔΩ_by_Ω = 0.02
        (; radial_params) = operators
        (; r_out, nr) = radial_params
        Ω0 = equatorial_rotation_angular_velocity(r_out / Rsun)
        return fill(ΔΩ_by_Ω * Ω0, nr, nθ), Ω0
    else
        error("$model is not a valid rotation model")
    end
end

function cutoff_degree(v, cutoff_power = 0.9)
    tr = sum(abs2, v)
    iszero(tr) && return 0
    s = zero(eltype(v))
    for (i, vi) in enumerate(v)
        s += abs2(vi)
        if (s / tr) >= cutoff_power
            return i - 1
        end
    end
    return length(v) - 1
end
function cutoff_legendre_degree(ΔΩ_nℓ, cutoff_power = 0.9)
    maximum(row -> cutoff_degree(row, cutoff_power), eachrow(ΔΩ_nℓ))
end
function cutoff_chebyshev_degree(ΔΩ_nℓ, cutoff_power = 0.9)
    maximum(col -> cutoff_degree(col, cutoff_power), eachcol(ΔΩ_nℓ))
end

function solar_differential_rotation_terms!(M, nr, nℓ, m;
    operators = radial_operators(nr, nℓ),
    rotation_profile = :constant)

    nparams = nr * nℓ
    (; transforms, diff_operators, rad_terms, identities, coordinates) = operators
    (; Iℓ, Ir) = identities
    (; ddr, DDr, d2dr2, rddr, r2d2dr2, DDr_minus_2byr) = diff_operators
    (; Tcrfwd, Tcrinv) = transforms
    (; g_cheby, onebyr_cheby, onebyr2_cheby, r2_cheby, r_cheby) = rad_terms
    (; r, r_chebyshev) = coordinates
    two_over_g = 2g_cheby^-1

    ddr_plus_2byr = ddr + 2onebyr_cheby

    # for the stream functions
    (; PLMfwd, PLMinv) = associatedlegendretransform_matrices(nℓ, m)
    fwdbig_SF = kron(PLMfwd, Tcrfwd)
    invbig_SF = kron(PLMinv, Tcrinv)
    spectralop = SpectralOperatorForm(fwdbig_SF, invbig_SF)

    VV = @view M[1:nparams, 1:nparams]
    VW = @view M[1:nparams, nparams.+(1:nparams)]
    WV = @view M[nparams.+(1:nparams), 1:nparams]
    WW = @view M[nparams.+(1:nparams), nparams.+(1:nparams)]
    SV = @view M[2nparams.+(1:nparams), 1:nparams]
    SW = @view M[2nparams.+(1:nparams), nparams.+(1:nparams)]

    # this is somewhat arbitrary, as we don't know the maximum ℓ of Ω before reading it in
    ntheta = ntheta_ℓmax(nℓ, m)
    (; thetaGL) = gausslegendre_theta_grid(ntheta)

    nℓ_Ω = nℓ
    ΔΩ_rθ, Ω0 = solar_differential_rotation_profile(operators, thetaGL, rotation_profile)
    ΔΩ_nℓ = chebyshev_normalizedlegendre_transform(ΔΩ_rθ)
    ΔΩ_nℓ = ΔΩ_nℓ[:, 1:nℓ_Ω]

    # temporary arrays used in the transformation to real space
    Pℓtemp = zeros(nℓ_Ω, length(thetaGL))
    Tctemp = zeros(nr, length(r_chebyshev))
    realspaceop = RealSpace(r_chebyshev, thetaGL, Pℓtemp, Tctemp)

    pseudospectral_op = spectralop ∘ realspaceop

    ΔΩ = pseudospectral_op(ΔΩ_nℓ)
    r²ΔΩ = pseudospectral_op(r2_cheby * ΔΩ_nℓ)
    ddr_r²ΔΩ = pseudospectral_op(ddr * r2_cheby * ΔΩ_nℓ)
    rddrΔΩ = pseudospectral_op(rddr * ΔΩ_nℓ)
    r2d2dr2ΔΩ = pseudospectral_op(r2d2dr2 * ΔΩ_nℓ)
    ddr_plus_2byr_ΔΩ_nℓ = ddr_plus_2byr * ΔΩ_nℓ
    ddr_plus_2byr_ΔΩ = pseudospectral_op(ddr_plus_2byr_ΔΩ_nℓ)

    cosθ_Ω = costheta_operator(nℓ_Ω, 0)
    sinθdθ_Ω = sintheta_dtheta_operator(nℓ_Ω, 0)
    cotθdθ_Ω = cottheta_dtheta_operator(nℓ_Ω)
    invsinθdθ_Ω = invsintheta_dtheta_operator(nℓ_Ω)

    ΔΩ_ℓn = permutedims(ΔΩ_nℓ)
    ddrΔΩ_ℓn = permutedims(ddr * ΔΩ_nℓ)

    ∇²Ω = laplacian_operator(nℓ_Ω, 0)
    ∇²ΔΩ = pseudospectral_op(ΔΩ_nℓ .* diag(∇²Ω)')
    sinθdθΔΩ_plus_2cosθΔΩ = pseudospectral_op(permutedims((sinθdθ_Ω + 2cosθ_Ω) * ΔΩ_ℓn))
    cotθdθΔΩ = pseudospectral_op(permutedims(cotθdθ_Ω * ΔΩ_ℓn))
    invsinθdθΔΩ = pseudospectral_op(permutedims(invsinθdθ_Ω * ΔΩ_ℓn))

    dzΔΩ = pseudospectral_op(
        reshape((kron(cosθ_Ω, ddr) - kron(sinθdθ_Ω, onebyr_cheby)) * vec(ΔΩ_nℓ), nr, nℓ_Ω)
    )

    ωΩr = sinθdθΔΩ_plus_2cosθΔΩ
    drωΩr = pseudospectral_op(permutedims((sinθdθ_Ω + 2cosθ_Ω) * ddrΔΩ_ℓn))
    ωΩθ_by_rsinθ = -ddr_plus_2byr_ΔΩ
    invsinθdθ_ωΩθ_by_rsinθ = pseudospectral_op(permutedims(-invsinθdθ_Ω * permutedims(ddr_plus_2byr_ΔΩ_nℓ)))
    invsinθdθ_ωΩr = ∇²ΔΩ - 2ΔΩ + 2cotθdθΔΩ
    ∇²ωΩθ_by_rsinθ = pseudospectral_op(ddr_plus_2byr_ΔΩ_nℓ .* adjoint(-diag(∇²Ω)))
    ∇²_plus_4rddr_plus_r2d2dr2_plus_2cotθdθ_ΔΩ = ∇²ΔΩ + 4rddrΔΩ + r2d2dr2ΔΩ + cotθdθΔΩ
    negrinvsinθ_curlωΩϕ = ∇²_plus_4rddr_plus_r2d2dr2_plus_2cotθdθ_ΔΩ

    sinθdθ = sintheta_dtheta_operator(nℓ, m)
    ∇² = laplacian_operator(nℓ, m)
    ∇²_by_r2 = kron2(∇², onebyr2_cheby)
    ddrDDr_minus_ℓ′ℓ′p1_by_r2 = kron(Iℓ, ddr * DDr) + ∇²_by_r2

    ir²ufr = VWArrays(0, kron(-∇², Ir))
    irsinθufθ = VWArrays(-m, kron(sinθdθ, DDr))
    r²ωfr = VWArrays(kron(-∇², Ir), 0)
    r²negmωfr = VWArrays(kron(m * ∇², Ir), 0)

    rsinθufϕ = VWArrays(kron(sinθdθ, -Ir), kron(m * Iℓ, DDr))
    rsinθωfθ = VWArrays(kron(sinθdθ, ddr), -m * ddrDDr_minus_ℓ′ℓ′p1_by_r2)
    r³sinθ_ωfθ = VWArrays(kron(sinθdθ, r2_cheby * ddr),
        -m * (kron(Iℓ, r2_cheby * ddr * DDr) + kron(∇², Ir)))

    r³sinθ_curlωfϕ = VWArrays(Matrix(kron2(sinθdθ, r2d2dr2) + kron2(sinθdθ * ∇², Ir)),
        Matrix(kron2(-m * Iℓ, r2d2dr2 * DDr) - kron2(m * ∇², DDr_minus_2byr)))

    (; ε, Wscaling) = operators.constants.scalings

    V_rhs = +ₛ((ωΩr * kron(-Iℓ, DDr_minus_2byr) + ωΩθ_by_rsinθ * kron2(sinθdθ, -Ir) + drωΩr) * ir²ufr,
        invsinθdθ_ωΩr * irsinθufθ,
        ΔΩ * r²negmωfr)

    r²_ωΩ_dot_ωf = ωΩr * r²ωfr +ₛ ωΩθ_by_rsinθ * r³sinθ_ωfθ
    r²_div_uf_cross_ωΩ = r²_ωΩ_dot_ωf +ₜ negrinvsinθ_curlωΩϕ * rsinθufϕ
    r²_div_uΩ_cross_ωf = r²_ωΩ_dot_ωf -ₛ ΔΩ * r³sinθ_curlωfϕ
    r²_div_u_cross_ω = r²_div_uf_cross_ωΩ +ₛ r²_div_uΩ_cross_ωf

    ddr_r²ΔΩ_plus_r²ΔΩ_ddr = ddr_r²ΔΩ + r²ΔΩ * kron(Iℓ, ddr)
    r²_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_V = ddr_r²ΔΩ_plus_r²ΔΩ_ddr * kron2(sinθdθ, -Ir)
    r²_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_W = kron(m * ∇², Ir) * ΔΩ +
                                          ddr_r²ΔΩ_plus_r²ΔΩ_ddr * kron(m * Iℓ, DDr)

    r²_u_cross_ω_r = VWArrays(
        r²_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_V,
        r²_uf_cross_ωΩ_plus_uΩ_cross_ωf_r_W)

    sinθdθ_big = kron2(sinθdθ, Ir)

    ∇²h_uf_cross_ωΩ_r = (ωΩθ_by_rsinθ * kron2(-∇², Ir) .- ∇²ωΩθ_by_rsinθ .-
                         2 .* invsinθdθ_ωΩθ_by_rsinθ * sinθdθ_big) * rsinθufϕ

    ∇²h_uΩ_cross_ωf_r = (-∇²ΔΩ .+ ΔΩ * kron2(-∇², Ir) .- 2 .* invsinθdθΔΩ * sinθdθ_big) * rsinθωfθ

    ∇²h_u_cross_ω_r = ∇²h_uf_cross_ωΩ_r +ₛ ∇²h_uΩ_cross_ωf_r

    negr²Ω0W_rhs = +ₛ(kron(-Iℓ, ddr) * r²_div_u_cross_ω,
        kron2(Iℓ, d2dr2) * r²_u_cross_ω_r,
        ∇²h_u_cross_ω_r)

    S_rhs_V = kron2(-m * Iℓ, two_over_g) * dzΔΩ
    S_rhs_W = dzΔΩ * kron2(sinθdθ, two_over_g * DDr)
    S_rhs = VWArrays(S_rhs_V, S_rhs_W)

    ℓs = range(m, length = nℓ)
    for ℓ in ℓs
        ℓℓp1 = ℓ * (ℓ + 1)
        invℓℓp1_invΩ0 = (1 / ℓℓp1) * (1 / Ω0)
        Gℓ = greenfn_cheby_numerical2(ℓ, operators)
        Gℓ_invℓℓp1_invΩ0 = Gℓ * invℓℓp1_invΩ0
        Gℓ_invℓℓp1_invΩ0_invWscaling = (1 / Wscaling) * Gℓ_invℓℓp1_invΩ0
        for ℓ′ in ℓs
            inds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)
            @views @. VV[inds_ℓℓ′] += invℓℓp1_invΩ0 * V_rhs.V[inds_ℓℓ′]
            @views @. VW[inds_ℓℓ′] += Wscaling * invℓℓp1_invΩ0 * V_rhs.W[inds_ℓℓ′]
            @views WV[inds_ℓℓ′] .+= Gℓ_invℓℓp1_invΩ0_invWscaling * negr²Ω0W_rhs.V[inds_ℓℓ′]
            @views WW[inds_ℓℓ′] .+= Gℓ_invℓℓp1_invΩ0 * negr²Ω0W_rhs.W[inds_ℓℓ′]

            @views @. SV[inds_ℓℓ′] += (1 / ε) * S_rhs.V[inds_ℓℓ′]
            @views @. SW[inds_ℓℓ′] .+= (Wscaling / ε) * S_rhs.W[inds_ℓℓ′]
        end
    end

    return M
end

function _differential_rotation_matrix!(M, nr, nℓ, m, rotation_profile; operators)
    if rotation_profile === :radial
        radial_differential_rotation_terms!(M, nr, nℓ, m; operators, rotation_profile = :solar_equator)
    elseif rotation_profile === :radial_constant
        radial_differential_rotation_terms!(M, nr, nℓ, m; operators, rotation_profile = :constant)
    elseif rotation_profile === :radial_linear
        radial_differential_rotation_terms!(M, nr, nℓ, m; operators, rotation_profile = :linear)
    elseif rotation_profile === :solar
        solar_differential_rotation_terms!(M, nr, nℓ, m; operators, rotation_profile)
    elseif rotation_profile === :solar_radial
        solar_differential_rotation_terms!(M, nr, nℓ, m; operators, rotation_profile = :radial)
    elseif rotation_profile === :solar_constant
        solar_differential_rotation_terms!(M, nr, nℓ, m; operators, rotation_profile = :constant)
    elseif rotation_profile === :linear
        linear_differential_rotation_terms!(M, nr, nℓ, m; operators)
    elseif rotation_profile === :constant
        constant_differential_rotation_terms!(M, nr, nℓ, m; operators)
    else
        throw(ArgumentError("Invalid rotation profile"))
    end
    return M
end
function differential_rotation_matrix(nr, nℓ, m; rotation_profile, operators, kw...)
    M = uniform_rotation_matrix(nr, nℓ, m; operators, kw...)
    _differential_rotation_matrix!(M, nr, nℓ, m, rotation_profile; operators)
    return M
end
function differential_rotation_matrix!(M, nr, nℓ, m; rotation_profile, operators, kw...)
    uniform_rotation_matrix!(M, nr, nℓ, m; operators, kw...)
    _differential_rotation_matrix!(M, nr, nℓ, m, rotation_profile; operators)
    return M
end

function constrained_eigensystem(M, operators, constraints = constraintmatrix(operators),
    cache = constrained_matmul_cache(constraints))

    (; ZC) = constraints
    (; M_constrained, MZCcache) = cache #= not thread safe =#
    mul!(M_constrained, permutedims(ZC), mul!(MZCcache, M, ZC))
    # M_constrained = permutedims(ZC) * M * ZC
    λ::Vector{ComplexF64}, w::Matrix{ComplexF64} = eigen!(M_constrained)
    v = ZC * w
    λ, v, M
end

function uniform_rotation_spectrum(nr, nℓ, m; operators,
    constraints = constraintmatrix(operators),
    cache = constrained_matmul_cache(constraints),
    kw...)

    M = uniform_rotation_matrix(nr, nℓ, m; operators, kw...)
    constrained_eigensystem(M, operators, constraints, cache)
end
function uniform_rotation_spectrum!(M, nr, nℓ, m; operators,
    constraints = constraintmatrix(operators),
    cache = constrained_matmul_cache(constraints),
    kw...)

    uniform_rotation_matrix!(M, nr, nℓ, m; operators, kw...)
    constrained_eigensystem(M, operators, constraints, cache)
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
    rotation_profile, operators,
    constraints = constraintmatrix(operators),
    cache = constrained_matmul_cache(constraints),
    kw...)

    M = differential_rotation_matrix(nr, nℓ, m; operators, rotation_profile, kw...)
    constrained_eigensystem(M, operators, constraints, cache)
end
function differential_rotation_spectrum!(M, nr, nℓ, m;
    rotation_profile,
    operators,
    constraints = constraintmatrix(operators),
    cache = constrained_matmul_cache(constraints),
    kw...)

    differential_rotation_matrix!(M, nr, nℓ, m; operators, rotation_profile, kw...)
    constrained_eigensystem(M, operators, constraints, cache)
end

rossby_ridge(m, ℓ = m; ΔΩ_by_Ω = 0) = 2m / (ℓ * (ℓ + 1)) * (1 + ΔΩ_by_Ω) - m * ΔΩ_by_Ω

function eigenvalue_filter(x, m;
    eig_imag_unstable_cutoff = -1e-1,
    eig_imag_to_real_ratio_cutoff = 1e-1,
    eig_imag_damped_cutoff = 5e-3,
    ΔΩ_by_Ω_low = 0,
    ΔΩ_by_Ω_high = ΔΩ_by_Ω_low)

    freq_sectoral = 2/(m+1)

    imagfilter = eig_imag_unstable_cutoff <= imag(x) <
                 min(freq_sectoral * eig_imag_to_real_ratio_cutoff, eig_imag_damped_cutoff)

    freq_real_1 = rossby_ridge(m; ΔΩ_by_Ω = ΔΩ_by_Ω_high)
    freq_real_2 = rossby_ridge(m; ΔΩ_by_Ω = ΔΩ_by_Ω_low)
    freq_real_low, freq_real_high = minmax(freq_real_1, freq_real_2)

    realfilter = freq_real_low - 0.05 <= real(x) <= freq_real_high + 0.05

    realfilter && imagfilter
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
    Plcosθ = SphericalHarmonics.allocate_p(ℓmax))

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
    Plcosθ = SphericalHarmonics.allocate_p(ℓmax))

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

function filterfn(λ, v, m, M, operators, additional_params)

    (; eig_imag_unstable_cutoff, eig_imag_to_real_ratio_cutoff,
        ΔΩ_by_Ω_low, ΔΩ_by_Ω_high, eig_imag_damped_cutoff,
        Δl_cutoff, Δl_power_cutoff, BC, BCVcache, atol_constraint,
        VWinv, n_cutoff, n_power_cutoff, nℓ, Plcosθ,
        θ_cutoff, equator_power_cutoff_frac, MVcache, eigen_rtol, VWinvsh, F) = additional_params

    ℓmax = m + nℓ - 1

    eigenvalue_filter(λ, m;
            eig_imag_unstable_cutoff, eig_imag_to_real_ratio_cutoff,
            ΔΩ_by_Ω_low, ΔΩ_by_Ω_high, eig_imag_damped_cutoff) &&
        sphericalharmonic_filter!(VWinvsh, F, v, operators, Δl_cutoff, Δl_power_cutoff) &&
        boundary_condition_filter(v, BC, BCVcache, atol_constraint) &&
        chebyshev_filter!(VWinv, F, v, m, operators, n_cutoff, n_power_cutoff; nℓ, ℓmax, Plcosθ) &&
        equator_filter!(VWinv, VWinvsh, F, v, m, operators, θ_cutoff, equator_power_cutoff_frac; nℓ, ℓmax, Plcosθ) &&
        eigensystem_satisfy_filter(λ, v, M, MVcache, eigen_rtol)
end

function allocate_filter_caches(m; operators, BC = constraintmatrix(operators).BC)
    (; nr, nℓ, nparams) = operators.radial_params
    # temporary cache arrays
    MVcache = Vector{ComplexF64}(undef, 3nparams)
    BCVcache = Vector{ComplexF64}(undef, size(BC, 1))

    nθ = length(spharm_θ_grid_uniform(m, nℓ).θ)
    VWinv = (; V = zeros(ComplexF64, nr, nθ), W = zeros(ComplexF64, nr, nθ))

    VWinvsh = (; V = zeros(ComplexF64, nr, nℓ), W = zeros(ComplexF64, nr, nℓ))

    ℓmax = m + nℓ - 1
    Plcosθ = SphericalHarmonics.allocate_p(ℓmax)

    F = (; V = zeros(ComplexF64, nr * nℓ), W = zeros(ComplexF64, nr * nℓ), S = zeros(ComplexF64, nr * nℓ))

    return (; MVcache, BCVcache, VWinv, VWinvsh, Plcosθ, F)
end

function filter_eigenvalues(λ::AbstractVector, v::AbstractMatrix,
    M::AbstractMatrix, m::Integer;
    operators,
    BC = constraintmatrix(operators).BC,
    filtercache = allocate_filter_caches(m; operators, BC),
    atol_constraint = 1e-5,
    Δl_cutoff = 7, Δl_power_cutoff = 0.9,
    eigen_rtol = 5e-2,
    n_cutoff = 7, n_power_cutoff = 0.9,
    eig_imag_unstable_cutoff = -1e-6,
    eig_imag_to_real_ratio_cutoff = 1e-1,
    eig_imag_damped_cutoff = 5e-3,
    ΔΩ_by_Ω_low = 0, ΔΩ_by_Ω_high = ΔΩ_by_Ω_low,
    θ_cutoff = deg2rad(75),
    equator_power_cutoff_frac = 0.3,
    kw...)

    (; nℓ) = operators.radial_params
    (; MVcache, BCVcache, VWinv, VWinvsh, Plcosθ, F) = filtercache

    additional_params = (; eig_imag_unstable_cutoff, eig_imag_to_real_ratio_cutoff,
        ΔΩ_by_Ω_low, ΔΩ_by_Ω_high, eig_imag_damped_cutoff,
        Δl_cutoff, Δl_power_cutoff, BC, BCVcache, atol_constraint,
        VWinv, n_cutoff, n_power_cutoff, nℓ, Plcosθ,
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

function filter_eigenvalues(λ::AbstractVector{<:AbstractVector},
    v::AbstractVector{<:AbstractMatrix},
    mr::AbstractVector;
    Mfn = uniform_rotation_matrix!,
    operators,
    constraints = constraintmatrix(operators),
    BC = constraints.BC,
    kw...)

    (; nr, nℓ, nparams) = operators.radial_params
    nparams = nr * nℓ
    Ms = [zeros(ComplexF64, 3nparams, 3nparams) for _ in 1:Threads.nthreads()]
    caches = [constrained_matmul_cache(constraints) for _ in 1:Threads.nthreads()]
    λv = @maybe_reduce_blas_threads(
        Folds.map(zip(λ, v, mr)) do (λm, vm, m)
            M = Ms[Threads.threadid()]
            cache = caches[Threads.threadid()]
            Mfn(M, nr, nℓ, m; operators)
            filter_eigenvalues(λm, vm, M, m;
                operators, BC, cache, kw...)
        end::Vector{Tuple{Vector{ComplexF64},Matrix{ComplexF64}}}
    )
    first.(λv), last.(λv)
end

function filter_eigenvalues(f, mr::AbstractVector; #= inplace function =#
    operators,
    constraints = constraintmatrix(operators),
    kw...)

    (; nr, nℓ, nparams) = operators.radial_params
    Ms = [zeros(ComplexF64, 3nparams, 3nparams) for _ in 1:Threads.nthreads()]
    caches = [constrained_matmul_cache(constraints) for _ in 1:Threads.nthreads()]
    (; BC) = constraints

    λv = @maybe_reduce_blas_threads(
        Folds.map(mr) do m
            M = Ms[Threads.threadid()]
            cache = caches[Threads.threadid()]
            λm, vm = f(M, nr, nℓ, m; operators, constraints, kw...)
            filter_eigenvalues(λm, vm, M, m; operators, BC, cache, kw...)
        end::Vector{Tuple{Vector{ComplexF64},Matrix{ComplexF64}}}
    )
    first.(λv), last.(λv)
end
function filter_eigenvalues(filename::String; kw...)
    λ, v, mr, nr, nℓ = load(filename, "lam", "vec", "mr", "nr", "nℓ")
    operators = radial_operators(nr, nℓ)
    filter_eigenvalues(λ, v, mr; operators, kw...)
end

rossbyeigenfilename(nr, nℓ, tag = "ur", posttag = "") = "$(tag)_nr$(nr)_nl$(nℓ)$(posttag).jld2"
function save_eigenvalues(f, nr, nℓ, mr;
    operators = radial_operators(nr, nℓ), kw...)

    lam, vec = filter_eigenvalues(f, mr; operators, kw...)
    isdiffrot = !iszero(get(kw, :ΔΩ_by_Ω_low, 0) * get(kw, :ΔΩ_by_Ω_high, 0))
    filenametag = isdiffrot == 0 ? "ur" : "dr"
    fname = datadir(rossbyeigenfilename(nr, nℓ, filenametag))
    @info "saving to $fname"
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

    V = VWinv.V
    V .= 0
    W = VWinv.W
    W .= 0

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

function eigenfunction_realspace(v, m, operators)
    (; nr, nℓ) = operators.radial_params
    θ = spharm_θ_grid_uniform(m, nℓ).θ
    nθ = length(θ)
    VWinv = (; V = zeros(ComplexF64, nr, nθ), W = zeros(ComplexF64, nr, nθ))
    V = VWinv.V
    VWinvsh = (; V = zeros(ComplexF64, nr, nℓ), W = zeros(ComplexF64, nr, nℓ))
    F = (; V = zeros(ComplexF64, nr * nℓ), W = zeros(ComplexF64, nr * nℓ), S = zeros(ComplexF64, nr * nℓ))

    eigenfunction_realspace!(VWinv, VWinvsh, F, v, m, operators)

    return (; V, θ)
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

end # module
