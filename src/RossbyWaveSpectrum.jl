module RossbyWaveSpectrum

using BlockArrays
using DelimitedFiles
using Dierckx
using Distributed
using FastGaussQuadrature
using FastSphericalHarmonics
using FastTransforms
using FillArrays
using Folds
using ForwardDiff
using LinearAlgebra
using LinearAlgebra: AbstractTriangular, BLAS
using JLD2
using Kronecker
using Kronecker: KroneckerProduct
using LaTeXStrings
using MKL
using OffsetArrays
using PyCall
using PyPlot
using SphericalHarmonics
using Trapz
using TimerOutputs
using WignerSymbols

export datadir

const SCRATCH = Ref("")
const DATADIR = Ref("")

const ticker = PyNULL()

function __init__()
    SCRATCH[] = get(ENV, "SCRATCH", homedir())
    DATADIR[] = get(ENV, "DATADIR", joinpath(SCRATCH[], "RossbyWaves"))
    copy!(ticker,  pyimport("matplotlib.ticker"))
end

datadir(f) = joinpath(DATADIR[], f)

# Legedre expansion constants
α⁺ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((ℓ-m+1)*(ℓ+m+1)/((2ℓ+1)*(2ℓ+3)))))
α⁻ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((ℓ-m)*(ℓ+m)/((2ℓ-1)*(2ℓ+1)))))

β⁻ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((2ℓ+1)/(2ℓ-1)*(ℓ^2-m^2))))
γ⁺ℓm(ℓ, m) = ℓ * α⁺ℓm(ℓ, m)
γ⁻ℓm(ℓ, m) = ℓ*α⁻ℓm(ℓ, m) - β⁻ℓm(ℓ, m)

# triple integrals involving normalized legendre and associated legendre polynomials
function intPl1mPl20Pl3m(ℓ1,ℓ2,ℓ3,m)
    iseven(ℓ1 + ℓ2 + ℓ3) || return 0.0
    return oftype(0.0, √((2ℓ1+1)*(2ℓ2+1)/(2*(2ℓ3+1))) *
        clebschgordan(Float64,ℓ1,0,ℓ2,0,ℓ3) * clebschgordan(Float64,ℓ1,m,ℓ2,0,ℓ3))
end

function intPl1mP′l20Pl3m(ℓ1,ℓ2,ℓ3,m)
    ℓ2 == 0 && return 0.0
    isodd(ℓ1 + ℓ2 + ℓ3) || return 0.0
    return oftype(0.0, sum(√((2ℓ2+1)*(2n+1))*intPl1mPl20Pl3m(ℓ1,n,ℓ3,m) for n in ℓ2-1:-2:0))
end

function chebyshevnodes(n, a = -1, b = 1)
    nodes = cos.(reverse(pi*((1:n) .- 0.5)./n))
    nodes_scaled = nodes*(b - a)/2 .+ (b + a)/2
    nodes, nodes_scaled
end

function legendretransform!(v::AbstractVector, PC = plan_chebyshevtransform!(v), PC2L = plan_cheb2leg(v))
    v2 = PC * v
    v2 .= PC2L * v2
    return v2
end
function legendretransform(v::AbstractVector, PC = plan_chebyshevtransform(v), PC2L = plan_cheb2leg(v))
    legendretransform!(copy(v), PC, PC2L)
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
        vo[l] *= √(2/(2l+1))
    end
    return v
end

function normalizedlegendretransform!(v::AbstractVector, PC = plan_chebyshevtransform!(v), PC2L = plan_cheb2leg(v))
    v2 = legendretransform!(v, PC, PC2L)
    normalizelegendre!(v2)
    return v2
end
function normalizedlegendretransform(v::AbstractVector, PC = plan_chebyshevtransform(v), PC2L = plan_cheb2leg(v))
    normalizedlegendretransform!(copy(v), PC, PC2L)
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
function chebyshev_normalizedlegendre_transform(A::AbstractMatrix)
    chebyshev_normalizedlegendre_transform!(copy(A))
end

function gausslegendre_theta_grid(ntheta)
    costheta, w = reverse.(gausslegendre(ntheta));
    thetaGL = acos.(costheta)
    (; thetaGL, costheta, w)
end

ntheta_ℓmax(nℓ, m) = 2(nℓ+m-1)

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
    ℓs = range(m, length = nℓ_transform); ℓmax = maximum(ℓs)
    Pcache = SphericalHarmonics.allocate_p(ℓmax)
    for (θind, costhetai) in enumerate(costheta)
        computePlmx!(Pcache, costhetai, ℓmax, m, norm = SphericalHarmonics.Orthonormal())
        for (ℓind, ℓ) in enumerate(ℓs)
            Pℓm = Pcache[(ℓ,m)]
            PLMfwd[ℓind, θind] = Pℓm
            PLMinv[θind, ℓind] = Pℓm
        end
    end
    PLMfwd .*= w'
    return (; PLMfwd, PLMinv)
end
function associatedlegendretransform_matrices(nℓ, m)
    ntheta = ntheta_ℓmax(nℓ, m)
    (; costheta, w) = gausslegendre_theta_grid(ntheta);
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
    Tc = zeros(n, n);
    Tc[:,1] .= 1
    Tc[:,2] = x
    @views for k = 3:n
        @. Tc[:,k] = 2x*Tc[:,k-1] - Tc[:,k-2]
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
    for col in 1:n, row in col+1:2:n
        v = (row-1) * (2 - (col == 1))
        M[row, col] = v
    end
    UpperTriangular(transpose(M))
end

function chebyshev_integrate(y)
    n = length(y)
    sum(yi * sin((2i-1)/2n*pi) * pi/n for (i, yi) in enumerate(y))
end

"""
    constraintmatrix(radial_params)

Evaluate the matrix ``F`` that captures the constriants on the Chebyshev coefficients
of ``V`` and ``W``, such that
```math
F * [V_{11}, ⋯, V_{1k}, ⋯, V_{n1}, ⋯, V_{nk}, W_{11}, ⋯, W_{1k}, ⋯, W_{n1}, ⋯, W_{nk}] = 0.
```
"""
function constraintmatrix(operators)
    (; radial_params) = operators;
    (; nr, r_in, r_out, nℓ) = radial_params;

    nparams = nr*nℓ;

    # Radial constraint
    MVn = zeros(2, nr);
    MVno = OffsetArray(MVn, :, 0:nr-1);
    MWn = zeros(2, nr);
    MWno = OffsetArray(MWn, :, 0:nr-1);
    # constraints on W
    for n in 0:nr-1
        MWno[1, n] = 1
        MWno[2, n] = (-1)^n
    end
    # constraints on V
    Δr = r_out - r_in;
    for n in 0:nr-1
        MVno[1, n] = 2n^2/Δr - 1/r_in
        MVno[2, n] = (-1)^n * (2n^2/Δr - 1/r_out)
    end

    BC = [
        Ones(1, nℓ) ⊗ MVn Zeros(2, nparams)
        Zeros(2, nparams) Ones(1, nℓ) ⊗ MWn
        ]

    ZC = constraintnullspacematrix(BC)
    ZW = constraintnullspacematrix(MWn)

    (; BC, MVn, MWn, ZC, ZW)
end

"""
    constraintnullspacematrix(M)

Given a constraint matrix `M` of size `m×n` that satisfies `Mv=0` for the sought eigenvectors `v`, return a matrix `C` of size `n×(n-m)`
whose columns lie in the null space of `M`, and for an arbitrary `y`, the vector `v = C*y` satisfies the constraint `Mv=0`.
"""
function constraintnullspacematrix(M)
    m, n = size(M)
    @assert n >= m
    _, R = qr(M);
    Q, _ = qr(R')
    return collect(Q)[:, end-(n-m)+1:end]
end

function parameters(nr, nℓ)
    nchebyr = nr;
    r_in = 0.2;
    r_out = 0.985;
    r_mid = (r_in + r_out)/2
    Δr = r_out - r_in
    nparams = nchebyr * nℓ;
    return (; nchebyr, r_in, r_out, Δr, nr, nparams, nℓ, r_mid)
end

include("identitymatrix.jl")
include("forwardinversetransforms.jl")
include("realspectralspace.jl")
include("paddedmatrix.jl")

function chebyshev_forward_inverse(n, boundaries...)
    nodes, r = chebyshevnodes(n, boundaries...);
    Tc = chebyshevpoly(n, nodes);
    Tcfwd = Tc' * 2 /n;
    Tcfwd[1, :] ./= 2
    Tcinv = Tc;
    r, Tcfwd, Tcinv
end

function spherical_harmonic_transform_plan(ntheta)
    # spherical harmonic transform parameters
    lmax = sph_lmax(ntheta);
    θϕ_sh = FastSphericalHarmonics.sph_points(lmax);
    shtemp = zeros(length.(θϕ_sh));
    PSPH2F = plan_sph2fourier(shtemp);
    PSPHA = plan_sph_analysis(shtemp);
    (; shtemp, PSPHA, PSPH2F, θϕ_sh, lmax);
end

function costheta_operator(nℓ, m, pad = nℓ)
    dl = [α⁻ℓm(ℓ,m) for ℓ in m.+(1:nℓ-1+pad)]
    d = zeros(nℓ+pad)
    M = SymTridiagonal(d, dl)'
    PaddedMatrix(M, pad)
end

function sintheta_dtheta_operator(nℓ, m, pad = nℓ)
    dl = [γ⁻ℓm(ℓ,m) for ℓ in m.+(1:nℓ-1+pad)]
    d = zeros(nℓ+pad)
    du = [γ⁺ℓm(ℓ,m) for ℓ in m.+(0:nℓ-2+pad)]
    M = Tridiagonal(dl, d, du)'
    PaddedMatrix(M, pad)
end

function ζfn(r, Binv_params)
    (; c0, c1, radial_params) = Binv_params;
    (; Δr) = radial_params;
    c0 + c1 * Δr /r
end
function ηfn(r, Binv_params)
    (; npol) = Binv_params
    npol * ForwardDiff.derivative(r -> log(ζfn(r, Binv_params)), r)
end
function ρfn(r, Binv_params)
    (; npol) = Binv_params;
    ρ0 = 1 # need to determine this
    # Dr is independent of this, although the Green function depends on this
    # ρ0 * exp(ηfn(r, Binv_params) * r)
    ρ0 * ζfn(r, Binv_params)^npol
end

w1(r, ℓ, r_in, r_out) = r^(ℓ+1)-(r_in^(2ℓ+1))/r^ℓ
w2(r, ℓ, r_in, r_out) = r^(ℓ+1)-(r_out^(2ℓ+1))/r^ℓ
w1w2(r, s, ℓ, r_in, r_out) = w1(r, ℓ, r_in, r_out)*w2(s, ℓ, r_in, r_out)

function greenfn_realspace(r, s, ℓ, Binv_params)
    @assert ℓ >= 0
    (; radial_params) = Binv_params;
    (; r_in, r_out) = radial_params;
    @assert r_out > r_in
    @assert r_in <= r <= r_out
    @assert r_in <= s <= r_out
    Wronskian = (2ℓ+1)*(r_out^(2ℓ+1) - r_in^(2ℓ+1))
    T = (s <= r) ? w1w2(s, r, ℓ, r_in, r_out) : w1w2(r, s, ℓ, r_in, r_out)
    prefactor = 1/ (ρfn(r, Binv_params) * ρfn(s, Binv_params))
    prefactor * (T / Wronskian)
end

function greenfn_realspace_F2(r, s, ℓ, Binv_params)
    prefactor = 1/ ρfn(s, Binv_params)^2
    prefactor * greenfn_realspace_ρs2F2(r, s, ℓ, Binv_params)
end
function greenfn_realspace_ρs2F2(r, s, ℓ, Binv_params)
    @assert ℓ >= 0
    (; radial_params) = Binv_params;
    (; r_in, r_out) = radial_params;
    @assert r_out > r_in
    @assert r_in <= r <= r_out
    @assert r_in <= s <= r_out
    Wronskian = (2ℓ+1)*(r_out^(2ℓ+1) - r_in^(2ℓ+1))
    T = (s <= r) ? w1w2(s, r, ℓ, r_in, r_out) : w1w2(r, s, ℓ, r_in, r_out)
    (T / Wronskian)
end
function greenfn_realspace_F(r, s, ℓ, Binv_params)
    G = greenfn_realspace(r, s, ℓ, Binv_params)
    ρfn(r, Binv_params)/ρfn(s, Binv_params) * G
end

∂r(f, r, args...) = ForwardDiff.derivative(r -> f(r, args...), r)
function ∂2r(f, r, s, ℓ, Binv_params)
    if r != s
        return ∂r(∂r(f), r, s, ℓ, Binv_params)
    else
        # finite difference
        rleft = prevfloat(r)
        rright = nextfloat(r)
        Δr = rright - rleft
        return (∂r(f)(rright, s, ℓ, Binv_params) - ∂r(f)(rleft, s, ℓ, Binv_params))/Δr
    end
end

∂s(f, r, s, args...) = ForwardDiff.derivative(s -> f(r, s, args...), s)
function ∂2s(f, r, s, ℓ, Binv_params)
    if r != s
        return ∂s(∂s(f), r, s, ℓ, Binv_params)
    else
        # finite difference
        sleft = prevfloat(s)
        sright = nextfloat(s)
        Δs = sright - sleft
        return (∂s(f)(r, sright, ℓ, Binv_params) - ∂s(f)(r, sleft, ℓ, Binv_params))/Δs
    end
end

for (var, fD, fd) in [(:r, :Dr, :∂r), (:r, :D2r, :∂2r), (:s, :Ds, :∂s), (:s, :D2s, :∂2s)]
    @eval function $fD(f, r, s, ℓ, Binv_params)
        ρf = (r, s, ℓ, Binv_params) -> ρfn($var, Binv_params) * f(r, s, ℓ, Binv_params)
        1/ρfn($var, Binv_params) * $fd(ρf, r, s, ℓ, Binv_params)
    end
end
function Bℓ(f, r, s, ℓ, Binv_params)
    D2r(f, r, s, ℓ, Binv_params) - ℓ*(ℓ+1)/r^2 * f(r, s, ℓ, Binv_params)
end
function Bℓadj(f, r, s, ℓ, Binv_params)
    D2s(f, r, s, ℓ, Binv_params) - ℓ*(ℓ+1)/s^2 * f(r, s, ℓ, Binv_params)
end
function Cℓ(f, r, s, ℓ, Binv_params)
    Dr(f, r, s, ℓ, Binv_params) - (ℓ*(ℓ+1)/r) * f(r, s, ℓ, Binv_params)
end
function Cℓadj(f, r, s, ℓ, Binv_params)
   -Ds(f, r, s, ℓ, Binv_params) - (ℓ*(ℓ+1)/s) * f(r, s, ℓ, Binv_params)
end

for f in [:∂r, :∂s, :∂2r, :∂2s, :Dr, :Ds, :D2r, :D2s, :Bℓ, :Bℓadj, :Cℓ, :Cℓadj]
    @eval $f(g) = (t...) -> $f(g, t...)
end

function greenfn_realspace_apply(f, ℓ, operators)
    (; Binv_params, coordinates) = operators;
    (; r) = coordinates;
    f.(r, r', ℓ, Ref(Binv_params))
end
function greenfn_cheby_apply(f, ℓ, operators)
    (; transforms) = operators;
    (; Tcrfwd, Tcrinv) = transforms;
    G = greenfn_realspace_apply(f, ℓ, operators)
    Tcrfwd * G * Tcrinv
end

function greenfn_realspace_analytical(ℓ, operators)
    (; radial_params, Binv_params, coordinates) = operators;
    (; r) = coordinates;
    (; Δr) = radial_params;
    ρs = ρfn.(r, (Binv_params,))
    G = greenfn_realspace_apply(greenfn_realspace, ℓ, operators)
    G .*= Δr/2 .* (ρs').^2
    return G
end
function greenfn_cheby_analytical(ℓ, operators)
    (; transforms) = operators;
    (; Tcrfwd, Tcrinv) = transforms;
    G = greenfn_realspace_analytical(ℓ, operators)
    Tcrfwd * G * Tcrinv
end
function greenfn_cheby_numerical(ℓ, operators)
    (; scratch, diff_operators, rad_terms) = operators;
    (; D2Dr2) = diff_operators;
    (; onebyr2_cheby) = rad_terms;
    (; Bℓ) = scratch;
    (; ZW) = constraintmatrix(operators);
    @. Bℓ = D2Dr2 - ℓ*(ℓ+1) * onebyr2_cheby
    ZW * ((ZW' * Bℓ * ZW) \ ZW')
end
function greenfn_realspace_numerical2(ℓ, operators)
    (; scratch, diff_operators, rad_terms, transforms) = operators;
    (; D2Dr2) = diff_operators;
    (; onebyr2_cheby) = rad_terms;
    (; Bℓ) = scratch;
    (; Tcrinv, Tcrfwd) = transforms;
    @. Bℓ = D2Dr2 - ℓ*(ℓ+1) * onebyr2_cheby;
    Bℓ_realspace = Tcrinv * Bℓ * Tcrfwd;
    G = pinv(Bℓ_realspace[2:end-1, 2:end-1])
    A = zero(Bℓ_realspace)
    A[2:end-1, 2:end-1] .= G
    return A
end
function greenfn_cheby_numerical2(ℓ, operators)
    (; transforms) = operators;
    (; Tcrinv, Tcrfwd) = transforms;
    G = greenfn_realspace_numerical2(ℓ, operators)
    Tcrfwd * G * Tcrinv
end

function radial_operators(nr, nℓ)
    radial_params = parameters(nr, nℓ);
    (; r_in, r_out, Δr, nchebyr) = radial_params;
    r, Tcrfwd, Tcrinv = chebyshev_forward_inverse(nr, r_in, r_out);
    r_cheby = Tcrfwd * Diagonal(r) * Tcrinv

    FTcrinv = factorize(Tcrinv);

    ddr = chebyshevderiv(nr) * (2/Δr);

    onebyr = 1 ./r;
    onebyr_cheby = Tcrfwd * Diagonal(onebyr) * Tcrinv;
    onebyr2_cheby = Tcrfwd * Diagonal(onebyr)^2 * Tcrinv;

    diml = 696e6;
    dimT = 1/(2 * pi * 453e-9);
    # dimrho = 0.2;
    # dimp = dimrho * diml^2/dimT^2;
    # dimtemp = 6000;
    Nrho = 3;
    npol = 2.6;
    betapol = r_in / r_out; rhoc = 2e-1;
    gravsurf=278; pc = 1e10;
    # R = 8.314 * dimrho * dimtemp/dimp;
    # G = 6.67e-11; Msol = 2e30;
    zeta0 = (betapol + 1)/(betapol * exp(Nrho/npol) + 1);
    # zetai = (1 + betapol - zeta0)/betapol;
    # Cp = (npol + 1) * R; gamma_ind = (npol+1)/npol; Cv = npol * R;
    c0 = (2 * zeta0 - betapol - 1)/(1-betapol);
    #^c1 = G * Msol * rhoc/((npol+1)*pc * Tc * (r_out-r_in));
    c1 = (1-zeta0)*(betapol + 1)/(1-betapol)^2;
    Binv_params = (; c0, c1, npol, radial_params);
    zeta = ζfn.(r, Ref(Binv_params));
    dzeta_dr = @. -c1 * Δr /r^2;
    # d2zeta_dr2 = @. 2 * c1 * Δr /r^3;
    grav = @. -pc / rhoc * (npol+1) * dzeta_dr;
    pc = gravsurf .* pc/minimum(grav);
    grav = grav .* gravsurf ./ minimum(grav) * dimT^2/diml;
    ηρ = @. npol/zeta*dzeta_dr;

    # drhrho = @. (- npol/zeta^2 * (dzeta_dr)^2 + npol/zeta * d2zeta_dr2);
    ηρ_cheby = Tcrfwd * Diagonal(ηρ) * Tcrinv;

    DDr = ddr + ηρ_cheby;
    rDDr = r_cheby * DDr
    rddr = r_cheby * ddr;
    D2Dr2 = DDr^2;
    DDr_minus_2byr = DDr - 2*onebyr_cheby;

    # scratch matrices
    # B = D2Dr2 - ℓ(ℓ+1)/r^2
    Bℓ = zero(D2Dr2);
    Cℓ′ = zero(DDr);
    Gℓ = zero(DDr);

    Ir = IdentityMatrix(nchebyr);
    Iℓ = PaddedMatrix(IdentityMatrix(2nℓ), nℓ);

    identities = (; Ir, Iℓ);
    coordinates = (; r, r_cheby);
    transforms = (; Tcrfwd, Tcrinv, FTcrinv);
    rad_terms = (; onebyr, onebyr_cheby, ηρ, ηρ_cheby, onebyr2_cheby);
    diff_operators = (; DDr, D2Dr2, DDr_minus_2byr, rDDr, rddr, ddr);
    scratch = (; Bℓ, Cℓ′, Gℓ);

    (; rad_terms, diff_operators, transforms, coordinates, radial_params, scratch, identities, Binv_params)
end

function twoΩcrossv(nr, nℓ, m; operators = radial_operators(nr, nℓ))
    (; radial_params) = operators;
    (; rad_terms, diff_operators, scratch) = operators;

    (; nchebyr) = radial_params;

    (; DDr, D2Dr2, DDr_minus_2byr) = diff_operators;
    (; onebyr_cheby, onebyr2_cheby) = rad_terms;
    (; Bℓ, Cℓ′, Gℓ) = scratch;
    TWVℓ′ = similar(Cℓ′);
    TVWℓ′ = similar(Cℓ′);
    GℓC1 = similar(Cℓ′);
    GℓCℓ′ = similar(Cℓ′);

    nparams = nchebyr * nℓ;
    ℓmin = m;
    ℓs = range(ℓmin, length = nℓ);

    A = 2m * Diagonal(@. 1/(ℓs*(ℓs + 1))) ⊗ I(nchebyr);
    D = A;

    B = zeros(nparams, nparams);
    C = zeros(nparams, nparams);

    C1 = DDr_minus_2byr;

    cosθ = OffsetArray(costheta_operator(nℓ, m), ℓs, ℓs);
    sinθdθ = OffsetArray(sintheta_dtheta_operator(nℓ, m), ℓs, ℓs);

    for ℓ in ℓs
        @. Bℓ = D2Dr2 - ℓ*(ℓ+1) * onebyr2_cheby;

        ABℓ′top = (ℓ - minimum(m)) * nchebyr + 1;
        ACℓ′vertinds = range(ABℓ′top, length = nr);

        # numerical green function
        Gℓ = greenfn_cheby_numerical2(ℓ, operators);
        GℓC1 .= Gℓ * C1;

        for ℓ′ in ℓ-1:2:ℓ+1
            ℓ′ in ℓs || continue
            @. Cℓ′ = DDr - ℓ′*(ℓ′+1) * onebyr_cheby;
            GℓCℓ′ .= Gℓ * Cℓ′;
            @. TVWℓ′ = -2/(ℓ*(ℓ+1))*(ℓ′*(ℓ′+1)*C1*cosθ[ℓ, ℓ′] + Cℓ′*sinθdθ[ℓ, ℓ′]);
            @. TWVℓ′ = -2/(ℓ*(ℓ+1))*(ℓ′*(ℓ′+1)*GℓC1*cosθ[ℓ, ℓ′] + GℓCℓ′*sinθdθ[ℓ, ℓ′]);

            ABℓ′left = (ℓ′ - minimum(m)) * nchebyr + 1
            ACℓ′horinds = range(ABℓ′left, length = nr);
            ACℓℓ′inds = CartesianIndices((ACℓ′vertinds, ACℓ′horinds));

            B[ACℓℓ′inds] .= TVWℓ′
            C[ACℓℓ′inds] .= TWVℓ′
        end
    end

    [A  B
     C  D];
end

function interp1d(xin, z, xout; s = 0.0)
    spline = Spline1D(xin, z; s)
    spline(xout)
end

function interp2d(xin, yin, z, xout, yout; s = 0.0)
    spline = Spline2D(xin, yin, z; s)
    evalgrid(spline, xout, yout)
end

function read_angular_velocity(operators, thetaGL;
        test = false, ΔΩscale = 1, r_ΔΩcutoff = 0.4)

    (; coordinates) = operators;
    (; r) = coordinates;

    parentdir = dirname(@__DIR__)
    r_ΔΩ_raw = vec(readdlm(joinpath(parentdir, "rmesh.orig"))::Matrix{Float64});
    r_ΔΩ_raw = r_ΔΩ_raw[1:4:end];

    ΔΩ_raw = readdlm(joinpath(parentdir, "rot2d.hmiv72d.ave"))::Matrix{Float64};
    ΔΩ_raw = [ΔΩ_raw reverse(ΔΩ_raw[:,2:end], dims = 2)];
    ΔΩ_raw = (ΔΩ_raw .- 453.1)./453.1;

    nθ = size(ΔΩ_raw,2);
    lats_raw = LinRange(0, pi, nθ)

    ΔΩ_r_thetaGL = permutedims(interp2d(lats_raw, r_ΔΩ_raw, ΔΩ_raw', thetaGL, r))
    theta_cheby = acos.(chebyshevpoints(length(lats_raw)))
    ΔΩ_r_rawtheta = permutedims(interp2d(lats_raw, r_ΔΩ_raw, ΔΩ_raw', lats_raw, r))
    ΔΩ_r_chebytheta = permutedims(interp2d(lats_raw, r_ΔΩ_raw, ΔΩ_raw', theta_cheby, r))

    # testing
    if test
        ΔΩ_r_chebytheta .= 1
        ΔΩ_r_thetaGL .= 1
        r_min, r_max = extrema(r)
        mask = @. 0.5 - 1/pi * atan(10*(r - r_ΔΩcutoff)/abs((r - r_min)*(r - r_max)));
        mask .*= (ΔΩscale - 1) / mask[1];
        ΔΩ_r_chebytheta .*= mask
        ΔΩ_r_thetaGL .*= mask
    end

    ΔΩ_r_Legendre = normalizedlegendretransform2(ΔΩ_r_chebytheta);

    (; lats_raw, ΔΩ_r_thetaGL, ΔΩ_r_rawtheta, ΔΩ_r_chebytheta, ΔΩ_r_Legendre)
end

function legendre_to_associatedlegendre(vℓ, ℓ1, ℓ2, m)
    vℓo = OffsetArray(vℓ, OffsetArrays.Origin(0))
    sum(vℓi * intPl1mPl20Pl3m(ℓ1, ℓ, ℓ2, m) for (ℓ, vℓi) in pairs(vℓo))
end

function dlegendre_to_associatedlegendre(vℓ, ℓ1, ℓ2, m)
    vℓo = OffsetArray(vℓ, OffsetArrays.Origin(0))
    sum(vℓi * intPl1mP′l20Pl3m(ℓ1, ℓ, ℓ2, m) for (ℓ, vℓi) in pairs(vℓo))
end

velocity_from_angular_velocity(Ω, r, sintheta) = Ω .* sintheta' .* r;

function invsinθ_dθ_ωΩr_operator(ΔΩ_r_Legendre, m, operators)
    (; transforms, radial_params) = operators;
    (; nparams, nℓ) = radial_params;
    (; Tcrfwd, Tcrinv) = transforms;
    ℓs = range(m, length = 2nℓ);
    ΔΩ_r_Legendre_of = OffsetArray(ΔΩ_r_Legendre, :, 0:size(ΔΩ_r_Legendre, 2)-1);
    invsinθ_dθ_ωΩr = zeros(2nparams, 2nparams);
    for ℓ in axes(ΔΩ_r_Legendre_of, 2)
        ΔΩ_r_ℓ = Diagonal(@view ΔΩ_r_Legendre_of[:, ℓ])
        ΔΩ_cheby_ℓ = Tcrfwd * ΔΩ_r_ℓ * Tcrinv
        nchebyr1, nchebyr2 = size(ΔΩ_cheby_ℓ)
        for (ℓ2ind, ℓ2) in enumerate(ℓs), (ℓ1ind, ℓ1) in enumerate(ℓs)
            Tℓ1ℓℓ2 = -ℓ*(ℓ-1)*intPl1mPl20Pl3m(ℓ1, ℓ, ℓ2, m) - 2*√((2ℓ+1)/(2ℓ+3))*intPl1mP′l20Pl3m(ℓ1, ℓ+1, ℓ2, m)
            iszero(Tℓ1ℓℓ2) && continue

            rowinds = (ℓ1ind-1)*nchebyr1 + 1:ℓ1ind*nchebyr1
            colinds = (ℓ2ind-1)*nchebyr2 + 1:ℓ2ind*nchebyr2
            ℓ2ℓ1inds = CartesianIndices((rowinds, colinds))
            @. invsinθ_dθ_ωΩr[ℓ2ℓ1inds] += ΔΩ_cheby_ℓ * Tℓ1ℓℓ2
        end
    end
    return PaddedMatrix(invsinθ_dθ_ωΩr, nparams)
end

function apply_radial_operator(op_cheby, ΔΩ_r_ℓℓ′, m, operators)
    (; radial_params, transforms) = operators;
    (; nℓ, nr, nparams) = radial_params;
    ℓs = range(m, length = 2nℓ);
    (; Tcrfwd, Tcrinv) = transforms;
    op_realspace = Tcrinv * op_cheby * Tcrfwd
    op_ΔΩ_kk′_ℓℓ′ = zeros(2nparams, 2nparams);
    for ℓ2ind in eachindex(ℓs), ℓ1ind in eachindex(ℓs)
        rowinds = (ℓ1ind-1)*nr + 1:ℓ1ind*nr
        colinds = (ℓ2ind-1)*nr + 1:ℓ2ind*nr
        inds = CartesianIndices((rowinds, colinds))
        D = @view parent(ΔΩ_r_ℓℓ′)[inds]
        fr = @view D[diagind(D)]
        op_ΔΩ_kk′_ℓℓ′[inds] = Tcrfwd * Diagonal(op_realspace * fr) * Tcrinv
    end
    return PaddedMatrix(op_ΔΩ_kk′_ℓℓ′, nparams)
end

function vorticity_terms(ΔΩ_kk′_ℓℓ′, ΔΩ_r_ℓℓ′, ΔΩ_r_Legendre, m, Cosθ, Sinθdθ, operators)
    (; identities, coordinates, diff_operators) = operators;
    (; Ir, Iℓ) = identities;
    (; r_cheby) = coordinates;
    (; ddr) = diff_operators;

    Sin²θ = I - Cosθ^2;
    CosθI = Cosθ ⊗ Ir;
    Sin²θI = Sin²θ ⊗ Ir;
    SinθdθI = Sinθdθ ⊗ Ir;

    sinθ_dθ_ΔΩ_kk′_ℓℓ′ = SinθdθI*ΔΩ_kk′_ℓℓ′ - ΔΩ_kk′_ℓℓ′*SinθdθI;

    ωΩr = 2CosθI*ΔΩ_kk′_ℓℓ′ + sinθ_dθ_ΔΩ_kk′_ℓℓ′;
    ddr_ΔΩ_kk′_ℓℓ′ = apply_radial_operator(ddr, ΔΩ_r_ℓℓ′, m, operators);
    drωΩr = (2CosθI + SinθdθI)*ddr_ΔΩ_kk′_ℓℓ′ - ddr_ΔΩ_kk′_ℓℓ′*SinθdθI;

    rddr_ΔΩ_kk′_ℓℓ′ = (Iℓ ⊗ r_cheby) * ddr_ΔΩ_kk′_ℓℓ′;
    ωΩθ_by_sinθ = -2ΔΩ_kk′_ℓℓ′ - rddr_ΔΩ_kk′_ℓℓ′;

    sinθ_ωΩθ =  Sin²θI * ωΩθ_by_sinθ;
    dθ_ωΩθ = (CosθI + SinθdθI)*ωΩθ_by_sinθ - ωΩθ_by_sinθ*SinθdθI;
    cotθ_ωΩθ = CosθI * ωΩθ_by_sinθ;

    invsinθ_dθ_ωΩr = invsinθ_dθ_ωΩr_operator(ΔΩ_r_Legendre, m, operators);

    (; ωΩr, ωΩθ_by_sinθ, sinθ_ωΩθ, invsinθ_dθ_ωΩr, dθ_ωΩθ, drωΩr, cotθ_ωΩθ, sinθ_dθ_ΔΩ_kk′_ℓℓ′)
end

function ΔΩ_terms(ΔΩ_r_ℓℓ′, m, operators)
    (; diff_operators, coordinates) = operators;
    (; r_cheby) = coordinates;
    (; DDr, rddr) = diff_operators;
    oneminrddr_2plusrddr_ΔΩ_kk′_ℓℓ′ = apply_radial_operator((I - rddr)*(2I + rddr), ΔΩ_r_ℓℓ′, m, operators)
    r2DDr_ΔΩ_kk′_ℓℓ′ = apply_radial_operator(r_cheby^2*DDr, ΔΩ_r_ℓℓ′, m, operators)
    (; oneminrddr_2plusrddr_ΔΩ_kk′_ℓℓ′, r2DDr_ΔΩ_kk′_ℓℓ′)
end

function curl_curl_matrix_terms(m, operators, ω_terms, ΔΩ_kk′_ℓℓ′, ΔΩ_r_ℓℓ′, Cosθ, Sinθdθ, ℓs)
    (; radial_params, rad_terms, diff_operators, identities, scratch) = operators;
    (; ZW) = constraintmatrix(operators);

    (; Ir, Iℓ) = identities;
    (; nchebyr, nℓ, nr) = radial_params;
    (; DDr, ddr, D2Dr2) = diff_operators;
    (; onebyr_cheby, onebyr2_cheby, ηρ_cheby) = rad_terms;
    (; Bℓ) = scratch;

    Sin²θ = I - Cosθ^2
    Sin⁻²θ = PaddedArray(inv(Matrix(parent(Sin²θ))), nℓ)

    ℓℓp1 = ℓs.*(ℓs.+1);
    ℓℓp1_diag = PaddedArray(Diagonal(ℓℓp1), nℓ)

    (; ωΩr, ωΩθ_by_sinθ, dθ_ωΩθ, cotθ_ωΩθ, sinθ_dθ_ΔΩ_kk′_ℓℓ′) = ω_terms;

    (; oneminrddr_2plusrddr_ΔΩ_kk′_ℓℓ′, r2DDr_ΔΩ_kk′_ℓℓ′) = ΔΩ_terms(ΔΩ_r_ℓℓ′, m, operators);

    im_vr_imW = ℓℓp1_diag ⊗ onebyr2_cheby;

    im_rsinθvθ_V = -m*(Iℓ ⊗ Ir);
    im_rsinθvθ_imW = Sinθdθ ⊗ DDr;

    rsinθvϕ_V = -Sinθdθ ⊗ Ir;
    rsinθvϕ_imW = m*(Iℓ ⊗ DDr);

    im_rsinθ_ωϕ_V = -m*(Iℓ ⊗ ddr);
    im_rsinθ_ωϕ_imW = Sinθdθ ⊗ (ddr*DDr) - (Sinθdθ * ℓℓp1_diag) ⊗ onebyr2_cheby;

    ωr_V = ℓℓp1_diag ⊗ onebyr2_cheby;

    rsinθ_ωθ_V = Sinθdθ ⊗ ddr;
    rsinθ_ωθ_imW = m*(ℓℓp1_diag ⊗ onebyr2_cheby - Iℓ ⊗ (ddr*DDr));

    # curl curl rho grad u
    T_imW_V1 = -(ΔΩ_kk′_ℓℓ′ * ((Sinθdθ * ℓℓp1_diag) ⊗ ηρ_cheby));
    T_imW_imW1 = m*(ΔΩ_kk′_ℓℓ′ * (ℓℓp1_diag ⊗ (ηρ_cheby * DDr)));

    im_sinθ_r_by_ρ_curl_ρu_cross_ω_θ_V =
        (ωΩθ_by_sinθ * ((Sinθdθ - Cosθ) ⊗ onebyr_cheby) + ωΩr * (Iℓ ⊗ (DDr - onebyr_cheby)) - (dθ_ωΩθ + ωΩr) * (Iℓ ⊗ onebyr_cheby)) * im_rsinθvθ_V +
        m*ΔΩ_kk′_ℓℓ′ * rsinθ_ωθ_V;

    im_sinθ_r_by_ρ_curl_ρu_cross_ω_θ_imW =
        (ωΩθ_by_sinθ * ((Sinθdθ - Cosθ) ⊗ onebyr_cheby) + ωΩr * (Iℓ ⊗ (DDr - onebyr_cheby)) - (dθ_ωΩθ + ωΩr) * (Iℓ ⊗ onebyr_cheby)) * im_rsinθvθ_imW -
        (Sin²θ ⊗ Ir) * oneminrddr_2plusrddr_ΔΩ_kk′_ℓℓ′ * im_vr_imW +
        m*ΔΩ_kk′_ℓℓ′ * rsinθ_ωθ_imW;

    sinθ_r_by_ρ_curl_ρu_cross_ω_ϕ_V =
        (ωΩθ_by_sinθ * ((Sinθdθ - Cosθ) ⊗ onebyr_cheby) + ωΩr * (Iℓ ⊗ (DDr - onebyr_cheby)) - (cotθ_ωΩθ + ωΩr) * (Iℓ ⊗ onebyr_cheby)) * rsinθvϕ_V +
        (Sin²θ ⊗ Ir) * r2DDr_ΔΩ_kk′_ℓℓ′ * ωr_V +
        sinθ_dθ_ΔΩ_kk′_ℓℓ′ * rsinθ_ωθ_V +
        -m*ΔΩ_kk′_ℓℓ′ * im_rsinθ_ωϕ_V;

    sinθ_r_by_ρ_curl_ρu_cross_ω_ϕ_imW =
        (ωΩθ_by_sinθ * ((Sinθdθ - Cosθ) ⊗ onebyr_cheby) + ωΩr * (Iℓ ⊗ (DDr - onebyr_cheby)) - (cotθ_ωΩθ + ωΩr) * (Iℓ ⊗ onebyr_cheby)) * rsinθvϕ_imW +
        sinθ_dθ_ΔΩ_kk′_ℓℓ′ * rsinθ_ωθ_imW +
        -m*ΔΩ_kk′_ℓℓ′ * im_rsinθ_ωϕ_imW;

    # curl curl rho u cross ω
    Sin²θ_T_imW_V2 = (Sinθdθ ⊗ Ir) * sinθ_r_by_ρ_curl_ρu_cross_ω_ϕ_V - m*im_sinθ_r_by_ρ_curl_ρu_cross_ω_θ_V;
    T_imW_V2 = (Sin⁻²θ ⊗ Ir) * Sin²θ_T_imW_V2;
    Sin²θ_T_imW_imW2 = (Sinθdθ ⊗ Ir) * sinθ_r_by_ρ_curl_ρu_cross_ω_ϕ_imW - m*im_sinθ_r_by_ρ_curl_ρu_cross_ω_θ_imW;
    T_imW_imW2 = (Sin⁻²θ ⊗ Ir) * Sin²θ_T_imW_imW2;

    T_imW_V_sum = T_imW_V1 + T_imW_V2;
    T_imW_V = Matrix(T_imW_V_sum);
    T_imW_imW_sum = T_imW_imW1 + T_imW_imW2;
    T_imW_imW = Matrix(T_imW_imW_sum);

    for ℓ in range(m, length = nℓ)
        @. Bℓ = -ℓ*(ℓ+1) * (D2Dr2 - ℓ*(ℓ+1) * onebyr2_cheby)
        B2 = ZW' * Bℓ * ZW;
        F = lu!(B2)
        ABℓ′top = (ℓ - minimum(m)) * nchebyr + 1;
        ACℓ′vertinds = range(ABℓ′top, length = nr);

        for ℓ′ in range(m, length = nℓ)
            ABℓ′left = (ℓ′ - minimum(m)) * nchebyr + 1
            ACℓ′horinds = range(ABℓ′left, length = nr);
            ACℓℓ′inds = CartesianIndices((ACℓ′vertinds, ACℓ′horinds));
            A = @view T_imW_V[ACℓℓ′inds]
            A .= ZW * (F \ (ZW' * A))
            A = @view T_imW_imW[ACℓℓ′inds]
            A .= ZW * (F \ (ZW' * A))
        end
    end

    (; T_imW_V, T_imW_imW, T_imW_V1, T_imW_imW1, T_imW_V2, T_imW_imW2, T_imW_V_sum, T_imW_imW_sum,
    im_sinθ_r_by_ρ_curl_ρu_cross_ω_θ_V, im_sinθ_r_by_ρ_curl_ρu_cross_ω_θ_imW, sinθ_r_by_ρ_curl_ρu_cross_ω_ϕ_V, sinθ_r_by_ρ_curl_ρu_cross_ω_ϕ_imW,
    Sin²θ_T_imW_V2, Sin²θ_T_imW_imW2)
end

function diffrotterms(nr, nℓ, m; operators = radial_operators(nr, nℓ), thetaop = theta_operators(nℓ, m), test = false)
    (; radial_params, rad_terms, diff_operators, transforms, identities) = operators;

    (; Ir, Iℓ) = identities;
    (; nparams) = radial_params;
    (; Tcrfwd, Tcrinv) = transforms;
    (; DDr, DDr_minus_2byr) = diff_operators;
    (; onebyr_cheby) = rad_terms;
    (; thetaGL, PLMfwd, PLMinv) = thetaop;
    (; nℓ, nr, nparams) = radial_params;

    fullfwd = kron(PLMfwd, Tcrfwd);
    fullinv = kron(PLMinv, Tcrinv);

    ℓs = range(m, length = 2nℓ);
    ℓℓp1 = ℓs.*(ℓs.+1);

    ℓℓp1_diag = PaddedMatrix(Diagonal(ℓℓp1), nℓ);
    invℓℓp1_diag = PaddedMatrix(Diagonal(1 ./ ℓℓp1), nℓ);

    Cosθ = costheta_operator(nℓ, m);
    Sinθdθ = sintheta_dtheta_operator(nℓ, m);

    (; ΔΩ_r_thetaGL, ΔΩ_r_Legendre) = read_angular_velocity(operators, thetaGL, test = test);
    ΔΩ_r_ℓℓ′ = PaddedMatrix((PLMfwd ⊗ Ir) * Diagonal(vec(ΔΩ_r_thetaGL)) * (PLMinv ⊗ Ir), nparams);
    ΔΩ_kk′_ℓℓ′ = PaddedMatrix(fullfwd * Diagonal(vec(ΔΩ_r_thetaGL)) * fullinv, nparams);

    ω_terms = vorticity_terms(ΔΩ_kk′_ℓℓ′, ΔΩ_r_ℓℓ′, ΔΩ_r_Legendre, m, Cosθ, Sinθdθ, operators);
    (; ωΩr, ωΩθ_by_sinθ, invsinθ_dθ_ωΩr, drωΩr) = ω_terms;

    T_V_V = -m*(invℓℓp1_diag ⊗ Ir)*(ΔΩ_kk′_ℓℓ′ * (ℓℓp1_diag ⊗ Ir) + invsinθ_dθ_ωΩr);
    T_V_imW = (invℓℓp1_diag ⊗ Ir) *(
        -(ωΩr * (Iℓ ⊗ DDr_minus_2byr) - drωΩr) * (ℓℓp1_diag ⊗ Ir) +
        invsinθ_dθ_ωΩr * (Sinθdθ ⊗ DDr) -
        ωΩθ_by_sinθ*((Sinθdθ * ℓℓp1_diag) ⊗ onebyr_cheby)
        );

    (; T_imW_V, T_imW_imW) = curl_curl_matrix_terms(m, operators, ω_terms, ΔΩ_kk′_ℓℓ′, ΔΩ_r_ℓℓ′, Cosθ, Sinθdθ, ℓs);

    [T_V_V  T_V_imW
    T_imW_V T_imW_imW]
end

function diffrotterms_constantΩ(nr, nℓ, m; operators = radial_operators(nr, nℓ), thetaop = theta_operators(nℓ, m), test = false)
    (; radial_params) = operators;
    (; rad_terms, diff_operators, scratch) = operators;

    (; nchebyr) = radial_params;

    (; DDr, D2Dr2, DDr_minus_2byr) = diff_operators;
    (; onebyr_cheby, onebyr2_cheby, ηρ_cheby) = rad_terms;
    (; Bℓ, Cℓ′, Gℓ) = scratch;
    TWVℓ′ = similar(Cℓ′);
    TVWℓ′ = similar(Cℓ′);
    GℓCη = similar(Cℓ′);
    GℓCℓ′ = similar(Cℓ′);

    nparams = nchebyr * nℓ;
    ℓmin = m;
    ℓs = range(ℓmin, length = nℓ);

    A = m * Diagonal(@. 2/(ℓs*(ℓs + 1))  - 1) ⊗ I(nchebyr);
    D = A;

    B = zeros(nparams, nparams);
    C = zeros(nparams, nparams);

    C1 = DDr_minus_2byr;
    Cη = DDr + ηρ_cheby;

    cosθ = OffsetArray(costheta_operator(nℓ, m), ℓs, ℓs);
    sinθdθ = OffsetArray(sintheta_dtheta_operator(nℓ, m), ℓs, ℓs);

    for ℓ in ℓs
        @. Bℓ = D2Dr2 - ℓ*(ℓ+1) * onebyr2_cheby

        ABℓ′top = (ℓ - minimum(m)) * nchebyr + 1;
        ACℓ′vertinds = range(ABℓ′top, length = nr);

        # numerical green function
        Gℓ = greenfn_cheby_numerical2(ℓ, operators)
        GℓCη .= Gℓ * Cη

        for ℓ′ in ℓ-1:2:ℓ+1
            ℓ′ in ℓs || continue
            @. Cℓ′ = DDr - ℓ′*(ℓ′+1) * onebyr_cheby
            GℓCℓ′ .= Gℓ * Cℓ′
            @. TVWℓ′ = -2/(ℓ*(ℓ+1))*(ℓ′*(ℓ′+1)*C1*cosθ[ℓ, ℓ′] + Cℓ′*sinθdθ[ℓ, ℓ′])
            @. TWVℓ′ = -2/(ℓ*(ℓ+1))*(ℓ′*(ℓ′+1)*GℓCη*cosθ[ℓ, ℓ′] + GℓCℓ′*sinθdθ[ℓ, ℓ′])

            ABℓ′left = (ℓ′ - minimum(m)) * nchebyr + 1
            ACℓ′horinds = range(ABℓ′left, length = nr);
            ACℓℓ′inds = CartesianIndices((ACℓ′vertinds, ACℓ′horinds));

            B[ACℓℓ′inds] .= TVWℓ′
            C[ACℓℓ′inds] .= TWVℓ′
        end
    end

    [A  B
     C D]
end

function constrained_eigensystem(M, operators)
    (; ZC) = constraintmatrix(operators);
    M_constrained = ZC'*M*ZC
    λ::Vector{ComplexF64}, w::Matrix{ComplexF64} = eigen!(M_constrained);
    v = ZC*w
    λ, v, M
end

function uniform_rotation_spectrum(nr, nℓ, m; operators = radial_operators(nr, nℓ), kw...)
    M = twoΩcrossv(nr, nℓ, m; operators);
    constrained_eigensystem(M, operators)
end

function real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop)
    (; PLMfwd, PLMinv) = thetaop;
    (; transforms, radial_params) = operators;
    (; nℓ, nchebyr) = radial_params;
    (; Tcrfwd, Tcrinv) = transforms;
    pad = nchebyr*nℓ
    PaddedMatrix((PLMfwd ⊗ Tcrfwd) * Diagonal(vec(ΔΩ_r_thetaGL)) * (PLMinv ⊗ Tcrinv), pad)
end

function real_to_r_assocleg(ΔΩ_r_thetaGL, operators, thetaop)
    (; PLMfwd, PLMinv) = thetaop;
    (; radial_params) = operators;
    (; nℓ, nchebyr) = radial_params;
    Ir = IdentityMatrix(nchebyr)
    pad = nchebyr*nℓ
    PaddedMatrix((PLMfwd ⊗ Ir) * Diagonal(vec(ΔΩ_r_thetaGL)) * (PLMinv ⊗ Ir), pad)
end

function differential_rotation_spectrum(nr, nℓ, m;
        operators = radial_operators(nr, nℓ),
        test = false, kw...)

    uniform_rot_operators = twoΩcrossv(nr, nℓ, m; operators);
    diff_rot_operators = diffrotterms(nr, nℓ, m; operators, test = test);

    M = uniform_rot_operators + diff_rot_operators;

    constrained_eigensystem(M, operators)
end

function differential_rotation_spectrum_constantΩ(nr, nℓ, m; operators = radial_operators(nr, nℓ), subtract_doppler = false, kw...)
    uniform_rot_operators = twoΩcrossv(nr, nℓ, m; operators);
    diff_rot_operators = diffrotterms_constantΩ(nr, nℓ, m; operators);

    M = uniform_rot_operators + diff_rot_operators;

    λ, v = constrained_eigensystem(M, operators)

    if subtract_doppler
        λ .+= m
    end
    λ, v
end

rossby_ridge(m, ℓ = m) = 2m/(ℓ*(ℓ+1))

eigenvalue_filter(x, m) = imag(x) >= 0 && imag(x) < real(x)*1e-2 && 0 < real(x) < 2rossby_ridge(m)
eigenvector_filter(v, C, atol = 1e-5) = norm(C * v) < atol
eigensystem_satisfy_filter(λ, v, M, rtol = 1e-1) = isapprox(M*v, λ*v, rtol=rtol)

function filter_eigenvalues(f, nr, nℓ, m::Integer;
        operators = radial_operators(nr, nℓ), kw...)

    λ::Vector{ComplexF64}, v::Matrix{ComplexF64}, M::Matrix{Float64} =
        f(nr, nℓ, m; operators, kw...)
    filter_eigenvalues(λ, v, M, m; operators, kw...)
end

function filter_eigenvalues(λ::AbstractVector, v::AbstractMatrix,
        M::AbstractMatrix{Float64}, m::Integer; operators,
        atol_constraint = 1e-5,
        Δl_cutoff = 3,
        power_cutoff = 0.9,
        eigen_rtol = 1e-1,
        kw...)

    (; BC) = constraintmatrix(operators);

    filterfn(λ, v)::Bool = begin
        eigenvalue_filter(λ, m) &&
        eigenvector_filter(v, BC, atol_constraint) &&
        sphericalharmonic_transform_filter(v, operators, Δl_cutoff, power_cutoff) &&
        eigensystem_satisfy_filter(λ, v, M, eigen_rtol)
    end
    filtinds = axes(λ, 1)[filterfn.(λ, eachcol(v))]
    real(λ[filtinds]), v[:, filtinds]
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

function filter_eigenvalues_mrange(f, nr, nℓ, mr::AbstractVector; kw...)
    @maybe_reduce_blas_threads(
        Folds.map(m -> filter_eigenvalues(f, nr, nℓ, m; kw...), mr))
end
function filter_eigenvalues_mrange(λ, v, mr::AbstractVector; kw...)
    @maybe_reduce_blas_threads(
        Folds.map(x -> filter_eigenvalues(x...; kw...), zip(λ, v, mr)))
end

function save_eigenvalues(f, nr, nℓ, mr; kw...)
    λv = filter_eigenvalues_mrange(f, nr, nℓ, mr; kw...)
    lam = first.(λv);
    vec = last.(λv);
    ΔΩscale = get(kw, :ΔΩscale, 1)
    eigen_rtol = get(kw, :eigen_rtol, 0.1)
    power_cutoff = get(kw, :power_cutoff, 0.9)
    Δl_cutoff = get(kw, :Δl_cutoff, 3)
    fname = joinpath(DATADIR[],
        "rossby_$(string(nameof(f)))_nr$(nr)_nell$(nℓ)"*
        "_Omegascale$(ΔΩscale)_eigen_rtol$(eigen_rtol)"*
        "_power_cutoff$(power_cutoff)_Δl_cutoff$(Δl_cutoff).jld2")
    jldsave(fname; lam, vec, mr)
end

plot_eigenvalues_singlem(m, lam) = plot.(m, lam, marker = "o", ms = 5, mfc = "0.8", mec= "0.2", zorder=2)

function plot_rossby_ridges(mr, ΔΩscale = 1)
    plot(mr, ΔΩscale * rossby_ridge.(mr), color = "k", label=L"ω = \frac{2Ω}{m+1}", lw=0.7, zorder=1)
    plot(mr, ΔΩscale*rossby_ridge.(mr, mr .+1), color = "0.3", label=L"ω = \frac{2mΩ}{(m+1)(m+2)}",
        marker="None", ls = "dashed", zorder=1)
    # plot(mr, 2*rossby_ridge.(mr) .- mr, color = "orange", label = "-m + 2/(m+1)")
end

function plot_eigenvalues(f, nr, nℓ, mr::AbstractVector; kw...)
    λv = filter_eigenvalues_mrange(f, nr, nℓ, mr; kw...)
    lam = map(first, λv)
    plot_eigenvalues(lam, mr; kw...)
end

function plot_eigenvalues(fname::String, mr; kw...)
    j = jldopen(fname, "r")
    lam = j["lam"]
    plot_eigenvalues(lam, mr; kw...)
end
function plot_eigenvalues(fname::String; kw...)
    j = jldopen(fname, "r")
    lam = j["lam"]
    mr = j["mr"]
    plot_eigenvalues(lam, mr; kw...)
end

function plot_eigenvalues(lam::AbstractArray, mr; ΔΩscale = 1, kw...)
    f, ax = subplots()
    plot_eigenvalues_singlem.(mr, lam)
    plot_rossby_ridges(mr, ΔΩscale)
    xlabel("m", fontsize = 12)
    ylabel(L"\omega/\Omega", fontsize = 12)
    ylim(0.5*2/(maximum(mr)+1), 1.1*2/(minimum(mr)+1))
    legend(loc=1, fontsize = 12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    f.set_size_inches(6,4)
    tight_layout()
end

function plot_eigenvalues_shfilter(fname::String, mr; operators, Δl_cutoff = 5, power_cutoff = 0.9)
    j = jldopen(fname, "r")
    lam = j["lam"]::Vector{Vector{Float64}}
    v = j["vec"]::Vector{Matrix{ComplexF64}}
    for (ind, (m, vm)) in enumerate(zip(mr, v))
        inds_filterred = sphericalharmonic_transform_filter.(eachcol(vm), (operators,), m, Δl_cutoff, power_cutoff)
        inds = axes(vm, 2)[inds_filterred]
        lam[ind] = lam[ind][inds]
        v[ind] = vm[:, inds]
    end
    figure()
    plot_eigenvalues(lam, mr)
end

function eigenfunction_cheby_ℓm_spectrum(v, operators)
    (; radial_params) = operators;
    (; nparams, nr, nℓ) = radial_params;
    V = reshape(v[1:nparams], nr, nℓ)
    W = reshape(v[nparams+1:end], nr, nℓ)
    (; V, W)
end

function eigenfunction_rad_sh(v, operators, VW = eigenfunction_cheby_ℓm_spectrum(v, operators))
    (; radial_params, transforms) = operators;
    (; nparams, nchebyr, nℓ) = radial_params;
    (; Tcrinv) = transforms;
    (; V, W ) = VW

    V .= Tcrinv * V
    W .= Tcrinv * W

    (; V, W)
end

function spharm_θ_grid_uniform(m, nℓ, ℓmax_mul = 4)
    ℓs = range(m, length = nℓ)
    ℓmax = maximum(ℓs);

    θ, _ = sph_points(ℓmax_mul * ℓmax);
    return (; ℓs, θ)
end

function eigenfunction_realspace(v, m, operators, VW = eigenfunction_rad_sh(v, operators))
    (; radial_params) = operators;
    (; nr) = radial_params;

    V_r_lm = VW.V
    W_r_lm = VW.W

    nℓ = size(V_r_lm, 2)

    (; ℓs, θ) = spharm_θ_grid_uniform(m, nℓ)

    ℓmax = maximum(ℓs);
    nθ = length(θ);
    V = zeros(eltype(V_r_lm), nr, nθ)
    W = zeros(eltype(W_r_lm), nr, nθ)

    Plcosθ = SphericalHarmonics.allocate_p(ℓmax);

    for (θind, θi) in enumerate(θ)
        computePlmcostheta!(Plcosθ, θi, ℓmax, m, norm = SphericalHarmonics.Orthonormal())
        for (ℓind, ℓ) in enumerate(ℓs)
            Plmcosθ = Plcosθ[(ℓ,m)]
            for r_ind in axes(V, 1)
                V[r_ind, θind] += V_r_lm[r_ind, ℓind] * Plmcosθ
                W[r_ind, θind] += W_r_lm[r_ind, ℓind] * Plmcosθ
            end
        end
    end

    (; V, W, θ)
end

function sphericalharmonic_transform_filter(v, operators, Δl_cutoff = 5, power_cutoff = 0.9)
    (; V) = eigenfunction_rad_sh(v, operators);

    l_cutoff_ind = 1 + Δl_cutoff
    # ensure that most of the power at the surface is below the cutoff
    V_lm_surface = @view V[end, :];
    P_frac = sum(abs2, @view V_lm_surface[1:l_cutoff_ind])/ sum(abs2, V_lm_surface)
    P_frac > power_cutoff
end

function eigenfunction_rossbyridge(lam, v, m, operators)
    minind = argmin(abs.(lam .- rossby_ridge(m)))
    (; V, θ) = eigenfunction_realspace((@view v[:, minind]), m, operators)
    Vr = real(V)
    Vrmax = maximum(abs, Vr)
    if Vrmax != 0
        Vr ./= Vrmax
    end
    (; coordinates) = operators;
    (; r) = coordinates;
    nθ = length(θ)
    f = figure()
    ax2 = subplot2grid((3,3), (1,1), colspan = 2, rowspan = 2)
    ax1 = subplot2grid((3,3), (0,1), colspan = 2, sharex = ax2)
    ax3 = subplot2grid((3,3), (1,0), rowspan = 2, sharey = ax2)
    ax2.pcolormesh(θ, r, Vr, shading="auto", cmap = "Greys",
    vmax = Vrmax, vmin = -Vrmax, rasterized = true)
    ax1.plot(θ, (@view Vr[end, :]), color = "k")
    ax1.plot(θ, sin.(θ).^m, "o", markevery=10, ms=4, mfc="0.8", mec="0.3")
    ax3.plot((@view Vr[:, nθ÷2]), r, color = "k")
    ax2.set_xlabel("θ [radians]", fontsize=12)
    ax2.set_ylabel(L"r/R_\odot", fontsize=12)

    ax2.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(4))

    f.set_size_inches(6,4)
    tight_layout()
end

function eigenfunction_rossbyridge(nr, nℓ, m; operators = radial_operators(nr, nℓ))
    λ, v = filter_eigenvalues(uniform_rotation_spectrum, nr, nℓ, m);
    eigenfunction_rossbyridge(λ, v, m, operators)
end

function eigenvalues_remove_spurious(f1::String, f2::String)
    lam1 = jldopen(f1)["lam"]::Vector{Vector{Float64}}
    lam2 = jldopen(f2)["lam"]::Vector{Vector{Float64}}
    eigenvalues_remove_spurious(lam1 ,lam2)
end
eigenvalues_remove_spurious(f1 ,f2, f3, f4...) = foldl(eigenvalues_remove_spurious, (f1, f2, f3, f4...))
function eigenvalues_remove_spurious(lam1::Vector{<:Vector}, lam2::Vector{<:Vector})
    lam_filt = similar(lam1)
    for (m_ind, (lam1_m, lam2_m)) in enumerate(zip(lam1, lam2))
        lam_filt[m_ind] = eltype(lam_filt)[]
        for λ1 in lam1_m, λ2 in lam2_m
            if abs2(λ1 - λ2) < abs2(λ1)*1e-2
                push!(lam_filt[m_ind], λ1)
            end
        end
    end
    return lam_filt
end

function rossbyridgefreqnearest(lam, mr)
    @assert length(lam) == length(mr)
    [argmin(x -> abs(rossby_ridge(m)-x), lam_m) for (m,lam_m) in zip(mr, lam)]
end
function rossbyridgefreqsmisfit(lam, mr)
    λ = rossbyridgefreqnearest(lam, mr)
    sum(((m, λm),) -> abs(rossby_ridge(m)-λm), zip(mr, λ))
end

function testrossbyridgefreqconvergence(nrs::AbstractVector, nℓs::AbstractVector, mr)
    misfits = zeros(length(nrs), length(nℓs))
    for (nℓind, nℓ) in enumerate(nℓs), (nrind, nr) in enumerate(nrs)
        λv = filter_eigenvalues_mrange(uniform_rotation_spectrum, nr, nℓ, mr)
        lam = first.(λv)
        misfit = rossbyridgefreqsmisfit(lam, mr)
        misfits[nrind, nℓind] = misfit
    end
    return misfits
end

end # module
