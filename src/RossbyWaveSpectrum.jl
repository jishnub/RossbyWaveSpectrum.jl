module RossbyWaveSpectrum

using LinearAlgebra
using LinearAlgebra: AbstractTriangular
using MKL
using FillArrays
using Kronecker
using Kronecker: KroneckerProduct
using BlockArrays
using UnPack
using PyPlot
using LaTeXStrings
using OffsetArrays
using DelimitedFiles
using Dierckx
using Distributed
using JLD2
using FastSphericalHarmonics
using FastTransforms
using FastGaussQuadrature
using SphericalHarmonics
using Trapz
using WignerSymbols

export datadir

const SCRATCH = Ref("")
const DATADIR = Ref("")

function __init__()
    SCRATCH[] = get(ENV, "SCRATCH", homedir())
    DATADIR[] = get(ENV, "DATADIR", joinpath(SCRATCH[], "RossbyWaves"))
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


function chebyshevnodes(n, a, b)
    nodes = cos.(reverse(pi*((1:n) .- 0.5)./n))
    nodes_scaled = nodes*(b - a)/2 .+ (b + a)/2
    nodes, nodes_scaled
end
chebyshevnodes(n) = chebyshevnodes(n, -1, 1)

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

"""
    constraintmatrix(params)

Evaluate the matrix ``F`` that captures the constriants on the Chebyshev coefficients
of ``V`` and ``W``, such that
```math
F * [V_{11}, ⋯, V_{1k}, ⋯, V_{n1}, ⋯, V_{nk}, W_{11}, ⋯, W_{1k}, ⋯, W_{n1}, ⋯, W_{nk}] = 0.
```
"""
function constraintmatrix(operators)
    (; params) = operators;
    (; nr, r_in, r_out, nℓ) = params;

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

    C = [
        Ones(1, nℓ) ⊗ MVn Zeros(2, nparams)
        Zeros(2, nparams) Ones(1, nℓ) ⊗ MWn
        ]

    ZC = constraintnullspacematrix(C)
    ZW = constraintnullspacematrix(MWn)

    (; C, MVn, MWn, ZC, ZW)
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
    r_in = 0.71;
    r_out = 0.985;
    Δr = r_out - r_in
    nparams = nchebyr * nℓ;
    return (; nchebyr, r_in, r_out, Δr, nr, nparams, nℓ)
end

include("identitymatrix.jl")
include("forwardinversetransforms.jl")
include("realspectralspace.jl")
include("paddedmatrix.jl")

function chebyshev_forward_inverse(n, boundaries...)
    nodes, r = chebyshevnodes(n, boundaries...);
    Tc = chebyshevpoly(n, nodes);
    Tcfwd = Tc' * 2 /n;
    Tcfwd[1, :] /= 2
    Tcinv = Tc;
    r, Tcfwd, Tcinv
end

isapproxdiagonal(M) = isapprox(M, Diagonal(M), atol=1e-14, rtol=1e-8)

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

function radial_operators(nr, nℓ)
    params = parameters(nr, nℓ);
    (; r_in, r_out, Δr, nchebyr) = params;
    r, Tcrfwd, Tcrinv = chebyshev_forward_inverse(nr, r_in, r_out);
    r_cheby = Tcrfwd * Diagonal(r) * Tcrinv

    FTcrinv = factorize(Tcrinv);

    ddr = chebyshevderiv(nr) * (2/Δr);

    onebyr = 1 ./r;
    onebyr_cheby = Tcrfwd * Diagonal(onebyr) * Tcrinv;
    onebyr2_cheby = Tcrfwd * Diagonal(onebyr)^2 * Tcrinv;

    diml = 696e6;
    dimT = 1/(2 * pi * 453e-9);
    dimrho = 0.2;
    dimp = dimrho * diml^2/dimT^2;
    dimtemp = 6000;
    Nrho = 3;
    npol = 2.6;
    betapol = r_in / r_out; rhoc = 2e-1;
    gravsurf=278; pc = 1e10;
    R = 8.314 * dimrho * dimtemp/dimp;
    G = 6.67e-11; Msol = 2e30;
    zeta0 = (betapol + 1)/(betapol * exp(Nrho/npol) + 1);
    zetai = (1 + betapol - zeta0)/betapol;
    Cp = (npol + 1) * R; gamma_ind = (npol+1)/npol; Cv = npol * R;
    c0 = (2 * zeta0 - betapol - 1)/(1-betapol);
    #^c1 = G * Msol * rhoc/((npol+1)*pc * Tc * (r_out-r_in));
    c1 = (1-zeta0)*(betapol + 1)/(1-betapol)^2;
    ζfn(r) = c0 + c1 * Δr /r
    zeta = ζfn.(r);
    dzeta_dr = @. -c1 * Δr /r^2;
    d2zeta_dr2 = @. 2 * c1 * Δr /r^3;
    grav = @. -pc / rhoc * (npol+1) * dzeta_dr;
    pc = gravsurf .* pc/minimum(grav);
    grav = grav .* gravsurf ./ minimum(grav) * dimT^2/diml;
    ηρ = @. npol/zeta*dzeta_dr;
    ηρ_integral = @. npol * log(zeta)
    ηρ_integral .-= first(ηρ_integral)
    ηρ_definite_integral(r, s) = npol * log(ζfn(s)/ζfn(r))
    # drhrho = @. (- npol/zeta^2 * (dzeta_dr)^2 + npol/zeta * d2zeta_dr2);
    ηρ_cheby = Tcrfwd * Diagonal(ηρ) * Tcrinv;

    DDr = ddr + ηρ_cheby;
    rDDr = r_cheby * DDr
    rddr = r_cheby * ddr;
    D2Dr2 = DDr^2;
    DDr_minus_2byr = DDr - 2*onebyr_cheby;

    # scratch matrices
    # B = D2Dr2 - ℓ(ℓ+1)/r^2
    B = zero(D2Dr2);
    Cℓ′ = zero(DDr);
    BinvCℓ′ = zero(DDr);

    Ir = IdentityMatrix(nchebyr);
    Iℓ = PaddedMatrix(IdentityMatrix(2nℓ), nℓ);

    identities = (; Ir, Iℓ);
    coordinates = (; r, r_cheby);
    transforms = (; Tcrfwd, Tcrinv, FTcrinv);
    rad_terms = (; onebyr, onebyr_cheby, ηρ, ηρ_cheby, onebyr2_cheby);
    diff_operators = (; DDr, D2Dr2, DDr_minus_2byr, rDDr, rddr, ddr);
    scratch = (; B, Cℓ′, BinvCℓ′);

    (; rad_terms, diff_operators, transforms, coordinates, params, scratch, identities)
end

w1(r, ℓ, r_in, r_out) = r^(ℓ+1)-(r_in^(2ℓ+1))/r^ℓ
w2(r, ℓ, r_in, r_out) = r^(ℓ+1)-(r_out^(2ℓ+1))/r^ℓ
w1w2(r, s, ℓ, r_in, r_out) = w1(r, ℓ, r_in, r_out)*w2(s, ℓ, r_in, r_out)
function Binv_realspace(r, s, ℓ, r_in, r_out, ηρ_definite_integral)
    @assert ℓ >= 0
    @assert r_out > r_in
    @assert r_in <= r <= r_out
    @assert r_in <= s <= r_out
    W = (2ℓ+1)*(r_out^(2ℓ+1) - r_in^(2ℓ+1))
    T = (s <= r) ? w1w2(s, r, ℓ, r_in, r_out) : w1w2(r, s, ℓ, r_in, r_out)
    p = exp(ηρ_definite_integral(r, s))
    p*T/W
end
function Binv_realspace(r::AbstractVector, ℓ, ηρ_definite_integral)
    r_in, r_out = extrema(r)
    Binv_realspace.(r, r', ℓ, r_in, r_out, ηρ_definite_integral)
end

function twoΩcrossv(nr, nℓ, m; operators = radial_operators(nr, nℓ))
    (; params) = operators;
    (; rad_terms, diff_operators, scratch) = operators;

    (; nchebyr) = params;

    (; DDr, D2Dr2, DDr_minus_2byr) = diff_operators;
    (; onebyr_cheby, onebyr2_cheby) = rad_terms;
    (; B, Cℓ′, BinvCℓ′) = scratch;

    nparams = nchebyr * nℓ;
    ℓmin = m;
    ℓs = range(ℓmin, length = nℓ);

    D = 2m * Diagonal(@. 1/(ℓs*(ℓs + 1))) ⊗ I(nchebyr);

    A = zeros(nparams, nparams);
    C = zeros(nparams, nparams);

    (; ZW) = constraintmatrix(operators);

    C1 = DDr_minus_2byr

    cosθ = OffsetArray(costheta_operator(nℓ, m), ℓs, ℓs)
    sinθdθ = OffsetArray(sintheta_dtheta_operator(nℓ, m), ℓs, ℓs)

    for ℓ in ℓs
        @. B = D2Dr2 - ℓ*(ℓ+1) * onebyr2_cheby
        B2 = ZW' * B * ZW;
        F = lu!(B2)
        ABℓ′top = (ℓ - minimum(m)) * nchebyr + 1;
        ACℓ′vertinds = range(ABℓ′top, length = nr);

        for ℓ′ in ℓ-1:2:ℓ+1
            ℓ′ in ℓs || continue
            @. Cℓ′ = DDr - ℓ′*(ℓ′+1) * onebyr_cheby
            @. Cℓ′ = -2/(ℓ*(ℓ+1))*(ℓ′*(ℓ′+1)*C1*cosθ[ℓ, ℓ′] + Cℓ′*sinθdθ[ℓ, ℓ′])
            BinvCℓ′ .= ZW * (F \ (ZW' * Cℓ′))
            ABℓ′left = (ℓ′ - minimum(m)) * nchebyr + 1
            ACℓ′horinds = range(ABℓ′left, length = nr);
            ACℓ′ℓ′inds = CartesianIndices((ACℓ′vertinds, ACℓ′horinds));
            A[ACℓ′ℓ′inds] .= Cℓ′
            C[ACℓ′ℓ′inds] .= BinvCℓ′
        end
    end

    [D  A
     C  D];
end

function interp1d(xin, z, xout)
    spline = Spline1D(xin, z)
    spline(xout)
end

function interp2d(xin, yin, z, xout, yout)
    spline = Spline2D(xin, yin, z)
    evalgrid(spline, xout, yout)
end

function read_angular_velocity(r, thetaGL)
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
    ΔΩ_r_chebytheta .= 1
    ΔΩ_r_thetaGL .= 1

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

function rotation_velocity_phi(operators, thetaGL)
    (; coordinates) = operators;
    (; r) = coordinates;

    (; ΔΩ_r_thetaGL, ΔΩ_r_Legendre) = read_angular_velocity(r, thetaGL);

    (; ΔΩ_r_thetaGL, ΔΩ_r_Legendre)
end

velocity_from_angular_velocity(Ω, r, sintheta) = Ω .* sintheta' .* r;

function invsinθ_dθ_ωΩr_operator(ΔΩ_r_Legendre, m, operators)
    (; transforms, params) = operators;
    (; nparams, nℓ) = params;
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

function ddr_operator(ΔΩ_r_ℓℓ′, m, operators)
    (; params, diff_operators, transforms) = operators;
    (; nℓ, nr, nparams) = params;
    (; ddr) = diff_operators;
    (; Tcrfwd, Tcrinv) = transforms;
    ddr_realspace = Tcrinv * ddr * Tcrfwd
    ℓs = range(m, length = 2nℓ);
    ddr_ΔΩ_kk′_ℓℓ′ = zeros(2nparams, 2nparams);
    for (ℓ2ind, ℓ2) in enumerate(ℓs), (ℓ1ind, ℓ1) in enumerate(ℓs)
        rowinds = (ℓ1ind-1)*nr + 1:ℓ1ind*nr
        colinds = (ℓ2ind-1)*nr + 1:ℓ2ind*nr
        inds = CartesianIndices((rowinds, colinds))
        D = @view parent(ΔΩ_r_ℓℓ′)[inds]
        fr = @view D[diagind(D)]
        ddr_ΔΩ_kk′_ℓℓ′[inds] = Tcrfwd * Diagonal(ddr_realspace * fr) * Tcrinv
    end
    return PaddedMatrix(ddr_ΔΩ_kk′_ℓℓ′, nparams)
end
function rddr_operator(ΔΩ_r_ℓℓ′, m, operators)
    (; coordinates, identities) = operators;
    (; r_cheby) = coordinates;
    (; Iℓ) = identities;
    (Iℓ ⊗ r_cheby) * ddr_operator(ΔΩ_r_ℓℓ′, m, operators)
end

function vorticity_terms(ΔΩ_kk′_ℓℓ′, ΔΩ_r_ℓℓ′, ΔΩ_r_Legendre, m, C, Sd, operators)
    (; identities, coordinates) = operators;
    (; Ir, Iℓ) = identities;
    (; r_cheby) = coordinates;

    S2 = I - C^2;
    CI = C ⊗ Ir;
    S2I = S2 ⊗ Ir;
    SdI = Sd ⊗ Ir;

    sinθ_dθ_ΔΩ_kk′_ℓℓ′ = SdI*ΔΩ_kk′_ℓℓ′ - ΔΩ_kk′_ℓℓ′*SdI;

    ωΩr = 2CI*ΔΩ_kk′_ℓℓ′ + sinθ_dθ_ΔΩ_kk′_ℓℓ′;
    ddr_ΔΩ_kk′_ℓℓ′ = ddr_operator(ΔΩ_r_ℓℓ′, m, operators);
    drωΩr = (2CI + SdI)*ddr_ΔΩ_kk′_ℓℓ′ - ddr_ΔΩ_kk′_ℓℓ′*SdI;

    rddr_ΔΩ_kk′_ℓℓ′ = (Iℓ ⊗ r_cheby) * ddr_ΔΩ_kk′_ℓℓ′;
    ωΩθ_by_sinθ = -2ΔΩ_kk′_ℓℓ′ - rddr_ΔΩ_kk′_ℓℓ′;

    sinθ_ωΩθ =  S2I * ωΩθ_by_sinθ;
    dθ_ωΩθ = (CI + SdI)*ωΩθ_by_sinθ - ωΩθ_by_sinθ*SdI;
    cotθ_ωΩθ = CI * ωΩθ_by_sinθ;
    ωΩr_plus_cotθ_ωΩθ = ωΩr + cotθ_ωΩθ;

    invsinθ_dθ_ωΩr = invsinθ_dθ_ωΩr_operator(ΔΩ_r_Legendre, m, operators);

    (; ωΩr, ωΩθ_by_sinθ, sinθ_ωΩθ, invsinθ_dθ_ωΩr, dθ_ωΩθ, drωΩr, cotθ_ωΩθ, ωΩr_plus_cotθ_ωΩθ)
end

function rθ_operators(nr, nℓ, m; operators = radial_operators(nr, nℓ),
    thetaop = theta_operators(nℓ, m))

    (; transforms) = operators;
    (; Tcrfwd, Tcrinv) = transforms;

    (; theta, PLMfwd, PLMinv) = thetaop;
    fullfwd = kron(PLMfwd, Tcrfwd);
    fullinv = kron(PLMinv, Tcrinv);

    full_transforms = (; fullfwd, fullinv)

    C = costheta_operator(nℓ, m)
    Sd = sintheta_dtheta_operator(nℓ, m)

    (; full_transforms)
end

function diffrotterms(nr, nℓ, m; operators = radial_operators(nr, nℓ),
    thetaop = theta_operators(nℓ, m))

    (; params, rad_terms, diff_operators, transforms, coordinates) = operators;

    (; nparams) = params;
    (; Tcrfwd, Tcrinv) = transforms;
    (; DDr, D2Dr2) = diff_operators;
    (; r) = coordinates;
    (; thetaGL, PLMfwd, PLMinv) = thetaop;

    fullfwd = kron(PLMfwd, Tcrfwd);
    fullinv = kron(PLMinv, Tcrinv);

    (; ΔΩ_r_thetaGL, ΔΩ_r_Legendre) = rotation_velocity_phi(operators, thetaGL);
    ΔΩ_by_r = ΔΩ_r_thetaGL ./ r
    ΔΩ_r_ℓℓ′ = PaddedMatrix((PLMfwd ⊗ Ir) * Diagonal(vec(ΔΩ_r_thetaGL)) * (PLMinv ⊗ Ir), nparams);
    ΔΩ_kk′_ℓℓ′ = fullfwd * Diagonal(vec(ΔΩ_r_thetaGL)) * fullinv
    ΔΩ_by_r_chebyleg = fullfwd * Diagonal(vec(ΔΩ_by_r)) * fullinv
end

function diffrotterms2(nr, ntheta, m, operators = radial_operators(nr, nℓ))
    @unpack trig_functions, rad_terms, diff_operators, transforms, params = operators;

    @unpack nparams = params;
    @unpack DD_V, DD_W, sintheta_ddtheta_mat, D2Dtheta2, DDtheta, DDr, D2Dr2, DDtheta_realspace, DDr_realspace = diff_operators;
    @unpack costheta_mat, sintheta_mat, cottheta_mat, onebysintheta_mat, sintheta_realspace, costheta_realspace, onebysintheta_realspace = trig_functions;
    @unpack r_mat, onebyr_mat, hrho_mat, onebyr2_mat, drhrho_mat, negr2_mat, r_realspace, onebyr_realspace = rad_terms;
    @unpack fullfwd, fullinv, fullfwdc, fullinvc = transforms;

    cos2theta_mat = costheta_mat^2 - sintheta_mat^2;
    sin2theta_mat = 2*sintheta_mat*costheta_mat;

    laplacianh = horizontal_laplacian(operators, m);

    ΔΩ_realspace, V0_realspace = rotation_velocity_phi(operators);
    V0 = fullfwd * V0_realspace * fullinv;
    ΔΩ = fullfwd * ΔΩ_realspace * fullinv;

    one_over_rsintheta = onebyr_mat * onebysintheta_mat;
    one_over_rsintheta_realspace = Diagonal(onebyr_realspace * onebysintheta_realspace);

    # Omega is the curl of V = (0,0,V0)
    Omega_r_realspace = one_over_rsintheta_realspace * Diagonal(DDtheta_realspace * diag(sintheta_realspace * V0_realspace));
    Omega_r = fullfwdc * Omega_r_realspace * fullinvc;
    Omega_theta_realspace = Diagonal(-onebyr_realspace * (DDr_realspace * diag(r_realspace * V0_realspace)));
    Omega_theta = fullfwdc * Omega_theta_realspace * fullinvc;

    DDr_Omega_r = fullfwdc * Diagonal(DDr_realspace * diag(Omega_r_realspace)) * fullinvc;
    Dtheta_Omega_r = fullfwdc * Diagonal(DDtheta_realspace * diag(Omega_r_realspace)) * fullinvc;
    DDr_Omega_theta = fullfwdc * Diagonal(DDr_realspace * diag(Omega_theta_realspace)) * fullinvc;
    Dtheta_Omega_theta = fullfwdc * Diagonal(DDtheta_realspace * diag(Omega_theta_realspace)) * fullinvc;

    DDV_V0 = fullfwdc * Diagonal(fullinv * DD_V * fullfwd * diag(V0_realspace)) * fullinvc;

    DDtheta_V0 = fullfwdc * Diagonal(DDtheta_realspace * diag(V0_realspace)) * fullinvc;

    DDtheta_sintheta_mat = sintheta_mat*DDtheta + costheta_mat;

    m_over_sintheta = m*onebysintheta_mat;

    # im_vr_V = 0
    im_vr_im_W = collect(-onebyr_mat^2*laplacianh);
    vr_W = im_vr_im_W;
    im_r_vr_im_W = collect(-onebyr_mat*laplacianh);
    im_r2_vr_im_W = collect(-laplacianh);
    vr_im_W = -im*im_vr_im_W;

    im_r_vtheta_V = collect(-m_over_sintheta);
    im_r_vtheta_im_W = collect(DDtheta*DD_W);
    im_vtheta_im_W = collect(onebyr_mat * im_r_vtheta_im_W);

    m_over_r_sintheta = m * one_over_rsintheta;

    im_vtheta_V = collect(-m_over_r_sintheta);
    vtheta_V = -im*im_vtheta_V;
    im_vtheta_im_W = collect(DDtheta*(onebyr_mat*DD_W));
    vtheta_W = im_vtheta_im_W;
    vtheta_im_W = -im*im_vtheta_im_W;

    vphi_V = collect(-onebyr_mat*DDtheta);
    im_vphi_V = im*vphi_V;
    r_vphi_V = collect(-DDtheta);
    vphi_im_W = collect(-m_over_r_sintheta*DD_W);
    im_vphi_im_W = im*vphi_im_W;
    r_vphi_im_W = collect(-m_over_sintheta*DD_W);

    omega_r_V = collect(-onebyr_mat^2*laplacianh);
    r2_omega_r_V = collect(-laplacianh);

    omega_theta_V = collect(onebyr_mat*DDr*DDtheta);
    omega_theta_im_W = collect(-m_over_r_sintheta*(laplacianh*onebyr2_mat + DDr*DD_W));

    im_omega_phi_V = collect(-m_over_r_sintheta*DD_V);
    im_omega_phi_im_W = collect(onebyr_mat*DDtheta*(DDr*DD_W + onebyr_mat^2 * laplacianh));

    # T2 terms

    T2r_V = 2sintheta_mat*vphi_V;
    T2r_im_W = 2sintheta_mat*vphi_im_W - m*im_vr_im_W;
    T2θ_V = 2costheta_mat*vphi_V - m*im_vtheta_V;
    T2θ_im_W = 2costheta_mat*vphi_im_W - m*im_vtheta_im_W;
    T2ϕ_V = -(m*im_vphi_V + 2costheta_mat*vtheta_V);
    T2ϕ_im_W = -(m*im_vphi_im_W + 2costheta_mat*vtheta_im_W + 2sintheta_mat*vr_im_W);

    # single curl terms

    curlT2_direct_r_V = r_mat*(m*(DDtheta + 3cottheta_mat)*vphi_V
    -((m^2*I + 2collect(cos2theta_mat))*onebysintheta_mat-2costheta_mat*DDtheta)*im_vtheta_V
    );

    curlT2_direct_r_im_W = r_mat*(m*(DDtheta + 3cottheta_mat)*vphi_im_W
    -((m^2*I + 2collect(cos2theta_mat))*onebysintheta_mat-2costheta_mat*DDtheta)*im_vtheta_im_W
    -(2sintheta_ddtheta_mat + 4costheta_mat)*im_vr_im_W
    );

    curlT2_direct_θ_V = onebyr_mat * (m*onebysintheta_mat*T2r_V + r_mat*(DD_V + onebyr_mat)*real(im*T2ϕ_V));
    curlT2_direct_θ_im_W = onebyr_mat * (m*onebysintheta_mat*T2r_im_W + r_mat*(DD_W + onebyr_mat)*real(im*T2ϕ_im_W));
    curlT2_direct_ϕ_V = onebyr_mat * (r_mat*(DD_V + onebyr_mat)*T2θ_V - DDtheta * T2r_V);
    curlT2_direct_ϕ_im_W = onebyr_mat * (r_mat*(DD_W + onebyr_mat)*T2θ_im_W - DDtheta * T2r_im_W);

    # double curl terms

    curl_curl_T2_direct_r_V = (-r_mat*onebysintheta_mat)*(
        DDtheta_sintheta_mat*real(im*curlT2_direct_ϕ_V) - m*curlT2_direct_θ_V
    );
    curl_curl_T2_direct_r_im_W = (-r_mat*onebysintheta_mat)*(
        DDtheta_sintheta_mat*real(im*curlT2_direct_ϕ_im_W) - m*curlT2_direct_θ_im_W
    );

    OperatorMatrix(curlT2_direct_r_V, curlT2_direct_r_im_W, curl_curl_T2_direct_r_V, curl_curl_T2_direct_r_im_W)
end

function uniform_rotation_spectrum(nr, nℓ, m, operators = radial_operators(nr, nℓ))
    (; C, ZC) = RossbyWaveSpectrum.constraintmatrix(operators);

    M = RossbyWaveSpectrum.twoΩcrossv(nr, nℓ, m; operators);

    # Two possible ways to evaluate the eigenvalues
    # The first seems to produce slightly better conditioned matrices
    M_constrained = ZC'*M*ZC
    λ, w = eigen!(M_constrained);
    v = ZC*w
    λ, v

    # Z = Zeros(size(C, 1), size(M,2) + size(C, 1) - size(C,2))
    # M_constrained = [M C'
    #                  C Z];
    # λ, v = eigen!(M_constrained);
    # λ, v[1:end - size(C,1), :]
end

function real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop)
    (; PLMfwd, PLMinv) = thetaop;
    (; transforms, params) = operators;
    (; nℓ, nchebyr) = params;
    (; Tcrfwd, Tcrinv) = transforms;
    pad = nchebyr*nℓ
    PaddedMatrix((PLMfwd ⊗ Tcrfwd) * Diagonal(vec(ΔΩ_r_thetaGL)) * (PLMinv ⊗ Tcrinv), pad)
end

function real_to_r_assocleg(ΔΩ_r_thetaGL, operators, thetaop)
    (; PLMfwd, PLMinv) = thetaop;
    (; params) = operators;
    (; nℓ, nchebyr) = params;
    Ir = IdentityMatrix(nchebyr)
    pad = nchebyr*nℓ
    PaddedMatrix((PLMfwd ⊗ Ir) * Diagonal(vec(ΔΩ_r_thetaGL)) * (PLMinv ⊗ Ir), pad)
end

function differential_rotation_spectrum(nr, nℓ, m, operators = radial_operators(nr, nℓ))
    (; C) = RossbyWaveSpectrum.constraintmatrix(operators);

    uniform_rot_operators = RossbyWaveSpectrum.twoΩcrossv(nr, nℓ, m; operators);
    diff_rot_operators = RossbyWaveSpectrum.diffrotterms2(nr, nℓ, m, operators);

    M = uniform_rot_operators + diff_rot_operators;

    Z = Zeros(size(C, 1), size(M,2) + size(C, 1) - size(C,2))

    M_constrained = [M C'
                    C Z];

    λ, v = eigen!(M_constrained);
    λ, v[1:end - size(C,1), :]
end

rossby_ridge(m, ℓ = m) = 2m/(ℓ*(ℓ+1))
eigenvalue_filter(x, m) = isreal(x) && 0 < real(x) < 4rossby_ridge(m)
eigenvector_filter(v, C, atol = 1e-5) = norm(C * v) < atol

function filter_eigenvalues(f, nr, nℓ, m::Integer; operators = radial_operators(nr, nℓ),
        atol_constraint = 1e-5, Δl_cutoff = 5, power_cutoff = 0.9)

    @show m, Libc.gethostname(), BLAS.get_num_threads()
    (; C) = constraintmatrix(operators);
    lam::Vector{ComplexF64}, v::Matrix{ComplexF64} = f(nr, nℓ, m, operators)
    filterfn(λ, v) = begin
        # remove the lagrange multiplier elements
        eigenvalue_filter(λ, m) &&
        eigenvector_filter(v, C, atol_constraint) &&
        sphericalharmonic_transform_filter(v, operators, Δl_cutoff, power_cutoff)
    end
    filtinds = axes(lam, 1)[filterfn.(lam, eachcol(v))]
    real(lam[filtinds]), v[:, filtinds]
end

function filter_eigenvalues_mrange(f, nr, nℓ, mr::AbstractVector; kw...)
    map(m -> filter_eigenvalues(f, nr, nℓ, m; kw...), mr)
end

function save_eigenvalues(f, nr, nℓ, mr;
    operators = radial_operators(nr, nℓ), atol_constraint = 1e-5,
    Δl_cutoff = 5, power_cutoff = 0.9)

    λv = filter_eigenvalues_mrange(f, nr, nℓ, mr;
        operators, atol_constraint, Δl_cutoff, power_cutoff)
    lam = first.(λv);
    vec = last.(λv);
    fname = joinpath(DATADIR[], "rossby_$(string(nameof(f)))_nr$(nr)_nell$(nℓ).jld2")
    jldsave(fname; lam, vec)
end

plot_eigenvalues_singlem(m, lam) = plot.(m, lam, marker = "o", ms = 4, mfc = "lightcoral", mec= "firebrick")

function plot_rossby_ridges(mr)
    plot(mr, rossby_ridge.(mr), color = "k", label="ℓ = m")
    linecolors = ["r", "b", "g"]
    for (Δℓind, Δℓ) in enumerate(1:3)
        plot(mr, (m -> rossby_ridge(m, m+Δℓ)).(mr), color = linecolors[Δℓind], label="ℓ = m + $Δℓ",
        marker="None")
    end
    # plot(mr, 1.5 .*rossby_ridge.(mr) .- mr, color = "green", label = "-m + 1.5 * 2/(m+1)")
end

function plot_eigenvalues(f, nr, nℓ, mr::AbstractVector; kw...)
    λv = filter_eigenvalues_mrange(f, nr, nℓ, mr; kw...)
    lam = map(first, λv)
    plot_eigenvalues(lam, mr)
end

function plot_eigenvalues(fname::String, mr)
    lam = jldopen(fname, "r")["lam"]
    plot_eigenvalues(lam, mr)
end

function plot_eigenvalues(lam::AbstractArray, mr)
    figure()
    plot_eigenvalues_singlem.(mr, lam)
    plot_rossby_ridges(mr)
    xlabel("m", fontsize = 12)
    ylabel(L"\omega/\Omega", fontsize = 12)
    ylim(0.5*2/(maximum(mr)+1), 1.1*2/(minimum(mr)+1))
    legend(loc=1)
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

function remove_spurious_eigenvalues(fname1, fname2, rtol=1e-3)
    lam1 = jldopen(fname1, "r")["lam"]::Vector{Vector{Float64}}
    lam2 = jldopen(fname2, "r")["lam"]::Vector{Vector{Float64}}
    lam = similar(lam1)
    for (ind, (l1, l2)) in enumerate(zip(lam1, lam2))
        lam_i = eltype(first(lam1))[]
        for l1_i in l1, l2_i in l2
            if isapprox(l1_i, l2_i, rtol=rtol)
                push!(lam_i, l1_i)
            end
        end
        lam[ind] = lam_i
    end
    return lam
end

function plot_eigenvalues_remove_spurious(fname1, fname2, mr, rtol=1e-3)
    lam_filt = remove_spurious_eigenvalues(fname1, fname2, rtol)
    for (m, lam) in zip(mr, lam_filt)
        plot(fill(m, length(lam)), lam, "o")
    end
    plot(mr, 1.5 .*rossby_ridge.(mr) .- mr, color = "green", label = "-m + 1.5 * 2/(m+1)")
    xlabel("m", fontsize = 12)
    ylabel(L"\omega/\Omega", fontsize = 12)
    legend()
end

function eigenfunction_cheby_ℓm_spectrum(v, operators)
    (; params) = operators;
    (; nparams, nr, nℓ) = params;
    V = reshape(v[1:nparams], nr, nℓ)
    W = reshape(v[nparams+1:end], nr, nℓ)
    (; V, W)
end

function eigenfunction_rad_sh(v, operators, VW = eigenfunction_cheby_ℓm_spectrum(v, operators))
    (; params, transforms) = operators;
    (; nparams, nchebyr, nℓ) = params;
    (; Tcrinv) = transforms;
    (; V, W ) = VW

    V .= Tcrinv * V
    W .= Tcrinv * W

    (; V, W)
end

function spharm_θ_grid_uniform(m, nℓ, ℓmax_mul = 1)
    ℓs = range(m, length = nℓ)
    ℓmax = maximum(ℓs);

    θ, _ = sph_points(ℓmax_mul * ℓmax);
    return (; ℓs, θ)
end

function eigenfunction_realspace(v, m, operators, VW = eigenfunction_rad_sh(v, operators))
    (; params) = operators;
    (; nr) = params;

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
    (; params) = operators;
    (; nr, nℓ) = params;
    (; V) = eigenfunction_rad_sh(v, operators);

    l_cutoff_ind = 1 + Δl_cutoff
    # ensure that most of the power at the surface is below the cutoff
    V_lm_surface = @view V[end, :];
    P_frac = sum(abs2, @view V_lm_surface[1:l_cutoff_ind])/ sum(abs2, V_lm_surface)
    P_frac > power_cutoff
end

function eigenfunction_rossbyridge(lam, v, m, operators)
    minind = argmin(abs.(lam .- rossby_ridge(m)))
    (; V, θ) = eigenfunction_realspace(v[:, minind], m, operators)
    Vr = real(V)
    (; coordinates, params) = operators;
    (; r) = coordinates;
    nθ = length(θ)
    f = figure()
    ax1 = subplot2grid((3,3), (0,1), colspan = 2)
    ax2 = subplot2grid((3,3), (1,1), colspan = 2, rowspan = 2)
    ax3 = subplot2grid((3,3), (1,0), rowspan = 2)
    ax2.pcolormesh(θ, r, Vr, shading="auto", cmap = "RdBu",
    vmax = maximum(abs, Vr), vmin = -maximum(abs, Vr))
    ax1.plot(θ, Vr[end, :])
    ax3.plot(Vr[:, nθ÷2], r)
    ax2.set_xlabel("θ", fontsize=12)
    ax2.set_ylabel("r", fontsize=12)
    tight_layout()
end

end # module
