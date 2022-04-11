module RossbyWaveSpectrum

using ApproxFun
using Dierckx
using FastGaussQuadrature
using FastTransforms
using Folds
using ForwardDiff
using Infinities
using JLD2
using Kronecker
const kron2 = Kronecker.KroneckerProduct
using LinearAlgebra
using LinearAlgebra: BLAS
using LegendrePolynomials
using MKL
using OffsetArrays
using SimpleDelimitedFiles: readdlm
using TimerOutputs

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
    if !ispath(RossbyWaveSpectrum.DATADIR[])
        mkdir(RossbyWaveSpectrum.DATADIR[])
    end
end

indexedfromzero(A) = OffsetArray(A, OffsetArrays.Origin(0))

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

struct RealSpace
    r_chebyshev::Vector{Float64}
    theta::Vector{Float64}
    Pℓ::Matrix{Float64}
    Tc::Matrix{Float64}
end

function (rop::RealSpace)(A)
    Diagonal(vec(
        inv_chebyshev_normalizedlegendre_transform(
            A, rop.r_chebyshev, rop.theta; rop.Pℓ, rop.Tc)))
end

datadir(f) = joinpath(DATADIR[], f)

# Legedre expansion constants
α⁺ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((ℓ - m + 1) * (ℓ + m + 1) / ((2ℓ + 1) * (2ℓ + 3)))))
α⁻ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((ℓ - m) * (ℓ + m) / ((2ℓ - 1) * (2ℓ + 1)))))

β⁻ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((2ℓ + 1) / (2ℓ - 1) * (ℓ^2 - m^2))))
γ⁺ℓm(ℓ, m) = ℓ * α⁺ℓm(ℓ, m)
γ⁻ℓm(ℓ, m) = ℓ * α⁻ℓm(ℓ, m) - β⁻ℓm(ℓ, m)

chebyshevnodes_lobatto(n) = [cos(pi*j/n) for j in n:-1:0]

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

function chebyshev_lobatto_forward_inverse(n)
    Tcinv = zeros(n+1, n+1)
    Tcfwd = zeros(n+1, n+1)
    Tcinvo = indexedfromzero(Tcinv)
    Tcfwdo = indexedfromzero(Tcfwd)
    for k in axes(Tcinvo, 2), j in axes(Tcinvo, 1)
        Tcinvo[j, k] = cos(pi*j*k/n)
    end
    for j in axes(Tcfwdo, 2), k in axes(Tcfwdo, 1)
        cj = j == 0 || j == n ? 2 : 1
        ck = k == 0 || k == n ? 2 : 1
        Tcfwdo[k, j] = 2/(n*cj*ck) * cos(pi*j*k/n)
    end
    # reverse the matrices, as we order the chebyshev nodes in increasing order
    return reverse(Tcfwd, dims=2), reverse(Tcinv, dims=1)
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

function chebyshevmatrix(f::Fun, nr, scalefactor = 2)::Matrix{Float64}
    chebyshevmatrix(ApproxFun.Multiplication(f), nr)
end

function chebyshevmatrix(A, nr, scalefactor = 2)::Matrix{Float64}
    B = A:ApproxFun.Chebyshev()
    BM = B[1:scalefactor*nr, 1:scalefactor*nr]
    C = zeros(eltype(BM), nr, nr)
    B_rangespace = rangespace(B)
    if B_rangespace isa ApproxFun.Ultraspherical
        order = B_rangespace.order
        for colind in axes(C, 2)
            col = @view BM[:, colind]
            C[:, colind] = @view FastTransforms.ultra2cheb(col, order)[axes(C, 1)]
        end
    elseif B_rangespace isa ApproxFun.Chebyshev
        C .= @view BM[axes(C)...]
    end
    return C
end

# differentiation operator using N+1 points, with 2 extremal points
function chebyderivGaussLobatto(n)
    D = zeros(n+1, n+1)
    Do = indexedfromzero(D)
    Do[0,0] = (2n^2 + 1)/6
    Do[n,n] = -Do[0,0]
    for j in 0:n, k in 0:n
        k == j && (k == 0 || k == n) && continue
        xk = cos(π*k/n)
        if k == j
            Do[k,k] = -1/2 * xk/(1-xk^2)
        else
            xj = cos(π*j/n)
            cj = 1 + (j == 0) + (j == n)
            ck = 1 + (k == 0) + (k == n)
            Do[k, j] = ck/cj * (-1)^(j+k)/(xk - xj)
        end
    end
    return D
end

# Legendre

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
    vo = indexedfromzero(v)
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
    Tc = zeros(size(Fnℓ, 1), length(r_chebyshev)))

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
    Pcache = zeros(ℓs)
    for (θind, costhetai) in enumerate(costheta)
        collectPlm!(Pcache, costhetai; m, norm = Val(:normalized))
        for (ℓind, ℓ) in enumerate(ℓs)
            Pℓm = Pcache[ℓ]
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

function constraintmatrix(operators; entropy_outer_boundary = :neumann)
    (; radial_params) = operators
    (; nr, r_in, r_out, nℓ, Δr, nparams) = radial_params

    nradconstraints = 2
    nconstraints = nradconstraints * nℓ

    # Radial constraint
    M = zeros(2nradconstraints, nr)
    MVn = @view M[1:nradconstraints, :]
    MVno = OffsetArray(MVn, :, 0:nr-1)
    MWn = @view M[nradconstraints.+(1:nradconstraints), :]
    MWno = OffsetArray(MWn, :, 0:nr-1)
    MSn = MWn
    MSn = zero(MWn)
    MSno = OffsetArray(MSn, :, 0:nr-1)

    (; nfields) = operators.constants
    BC = zeros(nfields * nconstraints, nfields * nparams)

    # constraints on V, Robin
    for n = 0:nr-1
        # inner boundary
        # impenetrable, stress-free
        MVno[1, n] = (-1)^n * (n^2 + Δr / r_in)
        # outer boundary
        # impenetrable, stress-free
        MVno[2, n] = n^2 - Δr / r_out
    end

    # constraints on W, Dirichlet
    for n = 0:nr-1
        # inner boundary
        # impenetrable, stress-free
        # equivalently, zero Dirichlet
        MWno[1, n] = (-1)^n
        # outer boundary
        # impenetrable, stress-free
        # equivalently, zero Dirichlet
        MWno[2, n] = 1
    end

    # constraints on S
    # zero Neumann
    for n = 0:nr-1
        # inner boundary
        MSno[1, n] = (-1)^n*n^2
        # outer boundary
        MSno[2, n] = n^2
    end

    fieldmatrices = [MVn, MWn, MSn][1:nfields]
    for ℓind = 1:nℓ
        indstart = (ℓind - 1)*nr + 1
        indend = (ℓind - 1)*nr + nr
        for (fieldno, M) in enumerate(fieldmatrices)
            rowinds = (fieldno - 1) * nconstraints + nradconstraints*(ℓind-1).+ (1:nradconstraints)
            colinds = (fieldno - 1) * nparams .+ (indstart:indend)
            BC[rowinds, colinds] = M
        end
    end

    # ZC = constraintnullspacematrix(BC)
    ZC = nullspace(BC)

    (; BC, ZC, nfields)
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
    Mo = indexedfromzero(M)
    for n in 0:nℓ-1, k in n-1:-2:0
        Mo[n, k] = -√((2n + 1) * (2k + 1))
    end
    return Matrix(M')
end

# defined for normalized Legendre polynomials
function cottheta_dtheta_operator(nℓ)
    M = zeros(nℓ, nℓ)
    Mo = indexedfromzero(M)
    for n in 0:nℓ-1
        Mo[n, n] = -n
        for k in n-2:-2:0
            Mo[n, k] = -√((2n + 1) * (2k + 1))
        end
    end
    Matrix(M')
end

# converts from Heinrichs basis to Chebyshev, so Cn = Anm Bm where Cn are the Chebyshev
# coefficients, and Bm are the Heinrichs ones
αheinrichs(n) = n < 0 ? 0 : n == 0 ? 2 : 1
function heinrichs_chebyshev_matrix(n)
    M = zeros(n, n-2)
    Mo = indexedfromzero(M)
    for n in axes(Mo,1), m in intersect(n-2:2:n+2, axes(Mo,2))
        T = 0.0
        if m == n-2
            T = -1/4 * αheinrichs(m)
        elseif m == n
            T = 1 - 1/4*(αheinrichs(n-1) + αheinrichs(n))
        elseif m == n + 2
            T = -1/4
        end
        Mo[n, m] = T
    end
    return M
end

function chebyshevneumann_chebyshev_matrix(n)
    M = zeros(n-2, n)
    Mo = indexedfromzero(M)
    for n in axes(Mo, 1)
        Mo[n, n] = 1
        Mo[n, n+2] = -(n/(n+2))^2
    end
    permutedims(M)
end

deltafn(x, y; scale) = exp(-(x/scale-y/scale)^2/2)

function deltafn_matrix(pts; scale)
    n = length(pts) - 1
    δ = deltafn.(pts, pts'; scale)
    for col in axes(δ, 2)
        v = @view δ[:, col]
        s = Spline1D(pts, v)
        is = Dierckx.integrate(s, pts[1], pts[end])
        v ./= is
    end
    # zero out the first and last rows to enfore boundary conditions
    δ[1, :] .= 0
    δ[end, :] .= 0
    # zero out the first and last columns to enfore symmetry of the Green function
    δ[:, 1] .= 0
    δ[:, end] .= 0
    return δ
end

function Bℓ(ℓ, operators)
    (; onebyr2_cheby) = operators.rad_terms

    # Chebyshev Lobatto points, used in computing the Green function
    (; r_chebyshev_lobatto) = operators.coordinates
    (; ddrDDr_lobatto) = operators.diff_operator_matrices

    Bℓ = ddrDDr_lobatto - ℓ * (ℓ + 1) * Diagonal(onebyr2_cheby.(r_chebyshev_lobatto))
    Bℓ .*= Rsun^2
    scale = maximum(abs, @view Bℓ[2:end-1, 2:end-1])
    Bℓ ./= scale

    # boundaries
    Bℓ[1, :] .= 0
    Bℓ[1, 1] = 1
    Bℓ[end, :] .= 0
    Bℓ[end, end] = 1

    return Bℓ, scale
end

function greenfn_radial_lobatto(ℓ, operators)
    (;  Δr) = operators.radial_params
    (; r_lobatto) = operators.coordinates
    (; deltafn_matrix_radial) = operators

    B, scale = Bℓ(ℓ, operators)

    H = B \ deltafn_matrix_radial
    H .*= (Δr/2) / scale
    # the Δr/2 factor is used to convert the subsequent integrals ∫H f dr to ∫(Δr/2)H f dx,
    # where x = (r - rmid)/(Δr/2)
    return H
end

struct UniformRotGfn end
struct RadDiffRotGfn end

function greenfn_cheby(::UniformRotGfn, ℓ, operators)
    (; r_chebyshev_lobatto, r_lobatto) = operators.coordinates
    (; lobattochebyshevtransform) = operators.transforms
    (; ddr_lobatto) = operators.diff_operator_matrices
    (; ηρ_cheby, ηρ_by_r, ηρ2_by_r2, ηρ_by_r2, g_cheby, ddr_ηρbyr2, ddr_ηρbyr,
            ηρ_by_r3, ddr_ηρ, d2dr2_ηρ, d3dr3_ηρ,
            onebyr_cheby, onebyr2_cheby, onebyr3_cheby) = operators.rad_terms;
    (; d2dr2, ddr) = operators.diff_operators;

    H = greenfn_radial_lobatto(ℓ, operators)

    ddrHrr = permutedims(ddr_lobatto * permutedims(H))

    ηρrHrr = H .* ηρ_cheby.(r_chebyshev_lobatto)'

    twoddr_plus_3ηρr_H = @. 2ddrHrr + 3ηρrHrr

    H_ddrηρ = H .* ddr_ηρ.(r_chebyshev_lobatto)'
    H_ηρbyr = H .* ηρ_by_r.(r_chebyshev_lobatto)'
    H_ηρbyr2 = H .* ηρ_by_r2.(r_chebyshev_lobatto)'
    H_ηρbyr3 = H .* ηρ_by_r3.(r_chebyshev_lobatto)'
    H_ddrηρbyr2 = H .* ddr_ηρbyr2.(r_chebyshev_lobatto)'
    H_ddrηρ_by_r = H_ddrηρ .* onebyr_cheby.(r_chebyshev_lobatto)'
    H_ddrηρ_by_r2 = H_ddrηρ .* onebyr2_cheby.(r_chebyshev_lobatto)'
    H_d2dr2ηρ_by_r = H .* (d2dr2_ηρ.(r_chebyshev_lobatto) .* onebyr_cheby.(r_chebyshev_lobatto))'
    H_ddrηρbyr2_plus_4ηρ_by_r3 = @. H_ddrηρbyr2 + 4 * H_ηρbyr3
    Hηρ2_by_r2 = H .* ηρ2_by_r2.(r_chebyshev_lobatto)'
    T = (r -> ηρ_cheby(r) * (ηρ_cheby(r) - 2onebyr_cheby(r))).(r_chebyshev_lobatto)
    H_ηρ²_min_2ηρbyr = H .* T'
    T = (r -> ddr_ηρ(r) * (ηρ_cheby(r) - 2onebyr_cheby(r))).(r_chebyshev_lobatto)
    H_ηρ_min_2byr_ddrηρ = H .* T'

    H_by_r = H ./ r_lobatto'

    H_times_2byr_min_ηρ = H .* (2 ./ r_lobatto .- ηρ_cheby.(r_chebyshev_lobatto))'

    T = (r -> (ddr_ηρ(r) - 2ηρ_by_r(r))*ddr_ηρ(r)).(r_chebyshev_lobatto)
    H_c1 = H .* T'

    T = (r -> 2(ddr_ηρ(r) - ηρ_by_r(r) + onebyr2_cheby(r))*ηρ_cheby(r)).(r_chebyshev_lobatto)
    H_c2_1 = H .* T'

    T = (r -> (-2*ddr_ηρbyr(r) + d2dr2_ηρ(r))*ηρ_cheby(r)).(r_chebyshev_lobatto)
    H_c2_2 = H .* T'

    T = (r -> 3ddr_ηρ(r) - 4ηρ_by_r(r)).(r_chebyshev_lobatto)
    H_a1_1 = H .* T'
    T = (r -> 3d2dr2_ηρ(r) - 8ddr_ηρ(r)*onebyr_cheby(r) + 8ηρ_cheby(r)*onebyr2_cheby(r)).(r_chebyshev_lobatto)
    H_a1_2 = H .* T'
    T = (r -> d3dr3_ηρ(r) - 4d2dr2_ηρ(r)*onebyr_cheby(r) +
            8ddr_ηρ(r)*onebyr2_cheby(r) - 8ηρ_cheby(r)*onebyr3_cheby(r) ).(r_chebyshev_lobatto)
    H_a1_3 = H .* T'

    H_g = H .* g_cheby.(r_chebyshev_lobatto)'

    rad_terms = (; H, twoddr_plus_3ηρr_H, H_times_2byr_min_ηρ,
        H_ηρbyr, H_ηρbyr2, H_ddrηρbyr2, H_by_r, Hηρ2_by_r2, ddrHrr, H_g)

    J = lobattochebyshevtransform(H)
    J_by_r = lobattochebyshevtransform(H_by_r)
    J_ηρ = lobattochebyshevtransform(ηρrHrr)
    J_ηρbyr = lobattochebyshevtransform(H_ηρbyr)
    J_ηρ2byr2 = lobattochebyshevtransform(Hηρ2_by_r2)
    J_ηρbyr2 = lobattochebyshevtransform(H_ηρbyr2)
    J_ηρbyr3 = lobattochebyshevtransform(H_ηρbyr3)
    J_4ηρbyr3 = 4 * J_ηρbyr3
    J_ddrηρ = lobattochebyshevtransform(H_ddrηρ)
    J_ddrηρ_by_r = lobattochebyshevtransform(H_ddrηρ_by_r)
    J_ddrηρ_by_r2 = lobattochebyshevtransform(H_ddrηρ_by_r2)
    J_d2dr2ηρ_by_r = lobattochebyshevtransform(H_d2dr2ηρ_by_r)
    J_d2dr2ηρ_by_r_min_2ddrηρ_by_r2 = @. J_d2dr2ηρ_by_r - 2J_ddrηρ_by_r2
    J_ddrηρbyr2_plus_4ηρbyr3 = lobattochebyshevtransform(H_ddrηρbyr2_plus_4ηρ_by_r3)
    J_ddrηρ_by_r_min_ηρbyr2 = @. J_ddrηρ_by_r - J_ηρbyr2
    J_ddrηρ_by_r2_min_4ηρbyr3 = @. J_ddrηρ_by_r2 - J_4ηρbyr3
    J_g = lobattochebyshevtransform(H_g)
    J_ηρ²_min_2ηρbyr = lobattochebyshevtransform(H_ηρ²_min_2ηρbyr)
    J_ηρ_min_2byr_ddrηρ = lobattochebyshevtransform(H_ηρ_min_2byr_ddrηρ)
    J_c1 = lobattochebyshevtransform(H_c1)
    J_c2_1 = lobattochebyshevtransform(H_c2_1)
    J_c2_2 = lobattochebyshevtransform(H_c2_2)
    J_a1_1 = lobattochebyshevtransform(H_a1_1)
    J_a1_2 = lobattochebyshevtransform(H_a1_2)
    J_a1_3 = lobattochebyshevtransform(H_a1_3)

    viscosity_terms = (; J_c1, J_c2_1, J_c2_2, J_a1_1, J_a1_2, J_a1_3,
        J_4ηρbyr3, J_ηρ,
        J_ηρ²_min_2ηρbyr, J_ηρ_min_2byr_ddrηρ, J_ddrηρ,
        J_ddrηρbyr2_plus_4ηρbyr3, J_ηρ2byr2,
        J_ηρbyr2, J_d2dr2ηρ_by_r_min_2ddrηρ_by_r2,
        J_ddrηρ_by_r, J_ddrηρ_by_r_min_ηρbyr2,
        J_ddrηρ_by_r2_min_4ηρbyr3)

    unirot_terms = (; J, J_by_r, J_g, J_ηρbyr)

    return (; rad_terms, unirot_terms, viscosity_terms)
end

function greenfn_cheby(::RadDiffRotGfn, ℓ, operators, ΔΩprofile_deriv,
        G = greenfn_cheby(UniformRotGfn(), ℓ, operators))

    (; r_chebyshev_lobatto, r_lobatto) = operators.coordinates
    (; lobattochebyshevtransform) = operators.transforms
    (; ηρ_cheby) = operators.rad_terms

    (; ΔΩ, ddrΔΩ, d2dr2ΔΩ) = ΔΩprofile_deriv

    (; rad_terms, unirot_terms) = G

    (; H, H_ηρbyr, H_times_2byr_min_ηρ, twoddr_plus_3ηρr_H) = rad_terms
    H_ηρbyr_ΔΩ = H_ηρbyr .* ΔΩ.(r_chebyshev_lobatto)'
    J_ηρbyr_ΔΩ = lobattochebyshevtransform(H_ηρbyr_ΔΩ)

    H_ddrΩ = H .* ddrΔΩ.(r_chebyshev_lobatto)'
    J_ddrΩ = lobattochebyshevtransform(H_ddrΩ)

    H_2byr_min_ηρ__min__twoddr_plus_3ηρr_H__times_drΔΩ =
        (H_times_2byr_min_ηρ .- twoddr_plus_3ηρr_H) .* ddrΔΩ.(r_chebyshev_lobatto)'
    J_2byr_min_ηρ__min__twoddr_plus_3ηρr_J__times_drΔΩ =
        lobattochebyshevtransform(H_2byr_min_ηρ__min__twoddr_plus_3ηρr_H__times_drΔΩ)

    twoddr_plus_3ηρr_H_ddrΔΩ = twoddr_plus_3ηρr_H .* ddrΔΩ.(r_chebyshev_lobatto)'
    twoddr_plus_3ηρr_J_ddrΔΩ = lobattochebyshevtransform(twoddr_plus_3ηρr_H_ddrΔΩ)

    H_twoΔΩ_by_r = H .*  (2 .* ΔΩ.(r_chebyshev_lobatto) ./ r_lobatto)'
    J_twoΔΩ_by_r = lobattochebyshevtransform(H_twoΔΩ_by_r)

    H_ΔΩ = H .* ΔΩ.(r_chebyshev_lobatto)'
    J_ΔΩ = lobattochebyshevtransform(H_ΔΩ)

    H_d2dr2ΔΩ = H .* d2dr2ΔΩ.(r_chebyshev_lobatto)'
    J_d2dr2ΔΩ = lobattochebyshevtransform(H_d2dr2ΔΩ)

    H_ddrΩ_ηρ = H_ddrΩ .* ηρ_cheby.(r_chebyshev_lobatto)'
    J_ddrΩ_ηρ = lobattochebyshevtransform(H_ddrΩ_ηρ)

    twoddr_plus_3ηρr_J = lobattochebyshevtransform(twoddr_plus_3ηρr_H)

    J_times_2byr_min_ηρ = lobattochebyshevtransform(H_times_2byr_min_ηρ)

    diffrot_terms = (; J_ηρbyr_ΔΩ, J_ddrΩ,
        J_twoΔΩ_by_r, J_d2dr2ΔΩ, J_ddrΩ_ηρ,
        J_2byr_min_ηρ__min__twoddr_plus_3ηρr_J__times_drΔΩ,
        twoddr_plus_3ηρr_J_ddrΔΩ, J_ΔΩ, twoddr_plus_3ηρr_J,
        J_times_2byr_min_ηρ)

    (; unirot_terms, diffrot_terms)
end

splderiv(v::Vector, r::Vector, rout = r; nu = 1) = splderiv(Spline1D(r, v), rout; nu = 1)
splderiv(spl::Spline1D, rout::Vector; nu = 1) = derivative(spl, rout; nu = 1)

smoothed_spline(r, v; s) = Spline1D(r, v, s = sum(abs2, v) * s)

function read_solar_model(; r_in = 0.7Rsun, r_out = Rsun, _stratified #= only for tests =# = true)
    ModelS = readdlm(joinpath(@__DIR__, "ModelS.detailed"))
    r_modelS = @view ModelS[:, 1]
    r_inds = r_in .<= r_modelS .<= r_out
    r_modelS = reverse(r_modelS[r_inds])
    q_modelS = exp.(reverse(ModelS[r_inds, 2]))
    T_modelS = reverse(ModelS[r_inds, 3])
    ρ_modelS = reverse(ModelS[r_inds, 5])
    if !_stratified
        ρ_modelS = fill(ρ_modelS[1], length(ρ_modelS))
        T_modelS = fill(T_modelS[1], length(T_modelS))
    end
    logρ_modelS = log.(ρ_modelS)
    logT_modelS = log.(T_modelS)

    sρ = Spline1D(r_modelS, ρ_modelS)
    slogρ = smoothed_spline(r_modelS, logρ_modelS, s = 1e-5)
    ddrlogρ = Dierckx.derivative(slogρ, r_modelS)
    if !_stratified
        ddrlogρ .= 0
    end
    sηρ = smoothed_spline(r_modelS, ddrlogρ, s = 1e-5)

    sT = Spline1D(r_modelS, T_modelS)
    slogT = smoothed_spline(r_modelS, logT_modelS, s = 1e-5)
    ddrlogT = Dierckx.derivative(slogT, r_modelS)
    if !_stratified
        ddrlogT .= 0
    end
    sηT = smoothed_spline(r_modelS, ddrlogT, s = 1e-5)

    g_modelS = @. G * Msun * q_modelS / r_modelS^2
    sg = Spline1D(r_modelS, g_modelS, s = sum(abs2, g_modelS))
    (; sρ, sT, sg, sηρ, sηT)
end

struct SpectralOperatorForm
    fwd::Matrix{Float64}
    inv::Matrix{Float64}
end

function (op::SpectralOperatorForm)(A::Diagonal)
    op.fwd * A * op.inv
end
function (op::SpectralOperatorForm)(A::AbstractVector)
    op(Diagonal(A))
end

const Tmul = TimesOperator{Float64,Tuple{InfiniteCardinal{0},InfiniteCardinal{0}}}
const Tplus = PlusOperator{Float64,Tuple{InfiniteCardinal{0},InfiniteCardinal{0}}}
const TFunSpline = Fun{Chebyshev{ChebyshevInterval{Float64},Float64},Float64,Vector{Float64}}

struct LobattoChebyshev
    TfGL_nr :: Matrix{Float64}
    TiGL_nr :: Matrix{Float64}
    normr :: Vector{Float64}
end
(L::LobattoChebyshev)(A::Matrix) = L.TfGL_nr * (A .* L.normr') * L.TiGL_nr

function radial_operators(nr, nℓ; r_in_frac = 0.7, r_out_frac = 1, _stratified = true, nfields = 3, ν = 1e10)
    _radial_operators(nr, nℓ, r_in_frac, r_out_frac, _stratified, nfields, ν)
end
function _radial_operators(nr, nℓ, r_in_frac, r_out_frac, _stratified, nfields, ν)
    r_in = r_in_frac * Rsun
    r_out = r_out_frac * Rsun
    radial_params = parameters(nr, nℓ; r_in, r_out)
    (; Δr, nchebyr, r_mid) = radial_params
    r, Tcrfwd, Tcrinv = chebyshev_forward_inverse(nr, r_in, r_out)

    pseudospectralop_radial = SpectralOperatorForm(Tcrfwd, Tcrinv)
    r_chebyshev = (r .- r_mid) ./ (Δr / 2)
    Tcrfwdc, Tcrinvc = complex.(Tcrfwd), complex.(Tcrinv)
    r_cheby = Fun(ApproxFun.Chebyshev(), [r_mid, Δr / 2])
    r2_cheby = r_cheby * r_cheby

    (; sρ, sg, sηρ, sT, sηT) = read_solar_model(; r_in, r_out, _stratified)

    T = sT.(r)

    ddr = ApproxFun.Derivative() * (2 / Δr)
    rddr = (r_cheby * ddr)::Tmul
    d2dr2 = (ddr * ddr)::Tmul
    d3dr3 = (ddr * d2dr2)::Tmul
    d4dr4 = (d2dr2 * d2dr2)::Tmul
    r2d2dr2 = (r2_cheby * d2dr2)::Tmul

    # density stratification
    ρ = sρ.(r)
    ηρ = sηρ.(r)
    ηρ_cheby = ApproxFun.chop(Fun(sηρ ∘ r_cheby, ApproxFun.Chebyshev()), 1e-3)::TFunSpline
    if ncoefficients(ηρ_cheby) == 0
        ηρ_cheby = Fun(ApproxFun.Chebyshev(), [1e-100])::TFunSpline
    end
    ηT = sηT.(r)
    ηT_cheby = ApproxFun.chop(Fun(sηT ∘ r_cheby, ApproxFun.Chebyshev()), 1e-2)::TFunSpline
    if ncoefficients(ηT_cheby) == 0
        ηT_cheby = Fun(ApproxFun.Chebyshev(), [1e-100])::TFunSpline
    end

    DDr = (ddr + ηρ_cheby)::Tplus
    rDDr = (r_cheby * DDr)::Tmul
    D2Dr2 = (DDr * DDr)::Tmul

    ddrDDr = (ddr * DDr)::Tmul

    onebyr = 1 ./ r
    onebyr_cheby = (1 / r_cheby)::typeof(r_cheby)
    onebyr2_cheby = (onebyr_cheby*onebyr_cheby)::typeof(r_cheby)
    onebyr3_cheby = (onebyr2_cheby*onebyr_cheby)::typeof(r_cheby)
    onebyr4_cheby = (onebyr2_cheby*onebyr2_cheby)::typeof(r_cheby)
    DDr_minus_2byr = (DDr - 2onebyr_cheby)::Tplus
    ddr_plus_2byr = (ddr + 2onebyr_cheby)::Tplus

    ηρ_by_r = onebyr_cheby * ηρ_cheby
    twoηρ_by_r = 2ηρ_by_r

    ηρ_by_r2 = onebyr2_cheby * ηρ_cheby
    ηρ_by_r3 = onebyr_cheby * ηρ_by_r2
    ddr_ηρ = ddr * ηρ_cheby
    ddr_ηρbyr = ddr * ηρ_by_r
    d2dr2_ηρ = (d2dr2 * ηρ_cheby)::typeof(ddr_ηρbyr)
    d3dr3_ηρ = (d3dr3 * ηρ_cheby)::typeof(ddr_ηρbyr)
    ddr_ηρbyr2 = ddr * ηρ_by_r2
    ηρ2_by_r2 = ApproxFun.chop(ηρ_by_r2 * ηρ_cheby, 1e-3)

    g = sg.(r)
    g_cheby = Fun(sg ∘ r_cheby, Chebyshev())::TFunSpline

    Ω0 = RossbyWaveSpectrum.equatorial_rotation_angular_velocity(r_out_frac)

    # viscosity
    ν /= Ω0*Rsun^2

    κ = ν
    ddr_lnρT = ηρ_cheby + ηT_cheby

    γ = 1.64
    cp = 1.7e8
    δ_superadiabatic = superadiabaticity.(r)
    ddr_S0_by_cp = ApproxFun.chop(Fun(ApproxFun.Chebyshev(),
        Tcrfwd * @. γ * δ_superadiabatic * ηρ / cp), 1e-3)

    Ir = I(nchebyr)
    Iℓ = I(nℓ)

    mat = x -> chebyshevmatrix(x, nr)

    # matrix forms of operators
    onebyr_chebyM = mat(onebyr_cheby)
    onebyr2_chebyM = mat(onebyr2_cheby)
    DDrM = mat(DDr)
    ddrM = mat(ddr)
    d2dr2M = mat(d2dr2)
    d3dr3M = mat(d3dr3)
    d4dr4M = mat(d4dr4)
    ddrDDrM = mat(ddrDDr)
    rddrM = mat(rddr)
    ddrDDrM = mat(ddrDDr)
    twoηρ_by_rM = mat(twoηρ_by_r)
    ddr_plus_2byrM = @. ddrM + 2 * onebyr_chebyM
    ddr_minus_2byrM = @. ddrM - 2 * onebyr_chebyM
    DDr_minus_2byrM = mat(DDr_minus_2byr)
    gM = mat(g_cheby)
    grddrM = mat((g_cheby * rddr)::Tmul)

    ηρ_by_rM = mat(ηρ_by_r)
    ηρ2_by_r2M = mat(ηρ2_by_r2)

    HeinrichsChebyshevMatrix = heinrichs_chebyshev_matrix(nr)

    # gauss lobatto points
    n_lobatto = 4nr
    Tcf, Tci = chebyshev_lobatto_forward_inverse(n_lobatto)
    TfGL_nr = Tcf[1:nr, :]
    TiGL_nr = Tci[:, 1:nr]
    r_chebyshev_lobatto = chebyshevnodes_lobatto(n_lobatto)
    r_lobatto = @. (Δr/2) * r_chebyshev_lobatto .+ r_mid
    ddr_lobatto = reverse(chebyderivGaussLobatto(n_lobatto) * (2 / Δr))
    d2dr2_lobatto = ddr_lobatto*ddr_lobatto
    ddrDDr_lobatto = d2dr2_lobatto + Diagonal(ηρ_cheby.(r_chebyshev_lobatto)) * ddr_lobatto +
                        Diagonal((ddr * ηρ_cheby).(r_chebyshev_lobatto))

    normr = sqrt.(1 .- r_chebyshev_lobatto.^2) * pi/n_lobatto
    lobattochebyshevtransform = LobattoChebyshev(TfGL_nr, TiGL_nr, normr)

    deltafn_matrix_radial = deltafn_matrix(r_lobatto, scale = Rsun*1e-3)

    # scalings = (; Sscaling = 1, Wscaling = 1)
    scalings = (; Sscaling = 1e6, Wscaling = 5e2)

    constants = (; κ, ν, nfields, Ω0, scalings)
    identities = (; Ir, Iℓ)
    coordinates = (; r, r_chebyshev, r_chebyshev_lobatto, r_lobatto)

    transforms = (; Tcrfwd, Tcrinv, Tcrfwdc, Tcrinvc, pseudospectralop_radial, TfGL_nr,
            TiGL_nr, n_lobatto, lobattochebyshevtransform)

    rad_terms = (; onebyr, onebyr_cheby, ηρ, ηρ_cheby, ηT_cheby,
        onebyr2_cheby, onebyr3_cheby, onebyr4_cheby,
        ddr_lnρT, ddr_S0_by_cp, g, g_cheby, r_cheby, r2_cheby, κ, twoηρ_by_r, sρ,
        ηρ_by_r, ηρ_by_r2, ηρ2_by_r2, ddr_ηρbyr, ddr_ηρbyr2, ηρ_by_r3,
        ddr_ηρ, d2dr2_ηρ, d3dr3_ηρ)

    diff_operators = (; DDr, D2Dr2, DDr_minus_2byr, rDDr, rddr,
        ddr, d2dr2, d3dr3, d4dr4, r2d2dr2, ddrDDr, ddr_plus_2byr)

    diff_operator_matrices = (; onebyr_chebyM, onebyr2_chebyM, DDrM,
        ddrM, d2dr2M, ddrDDrM, rddrM, twoηρ_by_rM, ddr_plus_2byrM,
        ddr_minus_2byrM, DDr_minus_2byrM,
        grddrM, gM,
        ddr_lobatto,
        ddrDDr_lobatto,
        ηρ_by_rM, ηρ2_by_r2M, d3dr3M, d4dr4M,
    )

    (;
        constants, rad_terms,
        diff_operators,
        transforms, coordinates,
        radial_params, identities,
        diff_operator_matrices,
        mat,
        HeinrichsChebyshevMatrix,
        _stratified,
        deltafn_matrix_radial
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

function uniform_rotation_matrix(nr, nℓ, m; operators, kw...)
    nparams = nr * nℓ
    nfields = operators.constants.nfields
    M = zeros(ComplexF64, nfields * nparams, nfields * nparams)
    uniform_rotation_matrix!(M, nr, nℓ, m; operators, kw...)
    return M
end

function uniform_rotation_matrix_terms_outer!((SWterm, SSterm),
    (ℓ, m), nchebyr,
    (ddrDDrM, onebyr2_IplusrηρM, gM, κ_∇r2_plus_ddr_lnρT_ddrM, κ_by_r2M,
        onebyr2_cheby_ddr_S0_by_cpM), Ω0)

    ℓℓp1 = ℓ*(ℓ+1)

    @. SWterm = ℓℓp1 * onebyr2_cheby_ddr_S0_by_cpM * Rsun^3
    @. SSterm = (κ_∇r2_plus_ddr_lnρT_ddrM - ℓℓp1 * κ_by_r2M) * Rsun^2

    SWterm, SSterm
end

function uniform_rotation_matrix!(M, nr, nℓ, m; operators, kw...)
    (; nfields, Ω0, scalings) = operators.constants;
    (; nchebyr, r_out) = operators.radial_params;
    (; ddr, d2dr2) = operators.diff_operators;
    (; onebyr2_cheby, onebyr_cheby, ddr_lnρT, κ,
        ηρ_cheby, r_cheby, ddr_S0_by_cp) = operators.rad_terms;
    (; ddrDDrM, ddrM, DDrM, onebyr2_chebyM,
        onebyr_chebyM, grddrM, gM, ddr_minus_2byrM,
        DDr_minus_2byrM, ηρ_by_rM) = operators.diff_operator_matrices;
    (; mat) = operators;
    (; Sscaling, Wscaling) = scalings;

    _greenfn = get(kw, :_greenfn, true)

    Cℓ′ = zeros(nr, nr)
    WVℓℓ′ = zeros(nr, nr)
    VWℓℓ′ = zeros(nr, nr)
    JC1 = zeros(nr, nr)
    JCℓ′ = zeros(nr, nr)
    Jddr = zeros(nr, nr)

    M .= 0

    VV = matrix_block(M, 1, 1, nfields)
    VW = matrix_block(M, 1, 2, nfields)
    WV = matrix_block(M, 2, 1, nfields)
    WW = matrix_block(M, 2, 2, nfields)
    # the following are only valid if S is included
    if nfields === 3
        WS = matrix_block(M, 2, 3, nfields)
        SW = matrix_block(M, 3, 2, nfields)
        SS = matrix_block(M, 3, 3, nfields)
    end

    ℓs = range(m, length = nℓ)
    cosθ = OffsetArray(costheta_operator(nℓ, m), ℓs, ℓs);
    sinθdθ = OffsetArray(sintheta_dtheta_operator(nℓ, m), ℓs, ℓs);

    onebyr2_IplusrηρM = mat((1 + ηρ_cheby * r_cheby) * onebyr2_cheby);

    onebyr2_cheby_ddr_S0_by_cpM = mat(onebyr2_cheby * ddr_S0_by_cp);
    ∇r2_plus_ddr_lnρT_ddr = d2dr2 + 2onebyr_cheby*ddr + ddr_lnρT * ddr;
    κ_∇r2_plus_ddr_lnρT_ddrM = κ * chebyshevmatrix(∇r2_plus_ddr_lnρT_ddr, nr, 3);
    κ_by_r2M = κ .* onebyr2_chebyM;

    SWterm = zeros(nr, nr);
    SSterm = zeros(nr, nr);

    @views for ℓ in ℓs
        # numerical green function
        (; J, J_ηρbyr, J_by_r, J_g) = greenfn_cheby(UniformRotGfn(), ℓ, operators).unirot_terms
        mul!(Jddr, J, ddrM)
        if _greenfn
            @. JC1 = Jddr - 2J_by_r
        else
            @. JC1 = ddrM - 2onebyr_chebyM
        end

        ℓℓp1 = ℓ * (ℓ + 1)

        blockdiaginds_ℓ = blockinds((m, nr), ℓ)

        uniform_rotation_matrix_terms_outer!((SWterm, SSterm),
                (ℓ, m), nchebyr,
                (ddrDDrM, onebyr2_IplusrηρM, gM, κ_∇r2_plus_ddr_lnρT_ddrM, κ_by_r2M,
                    onebyr2_cheby_ddr_S0_by_cpM), Ω0);

        diagterm = 2m/ℓℓp1

        VVblockdiag = VV[blockdiaginds_ℓ]
        VVblockdiag_diag = VVblockdiag[diagind(VVblockdiag)]
        VVblockdiag_diag .= diagterm

        WWblockdiag = WW[blockdiaginds_ℓ]
        WWblockdiag_diag = WWblockdiag[diagind(WWblockdiag)]
        WWblockdiag_diag .= diagterm
        @. WW[blockdiaginds_ℓ] -= Rsun^2 * 2m * (_greenfn ? J_ηρbyr : ηρ_by_rM)

        if nfields == 3
            @. WS[blockdiaginds_ℓ] = -(_greenfn ? gM : J_g) / (Ω0^2 * Rsun)  * Wscaling/Sscaling
            @. SW[blockdiaginds_ℓ] = SWterm * Sscaling/Wscaling
            @. SS[blockdiaginds_ℓ] = -im * SSterm
        end

        for ℓ′ in intersect(ℓs, ℓ-1:2:ℓ+1)
            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)

            @. VWℓℓ′ = (-2/ℓℓp1) * (ℓ′ℓ′p1 * DDr_minus_2byrM * cosθ[ℓ, ℓ′] +
                    (DDrM - ℓ′ℓ′p1 * onebyr_chebyM) * sinθdθ[ℓ, ℓ′]) * Rsun / Wscaling

            if _greenfn
                @. JCℓ′ = Jddr - ℓ′ℓ′p1 * J_by_r
            else
                @. JCℓ′ = ddrM - ℓ′ℓ′p1 * onebyr_chebyM
            end
            @. WVℓℓ′ = (-2/ℓℓp1) * (ℓ′ℓ′p1 * JC1 * cosθ[ℓ, ℓ′] + JCℓ′ * sinθdθ[ℓ, ℓ′]) * Rsun * Wscaling

            blockinds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)

            VW[blockinds_ℓℓ′] = VWℓℓ′
            WV[blockinds_ℓℓ′] = WVℓℓ′
        end
    end

    viscosity_terms!(M, nr, nℓ, m; operators, _greenfn)

    return M
end

function viscosity_terms!(M, nr, nℓ, m; operators, _greenfn = true)
    (; nchebyr) = operators.radial_params;
    (; DDr, ddr, d2dr2) = operators.diff_operators;
    (; ddrM, d2dr2M, d3dr3M, d4dr4M, onebyr2_chebyM, ηρ2_by_r2M) = operators.diff_operator_matrices;
    (; onebyr_cheby, onebyr2_cheby, onebyr3_cheby, ηρ_cheby, r_cheby,
        ηρ_by_r, ηρ_by_r2, ηρ_by_r3, ηρ2_by_r2) = operators.rad_terms;
    (; mat) = operators;
    (; ν, nfields) = operators.constants;

    VV = matrix_block(M, 1, 1, nfields)
    WW = matrix_block(M, 2, 2, nfields)

    ℓs = range(m, length = nℓ)

    ddr_minus_2byr = (ddr - 2onebyr_cheby)::Tplus;

    ηρ_ddr_minus_2byrM = mat((ηρ_cheby * ddr_minus_2byr)::Tmul);

    ddr_ηρ_by_r2 = ddr * ηρ_by_r2;
    ddr_ηρ_by_r2_times = ddr[ηρ_by_r2]::Tmul;
    ddr_ηρ_by_r2_timesM = mat(ddr_ηρ_by_r2_times);
    ddr_minus_2byr_DDr = (ddr_minus_2byr * DDr)::Tmul;
    ηρ_cheby_ddr_minus_2byr_DDr = (ηρ_cheby * ddr_minus_2byr_DDr)::Tmul;
    ηρ_by_r2_ddr_minus_2byr = (ηρ_by_r2 * ddr_minus_2byr)::Tmul;
    ddr_ηρ_cheby_ddr_minus_2byr_DDr = (ddr * ηρ_cheby_ddr_minus_2byr_DDr)::Tmul;
    ddr_ηρ_cheby_ddr_minus_2byr_DDrM = chebyshevmatrix(ddr_ηρ_cheby_ddr_minus_2byr_DDr, nr, 3);
    ηρ_by_r2_ddr_minus_2byrM = mat(ηρ_by_r2_ddr_minus_2byr);
    r_cheby_d2dr2_ηρ_by_r = (r_cheby * d2dr2[ηρ_by_r]::Tmul)::Tmul;

    ddr_minus_2byr_ηρ_by_r2 = ddr_minus_2byr[ηρ_by_r2]::Tmul;
    ddr_minus_2byr_ηρ_by_r2M = mat(ddr_minus_2byr_ηρ_by_r2);
    ddr_minus_2byr_r_cheby_d2dr2_ηρ_by_r = (ddr_minus_2byr * r_cheby_d2dr2_ηρ_by_r)::Tmul;
    ddr_minus_2byr_r_d2dr2_ηρ_by_rM = mat(ddr_minus_2byr_r_cheby_d2dr2_ηρ_by_r);
    d2dr2_plus_4ηρ_by_r = 4ηρ_by_r; #+ d2dr2

    d2dr2_d2dr2_plus_4ηρ_by_rM = mat(d2dr2[d2dr2_plus_4ηρ_by_r]);
    one_by_r2_d2dr2_plus_4ηρ_by_rM = mat(onebyr2_cheby * d2dr2_plus_4ηρ_by_r);
    d2dr2_one_by_r2M = mat(d2dr2[onebyr2_cheby]);
    onebyr4_chebyM = mat(onebyr2_cheby*onebyr2_cheby);

    neg_ηρ_by_r2_ddrM = mat(-ηρ_by_r2 * ddr);
    ddr_ηρbyr2_plus_4ηρ_by_r3M = mat(ddr_ηρ_by_r2 + 4ηρ_by_r3);

    ddrηρ_ddr_min_2byr_DDrM = mat((ddr*ηρ_cheby) * ddr_minus_2byr_DDr);
    ηρ_ddr_ddr_min_2byr_DDrM = mat(ηρ_cheby * ddr * ddr_minus_2byr_DDr);

    d2dr2_4ηρ_by_rM = mat(d2dr2[4ηρ_by_r]);
    ηρ_by_r3M = mat(ηρ_by_r * onebyr2_cheby);

    onebyr2_d2dr2M = mat(onebyr2_cheby*d2dr2);
    onebyr3_ddrM = mat(onebyr3_cheby*ddr);
    d2dr2_min_ℓℓp1_by_r2_squaredM = zeros(nr, nr);

    # caches for the WW term
    T3_1 = zeros(nr, nr);
    T3_2 = zeros(nr, nr);
    T3_ℓterms = zeros(nr, nr);
    T4 = zeros(nr, nr);
    WWop = zeros(nr, nr);

    for ℓ in ℓs
        blockdiaginds_ℓ = blockinds((m, nr), ℓ)

        ℓℓp1 = ℓ * (ℓ + 1)
        neg2by3_ℓℓp1 = -2ℓℓp1 / 3

        @views @. VV[blockdiaginds_ℓ] -= im * ν * (d2dr2M - ℓℓp1 * onebyr2_chebyM + ηρ_ddr_minus_2byrM) * Rsun^2

        G = greenfn_cheby(UniformRotGfn(), ℓ, operators);

        (; J, J_ηρbyr) = G.unirot_terms;

        (; J_ηρ2byr2, J_ηρbyr2, J_ddrηρ, J_ηρ, J_ηρ²_min_2ηρbyr, J_4ηρbyr3,
            J_ddrηρbyr2_plus_4ηρbyr3, J_ηρ_min_2byr_ddrηρ,
            J_c1, J_c2_1, J_c2_2,
            J_d2dr2ηρ_by_r_min_2ddrηρ_by_r2, J_ddrηρ_by_r_min_ηρbyr2,
            J_ddrηρ_by_r2_min_4ηρbyr3,
            J_a1_1, J_a1_2, J_a1_3) = G.viscosity_terms;

        ℓpre = (ℓ-2)*ℓ*(ℓ+1)*(ℓ+3)
        @. d2dr2_min_ℓℓp1_by_r2_squaredM = ℓpre * onebyr4_chebyM - 2ℓℓp1*onebyr2_d2dr2M + 4ℓℓp1*onebyr3_ddrM;
        # @. d2dr2_min_ℓℓp1_by_r2_squaredM = d4dr4M + ℓpre * onebyr4_chebyM - 2ℓℓp1*onebyr2_d2dr2M + 4ℓℓp1*onebyr3_ddrM;

        if _greenfn
            mul!(WWop, J, d2dr2_min_ℓℓp1_by_r2_squaredM)
            WWop .+= J_ηρ * d3dr3M .+ J_a1_1 * d2dr2M .+ J_a1_2 * ddrM .+ J_a1_3
            WWop .-= ℓℓp1 .* (J_ηρbyr2 * ddrM .+ J_ddrηρ_by_r2_min_4ηρbyr3)
            WWop .+= 4 .* (J_ηρbyr * d2dr2M .+ 2 .* J_ddrηρ_by_r_min_ηρbyr2 * ddrM .+ J_d2dr2ηρ_by_r_min_2ddrηρ_by_r2)
            @. WWop -= (ℓℓp1-2) * J_4ηρbyr3
        else
            @. WWop = d2dr2_min_ℓℓp1_by_r2_squaredM + ddr_minus_2byr_r_d2dr2_ηρ_by_rM -
                ℓℓp1 * ddr_minus_2byr_ηρ_by_r2M + d2dr2_4ηρ_by_rM - 4ℓℓp1 * ηρ_by_r3M
        end

        if _greenfn
            T3_1 .= J_ddrηρ * d2dr2M .+ J_ηρ_min_2byr_ddrηρ * ddrM .+ J_c1;
            T3_2 .= J_ηρ * d3dr3M .+ J_ηρ²_min_2ηρbyr * d2dr2M .+ J_c2_1 * ddrM .+ J_c2_2;
            T3_ℓterms .= ℓℓp1 .* (-J_ηρbyr2 * ddrM .+ J_ddrηρbyr2_plus_4ηρbyr3)

            @. T4 = neg2by3_ℓℓp1 * J_ηρ2byr2
        else
            @. T3_1 = ddrηρ_ddr_min_2byr_DDrM;
            @. T3_2 = ηρ_ddr_ddr_min_2byr_DDrM;
            @. T3_ℓterms = ℓℓp1 * (neg_ηρ_by_r2_ddrM + ddr_ηρbyr2_plus_4ηρ_by_r3M)

            @. T4 = neg2by3_ℓℓp1 * ηρ2_by_r2M
        end

        @. WWop += (T3_1 + T3_2 + T3_ℓterms) + T4

        @views @. WW[blockdiaginds_ℓ] -= im * ν * WWop * Rsun^4
    end

    return M
end

function interp1d(xin, z, xout; s = 0.0)
    spline = Spline1D(xin, z; s)
    spline(xout)
end

function interp2d(xin, yin, z, xout, yout; s = 0.0)
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
    (; coordinates, radial_params) = operators;
    (; r) = coordinates;
    (; r_out) = radial_params;

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

function matrix_block(M, rowind, colind, nfields = 3)
    nparams = div(size(M, 1), nfields)
    inds1 = (rowind - 1) * nparams .+ (1:nparams)
    inds2 = (colind - 1) * nparams .+ (1:nparams)
    inds = CartesianIndices((inds1, inds2))
    @view M[inds]
end

matrix_block_maximum(f, M, nfields = 3) = [maximum(f, matrix_block(M, i, j)) for i in 1:nfields, j in 1:nfields]

function constant_differential_rotation_terms!(M, nr, nℓ, m; operators = radial_operators(nr, nℓ))
    (; nfields, scalings) = operators.constants
    (; ddrDDrM, ddrM, DDrM, onebyr_chebyM,
        onebyr2_chebyM, twoηρ_by_rM, ddr_plus_2byrM) = operators.diff_operator_matrices

    (; onebyr_cheby, onebyr2_cheby, twoηρ_by_r) = operators.rad_terms
    (; DDr, ddr, ddrDDr, ddr_plus_2byr) = operators.diff_operators
    (; mat) = operators
    (; Wscaling) = scalings

    VV = matrix_block(M, 1, 1, nfields)
    VW = matrix_block(M, 1, 2, nfields)
    WV = matrix_block(M, 2, 1, nfields)
    WW = matrix_block(M, 2, 2, nfields)

    ΔΩ_by_Ω0 = 0.02

    ℓs = range(m, length = nℓ)

    cosθo = OffsetArray(costheta_operator(nℓ, m), ℓs, ℓs)
    sinθdθo = OffsetArray(sintheta_dtheta_operator(nℓ, m), ℓs, ℓs)
    laplacian_sinθdθo = OffsetArray(Diagonal(@. -ℓs * (ℓs + 1)) * parent(sinθdθo), ℓs, ℓs)

    DDr_min_2byrM = @. DDrM - 2 * onebyr_chebyM

    for ℓ in ℓs
        # numerical green function
        (; J, J_ηρbyr, J_by_r) = greenfn_cheby(UniformRotGfn(), ℓ, operators).unirot_terms
        ℓℓp1 = ℓ * (ℓ + 1)
        blockdiaginds_ℓ = blockinds((m, nr), ℓ)
        two_over_ℓℓp1 = 2 / ℓℓp1

        diagterm = m * (two_over_ℓℓp1 - 1) * ΔΩ_by_Ω0

        VVd = @view VV[blockdiaginds_ℓ]
        @views VVd[diagind(VVd)] .+= diagterm

        WWd = @view WW[blockdiaginds_ℓ]
        @views WWd[diagind(WWd)] .+= diagterm
        @. WWd -= m * 2ΔΩ_by_Ω0 * Rsun^2 * J_ηρbyr

        Jddr = J * ddrM
        Jddr_plus_2byrM = @. Jddr .+ 2J_by_r

        @views for ℓ′ in ℓ-1:2:ℓ+1
            ℓ′ in ℓs || continue
            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)
            inds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)

            @. VW[inds_ℓℓ′] -= two_over_ℓℓp1 *
                                    ΔΩ_by_Ω0 * (
                                        ℓ′ℓ′p1 * cosθo[ℓ, ℓ′] * DDr_min_2byrM +
                                        (DDrM - ℓ′ℓ′p1 * onebyr_chebyM) * sinθdθo[ℓ, ℓ′]
                                    ) * Rsun / Wscaling

            @. WV[inds_ℓℓ′] .-= Rsun * ΔΩ_by_Ω0 / ℓℓp1 *
                                    ((4ℓ′ℓ′p1 * cosθo[ℓ, ℓ′] + (ℓ′ℓ′p1 + 2) * sinθdθo[ℓ, ℓ′]) * Jddr
                                        +
                                        Jddr_plus_2byrM * laplacian_sinθdθo[ℓ, ℓ′]) * Wscaling
        end
    end
    return M
end
function radial_differential_rotation_profile(operators, thetaGL, model = :solar_equator;
    smoothing_param = 1e-3)

    (; r) = operators.coordinates
    (; r_out, nr, r_in) = operators.radial_params

    if model == :solar_equator
        ΔΩ_rθ, Ω0 = read_angular_velocity(operators, thetaGL)
        θind_equator = findmin(abs.(thetaGL .- pi / 2))[2]
        ΔΩ_r = ΔΩ_rθ[:, θind_equator];
        # smooth the profile in r
        s = sum(abs2, ΔΩ_r) * smoothing_param
        ΔΩ_r = Spline1D(r, ΔΩ_r, s = s)(r)
    elseif model == :linear
        Ω0 = equatorial_rotation_angular_velocity(r_out / Rsun)
        f = 0.02 / (r_in / Rsun - 1)
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

function radial_differential_rotation_profile_derivatives(m; operators, rotation_profile = :constant)
    (; r_cheby) = operators.rad_terms;
    (; r_chebyshev, r) = operators.coordinates;
    (; ddr) = operators.diff_operators;
    (; nℓ) = operators.radial_params;

    ntheta = ntheta_ℓmax(nℓ, m);
    (; thetaGL) = gausslegendre_theta_grid(ntheta);

    ΔΩ_r, Ω0 = radial_differential_rotation_profile(operators, thetaGL, rotation_profile);
    ΔΩ_spl = Spline1D(r, ΔΩ_r);
    ΔΩ = chop(Fun(ΔΩ_spl ∘ r_cheby, Chebyshev()), 1e-3) / Ω0;

    ddrΔΩ = ddr * ΔΩ
    drΔΩ_real = ddrΔΩ.(r_chebyshev)
    d2dr2ΔΩ = ddr * ddrΔΩ
    d2dr2ΔΩ_real = d2dr2ΔΩ.(r_chebyshev)
    (; ΔΩ_r, Ω0, ΔΩ, ΔΩ_spl, drΔΩ_real, d2dr2ΔΩ_real, ddrΔΩ, d2dr2ΔΩ)
end

function radial_differential_rotation_terms_inner!((VWterm, WVterm), (ℓ, ℓ′),
    (cosθ_ℓℓ′, sinθdθ_ℓℓ′, ∇²_sinθdθ_ℓℓ′),
    (ddrΔΩM,),
    (ΔΩ_by_rM, ΔΩ_DDrM, ΔΩ_DDr_min_2byrM, ddrΔΩ_plus_ΔΩddrM, twoΔΩ_by_rM))

    ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)
    ℓℓp1 = ℓ * (ℓ + 1)
    two_over_ℓℓp1 = 2/ℓℓp1

    @. VWterm = -two_over_ℓℓp1 *
            (ℓ′ℓ′p1 * cosθ_ℓℓ′ * (ΔΩ_DDr_min_2byrM - ddrΔΩM) +
            sinθdθ_ℓℓ′ * ((ΔΩ_DDrM - ℓ′ℓ′p1 * ΔΩ_by_rM) - ℓ′ℓ′p1 / 2 * ddrΔΩM))

    @. WVterm = -1/ℓℓp1 * (
                (4ℓ′ℓ′p1 * cosθ_ℓℓ′ + (ℓ′ℓ′p1 + 2) * sinθdθ_ℓℓ′ + ∇²_sinθdθ_ℓℓ′) * ddrΔΩ_plus_ΔΩddrM
                + ∇²_sinθdθ_ℓℓ′ * twoΔΩ_by_rM
            )

    VWterm, WVterm
end

function radial_differential_rotation_terms!(M, nr, nℓ, m;
    operators = radial_operators(nr, nℓ),
    rotation_profile = :constant)

    (; nfields, scalings) = operators.constants;
    (; r) = operators.coordinates;
    (; DDr, ddr, ddrDDr) = operators.diff_operators;
    (; twoηρ_by_rM, onebyr_chebyM, ddrM) = operators.diff_operator_matrices;
    (; onebyr_cheby, onebyr2_cheby, ηρ_cheby, g_cheby, r_cheby) = operators.rad_terms;
    (; mat) = operators;
    (; Sscaling, Wscaling) = scalings;

    VV = matrix_block(M, 1, 1, nfields)
    VW = matrix_block(M, 1, 2, nfields)
    WV = matrix_block(M, 2, 1, nfields)
    WW = matrix_block(M, 2, 2, nfields)
    if nfields === 3
        SV = matrix_block(M, 3, 1, nfields)
        SW = matrix_block(M, 3, 2, nfields)
    end

    ΔΩprofile_deriv = radial_differential_rotation_profile_derivatives(m; operators, rotation_profile);

    (; ΔΩ, Ω0, ddrΔΩ, d2dr2ΔΩ) = ΔΩprofile_deriv;

    ΔΩM = mat(ΔΩ);
    ddrΔΩM = mat(ddrΔΩ);

    ddrΔΩ_over_g = ddrΔΩ / g_cheby;
    ddrΔΩ_over_gM = mat(ddrΔΩ_over_g);
    ddrΔΩ_over_g_DDr = (ddrΔΩ_over_g * DDr)::Tmul;
    ddrΔΩ_over_g_DDrM = mat(ddrΔΩ_over_g_DDr);
    ddrΔΩ_plus_ΔΩddr = (ddrΔΩ + (ΔΩ * ddr)::Tmul)::Tplus;
    twoΔΩ_by_r = 2ΔΩ * onebyr_cheby;
    two_ΔΩbyr_ηρ = twoΔΩ_by_r * ηρ_cheby;
    ΔΩ_ddrDDr = (ΔΩ * ddrDDr)::Tmul;
    ΔΩ_by_r2 = ΔΩ * onebyr2_cheby;
    ddrΔΩ_ddr_plus_2byr = (ddrΔΩ * (ddr + 2onebyr_cheby)::Tplus)::Tmul;
    ddrΔΩ_DDr = (ddrΔΩ * DDr)::Tmul;

    ddrΔΩDDr_plus_d2dr2ΔΩM = mat(ddrΔΩ * DDr + d2dr2ΔΩ);

    ℓs = range(m, length = nℓ);

    cosθ = costheta_operator(nℓ, m);
    cosθo = OffsetArray(cosθ, ℓs, ℓs);
    sinθdθ = sintheta_dtheta_operator(nℓ, m);
    sinθdθo = OffsetArray(sinθdθ, ℓs, ℓs);
    cosθsinθdθ = (costheta_operator(nℓ + 1, m)*sintheta_dtheta_operator(nℓ + 1, m))[1:end-1, 1:end-1];
    cosθsinθdθo = OffsetArray(cosθsinθdθ, ℓs, ℓs);
    ∇²_sinθdθo = OffsetArray(Diagonal(@. -ℓs * (ℓs + 1)) * sinθdθ, ℓs, ℓs);

    DDr_min_2byr = (DDr - 2onebyr_cheby)::Tplus;
    ΔΩ_DDr_min_2byr = (ΔΩ * DDr_min_2byr)::Tmul;
    ΔΩ_DDr = (ΔΩ * DDr)::Tmul;
    ΔΩ_by_r = ΔΩ * onebyr_cheby;

    ηρ_chebyM = mat(ηρ_cheby);
    twobyr_min_ηρM = 2onebyr_chebyM - ηρ_chebyM;

    inner_matrices = map(mat, (ΔΩ_by_r, ΔΩ_DDr, ΔΩ_DDr_min_2byr, ddrΔΩ_plus_ΔΩddr, twoΔΩ_by_r));
    (ΔΩ_by_rM, ΔΩ_DDrM, ΔΩ_DDr_min_2byrM, ddrΔΩ_plus_ΔΩddrM, twoΔΩ_by_rM) = inner_matrices;

    VWterm, WVterm = (zeros(nr, nr) for i in 1:2);

    @views for ℓ in ℓs
        # numerical green function
        G = greenfn_cheby(RadDiffRotGfn(), ℓ, operators, ΔΩprofile_deriv);

        (; J, J_ηρbyr) = G.unirot_terms;

        (; J_ηρbyr_ΔΩ, twoddr_plus_3ηρr_J_ddrΔΩ,
            J_twoΔΩ_by_r, J_ΔΩ, J_ddrΩ, J_d2dr2ΔΩ,
            J_2byr_min_ηρ__min__twoddr_plus_3ηρr_J__times_drΔΩ,
            J_ddrΩ_ηρ, twoddr_plus_3ηρr_J,
            J_times_2byr_min_ηρ) = G.diffrot_terms;

        inds_ℓℓ = blockinds((m, nr), ℓ, ℓ);

        ℓℓp1 = ℓ * (ℓ + 1)
        two_over_ℓℓp1 = 2/ℓℓp1
        two_over_ℓℓp1_min_1 = two_over_ℓℓp1 - 1

        @. VV[inds_ℓℓ] += m * two_over_ℓℓp1_min_1 * ΔΩM

        J_ddrΔΩDDr_plus_d2dr2ΔΩM = J_ddrΩ * ddrM .+ J_ddrΩ_ηρ .+ J_d2dr2ΔΩ;

        @. WW[inds_ℓℓ] += m * (two_over_ℓℓp1_min_1 * ΔΩM - 2Rsun^2 * J_ηρbyr_ΔΩ +
            Rsun^2 * J_2byr_min_ηρ__min__twoddr_plus_3ηρr_J__times_drΔΩ +
            two_over_ℓℓp1 * Rsun^2 * (J_ddrΔΩDDr_plus_d2dr2ΔΩM + twoddr_plus_3ηρr_J_ddrΔΩ))

        J_ddrΔΩ_plus_ΔΩddrM = J_ddrΩ + J_ΔΩ * ddrM

        for ℓ′ in intersect(ℓs, ℓ-1:2:ℓ+1)
            inds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)

            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)

            cosθ_ℓℓ′ = cosθo[ℓ, ℓ′]
            sinθdθ_ℓℓ′ = sinθdθo[ℓ, ℓ′]
            ∇²_sinθdθ_ℓℓ′ = ∇²_sinθdθo[ℓ, ℓ′]

            @. VW[inds_ℓℓ′] += -Rsun * two_over_ℓℓp1 *
                    (ℓ′ℓ′p1 * cosθ_ℓℓ′ * (ΔΩ_DDr_min_2byrM - ddrΔΩM) +
                    sinθdθ_ℓℓ′ * ((ΔΩ_DDrM - ℓ′ℓ′p1 * ΔΩ_by_rM) - ℓ′ℓ′p1 / 2 * ddrΔΩM)) / Wscaling

            @. WV[inds_ℓℓ′] += -1/ℓℓp1 * Rsun * (
                        (4ℓ′ℓ′p1 * cosθ_ℓℓ′ + (ℓ′ℓ′p1 + 2) * sinθdθ_ℓℓ′ + ∇²_sinθdθ_ℓℓ′) * J_ddrΔΩ_plus_ΔΩddrM +
                        ∇²_sinθdθ_ℓℓ′ * J_twoΔΩ_by_r
                    ) * Wscaling

            # if nfields == 3
            #     @. SV[inds_ℓℓ′] -= (Ω0^2 * Rsun^2) * 2m * cosθo[ℓ, ℓ′] * ddrΔΩ_over_gM * Sscaling;
            # end
        end

        # for ℓ′ in intersect(ℓs, ℓ-2:2:ℓ+2)
        #     inds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)

        #     if nfields == 3
        #         @. SW[inds_ℓℓ′] += (Ω0^2 * Rsun^3) * 2cosθsinθdθo[ℓ, ℓ′] * ddrΔΩ_over_g_DDrM * Sscaling/Wscaling;
        #     end
        # end
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

    (; nfields) = operators.constants
    (; Iℓ, Ir) = operators.identities
    (; ddr, DDr, d2dr2, rddr, r2d2dr2, DDr_minus_2byr) = operators.diff_operators
    (; Tcrfwd, Tcrinv) = operators.transforms
    (; g_cheby, onebyr_cheby, onebyr2_cheby, r2_cheby, r_cheby) = operators.rad_terms
    (; r, r_chebyshev) = operators.coordinates
    two_over_g = 2g_cheby^-1

    ddr_plus_2byr = ddr + 2onebyr_cheby

    # for the stream functions
    (; PLMfwd, PLMinv) = associatedlegendretransform_matrices(nℓ, m)
    fwdbig_SF = kron(PLMfwd, Tcrfwd)
    invbig_SF = kron(PLMinv, Tcrinv)
    spectralop = SpectralOperatorForm(fwdbig_SF, invbig_SF)

    VV = matrix_block(M, 1, 1, nfields)
    VW = matrix_block(M, 1, 2, nfields)
    WV = matrix_block(M, 2, 1, nfields)
    WW = matrix_block(M, 2, 2, nfields)
    SV = matrix_block(M, 3, 1, nfields)
    SW = matrix_block(M, 3, 2, nfields)

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

    (; Sscaling, Wscaling) = operators.constants.scalings

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
        Gℓ = greenfn_cheby(ℓ, operators)
        Gℓ_invℓℓp1_invΩ0 = Gℓ * invℓℓp1_invΩ0
        Gℓ_invℓℓp1_invΩ0_Wscaling = Wscaling * Gℓ_invℓℓp1_invΩ0
        for ℓ′ in ℓs
            inds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)
            @views @. VV[inds_ℓℓ′] += invℓℓp1_invΩ0 * V_rhs.V[inds_ℓℓ′]
            @views @. VW[inds_ℓℓ′] += invℓℓp1_invΩ0 * V_rhs.W[inds_ℓℓ′] / Wscaling
            @views WV[inds_ℓℓ′] .+= Gℓ_invℓℓp1_invΩ0_Wscaling * negr²Ω0W_rhs.V[inds_ℓℓ′]
            @views WW[inds_ℓℓ′] .+= Gℓ_invℓℓp1_invΩ0 * negr²Ω0W_rhs.W[inds_ℓℓ′]

            @views @. SV[inds_ℓℓ′] += Sscaling * S_rhs.V[inds_ℓℓ′]
            @views @. SW[inds_ℓℓ′] .+= (Sscaling/Wscaling) * S_rhs.W[inds_ℓℓ′]
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

_maybetrimM(M, nfields, nparams) = nfields == 3 ? M : M[1:(nfields*nparams), 1:(nfields*nparams)]

function compute_matrix_scales(M, nfields)
    blockscalesreal = ones(nfields, nfields)

    Mrmax = zeros(nfields, nfields)
    for colind in 1:nfields, rowind in 1:nfields
        Mv = matrix_block(M, rowind, colind, nfields)
        Mrmax_i = maximum(abs∘real, Mv)
        Mrmax[rowind, colind] = Mrmax_i
    end
    VW_WV_min, VW_WV_max = minmax(Mrmax[1, 2], Mrmax[2, 1])
    Wscale = sqrt(VW_WV_max / VW_WV_min)
    if Mrmax[1, 2] == VW_WV_max
        blockscalesreal[1, 2] = 1/Wscale
        blockscalesreal[2, 1] = Wscale
    else
        blockscalesreal[1, 2] = Wscale
        blockscalesreal[2, 1] = 1/Wscale
    end

    SW_WS_min, SW_WS_max = minmax(Mrmax[3, 2], Mrmax[2, 3])
    Sscale = sqrt(SW_WS_max / SW_WS_min)
    if Mrmax[2, 3] == SW_WS_max
        blockscalesreal[2, 3] = 1/Sscale
        blockscalesreal[3, 2] = Sscale
    else
        blockscalesreal[2, 3] = Sscale
        blockscalesreal[3, 2] = 1/Sscale
    end

    blockscalesreal
end

function balance_matrix!(M, nfields)
    scales = compute_matrix_scales(M, nfields)
    for colind in 1:nfields, rowind in 1:nfields
        Mv = matrix_block(M, rowind, colind, nfields)
        Mv .*= scales[rowind, colind]
    end
    scales
end

function constrained_eigensystem(M, operators, constraints = constraintmatrix(operators),
    cache = constrained_matmul_cache(constraints), timer = TimerOutput())

    (; nparams) = operators.radial_params
    (; ZC, nfields) = constraints
    M = _maybetrimM(M, nfields, nparams)
    (; M_constrained, MZCcache) = cache
    scales = ones(nfields, nfields)
    scales = balance_matrix!(M, nfields)
    #= not thread safe =#
    @timeit timer "basis" mul!(M_constrained, permutedims(ZC), mul!(MZCcache, M, ZC))
    # M_constrained = permutedims(ZC) * M * ZC # thread-safe but allocating
    @timeit timer "eigen" λ::Vector{ComplexF64}, w::Matrix{ComplexF64} = eigen!(M_constrained)
    @timeit timer "projectback" v = ZC * w
    λ, v, M, scales
end

function uniform_rotation_spectrum(nr, nℓ, m; operators,
    constraints = constraintmatrix(operators),
    cache = constrained_matmul_cache(constraints),
    kw...)

    rp = operators.radial_params
    @assert (nr, nℓ) == (rp.nr, rp.nℓ) "Please regenerate operators for nr = $nr amd nℓ = $nℓ"

    to = TimerOutput()

    @timeit to "matrix" M = uniform_rotation_matrix(nr, nℓ, m; operators, kw...)
    X = @timeit to "eigen" constrained_eigensystem(M, operators, constraints, cache, to)
    if get(kw, :print_timer, false)
        println(to)
    end
    X
end
function uniform_rotation_spectrum!(M, nr, nℓ, m; operators,
    constraints = constraintmatrix(operators),
    cache = constrained_matmul_cache(constraints),
    kw...)

    rp = operators.radial_params
    @assert (nr, nℓ) == (rp.nr, rp.nℓ) "Please regenerate operators for nr = $nr amd nℓ = $nℓ"

    to = TimerOutput()

    @timeit to "matrix" uniform_rotation_matrix!(M, nr, nℓ, m; operators, kw...)
    X = @timeit to "eigen" constrained_eigensystem(M, operators, constraints, cache)
    if get(kw, :print_timer, false)
        println(to)
    end
    X
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

    rp = operators.radial_params
    @assert (nr, nℓ) == (rp.nr, rp.nℓ) "Please regenerate operators for nr = $nr amd nℓ = $nℓ"

    to = TimerOutput()

    @timeit to "matrix" M = differential_rotation_matrix(nr, nℓ, m; operators, rotation_profile, kw...)
    X = @timeit to "eigen" constrained_eigensystem(M, operators, constraints, cache, to)
    if get(kw, :print_timer, false)
        println(to)
    end
    X
end
function differential_rotation_spectrum!(M, nr, nℓ, m;
    rotation_profile,
    operators,
    constraints = constraintmatrix(operators),
    cache = constrained_matmul_cache(constraints),
    kw...)

    rp = operators.radial_params
    @assert (nr, nℓ) == (rp.nr, rp.nℓ) "Please regenerate operators for nr = $nr amd nℓ = $nℓ"

    to = TimerOutput()

    @timeit to "matrix" differential_rotation_matrix!(M, nr, nℓ, m; operators, rotation_profile, kw...)
    X = @timeit to "eigen" constrained_eigensystem(M, operators, constraints, cache, to)
    if get(kw, :print_timer, false)
        println(to)
    end
    X
end

rossby_ridge(m, ℓ = m; ΔΩ_by_Ω = 0) = 2m / (ℓ * (ℓ + 1)) * (1 + ΔΩ_by_Ω) - m * ΔΩ_by_Ω

function eigenvalue_filter(x, m;
    eig_imag_unstable_cutoff = -1e-3,
    eig_imag_to_real_ratio_cutoff = 1e-1,
    eig_imag_damped_cutoff = 5e-3,
    ΔΩ_by_Ω_low = 0,
    ΔΩ_by_Ω_high = ΔΩ_by_Ω_low)

    freq_sectoral = 2 / (m + 1)

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

function filterfields(coll, v, nparams, nfields; filterfieldpowercutoff = 1e-4)
    Vpow = sum(abs2, @view v[1:nparams])
    Wpow = sum(abs2, @view v[nparams .+ (1:nparams)])
    Spow = nfields == 3 ? sum(abs2, @view(v[2nparams .+ (1:nparams)])) : 0.0

    filterfields = typeof(coll.V)[]

    if Spow/max(Vpow, Wpow) > filterfieldpowercutoff
        push!(filterfields, coll.S)
    end
    if Vpow/max(Vpow, Wpow) > filterfieldpowercutoff
        push!(filterfields, coll.V)
    end

    if Wpow/max(Vpow, Wpow) > filterfieldpowercutoff
        push!(filterfields, coll.W)
    end
    return filterfields
end

function sphericalharmonic_filter!(VWSinvsh, F, v, operators,
        Δl_cutoff = 7, power_cutoff = 0.9, filterfieldpowercutoff = 1e-4)
    eigenfunction_rad_sh!(VWSinvsh, F, v, operators)
    l_cutoff_ind = 1 + Δl_cutoff

    flag = true

    (; nparams) = operators.radial_params
    (; nfields) = operators.constants
    fields = filterfields(VWSinvsh, v, nparams, nfields; filterfieldpowercutoff)

    for X in fields
        PV_frac = sum(abs2, @view X[:, 1:l_cutoff_ind]) / sum(abs2, X)
        flag &= PV_frac > power_cutoff
    end

    flag
end

function chebyshev_filter!(VWSinv, F, v, m, operators, n_cutoff = 7, n_power_cutoff = 0.9;
    nℓ = operators.radial_params.nℓ,
    Plcosθ = zeros(rnage(m, length=nℓ)),
    filterfieldpowercutoff = 1e-4)

    eigenfunction_n_theta!(VWSinv, F, v, m, operators; nℓ, Plcosθ)

    (; V) = VWSinv
    n_cutoff_ind = 1 + n_cutoff
    # ensure that most of the power at the equator is below the cutoff
    nθ = size(V, 2)
    equator_ind = nθ÷2

    Δθ_scan = div(nθ, 5)
    rangescan = intersect(equator_ind .+ (-Δθ_scan:Δθ_scan), axes(V, 2))

    flag = true

    (; nparams) = operators.radial_params
    (; nfields) = operators.constants
    fields = filterfields(VWSinv, v, nparams, nfields; filterfieldpowercutoff)

    for X in fields
        Xrange = view(X, :, rangescan)
        PV_frac = sum(abs2, @view X[1:n_cutoff_ind, rangescan]) / sum(abs2, X[:, rangescan])
        flag &= PV_frac > n_power_cutoff
    end

    flag
end

function spatial_filter!(VWSinv, VWSinvsh, F, v, m, operators,
    θ_cutoff = deg2rad(75), equator_power_cutoff_frac = 0.3,
    rad_cutoff = 0.15, rad_power_cutoff_frac = 0.8;
    nℓ = operators.radial_params.nℓ,
    Plcosθ = zeros(range(m, length=nℓ)),
    filterfieldpowercutoff = 1e-4)

    (; θ) = eigenfunction_realspace!(VWSinv, VWSinvsh, F, v, m, operators; nℓ, Plcosθ)

    eqfilter = true

    (; nparams) = operators.radial_params
    (; nfields) = operators.constants
    fields = filterfields(VWSinv, v, nparams, nfields; filterfieldpowercutoff)

    for X in fields
        peak_latprofile = @view X[end, :]
        θlowind = searchsortedfirst(θ, θ_cutoff)
        θhighind = searchsortedlast(θ, pi - θ_cutoff)
        powfrac = sum(abs2, @view peak_latprofile[θlowind:θhighind]) / sum(abs2, peak_latprofile)
        powflag = powfrac > equator_power_cutoff_frac
        peakflag = maximum(abs2, @view peak_latprofile[θlowind:θhighind]) == maximum(abs2, peak_latprofile)
        eqfilter &= powflag & peakflag
    end

    eqfilter, θ, fields
end

function nodes_filter(VWSinv, VWSinvsh, F, v, m, operators;
    nℓ = operators.radial_params.nℓ,
    Plcosθ = zeros(range(m, length=nℓ)),
    filterfieldpowercutoff = 1e-4,
    nnodesmax = 7)

    nodesfilter = true

    (; nparams) = operators.radial_params
    (; nfields) = operators.constants
    fields = filterfields(VWSinv, v, nparams, nfields; filterfieldpowercutoff)

    (; θ) = eigenfunction_realspace!(VWSinv, VWSinvsh, F, v, m, operators; nℓ, Plcosθ)
    eqind = argmin(abs.(θ .- pi/2))

    for X in fields
        radprof = @view X[:, eqind]
        nnodes_real = count(Bool.(sign.(abs.(diff(sign.(real(radprof)))))))
        nnodes_imag = count(Bool.(sign.(abs.(diff(sign.(imag(radprof)))))))
        nodesfilter &= nnodes_real <= nnodesmax && nnodes_imag <= nnodesmax
    end
    nodesfilter
end

function filterfn(λ, v, m, M, operators, additional_params; kw...)

    (; eig_imag_unstable_cutoff, eig_imag_to_real_ratio_cutoff,
        ΔΩ_by_Ω_low, ΔΩ_by_Ω_high, eig_imag_damped_cutoff,
        Δl_cutoff, Δl_power_cutoff, BC, BCVcache, atol_constraint,
        VWSinv, n_cutoff, n_power_cutoff, nℓ, Plcosθ,
        θ_cutoff, equator_power_cutoff_frac, MVcache, eigen_rtol, VWSinvsh, F,
        filters, filterfieldpowercutoff, nnodesmax) = additional_params

    if get(filters, :eigenvalue, true)
        f1 = eigenvalue_filter(λ, m;
            eig_imag_unstable_cutoff, eig_imag_to_real_ratio_cutoff,
            ΔΩ_by_Ω_low, ΔΩ_by_Ω_high, eig_imag_damped_cutoff)
        f1 || return false
    end

    if get(filters, :sphericalharmonic, true)
        f2 = sphericalharmonic_filter!(VWSinvsh, F, v, operators,
            Δl_cutoff, Δl_power_cutoff, filterfieldpowercutoff)
        f2 || return false
    end

    if get(filters, :boundarycondition, true)
        f3 = boundary_condition_filter(v, BC, BCVcache, atol_constraint)
        f3 || return false
    end

    if get(filters, :chebyshev, true)
        f4 = chebyshev_filter!(VWSinv, F, v, m, operators, n_cutoff,
            n_power_cutoff; nℓ, Plcosθ,filterfieldpowercutoff)
        f4 || return false
    end

    if get(filters, :spatial, true)
        f5, θ, fields = spatial_filter!(VWSinv, VWSinvsh, F, v, m, operators,
            θ_cutoff, equator_power_cutoff_frac; nℓ, Plcosθ,
            filterfieldpowercutoff)
        f5 || return false
    end

    if get(filters, :eigensystem_satisfy, true)
        f6 = eigensystem_satisfy_filter(λ, v, M, MVcache, eigen_rtol)
        f6 || return false
    end

    if get(filters, :nodes, true)
        f7 = nodes_filter(VWSinv, VWSinvsh, F, v, m, operators;
                nℓ, Plcosθ, filterfieldpowercutoff, nnodesmax)
        f7 || return false
    end

    return true
end

function allocate_field_caches(nr, nθ, nℓ)
    VWSinv = (; V = zeros(ComplexF64, nr, nθ), W = zeros(ComplexF64, nr, nθ), S = zeros(ComplexF64, nr, nθ))
    VWSinvsh = (; V = zeros(ComplexF64, nr, nℓ), W = zeros(ComplexF64, nr, nℓ), S = zeros(ComplexF64, nr, nℓ))
    F = (; V = zeros(ComplexF64, nr * nℓ), W = zeros(ComplexF64, nr * nℓ), S = zeros(ComplexF64, nr * nℓ))
    (; VWSinv, VWSinvsh, F)
end

function allocate_filter_caches(m; operators, constraints = constraintmatrix(operators))
    (; BC, nfields) = constraints
    (; nr, nℓ, nparams) = operators.radial_params
    # temporary cache arrays
    MVcache = Vector{ComplexF64}(undef, nfields * nparams)
    BCVcache = Vector{ComplexF64}(undef, size(BC, 1))

    nθ = length(spharm_θ_grid_uniform(m, nℓ).θ)

    (; VWSinv, VWSinvsh, F) = allocate_field_caches(nr, nθ, nℓ)

    Plcosθ = zeros(range(m, length=nℓ))

    return (; MVcache, BCVcache, VWSinv, VWSinvsh, Plcosθ, F)
end

function filter_eigenvalues(λ::AbstractVector, v::AbstractMatrix,
    M::AbstractMatrix, scales::Matrix, m::Integer;
    operators,
    constraints = constraintmatrix(operators),
    filtercache = allocate_filter_caches(m; operators, constraints),
    atol_constraint = 1e-5,
    Δl_cutoff = 7,
    Δl_power_cutoff = 0.9,
    eigen_rtol = 0.3,
    n_cutoff = 7,
    n_power_cutoff = 0.9,
    eig_imag_unstable_cutoff = -1e-3,
    eig_imag_to_real_ratio_cutoff = 1e-1,
    eig_imag_damped_cutoff = 5e-3,
    ΔΩ_by_Ω_low = 0,
    ΔΩ_by_Ω_high = ΔΩ_by_Ω_low,
    θ_cutoff = deg2rad(75),
    equator_power_cutoff_frac = 0.3,
    nnodesmax = 5,
    filterfieldpowercutoff = 1e-4,
    eigenvalue_filter = true,
    sphericalharmonic_filter = true,
    chebyshev_filter = true,
    boundarycondition_filter = true,
    spatial_filter = true,
    eigensystem_satisfy_filter = true,
    nodes_filter = true,
    kw...)

    filters = (;
        eigenvalue = eigenvalue_filter,
        sphericalharmonic = sphericalharmonic_filter,
        chebyshev = chebyshev_filter,
        boundarycondition = boundarycondition_filter,
        spatial = spatial_filter,
        eigensystem_satisfy = eigensystem_satisfy_filter,
        nodes = nodes_filter,
    )

    (; BC) = constraints
    (; nℓ, nparams) = operators.radial_params
    (; MVcache, BCVcache, VWSinv, VWSinvsh, Plcosθ, F) = filtercache

    additional_params = (; eig_imag_unstable_cutoff, eig_imag_to_real_ratio_cutoff,
        ΔΩ_by_Ω_low, ΔΩ_by_Ω_high, eig_imag_damped_cutoff,
        Δl_cutoff, Δl_power_cutoff, BC, BCVcache, atol_constraint,
        VWSinv, n_cutoff, n_power_cutoff, nℓ, Plcosθ,
        θ_cutoff, equator_power_cutoff_frac, MVcache,
        eigen_rtol, VWSinvsh, F, filters, filterfieldpowercutoff, nnodesmax,
    )

    inds_bool = filterfn.(λ, eachcol(v), m, (M,), (operators,), (additional_params,))
    filtinds = axes(λ, 1)[inds_bool]
    λ, v = λ[filtinds], v[:, filtinds]

    # re-apply scalings
    if get(kw, :scale_eigenvectors, true)
        V = @view v[1:nparams, :]
        W = @view v[nparams .+ (1:nparams), :]
        S = @view v[2nparams .+ (1:nparams), :]

        W .*= scales[1, 2]
        S .*= scales[1, 3]

        V .*= Rsun
        (; Wscaling, Sscaling) = operators.constants.scalings
        W .*= -im * Rsun^2 / Wscaling
        S ./= operators.constants.Ω0 * Rsun * Sscaling
    end

    λ, v
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
    kw...)

    (; nr, nℓ, nparams) = operators.radial_params
    nparams = nr * nℓ
    (; nfields) = constraints
    Ms = [zeros(ComplexF64, 3nparams, 3nparams) for _ in 1:Threads.nthreads()]
    λv = @maybe_reduce_blas_threads(
        Folds.map(zip(λ, v, mr)) do (λm, vm, m)
            M = Ms[Threads.threadid()]
            Mfn(M, nr, nℓ, m; operators)
            _M = _maybetrimM(M, nfields, nparams)
            scales = balance_matrix!(_M, nfields)
            filter_eigenvalues(λm, vm, _M, scales, m; operators, constraints, kw...)
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

    λv = @maybe_reduce_blas_threads(
        Folds.map(mr) do m
            M = Ms[Threads.threadid()]
            cache = caches[Threads.threadid()]
            X = f(M, nr, nℓ, m; operators, constraints, cache, kw...)
            filter_eigenvalues(X..., m; operators, constraints, kw...)
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
    (; nfields) = operators.constants

    F.V .= @view v[1:nparams]
    F.W .= @view v[nparams.+(1:nparams)]
    F.S .= nfields == 3 ? @view(v[2nparams.+(1:nparams)]) : 0.0

    V = reshape(F.V, nr, nℓ)
    W = reshape(F.W, nr, nℓ)
    S = reshape(F.S, nr, nℓ)

    (; V, W, S)
end

function eigenfunction_rad_sh!(VWSinvsh, F, v, operators)
    VWS = eigenfunction_cheby_ℓm_spectrum!(F, v, operators)
    (; transforms) = operators
    (; Tcrinvc) = transforms
    (; V, W, S) = VWS

    Vinv = VWSinvsh.V
    Winv = VWSinvsh.W
    Sinv = VWSinvsh.S

    mul!(Vinv, Tcrinvc, V)
    mul!(Winv, Tcrinvc, W)
    mul!(Sinv, Tcrinvc, S)

    return VWSinvsh
end

function spharm_θ_grid_uniform(m, nℓ, ℓmax_mul = 4)
    ℓs = range(m, length = nℓ)
    ℓmax = maximum(ℓs)

    θ, _ = sph_points(ℓmax_mul * ℓmax)
    return (; ℓs, θ)
end

function invshtransform2!(VWSinv, VWS, m;
    nℓ = size(VWS.V, 2),
    Plcosθ = zeros(range(m, length = nℓ)))

    V_r_lm = VWS.V
    W_r_lm = VWS.W
    S_r_lm = VWS.S

    (; ℓs, θ) = spharm_θ_grid_uniform(m, nℓ)

    V = VWSinv.V
    V .= 0
    W = VWSinv.W
    W .= 0
    S = VWSinv.S
    S .= 0

    for (θind, θi) in enumerate(θ)
        collectPlm!(Plcosθ, cos(θi); m, norm = Val(:normalized))
        for (ℓind, ℓ) in enumerate(ℓs)
            Plmcosθ = Plcosθ[ℓ]
            for r_ind in axes(V, 1)
                V[r_ind, θind] += V_r_lm[r_ind, ℓind] * Plmcosθ
                W[r_ind, θind] += W_r_lm[r_ind, ℓind] * Plmcosθ
                S[r_ind, θind] += S_r_lm[r_ind, ℓind] * Plmcosθ
            end
        end
    end

    (; VWSinv, θ)
end

function eigenfunction_realspace!(VWSinv, VWSinvsh, F, v, m, operators;
    nℓ = operators.radial_params.nℓ,
    Plcosθ = zeros(range(m, length=nℓ)))

    eigenfunction_rad_sh!(VWSinvsh, F, v, operators)
    invshtransform2!(VWSinv, VWSinvsh, m; nℓ, Plcosθ)
end

function eigenfunction_realspace(v, m, operators)
    (; nr, nℓ) = operators.radial_params
    θ = spharm_θ_grid_uniform(m, nℓ).θ
    nθ = length(θ)

    (; VWSinv, VWSinvsh, F) = allocate_field_caches(nr, nθ, nℓ)

    eigenfunction_realspace!(VWSinv, VWSinvsh, F, v, m, operators)

    return (; VWSinv, θ)
end

function eigenfunction_n_theta!(VWSinv, F, v, m, operators;
    nℓ = operators.radial_params.nℓ,
    Plcosθ = zeros(range(m, length=nℓ)))

    VW = eigenfunction_cheby_ℓm_spectrum!(F, v, operators)
    invshtransform2!(VWSinv, VW, m; nℓ, Plcosθ)
end

# precompile
precompile(_radial_operators, (Int, Int, Float64, Float64, Bool, Int, Float64))

end # module
