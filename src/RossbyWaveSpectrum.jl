module RossbyWaveSpectrum

using MKL

using ApproxFun
using ApproxFun: DomainSets
using BitFlags
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
using OffsetArrays
using SimpleDelimitedFiles: readdlm
using StructArrays
using TimerOutputs
using UnPack
using ZChop

include("eigen.jl")

export datadir
export F_EIGVAL
export F_EIGEN
export F_SPHARM
export F_CHEBY
export F_BC
export F_SPATIAL
export F_NODES

const SCRATCH = Ref("")
const DATADIR = Ref("")

# cgs units
const G = 6.6743e-8
const Msun = 1.989e+33
const Rsun = 6.959894677e+10

const Tmul = TimesOperator{Float64,Tuple{InfiniteCardinal{0},InfiniteCardinal{0}}}
const Tplus = PlusOperator{Float64,Tuple{InfiniteCardinal{0},InfiniteCardinal{0}}}
const TFun = Fun{Chebyshev{ChebyshevInterval{Float64},Float64},Float64,Vector{Float64}}
const TFunDeriv = ApproxFunBase.Fun{ApproxFunOrthogonalPolynomials.Ultraspherical{Int64, DomainSets.ChebyshevInterval{Float64}, Float64}, Float64, Vector{Float64}}

function __init__()
    SCRATCH[] = get(ENV, "SCRATCH", homedir())
    DATADIR[] = get(ENV, "DATADIR", joinpath(SCRATCH[], "RossbyWaves"))
    if !ispath(RossbyWaveSpectrum.DATADIR[])
        mkdir(RossbyWaveSpectrum.DATADIR[])
    end
end

indexedfromzero(A) = OffsetArray(A, OffsetArrays.Origin(0))

# Assume v is evaluated on an grid that is in increasing order (default in this project)
chebyshevgrid_to_Fun(v) = Fun(Chebyshev(), ApproxFun.transform(Chebyshev(), reverse(v)))

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

    (; nvariables) = operators.constants
    BC = zeros(nvariables * nconstraints, nvariables * nparams)

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

    fieldmatrices = [MVn, MWn, MSn][1:nvariables]
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

    (; BC, ZC, nvariables)
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
    dtrans = dtop = 0.03Rsun
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

# defined for normalized Legendre polynomials
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
struct ViscosityGfn end

struct LobattoChebyshev
    TfGL_nr :: Matrix{Float64}
    TiGL_nr :: Matrix{Float64}
    normr :: Vector{Float64}
    temp1 :: Matrix{Float64}
    temp2 :: Matrix{Float64}
    function LobattoChebyshev(TfGL_nr, TiGL_nr, normr)
        n_lobatto, nr = size(TiGL_nr)
        temp1 = Matrix{Float64}(undef, n_lobatto, n_lobatto)
        temp2 = Matrix{Float64}(undef, n_lobatto, nr)
        new(TfGL_nr, TiGL_nr, normr, temp1, temp2)
    end
end
function (L::LobattoChebyshev)(A::Matrix)
    L.temp1 .= A .* L.normr'
    mul!(L.temp2, L.temp1, L.TiGL_nr)
    L.TfGL_nr * L.temp2
end
function lobattochebyshev!(out::Matrix, L::LobattoChebyshev, A::Matrix)
    L.temp1 .= A .* L.normr'
    mul!(L.temp2, L.temp1, L.TiGL_nr)
    mul!(out, L.TfGL_nr, L.temp2)
end

function greenfn_cheby(::UniformRotGfn, ℓ, operators,
        lobattochebyshevtransform =
            LobattoChebyshev(operators.transforms.TfGL_nr,
                    operators.transforms.TiGL_nr,
                    operators.transforms.normr),
    )

    (; r_chebyshev_lobatto, r_lobatto) = operators.coordinates;
    (; ddr_lobatto) = operators.diff_operator_matrices;
    (; ηρ_cheby, ηρ_by_r, g_cheby, ηρ_by_r3, ddr_ηρ) = operators.rad_terms;

    H = greenfn_radial_lobatto(ℓ, operators)
    J = lobattochebyshevtransform(H)

    ddr′Hrr′ = H * ddr_lobatto'

    tempmat = zeros(size(H));
    tempvec = zeros(length(r_chebyshev_lobatto));

    @. tempvec = ηρ_cheby(r_chebyshev_lobatto)
    ηρrHrr = H .* tempvec'

    twoddr_plus_3ηρr_H = @. 2*ddr′Hrr′ + 3*ηρrHrr

    @. tempvec = ηρ_by_r(r_chebyshev_lobatto)
    H_ηρbyr = H .* tempvec'
    J_ηρbyr = lobattochebyshevtransform(H_ηρbyr)

    H_by_r = H ./ r_lobatto'
    J_by_r = lobattochebyshevtransform(H_by_r)

    @. tempvec = 2/r_lobatto - ηρ_cheby(r_chebyshev_lobatto)
    H_times_2byr_min_ηρ = H .* tempvec'

    @. tempvec = g_cheby(r_chebyshev_lobatto)
    H_g = H .* tempvec'
    J_g = lobattochebyshevtransform(H_g)

    @. tempvec = ddr_ηρ(r_chebyshev_lobatto)
    H_ddrηρ = H .* tempvec'

    @. tempvec = ηρ_by_r3(r_chebyshev_lobatto)
    H_4ηρbyr3 = 4 .* H .* tempvec'

    rad_terms = (; H, twoddr_plus_3ηρr_H, H_times_2byr_min_ηρ, ηρrHrr, H_ηρbyr, H_g,
        H_ddrηρ, H_4ηρbyr3, ddr′Hrr′)

    unirot_terms = (; J, J_by_r, J_g, J_ηρbyr)

    return (; rad_terms, unirot_terms)
end

function greenfn_cheby!(::ViscosityGfn, ℓ, operators, viscosity_terms, funs,
        G = greenfn_cheby(UniformRotGfn(), ℓ, operators),
        tempmat = zeros(size(G.rad_terms.H)),
        tempvec = zeros(length(operators.coordinates.r_chebyshev_lobatto)),
        lobattochebyshevtransform =
            LobattoChebyshev(operators.transforms.TfGL_nr,
                    operators.transforms.TiGL_nr,
                    operators.transforms.normr),
    )

    (; r_chebyshev_lobatto) = operators.coordinates;
    (; ηρ2_by_r2, ηρ_by_r2, ddr_ηρbyr2, d2dr2_ηρ, onebyr_cheby, onebyr2_cheby) = operators.rad_terms;

    (; unirot_terms) = G
    L = lobattochebyshevtransform
    (; H, ηρrHrr, H_ddrηρ, H_4ηρbyr3) = G.rad_terms

    (;
        J_c1,
        J_c2_1,
        J_c2_2,
        J_a1_1,
        J_a1_2,
        J_a1_3,
        J_4ηρbyr3,
        J_ηρ,
        J_ηρ²_min_2ηρbyr,
        J_ηρ_min_2byr_ddrηρ,
        J_ddrηρ,
        J_ddrηρbyr2_plus_4ηρbyr3,
        J_ηρ2byr2,
        J_ηρbyr2,
        J_d2dr2ηρ_by_r_min_2ddrηρ_by_r2,
        J_ddrηρ_by_r,
        J_ddrηρ_by_r_min_ηρbyr2,
        J_ddrηρ_by_r2_min_4ηρbyr3,
        J_ddrηρ_by_r2,
        J_d2dr2ηρ_by_r,
    ) = viscosity_terms

    lobattochebyshev!(J_ηρ, L, ηρrHrr)
    lobattochebyshev!(J_ddrηρ, L, H_ddrηρ)
    lobattochebyshev!(J_4ηρbyr3, L, H_4ηρbyr3)

    @. tempvec = onebyr_cheby(r_chebyshev_lobatto)
    tempmat .= H_ddrηρ .* tempvec'
    lobattochebyshev!(J_ddrηρ_by_r, L, tempmat)

    @. tempvec = onebyr2_cheby(r_chebyshev_lobatto)
    tempmat .= H_ddrηρ .* tempvec'
    lobattochebyshev!(J_ddrηρ_by_r2, L, tempmat)

    @. tempvec = ηρ_by_r2(r_chebyshev_lobatto)
    tempmat .= H .* tempvec'
    H_ηρbyr2 = tempmat
    lobattochebyshev!(J_ηρbyr2, L, H_ηρbyr2)

    @. tempvec = ddr_ηρbyr2(r_chebyshev_lobatto)
    tempmat .= H .* tempvec'
    H_ddrηρbyr2 = tempmat
    @. tempmat = H_ddrηρbyr2 + H_4ηρbyr3
    H_ddrηρbyr2_plus_4ηρ_by_r3 = tempmat
    lobattochebyshev!(J_ddrηρbyr2_plus_4ηρbyr3, L, H_ddrηρbyr2_plus_4ηρ_by_r3)

    @. tempvec = d2dr2_ηρ(r_chebyshev_lobatto) * onebyr_cheby(r_chebyshev_lobatto)
    tempmat .= H .* tempvec'
    H_d2dr2ηρ_by_r = tempmat
    lobattochebyshev!(J_d2dr2ηρ_by_r, L, H_d2dr2ηρ_by_r)

    @. tempvec = ηρ2_by_r2(r_chebyshev_lobatto)
    tempmat .= H .* tempvec'
    Hηρ2_by_r2 = tempmat
    lobattochebyshev!(J_ηρ2byr2, L, Hηρ2_by_r2)

    # funs = (ηρ_cheby * (ηρ_cheby - 2onebyr_cheby)::TFun,
    #     (ddr_ηρ * (ηρ_cheby - 2onebyr_cheby))::TFun,
    #     ((ddr_ηρ - 2ηρ_by_r)*ddr_ηρ) ::TFun,
    #     (2(ddr_ηρ - ηρ_by_r + onebyr2_cheby)*ηρ_cheby) ::TFun,
    #     ((-2*ddr_ηρbyr + d2dr2_ηρ) * ηρ_cheby)::TFunDeriv,
    #     (3ddr_ηρ - 4ηρ_by_r)::TFun,
    #     (3d2dr2_ηρ - 8ddr_ηρ*onebyr_cheby + 8ηρ_cheby*onebyr2_cheby)::TFun,
    #     (d3dr3_ηρ - 4d2dr2_ηρ*onebyr_cheby + 8ddr_ηρ*onebyr2_cheby - 8ηρ_cheby*onebyr3_cheby)::TFun,
    #     )

    # T::TFun = ηρ_cheby * (ηρ_cheby - 2onebyr_cheby)::TFun
    T::TFun = funs[1]::TFun
    @. tempvec = T(r_chebyshev_lobatto)
    tempmat .= H .* tempvec'
    lobattochebyshev!(J_ηρ²_min_2ηρbyr, L, tempmat)

    # T = (ddr_ηρ * (ηρ_cheby - 2onebyr_cheby))::TFun
    T = funs[2]::TFun
    @. tempvec = T(r_chebyshev_lobatto)
    tempmat .= H .* tempvec'
    lobattochebyshev!(J_ηρ_min_2byr_ddrηρ, L, tempmat)

    # T = ((ddr_ηρ - 2ηρ_by_r)*ddr_ηρ) ::TFun
    T = funs[3]::TFun
    @. tempvec = T(r_chebyshev_lobatto)
    tempmat .= H .* tempvec'
    lobattochebyshev!(J_c1, L, tempmat)

    # T = (2(ddr_ηρ - ηρ_by_r + onebyr2_cheby)*ηρ_cheby) ::TFun
    T = funs[4] ::TFun
    @. tempvec = T(r_chebyshev_lobatto)
    tempmat .= H .* tempvec'
    lobattochebyshev!(J_c2_1, L, tempmat)

    # T2 = ((-2*ddr_ηρbyr + d2dr2_ηρ) * ηρ_cheby)::TFunDeriv
    T2 = funs[5]::TFun
    @. tempvec = T2(r_chebyshev_lobatto)
    tempmat .= H .* tempvec'
    lobattochebyshev!(J_c2_2, L, tempmat)

    # T = (3ddr_ηρ - 4ηρ_by_r)::TFun
    T = funs[6]::TFun
    @. tempvec = T(r_chebyshev_lobatto)
    tempmat .= H .* tempvec'
    lobattochebyshev!(J_a1_1, L, tempmat)

    # T = (3d2dr2_ηρ - 8ddr_ηρ*onebyr_cheby + 8ηρ_cheby*onebyr2_cheby)::TFun
    T = funs[7]::TFun
    @. tempvec = T(r_chebyshev_lobatto)
    tempmat .= H .* tempvec'
    lobattochebyshev!(J_a1_2, L, tempmat)

    # T = (d3dr3_ηρ - 4d2dr2_ηρ*onebyr_cheby + 8ddr_ηρ*onebyr2_cheby - 8ηρ_cheby*onebyr3_cheby)::TFun
    T = funs[8]::TFun
    @. tempvec = T(r_chebyshev_lobatto)
    tempmat .= H .* tempvec'
    lobattochebyshev!(J_a1_3, L, tempmat)

    @. J_d2dr2ηρ_by_r_min_2ddrηρ_by_r2 = J_d2dr2ηρ_by_r - 2 * J_ddrηρ_by_r2
    @. J_ddrηρ_by_r_min_ηρbyr2 = J_ddrηρ_by_r - J_ηρbyr2
    @. J_ddrηρ_by_r2_min_4ηρbyr3 = J_ddrηρ_by_r2 - J_4ηρbyr3

    return (; unirot_terms, viscosity_terms)
end

function greenfn_cheby!(::RadDiffRotGfn, ℓ, operators, ΔΩprofile_deriv, diffrot_terms,
        G = greenfn_cheby(UniformRotGfn(), ℓ, operators),
        tempmat = zeros(size(G.rad_terms.H)),
        tempvec = zeros(length(operators.coordinates.r_chebyshev_lobatto)),
        lobattochebyshevtransform =
            LobattoChebyshev(operators.transforms.TfGL_nr,
                    operators.transforms.TiGL_nr,
                    operators.transforms.normr),
        )

    (; r_chebyshev_lobatto, r_lobatto) = operators.coordinates
    (; ηρ_cheby) = operators.rad_terms
    L = lobattochebyshevtransform
    (; rad_terms, unirot_terms) = G

    (;
        J_ηρbyr_ΔΩ,
        J_ddrΩ,
        J_twoΔΩ_by_r,
        J_d2dr2ΔΩ,
        J_ddrΩ_ηρ,
        J_2byr_min_ηρ__min__twoddr_plus_3ηρr_J__times_drΔΩ,
        twoddr_plus_3ηρr_J_ddrΔΩ,
        J_ΔΩ,
        twoddr_plus_3ηρr_J,
        J_times_2byr_min_ηρ,
    ) = diffrot_terms

    (; ΔΩ, ddrΔΩ, d2dr2ΔΩ) = ΔΩprofile_deriv

    (; H, H_ηρbyr, H_times_2byr_min_ηρ, twoddr_plus_3ηρr_H) = rad_terms

    @. tempvec = ΔΩ(r_chebyshev_lobatto)
    tempmat .= H_ηρbyr .* tempvec'
    lobattochebyshev!(J_ηρbyr_ΔΩ, L, tempmat)
    tempmat .= H .* tempvec'
    lobattochebyshev!(J_ΔΩ, L, tempmat)

    @. tempvec = ddrΔΩ(r_chebyshev_lobatto)
    H_ddrΩ = H .* tempvec'
    lobattochebyshev!(J_ddrΩ, L, H_ddrΩ)

    @. tempmat = H_times_2byr_min_ηρ - twoddr_plus_3ηρr_H
    tempmat .*= tempvec'
    lobattochebyshev!(J_2byr_min_ηρ__min__twoddr_plus_3ηρr_J__times_drΔΩ, L, tempmat)

    tempmat .= twoddr_plus_3ηρr_H .* tempvec'
    lobattochebyshev!(twoddr_plus_3ηρr_J_ddrΔΩ, L, tempmat)

    @. tempvec = ηρ_cheby(r_chebyshev_lobatto)
    tempmat .= H_ddrΩ .* tempvec'
    lobattochebyshev!(J_ddrΩ_ηρ, L, tempmat)

    @. tempvec = 2ΔΩ(r_chebyshev_lobatto) / r_lobatto
    tempmat .= H .*  tempvec'
    lobattochebyshev!(J_twoΔΩ_by_r, L, tempmat)

    @. tempvec .= d2dr2ΔΩ(r_chebyshev_lobatto)
    tempmat .= H .* tempvec'
    lobattochebyshev!(J_d2dr2ΔΩ, L, tempmat)

    lobattochebyshev!(twoddr_plus_3ηρr_J, L, twoddr_plus_3ηρr_H)

    lobattochebyshev!(J_times_2byr_min_ηρ, L, H_times_2byr_min_ηρ)

    diffrot_terms = (; J_ηρbyr_ΔΩ, J_ddrΩ,
        J_twoΔΩ_by_r, J_d2dr2ΔΩ, J_ddrΩ_ηρ,
        J_2byr_min_ηρ__min__twoddr_plus_3ηρr_J__times_drΔΩ,
        twoddr_plus_3ηρr_J_ddrΔΩ, J_ΔΩ, twoddr_plus_3ηρr_J,
        J_times_2byr_min_ηρ)

    (; unirot_terms, diffrot_terms)
end

splderiv(v::Vector, r::Vector, rout = r; nu = 1) = Dierckx.derivative(Spline1D(r, v), rout; nu = 1)

smoothed_spline(r, v; s) = Spline1D(r, v, s = sum(abs2, v) * s)

function read_solar_model(; r_in = 0.7Rsun, r_out = Rsun, _stratified #= only for tests =# = true)
    ModelS = readdlm(joinpath(@__DIR__, "ModelS.detailed"))
    r_modelS = @view ModelS[:, 1];
    r_inds = r_in .<= r_modelS .<= r_out;
    r_modelS = reverse(r_modelS[r_inds]);
    q_modelS = exp.(reverse(ModelS[r_inds, 2]));
    T_modelS = reverse(ModelS[r_inds, 3]);
    ρ_modelS = reverse(ModelS[r_inds, 5]);
    if !_stratified
        ρ_modelS = fill(ρ_modelS[1], length(ρ_modelS));
        T_modelS = fill(T_modelS[1], length(T_modelS));
    end
    logρ_modelS = log.(ρ_modelS);
    logT_modelS = log.(T_modelS);

    sρ = Spline1D(r_modelS, ρ_modelS);
    slogρ = smoothed_spline(r_modelS, logρ_modelS, s = 1e-6);
    ddrlogρ_modelS = Dierckx.derivative(slogρ, r_modelS);
    if !_stratified
        ddrlogρ_modelS .= 0
    end
    sηρ = smoothed_spline(r_modelS, ddrlogρ_modelS, s = 1e-4);
    sηρ_by_r = smoothed_spline(r_modelS, ddrlogρ_modelS ./ r_modelS, s = 1e-5);
    sηρ_by_r2 = smoothed_spline(r_modelS, ddrlogρ_modelS ./ r_modelS.^2, s = 1e-6);
    ddrsηρ = smoothed_spline(r_modelS, derivative(sηρ, r_modelS), s = 1e-5);
    ddrsηρ_by_r = smoothed_spline(r_modelS, derivative(sηρ_by_r, r_modelS), s = 1e-4);
    ddrsηρ_by_r2 = smoothed_spline(r_modelS, derivative(sηρ_by_r2, r_modelS), s = 1e-4);
    d2dr2sηρ = smoothed_spline(r_modelS, derivative(ddrsηρ, r_modelS), s = 1e-4)
    d3dr3sηρ = smoothed_spline(r_modelS, derivative(ddrsηρ, r_modelS, nu=2), s = 1e-4)

    sT = Spline1D(r_modelS, T_modelS);
    slogT = smoothed_spline(r_modelS, logT_modelS, s = 1e-7);
    ddrlogT = Dierckx.derivative(slogT, r_modelS);
    if !_stratified
        ddrlogT .= 0
    end
    sηT = smoothed_spline(r_modelS, ddrlogT, s = 1e-7);

    g_modelS = @. G * Msun * q_modelS / r_modelS^2;
    sg = smoothed_spline(r_modelS, g_modelS, s = 1e-2);

    rad_terms = (; r_modelS, ρ_modelS, logρ_modelS, ddrlogρ_modelS, T_modelS, g_modelS)
    splines = (; sρ, sT, sg, slogρ, sηρ, sηρ_by_r, ddrsηρ_by_r, ddrsηρ_by_r2, ddrsηρ, d2dr2sηρ, d3dr3sηρ, sηT)
    (; splines, rad_terms)
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

iszerofun(v) = ncoefficients(v) == 0 || (ncoefficients(v) == 1 && coefficients(v)[] == 0.0)

function radial_operators(nr, nℓ; r_in_frac = 0.7, r_out_frac = 0.985, _stratified = true, nvariables = 3, ν = 1e10)
    _radial_operators(nr, nℓ, r_in_frac, r_out_frac, _stratified, nvariables, ν)
end
function _radial_operators(nr, nℓ, r_in_frac, r_out_frac, _stratified, nvariables, ν)
    r_in = r_in_frac * Rsun;
    r_out = r_out_frac * Rsun;
    radial_params = parameters(nr, nℓ; r_in, r_out);
    (; Δr, nchebyr, r_mid) = radial_params;
    r, Tcrfwd, Tcrinv = chebyshev_forward_inverse(nr, r_in, r_out);

    pseudospectralop_radial = SpectralOperatorForm(Tcrfwd, Tcrinv);
    r_chebyshev = (r .- r_mid) ./ (Δr / 2);
    Tcrfwdc, Tcrinvc = complex.(Tcrfwd), complex.(Tcrinv);
    r_cheby = Fun(ApproxFun.Chebyshev(), [r_mid, Δr / 2]);
    r2_cheby = r_cheby * r_cheby;

    (; splines) = read_solar_model(; r_in, r_out, _stratified);

    (; sρ, sg, sηρ, ddrsηρ, d2dr2sηρ,
        sηρ_by_r, ddrsηρ_by_r, ddrsηρ_by_r2, d3dr3sηρ,
        sT, sηT) = splines

    ddr = ApproxFun.Derivative() * (2 / Δr)
    rddr = (r_cheby * ddr)::Tmul
    d2dr2 = (ddr * ddr)::Tmul
    d3dr3 = (ddr * d2dr2)::Tmul
    d4dr4 = (d2dr2 * d2dr2)::Tmul
    r2d2dr2 = (r2_cheby * d2dr2)::Tmul

    # density stratification
    ρ = sρ.(r);
    ηρ = sηρ.(r);
    # ηρ_cheby = ApproxFun.chop(chebyshevgrid_to_Fun(ηρ), 1e-2)::TFun;
    ηρ_cheby = ApproxFun.chop(Fun(sηρ ∘ r_cheby, ApproxFun.Chebyshev()), 1e-3)::TFun
    if iszerofun(ηρ_cheby)
        ηρ_cheby = Fun(ApproxFun.Chebyshev(), [1e-100])::TFun
    end
    if ncoefficients(ηρ_cheby) > 2/3*nr
        @warn "number of coefficients in ηρ_cheby is $(ncoefficients(ηρ_cheby))"
    end
    ηT = sηT.(r);
    # ηT_cheby = ApproxFun.chop(chebyshevgrid_to_Fun(ηT), 1e-2)::TFun
    ηT_cheby = ApproxFun.chop(Fun(sηT ∘ r_cheby, ApproxFun.Chebyshev()), 1e-2)::TFun
    if iszerofun(ηT_cheby)
        ηT_cheby = Fun(ApproxFun.Chebyshev(), [1e-100])::TFun
    end
    if ncoefficients(ηT_cheby) > 2/3*nr
        @warn "number of coefficients in ηT_cheby is $(ncoefficients(ηT_cheby))"
    end
    ddr_lnρT = ηρ_cheby + ηT_cheby

    DDr = (ddr + ηρ_cheby)::Tplus
    rDDr = (r_cheby * DDr)::Tmul

    onebyr = 1 ./ r
    onebyr_cheby = (1 / r_cheby)::typeof(r_cheby)
    onebyr2_cheby = (onebyr_cheby*onebyr_cheby)::typeof(r_cheby)
    onebyr3_cheby = (onebyr2_cheby*onebyr_cheby)::typeof(r_cheby)
    onebyr4_cheby = (onebyr2_cheby*onebyr2_cheby)::typeof(r_cheby)
    DDr_minus_2byr = (DDr - 2onebyr_cheby)::Tplus
    ddr_plus_2byr = (ddr + 2onebyr_cheby)::Tplus

    # ηρ_by_r = onebyr_cheby * ηρ_cheby
    ηρ_by_r = chop(Fun(sηρ_by_r ∘ r_cheby, Chebyshev()), 1e-2)::TFun;
    if ncoefficients(ηρ_by_r) > 2/3*nr
        @warn "number of coefficients in ηρ_by_r is $(ncoefficients(ηρ_by_r))"
    end
    twoηρ_by_r = 2ηρ_by_r

    ηρ_by_r2 = onebyr2_cheby * ηρ_cheby
    ηρ_by_r3 = onebyr_cheby * ηρ_by_r2
    ddr_ηρ = chop(Fun(ddrsηρ ∘ r_cheby, Chebyshev()), 1e-2)::TFun
    if ncoefficients(ddr_ηρ) > 2/3*nr
        @warn "number of coefficients in ddr_ηρ is $(ncoefficients(ddr_ηρ))"
    end
    ddr_ηρbyr = chop(Fun(ddrsηρ_by_r ∘ r_cheby, Chebyshev()), 1e-2)::TFun
    if ncoefficients(ddr_ηρbyr) > 2/3*nr
        @warn "number of coefficients in ddr_ηρbyr is $(ncoefficients(ddr_ηρbyr))"
    end
    # ddr_ηρbyr = ddr * ηρ_by_r
    d2dr2_ηρ = chop(Fun(d2dr2sηρ ∘ r_cheby, Chebyshev()), 1e-2)::TFun
    if ncoefficients(d2dr2_ηρ) > 2/3*nr
        @warn "number of coefficients in d2dr2_ηρ is $(ncoefficients(d2dr2_ηρ))"
    end
    d3dr3_ηρ = chop(Fun(d3dr3sηρ ∘ r_cheby, Chebyshev()), 1e-2)::TFun
    if ncoefficients(d3dr3_ηρ) > 2/3*nr
        @warn "number of coefficients in d3dr3_ηρ is $(ncoefficients(d3dr3_ηρ))"
    end
    ddr_ηρbyr2 = chop(Fun(ddrsηρ_by_r2 ∘ r_cheby, Chebyshev()), 5e-3)::TFun
    if ncoefficients(ddr_ηρbyr2) > 2/3*nr
        @warn "number of coefficients in ddr_ηρbyr2 is $(ncoefficients(ddr_ηρbyr2))"
    end
    # ddr_ηρbyr2 = ddr * ηρ_by_r2
    ηρ2_by_r2 = ApproxFun.chop(ηρ_by_r2 * ηρ_cheby, 1e-3)::TFun
    if iszerofun(ηρ2_by_r2)
        ηρ2_by_r2 = Fun(ApproxFun.Chebyshev(), [1e-100])::TFun
    end
    if ncoefficients(ηρ2_by_r2) > 2/3*nr
        @warn "number of coefficients in ηρ2_by_r2 is $(ncoefficients(ηρ2_by_r2))"
    end

    g = sg.(r);
    g_cheby = Fun(sg ∘ r_cheby, Chebyshev())::TFun
    # g_cheby = chop(chebyshevgrid_to_Fun(g), 1e-3)::TFun

    Ω0 = RossbyWaveSpectrum.equatorial_rotation_angular_velocity(r_out_frac)

    # viscosity
    ν /= Ω0*Rsun^2
    κ = ν

    γ = 1.64
    cp = 1.7e8
    δ_superadiabatic = superadiabaticity.(r)
    # ddr_S0_by_cp = ApproxFun.chop(chebyshevgrid_to_Fun(@. γ * δ_superadiabatic * ηρ / cp), 1e-3)
    ddr_S0_by_cp = ApproxFun.chop(Fun(ApproxFun.Chebyshev(),
        Tcrfwd * @. γ * δ_superadiabatic * ηρ / cp), 1e-3)
    if ncoefficients(ddr_S0_by_cp) > 2/3*nr
        @warn "number of coefficients in ddr_S0_by_cp is $(ncoefficients(ddr_S0_by_cp))"
    end

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
    rddrM = mat(rddr)
    twoηρ_by_rM = mat(twoηρ_by_r)
    ddr_plus_2byrM = @. ddrM + 2 * onebyr_chebyM
    ddr_minus_2byrM = @. ddrM - 2 * onebyr_chebyM
    DDr_minus_2byrM = mat(DDr_minus_2byr)
    gM = mat(g_cheby)

    # uniform rotation terms
    onebyr2_IplusrηρM = mat((1 + ηρ_cheby * r_cheby) * onebyr2_cheby);
    onebyr2_cheby_ddr_S0_by_cpM = mat(chop(onebyr2_cheby * ddr_S0_by_cp, 1e-4));
    ∇r2_plus_ddr_lnρT_ddr = (d2dr2 + 2onebyr_cheby*ddr + ddr_lnρT * ddr)::Tplus;
    κ_∇r2_plus_ddr_lnρT_ddrM = κ * chebyshevmatrix(∇r2_plus_ddr_lnρT_ddr, nr, 4);
    κ_by_r2M = κ .* onebyr2_chebyM;

    # terms for viscosity
    ddr_minus_2byr = (ddr - 2onebyr_cheby)::Tplus;
    ηρ_ddr_minus_2byrM = mat((ηρ_cheby * ddr_minus_2byr)::Tmul);
    onebyr2_d2dr2M = mat(onebyr2_cheby*d2dr2);
    onebyr3_ddrM = mat(onebyr3_cheby*ddr);
    onebyr4_chebyM = mat(onebyr2_cheby*onebyr2_cheby);

    ηρ_by_rM = mat(ηρ_by_r)
    ηρ2_by_r2M = mat(ηρ2_by_r2)

    HeinrichsChebyshevMatrix = heinrichs_chebyshev_matrix(nr)

    # gauss lobatto points
    n_lobatto = 4nr # one less than the number of points
    Tcf, Tci = chebyshev_lobatto_forward_inverse(n_lobatto)
    TfGL_nr = Tcf[1:nr, :]
    TiGL_nr = Tci[:, 1:nr]
    r_chebyshev_lobatto = chebyshevnodes_lobatto(n_lobatto)
    r_lobatto = @. (Δr/2) * r_chebyshev_lobatto .+ r_mid
    ddr_lobatto = reverse(chebyderivGaussLobatto(n_lobatto) * (2 / Δr))
    d2dr2_lobatto = ddr_lobatto*ddr_lobatto
    ddrDDr_lobatto = d2dr2_lobatto + Diagonal(ηρ_cheby.(r_chebyshev_lobatto)) * ddr_lobatto +
                        Diagonal((ddr * ηρ_cheby).(r_chebyshev_lobatto))

    normr = sqrt.(1 .- r_chebyshev_lobatto.^2) .* pi/n_lobatto

    deltafn_matrix_radial = deltafn_matrix(r_lobatto, scale = Rsun*1e-5)

    # scalings = (; Sscaling = 1, Wscaling = 1)
    scalings = (; Sscaling = 1e6, Wscaling = 5e2)

    constants = (; κ, ν, nvariables, Ω0, scalings)
    identities = (; Ir, Iℓ)
    coordinates = (; r, r_chebyshev, r_chebyshev_lobatto, r_lobatto)

    transforms = (; Tcrfwd, Tcrinv, Tcrfwdc, Tcrinvc, pseudospectralop_radial, TfGL_nr,
            TiGL_nr, n_lobatto, normr)

    rad_terms = (; onebyr, onebyr_cheby, ηρ, ηρ_cheby, ηT_cheby,
        onebyr2_cheby, onebyr3_cheby, onebyr4_cheby,
        ddr_lnρT, ddr_S0_by_cp, g, g_cheby, r_cheby, r2_cheby, κ, twoηρ_by_r, sρ,
        ηρ_by_r, ηρ_by_r2, ηρ2_by_r2, ddr_ηρbyr, ddr_ηρbyr2, ηρ_by_r3,
        ddr_ηρ, d2dr2_ηρ, d3dr3_ηρ)

    diff_operators = (; DDr, DDr_minus_2byr, rDDr, rddr,
        ddr, d2dr2, d3dr3, d4dr4, r2d2dr2, ddr_plus_2byr)

    diff_operator_matrices = (; onebyr_chebyM, onebyr2_chebyM, DDrM,
        ddrM, d2dr2M, rddrM, twoηρ_by_rM, ddr_plus_2byrM,
        ddr_minus_2byrM, DDr_minus_2byrM,
        gM,
        ddr_lobatto,
        ddrDDr_lobatto,
        ηρ_by_rM, ηρ2_by_r2M, d3dr3M, d4dr4M,
        # uniform rotation terms
        onebyr2_IplusrηρM, onebyr2_cheby_ddr_S0_by_cpM, κ_∇r2_plus_ddr_lnρT_ddrM, κ_by_r2M,
        # viscosity terms
        ηρ_ddr_minus_2byrM, onebyr2_d2dr2M, onebyr3_ddrM, onebyr4_chebyM,
    )

    (;
        constants, rad_terms,
        splines,
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

function allocate_matrix(operators)
    (; nparams) = operators.radial_params
    (; nvariables) = operators.constants
    nrows = nvariables * nparams
    sz = (nrows, nrows)
    M = StructArray{ComplexF64}((zeros(sz), zeros(sz)))
end

function uniform_rotation_matrix(nr, nℓ, m; operators, kw...)
    M = allocate_matrix(operators)
    uniform_rotation_matrix!(M, nr, nℓ, m; operators, kw...)
    return M
end

function uniform_rotation_matrix_terms_outer!((SWterm, SSterm),
    (ℓ, m),
    (κ_∇r2_plus_ddr_lnρT_ddrM, κ_by_r2M, onebyr2_cheby_ddr_S0_by_cpM))

    ℓℓp1 = ℓ*(ℓ+1)

    @. SWterm = ℓℓp1 * onebyr2_cheby_ddr_S0_by_cpM * Rsun^3
    @. SSterm = (κ_∇r2_plus_ddr_lnρT_ddrM - ℓℓp1 * κ_by_r2M) * Rsun^2

    SWterm, SSterm
end

function uniform_rotation_matrix!(M, nr, nℓ, m;
        ℓs = range(m, length = nℓ),
        operators,
        # Precompute certain green function integrals
        Jtermsunirot = begin
            lobattochebyshevtransform =
                LobattoChebyshev(operators.transforms.TfGL_nr,
                    operators.transforms.TiGL_nr,
                    operators.transforms.normr)
            OffsetArray(
                map(ℓ -> greenfn_cheby(UniformRotGfn(), ℓ, operators, lobattochebyshevtransform), ℓs), ℓs)
        end,
        kw...
        )

    (; nvariables, Ω0, scalings) = operators.constants;
    (; ddrM, DDrM, onebyr_chebyM, DDr_minus_2byrM,
        onebyr2_cheby_ddr_S0_by_cpM,
        κ_∇r2_plus_ddr_lnρT_ddrM, κ_by_r2M) = operators.diff_operator_matrices;
    (; Sscaling, Wscaling) = scalings;

    WVℓℓ′ = zeros(nr, nr)
    VWℓℓ′ = zeros(nr, nr)
    JC1 = zeros(nr, nr)
    JCℓ′ = zeros(nr, nr)
    Jddr = zeros(nr, nr)

    M .= 0

    VV = matrix_block(M.re, 1, 1, nvariables)
    VW = matrix_block(M.re, 1, 2, nvariables)
    WV = matrix_block(M.re, 2, 1, nvariables)
    WW = matrix_block(M.re, 2, 2, nvariables)
    # the following are only valid if S is included
    if nvariables == 3
        WS = matrix_block(M.re, 2, 3, nvariables)
        SW = matrix_block(M.re, 3, 2, nvariables)
        SS = matrix_block(M.im, 3, 3, nvariables)
    end

    cosθ = OffsetArray(costheta_operator(nℓ, m), ℓs, ℓs);
    sinθdθ = OffsetArray(sintheta_dtheta_operator(nℓ, m), ℓs, ℓs);

    SWterm = zeros(nr, nr);
    SSterm = zeros(nr, nr);

    @views for ℓ in ℓs
        (; J, J_ηρbyr, J_by_r, J_g) = Jtermsunirot[ℓ].unirot_terms
        mul!(Jddr, J, ddrM)
        @. JC1 = Jddr - 2J_by_r

        ℓℓp1 = ℓ * (ℓ + 1)

        blockdiaginds_ℓ = blockinds((m, nr), ℓ)

        uniform_rotation_matrix_terms_outer!((SWterm, SSterm),
                (ℓ, m),
                (κ_∇r2_plus_ddr_lnρT_ddrM, κ_by_r2M,
                    onebyr2_cheby_ddr_S0_by_cpM));

        diagterm = 2m/ℓℓp1

        VVblockdiag = VV[blockdiaginds_ℓ]
        VVblockdiag_diag = VVblockdiag[diagind(VVblockdiag)]
        VVblockdiag_diag .= diagterm

        WWblockdiag = WW[blockdiaginds_ℓ]
        WWblockdiag_diag = WWblockdiag[diagind(WWblockdiag)]
        WWblockdiag_diag .= diagterm
        @. WW[blockdiaginds_ℓ] -= Rsun^2 * 2m * J_ηρbyr

        if nvariables == 3
            @. WS[blockdiaginds_ℓ] = -J_g / (Ω0^2 * Rsun)  * Wscaling/Sscaling
            @. SW[blockdiaginds_ℓ] = SWterm * Sscaling/Wscaling
            @. SS[blockdiaginds_ℓ] = -SSterm
        end

        for ℓ′ in intersect(ℓs, ℓ-1:2:ℓ+1)
            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)

            @. VWℓℓ′ = (-2/ℓℓp1) * (ℓ′ℓ′p1 * DDr_minus_2byrM * cosθ[ℓ, ℓ′] +
                    (DDrM - ℓ′ℓ′p1 * onebyr_chebyM) * sinθdθ[ℓ, ℓ′]) * Rsun / Wscaling

            @. JCℓ′ = Jddr - ℓ′ℓ′p1 * J_by_r
            @. WVℓℓ′ = (-2/ℓℓp1) * (ℓ′ℓ′p1 * JC1 * cosθ[ℓ, ℓ′] + JCℓ′ * sinθdθ[ℓ, ℓ′]) * Rsun * Wscaling

            blockinds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)

            VW[blockinds_ℓℓ′] = VWℓℓ′
            WV[blockinds_ℓℓ′] = WVℓℓ′
        end
    end

    viscosity_terms!(M, nr, nℓ, m; operators, Jtermsunirot)

    return M
end

function viscosity_functions(operators)
    (; onebyr_cheby, onebyr2_cheby, onebyr3_cheby, ηρ_cheby,
        ηρ_by_r, ddr_ηρbyr, ddr_ηρ, d2dr2_ηρ, d3dr3_ηρ) = operators.rad_terms;

    (ηρ_cheby * (ηρ_cheby - 2onebyr_cheby)::TFun,
        (ddr_ηρ * (ηρ_cheby - 2onebyr_cheby))::TFun,
        ((ddr_ηρ - 2ηρ_by_r)*ddr_ηρ) ::TFun,
        (2(ddr_ηρ - ηρ_by_r + onebyr2_cheby)*ηρ_cheby) ::TFun,
        ((-2*ddr_ηρbyr + d2dr2_ηρ) * ηρ_cheby)::TFun,
        (3ddr_ηρ - 4ηρ_by_r)::TFun,
        (3d2dr2_ηρ - 8ddr_ηρ*onebyr_cheby + 8ηρ_cheby*onebyr2_cheby)::TFun,
        (d3dr3_ηρ - 4d2dr2_ηρ*onebyr_cheby + 8ddr_ηρ*onebyr2_cheby - 8ηρ_cheby*onebyr3_cheby)::TFun,
        )
end

function viscosity_terms!(M, nr, nℓ, m; operators,
        ℓs = range(m, length = nℓ),
        Jtermsunirot = begin
            lobattochebyshevtransform =
                LobattoChebyshev(operators.transforms.TfGL_nr,
                    operators.transforms.TiGL_nr,
                    operators.transforms.normr)
            OffsetArray(
                map(ℓ -> greenfn_cheby(UniformRotGfn(), ℓ, operators, lobattochebyshevtransform), ℓs), ℓs)
        end,
        kw...
        )

    (; ddrM, d2dr2M, d3dr3M, onebyr2_chebyM, ηρ_ddr_minus_2byrM, onebyr2_d2dr2M,
        onebyr3_ddrM, onebyr4_chebyM) = operators.diff_operator_matrices;
    (; ν, nvariables) = operators.constants;

    VV = matrix_block(M.im, 1, 1, nvariables)
    WW = matrix_block(M.im, 2, 2, nvariables)

    d2dr2_min_ℓℓp1_by_r2_squaredM = zeros(nr, nr);

    # caches for the WW term
    T3_1 = zeros(nr, nr);
    T3_2 = zeros(nr, nr);
    T3_ℓterms = zeros(nr, nr);
    T4 = zeros(nr, nr);
    WWop = zeros(nr, nr);

    Mcache1 = zeros(nr, nr)
    Mcache2 = zeros(nr, nr)
    Mcache3 = zeros(nr, nr)

    J_c1 = zeros(nr, nr);
    J_c2_1 = zeros(nr, nr);
    J_c2_2 = zeros(nr, nr);
    J_a1_1 = zeros(nr, nr);
    J_a1_2 = zeros(nr, nr);
    J_a1_3 = zeros(nr, nr);
    J_4ηρbyr3 = zeros(nr, nr);
    J_ηρ = zeros(nr, nr);
    J_ηρ²_min_2ηρbyr = zeros(nr, nr);
    J_ηρ_min_2byr_ddrηρ = zeros(nr, nr);
    J_ddrηρ = zeros(nr, nr);
    J_ddrηρbyr2_plus_4ηρbyr3 = zeros(nr, nr);
    J_ηρ2byr2 = zeros(nr, nr);
    J_ηρbyr2 = zeros(nr, nr);
    J_d2dr2ηρ_by_r_min_2ddrηρ_by_r2 = zeros(nr, nr);
    J_ddrηρ_by_r = zeros(nr, nr);
    J_ddrηρ_by_r_min_ηρbyr2 = zeros(nr, nr);
    J_ddrηρ_by_r2_min_4ηρbyr3 = zeros(nr, nr);
    J_ddrηρ_by_r2 = zeros(nr, nr);
    J_d2dr2ηρ_by_r = zeros(nr, nr);

    viscosity_terms = (;
        J_c1,
        J_c2_1,
        J_c2_2,
        J_a1_1,
        J_a1_2,
        J_a1_3,
        J_4ηρbyr3,
        J_ηρ,
        J_ηρ²_min_2ηρbyr,
        J_ηρ_min_2byr_ddrηρ,
        J_ddrηρ,
        J_ddrηρbyr2_plus_4ηρbyr3,
        J_ηρ2byr2,
        J_ηρbyr2,
        J_d2dr2ηρ_by_r_min_2ddrηρ_by_r2,
        J_ddrηρ_by_r,
        J_ddrηρ_by_r_min_ηρbyr2,
        J_ddrηρ_by_r2_min_4ηρbyr3,
        J_ddrηρ_by_r2,
        J_d2dr2ηρ_by_r,
    );

    funs = viscosity_functions(operators)

    Nlobatto = length(operators.coordinates.r_chebyshev_lobatto)
    tempmatGfn = zeros(Nlobatto, Nlobatto);
    tempvecGfn = zeros(Nlobatto);
    lobattochebyshevtransform =
        LobattoChebyshev(operators.transforms.TfGL_nr,
                operators.transforms.TiGL_nr,
                operators.transforms.normr);

    for ℓ in ℓs
        blockdiaginds_ℓ = blockinds((m, nr), ℓ)

        ℓℓp1 = ℓ * (ℓ + 1)
        neg2by3_ℓℓp1 = -2ℓℓp1 / 3

        @views @. VV[blockdiaginds_ℓ] -= ν * (d2dr2M - ℓℓp1 * onebyr2_chebyM + ηρ_ddr_minus_2byrM) * Rsun^2

        G = greenfn_cheby!(ViscosityGfn(), ℓ, operators, viscosity_terms, funs, Jtermsunirot[ℓ],
            tempmatGfn, tempvecGfn, lobattochebyshevtransform);

        (; J, J_ηρbyr) = G.unirot_terms;

        ℓpre = (ℓ-2)*ℓ*(ℓ+1)*(ℓ+3)
        @. d2dr2_min_ℓℓp1_by_r2_squaredM = ℓpre * onebyr4_chebyM - 2ℓℓp1*onebyr2_d2dr2M + 4ℓℓp1*onebyr3_ddrM;
        # @. d2dr2_min_ℓℓp1_by_r2_squaredM = d4dr4M + ℓpre * onebyr4_chebyM - 2ℓℓp1*onebyr2_d2dr2M + 4ℓℓp1*onebyr3_ddrM;

        mul!(WWop, J, d2dr2_min_ℓℓp1_by_r2_squaredM)
        WWop .+= mul!(Mcache1, J_ηρ, d3dr3M) .+ mul!(Mcache2, J_a1_1, d2dr2M) .+ mul!(Mcache3, J_a1_2, ddrM) .+ J_a1_3
        WWop .-= ℓℓp1 .* (mul!(Mcache1, J_ηρbyr2, ddrM) .+ J_ddrηρ_by_r2_min_4ηρbyr3)
        WWop .+= 4 .* (mul!(Mcache1, J_ηρbyr, d2dr2M) .+
                2 .* mul!(Mcache2, J_ddrηρ_by_r_min_ηρbyr2, ddrM) .+ J_d2dr2ηρ_by_r_min_2ddrηρ_by_r2)
        @. WWop -= (ℓℓp1-2) * J_4ηρbyr3

        T3_1 .= mul!(Mcache1, J_ddrηρ, d2dr2M) .+ mul!(Mcache2, J_ηρ_min_2byr_ddrηρ, ddrM) .+ J_c1;
        T3_2 .= mul!(Mcache1, J_ηρ, d3dr3M) .+ mul!(Mcache2, J_ηρ²_min_2ηρbyr, d2dr2M) .+
                    mul!(Mcache3, J_c2_1, ddrM) .+ J_c2_2;
        T3_ℓterms .= ℓℓp1 .* (.- mul!(Mcache1, J_ηρbyr2, ddrM) .+ J_ddrηρbyr2_plus_4ηρbyr3)

        @. T4 = neg2by3_ℓℓp1 * J_ηρ2byr2

        @. WWop += (T3_1 + T3_2 + T3_ℓterms) + T4

        @views @. WW[blockdiaginds_ℓ] -= ν * WWop * Rsun^4
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

function read_angular_velocity(operators, thetaGL; smoothing_param = 1e-3)
    (; r) = operators.coordinates;
    (; r_out) = operators.radial_params;

    parentdir = dirname(@__DIR__)
    r_ΔΩ_raw = read_angular_velocity_radii(parentdir)
    Ω_raw = read_angular_velocity_raw(parentdir)
    Ω0 = equatorial_rotation_angular_velocity(r_out / Rsun, r_ΔΩ_raw, Ω_raw)
    ΔΩ_raw = Ω_raw .- Ω0

    nθ = size(ΔΩ_raw, 2)
    lats_raw = LinRange(0, pi, nθ)

    splΔΩ2D = Spline2D(r_ΔΩ_raw * Rsun, lats_raw, ΔΩ_raw; s = sum(abs2, ΔΩ_raw)*smoothing_param)
    ΔΩ2D_grid = evalgrid(splΔΩ2D, r, thetaGL);

    (; ΔΩ2D_grid, Ω0)
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

function matrix_block(M, rowind, colind, nvariables = 3)
    nparams = div(size(M, 1), nvariables)
    inds1 = (rowind - 1) * nparams .+ (1:nparams)
    inds2 = (colind - 1) * nparams .+ (1:nparams)
    inds = CartesianIndices((inds1, inds2))
    @view M[inds]
end

matrix_block_maximum(f, M::AbstractMatrix, nvariables = 3) = [maximum(f, matrix_block(M, i, j)) for i in 1:nvariables, j in 1:nvariables]
function matrix_block_maximum(M::AbstractMatrix, operators::NamedTuple)
    (; nvariables) = operators.constants
    R = RossbyWaveSpectrum.matrix_block_maximum(abs∘real, M, nvariables)
    I = RossbyWaveSpectrum.matrix_block_maximum(abs∘imag, M, nvariables)
    [R I]
end

function constant_differential_rotation_terms!(M, nr, nℓ, m;
        operators,
        ℓs = range(m, length = nℓ),
        Jtermsunirot = begin
            lobattochebyshevtransform =
                LobattoChebyshev(operators.transforms.TfGL_nr,
                    operators.transforms.TiGL_nr,
                    operators.transforms.normr)
            OffsetArray(
                map(ℓ -> greenfn_cheby(UniformRotGfn(), ℓ, operators, lobattochebyshevtransform), ℓs), ℓs)
        end,
        ΔΩ_by_Ω0 = 0.02,
        kw...
    )

    (; nvariables, scalings) = operators.constants
    (; ddrM, DDrM, onebyr_chebyM) = operators.diff_operator_matrices

    (; Wscaling) = scalings

    VV = matrix_block(M.re, 1, 1, nvariables)
    VW = matrix_block(M.re, 1, 2, nvariables)
    WV = matrix_block(M.re, 2, 1, nvariables)
    WW = matrix_block(M.re, 2, 2, nvariables)
    if nvariables == 3
        SS = matrix_block(M.re, 3, 3, nvariables)
    end

    cosθo = OffsetArray(costheta_operator(nℓ, m), ℓs, ℓs)
    sinθdθo = OffsetArray(sintheta_dtheta_operator(nℓ, m), ℓs, ℓs)
    laplacian_sinθdθo = OffsetArray(Diagonal(@. -ℓs * (ℓs + 1)) * parent(sinθdθo), ℓs, ℓs)

    DDr_min_2byrM = @. DDrM - 2 * onebyr_chebyM

    Jddr = zeros(nr, nr)
    Jddr_plus_2byrM = zeros(nr, nr)

    for ℓ in ℓs
        # numerical green function
        (; J, J_ηρbyr, J_by_r) = Jtermsunirot[ℓ].unirot_terms
        ℓℓp1 = ℓ * (ℓ + 1)
        blockdiaginds_ℓ = blockinds((m, nr), ℓ)
        two_over_ℓℓp1 = 2 / ℓℓp1

        dopplerterm = m * ΔΩ_by_Ω0
        diagterm = m * two_over_ℓℓp1 * ΔΩ_by_Ω0 - dopplerterm

        VVd = @view VV[blockdiaginds_ℓ]
        @views VVd[diagind(VVd)] .+= diagterm

        WWd = @view WW[blockdiaginds_ℓ]
        @views WWd[diagind(WWd)] .+= diagterm
        @. WWd -= m * 2ΔΩ_by_Ω0 * Rsun^2 * J_ηρbyr

        mul!(Jddr, J, ddrM)
        @. Jddr_plus_2byrM = Jddr + 2J_by_r

        if nvariables == 3
            SSd = @view SS[blockdiaginds_ℓ]
            @views SSd[diagind(SSd)] .-= dopplerterm
        end

        @views for ℓ′ in ℓ-1:2:ℓ+1
            ℓ′ in ℓs || continue
            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)
            inds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)

            @. VW[inds_ℓℓ′] += -two_over_ℓℓp1 *
                                    ΔΩ_by_Ω0 * (
                                        ℓ′ℓ′p1 * cosθo[ℓ, ℓ′] * DDr_min_2byrM +
                                        (DDrM - ℓ′ℓ′p1 * onebyr_chebyM) * sinθdθo[ℓ, ℓ′]
                                    ) * Rsun / Wscaling

            @. WV[inds_ℓℓ′] += -Rsun * ΔΩ_by_Ω0 / ℓℓp1 *
                                    ((4ℓ′ℓ′p1 * cosθo[ℓ, ℓ′] + (ℓ′ℓ′p1 + 2) * sinθdθo[ℓ, ℓ′]) * Jddr
                                        +
                                        Jddr_plus_2byrM * laplacian_sinθdθo[ℓ, ℓ′]) * Wscaling
        end
    end
    return M
end

function equatorial_radial_rotation_profile(operators, thetaGL; smoothing_param = 1e-3)
    (; r) = operators.coordinates
    ΔΩ_rθ, Ω0 = read_angular_velocity(operators, thetaGL; smoothing_param)
    s = Spline2D(r, thetaGL, ΔΩ_rθ)
    ΔΩ_r = reshape(evalgrid(s, r, [pi/2]), Val(1))
end

function radial_differential_rotation_profile(operators, thetaGL, model = :solar_equator;
    smoothing_param = 1e-3)

    (; r) = operators.coordinates
    (; r_out, nr, r_in) = operators.radial_params

    if model == :solar_equator
        Ω0 = equatorial_rotation_angular_velocity(r_out / Rsun)
        ΔΩ_r = equatorial_radial_rotation_profile(operators, thetaGL; smoothing_param)
    elseif model == :linear # for testing
        Ω0 = equatorial_rotation_angular_velocity(r_out / Rsun)
        f = 0.02 / (r_in / Rsun - 1)
        ΔΩ_r = @. Ω0 * f * (r / Rsun - 1)
    elseif model == :constant # for testing
        ΔΩ_by_Ω = 0.02
        Ω0 = equatorial_rotation_angular_velocity(r_out / Rsun)
        ΔΩ_r = fill(ΔΩ_by_Ω * Ω0, nr)
    elseif model == :core
        ΔΩ_by_Ω = 0.3
        Ω0 = equatorial_rotation_angular_velocity(r_out / Rsun)
        ΔΩ_r = @. (Ω0*ΔΩ_by_Ω)*1/2*(1 - tanh((r - 0.6Rsun)/(0.08Rsun)))
    else
        error("$model is not a valid rotation model")
    end
    return ΔΩ_r, Ω0
end

function rotationprofile_radialderiv(r, ΔΩ_r, nr, Δr)
    ΔΩ = chop(chebyshevgrid_to_Fun(ΔΩ_r), 1e-3);
    if ncoefficients(ΔΩ) > 2nr/3
        @warn "ncoefficients(ΔΩ) = $(ncoefficients(ΔΩ))"
    end

    ΔΩ_spl = Spline1D(r, ΔΩ_r);
    ddrΔΩ_r = derivative(ΔΩ_spl, r);
    zchop!(ddrΔΩ_r, 1e-10*(2/Δr))
    d2dr2ΔΩ_r = derivative(ΔΩ_spl, r, nu=2);
    zchop!(d2dr2ΔΩ_r, 1e-10*(2/Δr)^2)

    ddrΔΩ = chop(chebyshevgrid_to_Fun(ddrΔΩ_r), 1e-2);
    if ncoefficients(ddrΔΩ) > 2nr/3
        @warn "ncoefficients(ddrΔΩ) = $(ncoefficients(ddrΔΩ))"
    end
    d2dr2ΔΩ = chop(chebyshevgrid_to_Fun(d2dr2ΔΩ_r), 5e-2);
    if ncoefficients(d2dr2ΔΩ) > 2nr/3
        @warn "ncoefficients(d2dr2ΔΩ) = $(ncoefficients(d2dr2ΔΩ))"
    end

    (ΔΩ, ddrΔΩ, d2dr2ΔΩ)
end

function replaceemptywithzero(f::Fun)
    T = eltype(coefficients(f))
    ncoefficients(f) == 0 ? typeof(f)(space(f), T[0]) : f
end

function radial_differential_rotation_profile_derivatives(m; operators,
        rotation_profile = :radial, smoothing_param = 1e-3)
    (; r) = operators.coordinates;
    (; nr, nℓ, Δr) = operators.radial_params;

    ntheta = ntheta_ℓmax(nℓ, m);
    (; thetaGL) = gausslegendre_theta_grid(ntheta);

    ΔΩ_r, Ω0 = radial_differential_rotation_profile(operators, thetaGL, rotation_profile; smoothing_param);
    ΔΩ_r ./= Ω0;

    (ΔΩ, ddrΔΩ, d2dr2ΔΩ) = replaceemptywithzero.(rotationprofile_radialderiv(r, ΔΩ_r, nr, Δr))
    (; Ω0, ΔΩ, ddrΔΩ, d2dr2ΔΩ)
end

function radial_differential_rotation_terms_inner!((VWterm, WVterm), (ℓ, ℓ′),
    (cosθ_ℓℓ′, sinθdθ_ℓℓ′, ∇²_sinθdθ_ℓℓ′),
    (ddrΔΩM,),
    (ΔΩ_by_rM, ΔΩ_DDrM, ΔΩ_DDr_min_2byrM, ddrΔΩ_plus_ΔΩddrM))

    ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)
    ℓℓp1 = ℓ * (ℓ + 1)
    two_over_ℓℓp1 = 2/ℓℓp1

    @. VWterm = -two_over_ℓℓp1 *
            (ℓ′ℓ′p1 * cosθ_ℓℓ′ * (ΔΩ_DDr_min_2byrM - ddrΔΩM) +
            sinθdθ_ℓℓ′ * ((ΔΩ_DDrM - ℓ′ℓ′p1 * ΔΩ_by_rM) - ℓ′ℓ′p1 / 2 * ddrΔΩM))

    @. WVterm = -1/ℓℓp1 * (
                (4ℓ′ℓ′p1 * cosθ_ℓℓ′ + (ℓ′ℓ′p1 + 2) * sinθdθ_ℓℓ′ + ∇²_sinθdθ_ℓℓ′) * ddrΔΩ_plus_ΔΩddrM
                + ∇²_sinθdθ_ℓℓ′ * 2ΔΩ_by_rM
            )

    VWterm, WVterm
end

function radial_differential_rotation_terms!(M, nr, nℓ, m;
        operators,
        ℓs = range(m, length = nℓ),
        Jtermsunirot = begin
            lobattochebyshevtransform =
                LobattoChebyshev(operators.transforms.TfGL_nr,
                    operators.transforms.TiGL_nr,
                    operators.transforms.normr)
            OffsetArray(
                map(ℓ -> greenfn_cheby(UniformRotGfn(), ℓ, operators, lobattochebyshevtransform), ℓs), ℓs)
        end,
        rotation_profile = :constant,
        kw...
    )

    (; nvariables, scalings) = operators.constants;
    (; DDr, ddr) = operators.diff_operators;
    (; ddrM) = operators.diff_operator_matrices;
    (; onebyr_cheby, g_cheby) = operators.rad_terms;
    (; mat) = operators;
    (; Sscaling, Wscaling) = scalings;

    VV = matrix_block(M.re, 1, 1, nvariables)
    VW = matrix_block(M.re, 1, 2, nvariables)
    WV = matrix_block(M.re, 2, 1, nvariables)
    WW = matrix_block(M.re, 2, 2, nvariables)
    if nvariables === 3
        SV = matrix_block(M.re, 3, 1, nvariables)
        SW = matrix_block(M.re, 3, 2, nvariables)
        SS = matrix_block(M.re, 3, 3, nvariables)
    end

    ΔΩprofile_deriv = radial_differential_rotation_profile_derivatives(m; operators, rotation_profile);

    (; ΔΩ, ddrΔΩ, Ω0) = ΔΩprofile_deriv;

    ΔΩM = mat(ΔΩ);
    ddrΔΩM = mat(ddrΔΩ);

    ddrΔΩ_over_g = ddrΔΩ / g_cheby;
    ddrΔΩ_over_gM = mat(ddrΔΩ_over_g);
    ddrΔΩ_over_g_DDr = (ddrΔΩ_over_g * DDr)::Tmul;
    ddrΔΩ_over_g_DDrM = mat(ddrΔΩ_over_g_DDr);
    ddrΔΩ_plus_ΔΩddr = (ddrΔΩ + (ΔΩ * ddr)::Tmul)::Tplus;

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

    inner_matrices = map(mat, (ΔΩ_by_r, ΔΩ_DDr, ΔΩ_DDr_min_2byr, ddrΔΩ_plus_ΔΩddr));
    (ΔΩ_by_rM, ΔΩ_DDrM, ΔΩ_DDr_min_2byrM) = inner_matrices;

    Nlobatto = length(operators.coordinates.r_chebyshev_lobatto)
    tempmatGfn = zeros(Nlobatto, Nlobatto);
    tempvecGfn = zeros(Nlobatto);
    lobattochebyshevtransform =
        LobattoChebyshev(operators.transforms.TfGL_nr,
                operators.transforms.TiGL_nr,
                operators.transforms.normr);

    J_ηρbyr_ΔΩ = zeros(nr, nr);
    J_ddrΩ = zeros(nr, nr);
    J_twoΔΩ_by_r = zeros(nr, nr);
    J_d2dr2ΔΩ = zeros(nr, nr);
    J_ddrΩ_ηρ = zeros(nr, nr);
    J_2byr_min_ηρ__min__twoddr_plus_3ηρr_J__times_drΔΩ = zeros(nr, nr);
    twoddr_plus_3ηρr_J_ddrΔΩ = zeros(nr, nr);
    J_ΔΩ = zeros(nr, nr);
    twoddr_plus_3ηρr_J = zeros(nr, nr);
    J_times_2byr_min_ηρ = zeros(nr, nr);

    diffrot_terms = (;
        J_ηρbyr_ΔΩ,
        J_ddrΩ,
        J_twoΔΩ_by_r,
        J_d2dr2ΔΩ,
        J_ddrΩ_ηρ,
        J_2byr_min_ηρ__min__twoddr_plus_3ηρr_J__times_drΔΩ,
        twoddr_plus_3ηρr_J_ddrΔΩ,
        J_ΔΩ,
        twoddr_plus_3ηρr_J,
        J_times_2byr_min_ηρ,
    );

    @views for ℓ in ℓs
        # numerical green function
        G = greenfn_cheby!(RadDiffRotGfn(), ℓ, operators, ΔΩprofile_deriv,
            diffrot_terms, Jtermsunirot[ℓ],
            tempmatGfn, tempvecGfn, lobattochebyshevtransform);

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

        if nvariables == 3
            @. SS[inds_ℓℓ] -= m * ΔΩM
        end

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

            if nvariables == 3
                @. SV[inds_ℓℓ′] -= (Ω0^2 * Rsun^2) * 2m * cosθo[ℓ, ℓ′] * ddrΔΩ_over_gM * Sscaling;
            end
        end

        for ℓ′ in intersect(ℓs, ℓ-2:2:ℓ+2)
            inds_ℓℓ′ = blockinds((m, nr), ℓ, ℓ′)

            if nvariables == 3
                @. SW[inds_ℓℓ′] += (Ω0^2 * Rsun^3) * 2cosθsinθdθo[ℓ, ℓ′] * ddrΔΩ_over_g_DDrM * Sscaling/Wscaling;
            end
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

    (; nvariables) = operators.constants
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

    VV = matrix_block(M, 1, 1, nvariables)
    VW = matrix_block(M, 1, 2, nvariables)
    WV = matrix_block(M, 2, 1, nvariables)
    WW = matrix_block(M, 2, 2, nvariables)
    SV = matrix_block(M, 3, 1, nvariables)
    SW = matrix_block(M, 3, 2, nvariables)

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

function _differential_rotation_matrix!(M, nr, nℓ, m, rotation_profile; kw...)
    if rotation_profile === :radial
        radial_differential_rotation_terms!(M, nr, nℓ, m; rotation_profile = :solar_equator, kw...)
    elseif rotation_profile === :radial_constant
        radial_differential_rotation_terms!(M, nr, nℓ, m; rotation_profile = :constant, kw...)
    elseif rotation_profile === :radial_linear
        radial_differential_rotation_terms!(M, nr, nℓ, m; rotation_profile = :linear, kw...)
    elseif rotation_profile === :radial_core
        radial_differential_rotation_terms!(M, nr, nℓ, m; rotation_profile = :core, kw...)
    elseif rotation_profile === :solar
        solar_differential_rotation_terms!(M, nr, nℓ, m; rotation_profile, kw...)
    elseif rotation_profile === :solar_radial
        solar_differential_rotation_terms!(M, nr, nℓ, m; rotation_profile = :radial, kw...)
    elseif rotation_profile === :solar_constant
        solar_differential_rotation_terms!(M, nr, nℓ, m; rotation_profile = :constant, kw...)
    elseif rotation_profile === :constant
        constant_differential_rotation_terms!(M, nr, nℓ, m; kw...)
    else
        throw(ArgumentError("Invalid rotation profile"))
    end
    return M
end
function differential_rotation_matrix(nr, nℓ, m; rotation_profile, operators, kw...)
    ℓs = range(m, length = nℓ)
    lobattochebyshevtransform = LobattoChebyshev(operators.transforms.TfGL_nr,
                                    operators.transforms.TiGL_nr,
                                    operators.transforms.normr)
    Jtermsunirot = OffsetArray(
            map(ℓ -> greenfn_cheby(UniformRotGfn(), ℓ, operators, lobattochebyshevtransform), ℓs), ℓs)
    M = uniform_rotation_matrix(nr, nℓ, m; operators, Jtermsunirot, ℓs, kw...)
    _differential_rotation_matrix!(M, nr, nℓ, m, rotation_profile; operators, Jtermsunirot, ℓs)
    return M
end
function differential_rotation_matrix!(M, nr, nℓ, m; rotation_profile, operators,
    ℓs = range(m, length = nℓ),
    Jtermsunirot = begin
        lobattochebyshevtransform = LobattoChebyshev(operators.transforms.TfGL_nr,
                                operators.transforms.TiGL_nr,
                                operators.transforms.normr)
        OffsetArray(
        map(ℓ -> greenfn_cheby(UniformRotGfn(), ℓ, operators, lobattochebyshevtransform), ℓs), ℓs)
    end,
    kw...)

    uniform_rotation_matrix!(M, nr, nℓ, m; operators, Jtermsunirot, ℓs, kw...)
    _differential_rotation_matrix!(M, nr, nℓ, m, rotation_profile; operators, Jtermsunirot, ℓs)
    return M
end

_maybetrimM(M, nvariables, nparams) = nvariables == 3 ? M : M[1:(nvariables*nparams), 1:(nvariables*nparams)]

function constrained_matmul_cache(constraints)
    (; ZC) = constraints
    sz_constrained = (size(ZC, 2), size(ZC, 2))
    MZCcache = zeros(size(ZC))
    M_constrained_reim = zeros(sz_constrained)
    M_constrained = zeros(ComplexF64, sz_constrained)
    return (; MZCcache, M_constrained, M_constrained_reim)
end

function compute_constrained_matrix(M, constraints,
        cache = constrained_matmul_cache(constraints))

    (; M_constrained, MZCcache, M_constrained_reim) = cache
    (; ZC) = constraints

    #= not thread-safe if cache is preallocated =#
    mul!(M_constrained_reim, ZC', mul!(MZCcache, M.re, ZC))
    for i in eachindex(M_constrained)
        M_constrained[i] = M_constrained_reim[i]
    end
    mul!(M_constrained_reim, ZC', mul!(MZCcache, M.im, ZC))
    for i in eachindex(M_constrained)
        M_constrained[i] += im*M_constrained_reim[i]
    end

    # thread-safe version but allocating
    # M_constrained = Complex.(ZC' * real(M) * ZC, ZC' * imag(M) * ZC)
    return M_constrained
end

function compute_matrix_scales(M, nvariables)
    blockscalesreal = ones(nvariables, nvariables)

    Mrmax = zeros(nvariables, nvariables)
    for colind in 1:nvariables, rowind in 1:nvariables
        Mv = matrix_block(M, rowind, colind, nvariables)
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

    (; Wscaling = blockscalesreal[2, 1], Sscaling = blockscalesreal[3, 1])
end

function balance_matrix!(M, nvariables, scales)
    for colind in 1:nvariables, rowind in 1:nvariables
        Mv = matrix_block(M, rowind, colind, nvariables)
        Mv .*= scales[rowind, colind]
    end
    return nothing
end

function realmatcomplexmatmul(A, B, (v, temp)::NTuple{2,AbstractMatrix})
    @. temp = real(B)
    mul!(v.re, A, temp)
    @. temp = imag(B)
    mul!(v.im, A, temp)
    v
end

function allocate_projectback_temp_matrices(sz)
    vr = zeros(sz)
    vi = zeros(sz)
    v = StructArray{ComplexF64}((vr, vi))
    temp = zeros(sz[2], sz[2])
    v, temp
end

function constrained_eigensystem(M;
    operators,
    constraints = constraintmatrix(operators),
    cache = constrained_matmul_cache(constraints),
    rebalance_matrix = false,
    scalings = (; Wscaling = 1, Sscaling = 1),
    temp_projectback = allocate_projectback_temp_matrices(size(constraints.ZC)),
    eigencache = allocate_eigen_cache(cache.M_constrained),
    λ = similar(M, ComplexF64, size(cache.M_constrained, 1)),
    w = similar(M, ComplexF64, size(cache.M_constrained)),
    timer = TimerOutput(),
    kw...
    )

    (; nparams) = operators.radial_params
    (; ZC, nvariables) = constraints
    M = _maybetrimM(M, nvariables, nparams)
    (; Wscaling, Sscaling) = merge((; Wscaling = 1, Sscaling = 1), scalings)
    scales = [
        1               1/Wscaling              1/Sscaling
        Wscaling        1                       Wscaling/Sscaling
        Sscaling        Sscaling/Wscaling           1
    ]
    if rebalance_matrix
        balance_matrix!(M, nvariables, scales)
    end
    @timeit timer "basis" M_constrained = compute_constrained_matrix(M, constraints, cache)
    @timeit timer "eigen" eigenCF64!(M_constrained; cache = eigencache, lams = λ, vecs = w)
    @timeit timer "projectback" v = realmatcomplexmatmul(ZC, w, temp_projectback)
    λ, v, M
end

function uniform_rotation_spectrum(nr, nℓ, m; operators,
    constraints = constraintmatrix(operators),
    kw...)

    rp = operators.radial_params
    @assert (nr, nℓ) == (rp.nr, rp.nℓ) "Please regenerate operators for nr = $nr amd nℓ = $nℓ"

    to = TimerOutput()

    @timeit to "matrix" M = uniform_rotation_matrix(nr, nℓ, m; operators, kw...)
    X = @timeit to "eigen" constrained_eigensystem(M; operators, constraints, timer = to, kw...)
    if get(kw, :print_timer, false)
        println(to)
    end
    X
end
function uniform_rotation_spectrum!(M, nr, nℓ, m; operators,
    constraints = constraintmatrix(operators),
    kw...)

    rp = operators.radial_params
    @assert (nr, nℓ) == (rp.nr, rp.nℓ) "Please regenerate operators for nr = $nr amd nℓ = $nℓ"

    to = TimerOutput()

    @timeit to "matrix" uniform_rotation_matrix!(M, nr, nℓ, m; operators, kw...)
    X = @timeit to "eigen" constrained_eigensystem(M; operators, constraints, timer = to, kw...)
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
    kw...)

    rp = operators.radial_params
    @assert (nr, nℓ) == (rp.nr, rp.nℓ) "Please regenerate operators for nr = $nr amd nℓ = $nℓ"

    to = TimerOutput()

    @timeit to "matrix" M = differential_rotation_matrix(nr, nℓ, m; operators, rotation_profile, kw...)
    X = @timeit to "eigen" constrained_eigensystem(M; operators, constraints, timer = to, kw...)
    if get(kw, :print_timer, false)
        println(to)
    end
    X
end
function differential_rotation_spectrum!(M, nr, nℓ, m;
    rotation_profile, operators,
    constraints = constraintmatrix(operators),
    kw...)

    rp = operators.radial_params
    @assert (nr, nℓ) == (rp.nr, rp.nℓ) "Please regenerate operators for nr = $nr amd nℓ = $nℓ"

    to = TimerOutput()

    @timeit to "matrix" differential_rotation_matrix!(M, nr, nℓ, m; operators, rotation_profile, kw...)
    X = @timeit to "eigen" constrained_eigensystem(M; operators, constraints, timer = to, kw...)
    if get(kw, :print_timer, false)
        println(to)
    end
    X
end

rossby_ridge(m; ΔΩ_by_Ω = 0) = 2 / (m + 1) * (1 + ΔΩ_by_Ω) - m * ΔΩ_by_Ω

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
function boundary_condition_filter(v, BC, BCVcache, atol = 1e-5)
    mul!(BCVcache.re, BC, v.re)
    mul!(BCVcache.im, BC, v.im)
    norm(BCVcache) < atol
end
function isapprox2(x, y; rtol)
    N = max(norm(x), norm(y))
    Ndiff = norm((xi - yi for (xi, yi) in zip(x, y)))
    Ndiff <= rtol * N
end
function eigensystem_satisfy_filter(λ, v, M, MVcache, rtol = 1e-1)
    mul!(MVcache.re, M.re, v.re)
    mul!(MVcache.re, M.im, v.im, -1.0, 1.0)
    mul!(MVcache.im, M.re, v.im)
    mul!(MVcache.im, M.im, v.re,  1.0, 1.0)
    MVcache ./= λ
    isapprox2(MVcache, v; rtol) && return true
    # isapprox2(MVcache.re, v.re; rtol) && isapprox2(MVcache.im, v.im; rtol) && return true
    return false
end

function filterfields(coll, v, nparams, nvariables; filterfieldpowercutoff = 1e-4)
    Vpow = sum(abs2, @view v[1:nparams])
    Wpow = sum(abs2, @view v[nparams .+ (1:nparams)])
    Spow = nvariables == 3 ? sum(abs2, @view(v[2nparams .+ (1:nparams)])) : 0.0

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
    (; nvariables) = operators.constants
    fields = filterfields(VWSinvsh, v, nparams, nvariables; filterfieldpowercutoff)

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
    (; nvariables) = operators.constants
    fields = filterfields(VWSinv, v, nparams, nvariables; filterfieldpowercutoff)

    for X in fields
        Xrange = view(X, :, rangescan)
        PV_frac = sum(abs2, @view X[1:n_cutoff_ind, rangescan]) / sum(abs2, X[:, rangescan])
        flag &= PV_frac > n_power_cutoff
    end

    flag
end

function spatial_filter!(VWSinv, VWSinvsh, F, v, m, operators,
    θ_cutoff = deg2rad(75), equator_power_cutoff_frac = 0.3;
    nℓ = operators.radial_params.nℓ,
    Plcosθ = zeros(range(m, length=nℓ)),
    filterfieldpowercutoff = 1e-4)

    (; θ) = eigenfunction_realspace!(VWSinv, VWSinvsh, F, v, m, operators; nℓ, Plcosθ)

    eqfilter = true

    (; nparams) = operators.radial_params
    (; nvariables) = operators.constants
    fields = filterfields(VWSinv, v, nparams, nvariables; filterfieldpowercutoff)

    for X in fields
        peak_latprofile = @view X[end, :]
        θlowind = searchsortedfirst(θ, θ_cutoff)
        θhighind = searchsortedlast(θ, pi - θ_cutoff)
        powfrac = sum(abs2, @view peak_latprofile[θlowind:θhighind]) / sum(abs2, peak_latprofile)
        powflag = powfrac > equator_power_cutoff_frac
        peakflag = maximum(abs2, @view peak_latprofile[θlowind:θhighind]) == maximum(abs2, peak_latprofile)
        eqfilter &= powflag & peakflag
    end

    eqfilter
end

function nodes_filter(VWSinv, VWSinvsh, F, v, m, operators;
    nℓ = operators.radial_params.nℓ,
    Plcosθ = zeros(range(m, length=nℓ)),
    filterfieldpowercutoff = 1e-4,
    nnodesmax = 7)

    nodesfilter = true

    (; nparams) = operators.radial_params
    (; nvariables) = operators.constants
    fields = filterfields(VWSinv, v, nparams, nvariables; filterfieldpowercutoff)

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

@bitflag FilterFlag::UInt8 begin
    F_NONE=0
    F_EIGVAL
    F_EIGEN
    F_SPHARM
    F_CHEBY
    F_BC
    F_SPATIAL
    F_NODES
end
FilterFlag(F::FilterFlag) = F
Base.:(!)(F::FilterFlag) = FilterFlag(Int(typemax(UInt8) >> 1) - Int(F))
Base.in(t::FilterFlag, F::FilterFlag) = (t & F) != F_NONE
Base.broadcastable(x::FilterFlag) = Ref(x)

function filterfn(λ, v, m, M, (operators, constraints, filtercache, kw)::NTuple{4,Any}, filterflags)

    (; BC) = constraints
    (; nℓ) = operators.radial_params;

    @unpack eig_imag_unstable_cutoff = kw
    @unpack eig_imag_to_real_ratio_cutoff = kw
    @unpack ΔΩ_by_Ω_low = kw
    @unpack ΔΩ_by_Ω_high = kw
    @unpack eig_imag_damped_cutoff = kw
    @unpack Δl_cutoff = kw
    @unpack Δl_power_cutoff = kw
    @unpack atol_constraint = kw
    @unpack n_cutoff = kw
    @unpack n_power_cutoff = kw
    @unpack θ_cutoff = kw
    @unpack equator_power_cutoff_frac = kw
    @unpack eigen_rtol = kw
    @unpack filterfieldpowercutoff = kw
    @unpack nnodesmax = kw
    @unpack scalings = kw

    (; MVcache, Vcache, BCVcache, VWSinv, VWSinvsh, Plcosθ, F) = filtercache;

    FILTERS = FilterFlag(filterflags)

    if F_EIGEN in FILTERS
        f1 = eigenvalue_filter(λ, m;
        eig_imag_unstable_cutoff, eig_imag_to_real_ratio_cutoff,
        ΔΩ_by_Ω_low, ΔΩ_by_Ω_high, eig_imag_damped_cutoff)
        f1 || return false
    end

    Vcache .= v
    if F_SPHARM in FILTERS
        f2 = sphericalharmonic_filter!(VWSinvsh, F, Vcache, operators,
            Δl_cutoff, Δl_power_cutoff, filterfieldpowercutoff)
        f2 || return false
    end

    if F_BC in FILTERS
        f3 = boundary_condition_filter(Vcache, BC, BCVcache, atol_constraint)
        f3 || return false
    end

    if F_CHEBY in FILTERS
        f4 = chebyshev_filter!(VWSinv, F, Vcache, m, operators, n_cutoff,
            n_power_cutoff; nℓ, Plcosθ,filterfieldpowercutoff)
        f4 || return false
    end

    if F_SPATIAL in FILTERS
        f5 = spatial_filter!(VWSinv, VWSinvsh, F, Vcache, m, operators,
            θ_cutoff, equator_power_cutoff_frac; nℓ, Plcosθ,
            filterfieldpowercutoff)
        f5 || return false
    end

    if F_EIGEN in FILTERS
        f6 = eigensystem_satisfy_filter(λ, Vcache, M, MVcache, eigen_rtol)
        f6 || return false
    end

    if F_NODES in FILTERS
        f7 = nodes_filter(VWSinv, VWSinvsh, F, Vcache, m, operators;
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
    (; BC, nvariables) = constraints
    (; nr, nℓ, nparams) = operators.radial_params
    # temporary cache arrays
    nrows = nvariables * nparams
    MVcache = StructArray{ComplexF64}((zeros(nrows), zeros(nrows)))
    Vcache = StructArray{ComplexF64}((zeros(nrows), zeros(nrows)))

    n_bc = size(BC, 1)
    BCVcache = StructArray{ComplexF64}((zeros(n_bc), zeros(n_bc)))

    nθ = length(spharm_θ_grid_uniform(m, nℓ).θ)

    (; VWSinv, VWSinvsh, F) = allocate_field_caches(nr, nθ, nℓ)

    Plcosθ = zeros(range(m, length=nℓ))

    return (; MVcache, Vcache, BCVcache, VWSinv, VWSinvsh, Plcosθ, F)
end

const DefaultFilterParams = Dict(
    :atol_constraint => 1e-5,
    :Δl_cutoff => 7,
    :Δl_power_cutoff => 0.9,
    :eigen_rtol => 0.01,
    :n_cutoff => 10,
    :n_power_cutoff => 0.9,
    :eig_imag_unstable_cutoff => -1e-3,
    :eig_imag_to_real_ratio_cutoff => 1e-1,
    :eig_imag_damped_cutoff => 5e-3,
    :ΔΩ_by_Ω_low => 0,
    :ΔΩ_by_Ω_high => 0,
    :θ_cutoff => deg2rad(75),
    :equator_power_cutoff_frac => 0.3,
    :nnodesmax => 10,
    :filterfieldpowercutoff => 1e-4,
    :scalings => (; Wscaling = 1, Sscaling = 1),
    )
const DefaultFilter = F_EIGVAL | F_EIGEN | F_SPHARM | F_CHEBY | F_BC | F_SPATIAL

function filter_eigenvalues(λ::AbstractVector, v::AbstractMatrix,
    M::AbstractMatrix, m::Integer;
    operators,
    constraints = constraintmatrix(operators),
    filtercache = allocate_filter_caches(m; operators, constraints),
    filterflags = DefaultFilter,
    kw...)

    (; nparams) = operators.radial_params;
    kw = merge(DefaultFilterParams, kw)
    additional_params = (operators, constraints, filtercache, kw)

    inds_bool = filterfn.(λ, eachcol(v), m, (M,), (additional_params,), filterflags)
    filtinds = axes(λ, 1)[inds_bool]
    λ, v = λ[filtinds], v[:, filtinds]

    # re-apply scalings
    if get(kw, :scale_eigenvectors, false)
        scalings = merge((; Wscaling = 1, Sscaling = 1), kw[:scalings])
        V = @view v[1:nparams, :]
        W = @view v[nparams .+ (1:nparams), :]
        S = @view v[2nparams .+ (1:nparams), :]

        W ./= scalings.Wscaling
        S ./= scalings.Sscaling

        V .*= Rsun
        (; Wscaling, Sscaling) = operators.constants.scalings
        W .*= -im * Rsun^2 / Wscaling
        S ./= operators.constants.Ω0 * Rsun * Sscaling
    end

    λ, v
end

macro maybe_reduce_blas_threads(nt, ex)
    ex_esc = esc(ex)
    nt_esc = esc(nt)
    quote
        nblasthreads = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(max(1, round(Int, nblasthreads/$nt_esc)))
            $ex_esc
        finally
            BLAS.set_num_threads(nblasthreads)
        end
    end
end

diffrotmatrixfn!(rotation_profile) = (x...; kw...) -> differential_rotation_matrix!(x...; rotation_profile, kw..., )
diffrotspectrumfn!(rotation_profile) = (x...; kw...) -> differential_rotation_spectrum!(x...; rotation_profile, kw..., )

function filter_eigenvalues(λs::AbstractVector{<:AbstractVector},
    vs::AbstractVector{<:AbstractMatrix}, mr::AbstractVector;
    matrixfn! = uniform_rotation_matrix!,
    operators, constraints = constraintmatrix(operators),kw...)

    (; nr, nℓ, nparams) = operators.radial_params
    (; nvariables) = operators.constants
    Ms = [allocate_matrix(operators) for _ in 1:Threads.nthreads()]
    ℓs = minimum(mr):maximum(mr) + nℓ - 1
    Jtermsunirot = begin
        lobattochebyshevtransform =
            LobattoChebyshev(operators.transforms.TfGL_nr,
                operators.transforms.TiGL_nr,
                operators.transforms.normr)
        OffsetArray(
            map(ℓ -> greenfn_cheby(UniformRotGfn(), ℓ, operators, lobattochebyshevtransform), ℓs), ℓs)
    end

    λv = @maybe_reduce_blas_threads(Threads.nthreads(),
        Folds.map(zip(λs, vs, mr)) do (λm, vm, m)
            M = Ms[Threads.threadid()]
            matrixfn!(M, nr, nℓ, m; operators, Jtermsunirot)
            _M = _maybetrimM(M, nvariables, nparams)
            filter_eigenvalues(λm, vm, _M, m; operators, constraints, kw...)
        end
    )
    first.(λv), last.(λv)
end

function fmap(spectrumfn!, (nr, nℓ, m), (Ms, caches, temp_projectback_mats, eigencaches, λs, ws, operators, constraints, Jtermsunirot); kw...)
    threadid = Threads.threadid()
    M = Ms[threadid];
    cache = caches[threadid];
    temp_projectback = temp_projectback_mats[threadid];
    eigencache = eigencaches[threadid];
    λ = λs[threadid];
    w = ws[threadid];
    X = spectrumfn!(M, nr, nℓ, m; operators, constraints, cache,
            temp_projectback, Jtermsunirot, eigencache, λ, w, kw...);
    filter_eigenvalues(X..., m; operators, constraints, kw...)
end

function filter_eigenvalues(spectrumfn!, mr::AbstractVector;
    operators, constraints = constraintmatrix(operators), kw...)

    to = TimerOutput()

    @timeit to "alloc" begin
        (; nvariables) = operators.constants;
        (; nr, nℓ, nparams) = operators.radial_params;
        ℓs = minimum(mr):maximum(mr) + nℓ - 1
        nthreads = Threads.nthreads()
        @timeit to "M" Ms = [allocate_matrix(operators) for _ in 1:nthreads];
        @timeit to "caches" caches = [constrained_matmul_cache(constraints) for _ in 1:nthreads];
        @timeit to "projectback" temp_projectback_mats = [allocate_projectback_temp_matrices(size(constraints.ZC)) for _ in 1:nthreads];
        @timeit to "J" Jtermsunirot = begin
                lobattochebyshevtransform =
                    LobattoChebyshev(operators.transforms.TfGL_nr,
                        operators.transforms.TiGL_nr,
                        operators.transforms.normr)
                OffsetArray(
                    map(ℓ -> greenfn_cheby(UniformRotGfn(), ℓ, operators, lobattochebyshevtransform), ℓs), ℓs)
            end;

        Mc = caches[1].M_constrained;
        eigencaches = [allocate_eigen_cache(Mc) for _ in 1:nthreads];
        λs = [Vector{ComplexF64}(undef, size(Mc, 1)) for _ in 1:nthreads];
        @timeit to "w" ws = [Matrix{ComplexF64}(undef, size(Mc)) for _ in 1:nthreads];
    end

    @timeit to "spectrum" begin
        addl_params = (Ms, caches, temp_projectback_mats, eigencaches, λs, ws, operators, constraints, Jtermsunirot);

        nblasthreads = BLAS.get_num_threads()

        nthreads_trailing_elems = rem(length(mr), nthreads)

        if nthreads_trailing_elems > 0 && div(nblasthreads, nthreads_trailing_elems) > max(1, div(nblasthreads, nthreads))
            # in this case the extra elements may be run using a higher number of blas threads
            mr1 = @view mr[1:end-nthreads_trailing_elems]
            λv1 = @maybe_reduce_blas_threads(Threads.nthreads(),
                Folds.map(mr1) do m
                    fmap(spectrumfn!, (nr, nℓ, m), addl_params; kw...)
                end
            )
            λs, vs = first.(λv1), last.(λv1)

            mr2 = @view mr[end-nthreads_trailing_elems+1:end]

            λv2 = @maybe_reduce_blas_threads(nthreads_trailing_elems,
                Folds.map(mr2) do m
                    fmap(spectrumfn!, (nr, nℓ, m), addl_params; kw...)
                end
            )

            λs2, vs2 = first.(λv2), last.(λv2)
            append!(λs, λs2)
            append!(vs, vs2)
        else
            λv = @maybe_reduce_blas_threads(Threads.nthreads(),
                Folds.map(mr) do m
                    fmap(spectrumfn!, (nr, nℓ, m), addl_params; kw...)
                end
            )
            λs, vs = first.(λv), last.(λv)
        end
    end
    println(to)
    λs, vs
end
function filter_eigenvalues(filename::String; kw...)
    λs, vs, mr, nr, nℓ, kwold = load(filename, "lam", "vec", "mr", "nr", "nℓ", "kw");
    kw = merge(kwold, kw)
    operators = radial_operators(nr, nℓ)
    constraints = constraintmatrix(operators)
    filter_eigenvalues(λs, vs, mr; operators, constraints, kw...)
end

rossbyeigenfilename(nr, nℓ, tag = "ur", posttag = "") = "$(tag)_nr$(nr)_nl$(nℓ)$(posttag).jld2"
function save_eigenvalues(f, nr, nℓ, mr; operators = radial_operators(nr, nℓ), kw...)
    kw = merge(DefaultFilterParams, kw)
    lam, vec = filter_eigenvalues(f, mr; operators, kw...)
    isdiffrot = get(kw, :diffrot, false)
    filenametag = isdiffrot ? "dr" : "ur"
    fname = datadir(rossbyeigenfilename(nr, nℓ, filenametag))
    @info "saving to $fname"
    jldsave(fname; lam, vec, mr, nr, nℓ, kw)
end

function eigenfunction_cheby_ℓm_spectrum!(F, v, operators)
    (; radial_params) = operators
    (; nparams, nr, nℓ) = radial_params
    (; nvariables) = operators.constants

    F.V .= @view v[1:nparams]
    F.W .= @view v[nparams.+(1:nparams)]
    F.S .= nvariables == 3 ? @view(v[2nparams.+(1:nparams)]) : 0.0

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
