module RossbyWaveSpectrum

using MKL

using ApproxFun
using ApproxFun: DomainSets
using BandedMatrices
using BlockArrays
using BlockBandedMatrices
using Dierckx
using FastGaussQuadrature
using FastTransforms
using FillArrays
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
using OffsetArrays: Origin
using SimpleDelimitedFiles: readdlm
using SparseArrays
using StructArrays
using TimerOutputs
using UnPack
using ZChop

export datadir
export Filters

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

const StructMatrix{T} = StructArray{T,2}

function __init__()
    SCRATCH[] = get(ENV, "SCRATCH", homedir())
    DATADIR[] = get(ENV, "DATADIR", joinpath(SCRATCH[], "RossbyWaves"))
    if !ispath(RossbyWaveSpectrum.DATADIR[])
        mkdir(RossbyWaveSpectrum.DATADIR[])
    end
end

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

function operatormatrix(f::Fun, nr, rangespace)::Matrix{Float64}
    operatormatrix(ApproxFun.Multiplication(f), nr, rangespace)
end

function operatormatrix(A, nr, rangespace)::Matrix{Float64}
    B = A:ApproxFun.Chebyshev()
    C = ApproxFunBase.promoterangespace(B, rangespace)
    C[1:nr, 1:nr]
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
    vo = Origin(0)(v)
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
    @unpack costheta, w = gausslegendre_theta_grid(ntheta)
    associatedlegendretransform_matrices(nℓ, m, costheta, w)
end

for f in [:Vboundary, :Wboundary, :Sboundary]
    f! = Symbol("$(f)!")
    @eval $(f)(nconstraints, nchebyr, args...) = ($f!)(zeros(nconstraints, nchebyr), args...)
end

function Vboundary!(MVn, radial_params)
    @unpack r_in, r_out, Δr = radial_params;
    MVno = Origin(1,0)(MVn);
    # constraints on V, Robin
    for n in axes(MVno, 2)
        # inner boundary
        # impenetrable, stress-free
        MVno[1, n] = (-1)^n * (n^2 + Δr / r_in)
        # outer boundary
        # impenetrable, stress-free
        MVno[2, n] = n^2 - Δr / r_out
    end
    return MVn
end

function Wboundary!(MWn, args...)
    # constraints on W, Dirichlet
    MWno = Origin(1,0)(MWn);
    for n in axes(MWno, 2)
        # inner boundary
        # impenetrable, stress-free
        # equivalently, zero Dirichlet
        MWno[1, n] = (-1)^n
        # outer boundary
        # impenetrable, stress-free
        # equivalently, zero Dirichlet
        MWno[2, n] = 1

        MWno[3, n] = (-1)^n * n^2
        MWno[4, n] = n^2
    end
    return MWn
end

function Sboundary!(MSn, args...)
    # constraints on W, Dirichlet
    MSno = Origin(1,0)(MSn);
    # constraints on S
    # zero Neumann
    for n in axes(MSno, 2)
        # inner boundary
        MSno[1, n] = (-1)^n*n^2
        # outer boundary
        MSno[2, n] = n^2
    end

    return MSn
end

function constraintmatrix(operators; W_basis = 1, S_basis = 1)
    @unpack radial_params = operators;
    @unpack nr, nℓ = radial_params;
    @unpack nvariables = operators.constants;

    nradconstraintsVS = 2;
    nradconstraintsW = 4;

    MVn = Vboundary(nradconstraintsVS, nr, radial_params)
    ZMVn = nullspace(MVn)

    MWn = Wboundary(nradconstraintsW, nr)
    ZMWn = W_basis == 1 ? nullspace(MWn) : normalizecols!(dirichlet_neumann_chebyshev_matrix(nr));

    MSn = Sboundary(nradconstraintsVS, nr)
    ZMSn = S_basis == 1 ? nullspace(MSn) : normalizecols!(neumann_chebyshev_matrix(nr));

    rowsB_VS = Fill(nradconstraintsVS, nℓ)
    rowsB_W = Fill(nradconstraintsW, nℓ)
    colsB = Fill(nr, nℓ)
    rows = hvcatrows(nvariables)
    B_blocks = if nvariables == 3
        [blockdiagzero(rowsB_VS, colsB), blockdiagzero(rowsB_W, colsB), blockdiagzero(rowsB_VS, colsB)]
    else
        [blockdiagzero(rowsB_VS, colsB), blockdiagzero(rowsB_W, colsB)]
    end

    BC_block = mortar(Diagonal(B_blocks))
    # BC_block = allocate_block_matrix(nvariables, 0, rowsB, colsB)

    rowsZ = Fill(nr, nℓ)
    colsZ_VS = Fill(nr - nradconstraintsVS, nℓ)
    colsZ_W = Fill(nr - nradconstraintsW, nℓ)
    Z_blocks = if nvariables == 3
        [blockdiagzero(rowsZ, colsZ_VS), blockdiagzero(rowsZ, colsZ_W), blockdiagzero(rowsZ, colsZ_VS)]
    else
        [blockdiagzero(rowsZ, colsZ_VS), blockdiagzero(rowsZ, colsZ_W)]
    end
    ZC_block = mortar(Diagonal(Z_blocks))
    # ZC_block = allocate_block_matrix(nvariables, 0, rowsZ, colsZ)

    fieldmatrices = nvariables == 3 ? [MVn, MWn, MSn] : [MVn, MWn]
    for (Mind, M) in enumerate(fieldmatrices)
        BCi = BC_block[Block(Mind, Mind)]
        for ℓind = 1:nℓ
            BCi[Block(ℓind, ℓind)] = M
        end
    end
    nullspacematrices = nvariables == 3 ? [ZMVn, ZMWn, ZMSn] : [ZMVn, ZMWn];
    for (Zind, Z) in enumerate(nullspacematrices)
        ZCi = ZC_block[Block(Zind, Zind)]
        for ℓind = 1:nℓ
            ZCi[Block(ℓind, ℓind)] = Z
        end
    end

    BC = computesparse(BC_block)
    ZC = computesparse(ZC_block)

    (; BC, ZC, nvariables, ZC_block)
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
    δrad = -1e-1
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
    Mo = Origin(0)(M)
    for n in 0:nℓ-1, k in n-1:-2:0
        Mo[n, k] = -√((2n + 1) * (2k + 1))
    end
    return Matrix(M')
end

# defined for normalized Legendre polynomials
function cottheta_dtheta_operator(nℓ)
    M = zeros(nℓ, nℓ)
    Mo = Origin(0)(M)
    for n in 0:nℓ-1
        Mo[n, n] = -n
        for k in n-2:-2:0
            Mo[n, k] = -√((2n + 1) * (2k + 1))
        end
    end
    Matrix(M')
end

function normalizecols!(M)
    foreach(normalize!, eachcol(M))
    return M
end
function chebynormalize!(M)
    for col in eachcol(M)
        N2 = col[1]^2 * pi
        for ind in 2:lastindex(col)
            N2 += col[ind]^2 * pi/2
        end
        N = sqrt(N2)
        col ./= N
    end
    return M
end

# converts from Heinrichs basis to Chebyshev, so Cn = Anm Bm where Cn are the Chebyshev
# coefficients, and Bm are the Heinrichs ones
αheinrichs(n) = n < 0 ? 0 : n == 0 ? 2 : 1
function dirichlet_chebyshev_matrix(n)
    M = BandedMatrix(Zeros(n, n-2), (2, 2))
    Mo = Origin(0)(M)
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

function dirichlet_neumann_chebyshev_matrix(n)
    M = BandedMatrix(Zeros(n, n-4), (4, 4))
    Mo = Origin(0)(M)
    pT4coeffs = OffsetArray([1/16, 0, -1/4, 0, 3/8, 0, -1/4, 0, 1/16], -4:4)
    for n in 0:3
        for (ind, pc) in pairs(pT4coeffs)
            Mo[abs(n + ind), n] += pc
        end
    end
    for n in 4:lastindex(Mo,2)
        Mo[n .+ (-4:4), n] = parent(pT4coeffs)
    end
    return M
end

n2coeff_neumann(n) = -(n/(n+2))^2
function neumann_chebyshev_matrix(n)
    M = BandedMatrix(Zeros(n, n-2), (2, 0))
    Mo = Origin(0)(M)
    for n in axes(Mo, 2)
        Mo[n, n] = 1
        Mo[n+2, n] = n2coeff_neumann(n)
    end
    M
end

function r2neumann_chebyshev_matrix(ncheby, radial_params)
    @unpack Δr, r_mid = radial_params
    M = BandedMatrix(Zeros(ncheby, ncheby-2), (4, 2))
    Mo = Origin(0)(M)

    T0 = 1/2*(Δr/2)^2 + r_mid^2
    T1 = r_mid * (Δr/2)
    T2 = 1/4*(Δr/2)^2
    C = OffsetArray([T2, T1, T0, T1, T2] ./ Rsun^2, -2:2)

    # B1
    Mo[0, 0] = C[0]
    @views @. Mo[1:2, 0] = 2C[1:2]
    # B2
    @views @. Mo[0:3, 1] = C[-1:2]
    Mo[1, 1] += C[2]
    @views @. Mo[1:5, 1] -= C[-2:2]/3

    # B3 onwards
    for n in 2:last(axes(Mo,2))
        colv = @view Mo[n-2:2:n+2, n]
        colv[1] = 1/(n-1)
        colv[2] = -(1/(n-1) + 1/(n+1))
        colv[3] = 1/(n+1)
    end
    return M
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

    rad_terms = Dict{Symbol, Vector{Float64}}()
    @pack! rad_terms = r_modelS, ρ_modelS, logρ_modelS, ddrlogρ_modelS, T_modelS, g_modelS

    splines = Dict{Symbol, typeof(sρ)}()
    @pack! splines = sρ, sT, sg, slogρ, sηρ, sηρ_by_r, ddrsηρ_by_r, ddrsηρ_by_r2, ddrsηρ, d2dr2sηρ, d3dr3sηρ, sηT
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
function replaceemptywitheps(f::Fun, eps = 1e-100)
    T = eltype(coefficients(f))
    iszerofun(f) ? typeof(f)(space(f), T[eps]) : f
end

function checkncoeff(v, vname, nr)
    if ncoefficients(v) > 2/3*nr
        @debug "number of coefficients in $vname is $(ncoefficients(v)), but nr = $nr"
    end
end
macro checkncoeff(v, nr)
    :(checkncoeff($(esc(v)), $(String(v)), $(esc(nr))))
end

const DefaultScalings = (; Wscaling = 1e1, Sscaling = 1e6, Weqglobalscaling = 1e-3, Seqglobalscaling = 1.0)
function radial_operators(nr, nℓ; r_in_frac = 0.7, r_out_frac = 0.985, _stratified = true, nvariables = 3, ν = 1e10,
    scalings = DefaultScalings)
    scalings = merge(DefaultScalings, scalings)
    _radial_operators(nr, nℓ, r_in_frac, r_out_frac, _stratified, nvariables, ν, Tuple(scalings))
end
function _radial_operators(nr, nℓ, r_in_frac, r_out_frac, _stratified, nvariables, ν,
        (Wscaling, Sscaling, Weqglobalscaling, Seqglobalscaling))
    r_in = r_in_frac * Rsun;
    r_out = r_out_frac * Rsun;
    radial_params = parameters(nr, nℓ; r_in, r_out);
    @unpack Δr, nchebyr, r_mid = radial_params;
    r, Tcrfwd, Tcrinv = chebyshev_forward_inverse(nr, r_in, r_out);
    Tcrinvc = complex.(Tcrinv)
    r_chebyshev = (r .- r_mid) ./ (Δr / 2);

    pseudospectralop_radial = SpectralOperatorForm(Tcrfwd, Tcrinv);

    r_cheby = Fun(ApproxFun.Chebyshev(), [r_mid, Δr / 2]);
    r2_cheby = r_cheby * r_cheby;

    @unpack splines = read_solar_model(; r_in, r_out, _stratified);

    @unpack sρ, sg, sηρ, ddrsηρ, d2dr2sηρ,
        sηρ_by_r, ddrsηρ_by_r, ddrsηρ_by_r2, d3dr3sηρ, sT, sηT = splines

    ddr = ApproxFun.Derivative() * (2 / Δr)
    rddr = (r_cheby * ddr)::Tmul
    d2dr2 = (ddr * ddr)::Tmul
    d3dr3 = (ddr * d2dr2)::Tmul
    d4dr4 = (d2dr2 * d2dr2)::Tmul
    r2d2dr2 = (r2_cheby * d2dr2)::Tmul

    # density stratification
    ηρ = replaceemptywitheps(ApproxFun.chop(Fun(sηρ ∘ r_cheby, ApproxFun.Chebyshev()), 1e-3))::TFun
    @checkncoeff ηρ nr

    ηT = replaceemptywitheps(ApproxFun.chop(Fun(sηT ∘ r_cheby, ApproxFun.Chebyshev()), 1e-2))::TFun
    @checkncoeff ηT nr

    ddr_lnρT = (ηρ + ηT)::TFun

    DDr = (ddr + ηρ)::Tplus
    rDDr = (r_cheby * DDr)::Tmul

    onebyr = (1 / r_cheby)::typeof(r_cheby)
    onebyr2 = (onebyr*onebyr)::typeof(r_cheby)
    onebyr3 = (onebyr2*onebyr)::typeof(r_cheby)
    onebyr4 = (onebyr2*onebyr2)::typeof(r_cheby)
    DDr_minus_2byr = (DDr - 2onebyr)::Tplus
    ddr_plus_2byr = (ddr + 2onebyr)::Tplus

    # ηρ_by_r = onebyr * ηρ
    ηρ_by_r = chop(Fun(sηρ_by_r ∘ r_cheby, Chebyshev()), 1e-2)::TFun;
    @checkncoeff ηρ_by_r nr

    ηρ_by_r2 = (ηρ * onebyr2)::TFun
    @checkncoeff ηρ_by_r2 nr

    ηρ_by_r3 = (ηρ_by_r2 * onebyr)::TFun
    @checkncoeff ηρ_by_r3 nr

    ddr_ηρ = chop(Fun(ddrsηρ ∘ r_cheby, Chebyshev()), 1e-2)::TFun
    @checkncoeff ddr_ηρ nr

    ddr_ηρbyr = chop(Fun(ddrsηρ_by_r ∘ r_cheby, Chebyshev()), 1e-2)::TFun
    @checkncoeff ddr_ηρbyr nr

    # ddr_ηρbyr = ddr * ηρ_by_r
    d2dr2_ηρ = chop(Fun(d2dr2sηρ ∘ r_cheby, Chebyshev()), 1e-2)::TFun
    @checkncoeff d2dr2_ηρ nr

    d3dr3_ηρ = chop(Fun(d3dr3sηρ ∘ r_cheby, Chebyshev()), 1e-2)::TFun
    @checkncoeff d3dr3_ηρ nr

    ddr_ηρbyr2 = chop(Fun(ddrsηρ_by_r2 ∘ r_cheby, Chebyshev()), 5e-3)::TFun
    @checkncoeff ddr_ηρbyr2 nr

    # ddr_ηρbyr2 = ddr * ηρ_by_r2
    ηρ2_by_r2 = ApproxFun.chop(ηρ_by_r2 * ηρ, 1e-3)::TFun
    @checkncoeff ηρ2_by_r2 nr

    ddrηρ_by_r = (ddr_ηρ * onebyr)::TFun
    d2dr2ηρ_by_r = (d2dr2_ηρ * onebyr)::TFun

    ddrDDr = (d2dr2 + ηρ * ddr + ddr_ηρ)::Tplus
    d2dr2DDr = (d3dr3 + ηρ * d2dr2 + ddr_ηρ * ddr + d2dr2_ηρ)::Tplus

    g = Fun(sg ∘ r_cheby, Chebyshev())::TFun

    Ω0 = RossbyWaveSpectrum.equatorial_rotation_angular_velocity(r_out_frac)

    # viscosity
    ν /= Ω0*Rsun^2
    κ = ν

    γ = 1.64
    cp = 1.7e8
    δ_superadiabatic = superadiabaticity.(r)
    ddr_S0_by_cp = ApproxFun.chop(Fun(x -> γ * superadiabaticity(r_cheby(x)) * ηρ(x) / cp, Chebyshev()), 1e-3)::TFun
    @checkncoeff ddr_S0_by_cp nr

    ddr_S0_by_cp_by_r2 = chop(onebyr2 * ddr_S0_by_cp, 1e-4)::TFun

    Ir = I(nchebyr)
    Iℓ = I(nℓ)

    # matrix representations

    matCU2 = x -> operatormatrix(x, nr, Ultraspherical(2))
    matCU4 = x -> operatormatrix(x, nr, Ultraspherical(4))

    # matrix forms of operators
    onebyrMCU2 = matCU2(onebyr)
    onebyrMCU4 = matCU4(onebyr)
    onebyr2MCU2 = matCU2(onebyr2)
    onebyr2MCU4 = matCU4(onebyr2);

    ddrMCU4 = matCU4(ddr)
    rMCU4 = matCU4(r_cheby)
    ddr_minus_2byrMCU4 = @. ddrMCU4 - 2*onebyrMCU4
    d2dr2MCU2 = matCU2(d2dr2)
    d2dr2MCU4 = matCU4(d2dr2)
    d3dr3MCU4 = matCU4(d3dr3)
    d4dr4MCU4 = matCU4(d4dr4)
    DDrMCU2 = matCU2(DDr)
    DDr_minus_2byrMCU2 = matCU2(DDr_minus_2byr)
    ddrDDrMCU4 = matCU4(ddrDDr);
    gMCU4 = matCU4(g)

    # uniform rotation terms
    onebyr2_IplusrηρMCU4 = matCU4((1 + ηρ * r_cheby) * onebyr2);
    ∇r2_plus_ddr_lnρT_ddr = (d2dr2 + 2onebyr * ddr + ddr_lnρT * ddr)::Tplus;
    κ_∇r2_plus_ddr_lnρT_ddrMCU2 = κ * matCU2(∇r2_plus_ddr_lnρT_ddr);
    κ_by_r2MCU2 = κ .* matCU2(onebyr2);
    ddr_S0_by_cp_by_r2MCU2 = matCU2(ddr_S0_by_cp_by_r2);

    # terms for viscosity
    ddr_minus_2byr = (ddr - 2onebyr)::Tplus;
    ηρ_ddr_minus_2byrMCU2 = matCU2((ηρ * ddr_minus_2byr)::Tmul);
    onebyr2_d2dr2MCU4 = matCU4(onebyr2*d2dr2);
    onebyr3_ddrMCU4 = matCU4(onebyr3*ddr);
    onebyr4_chebyMCU4 = matCU4(onebyr2*onebyr2);

    ηρ_by_rMCU4 = matCU4(ηρ_by_r)
    ηρ2_by_r2MCU4 = matCU4(ηρ2_by_r2)
    ηρ_by_r3MCU4 = matCU4(ηρ * onebyr3)

    IU2 = matCU2(I);

    scalings = Dict{Symbol, Float64}()
    @pack! scalings = Sscaling, Wscaling, Weqglobalscaling, Seqglobalscaling

    constants = (; κ, ν, nvariables, Ω0)
    identities = (; Ir, Iℓ, IU2)

    coordinates = Dict{Symbol, Vector{Float64}}()
    @pack! coordinates = r, r_chebyshev

    transforms = (; Tcrfwd, Tcrinv, Tcrinvc, pseudospectralop_radial)

    rad_terms = Dict{Symbol, TFun}();
    @pack! rad_terms = onebyr, ηρ, ηT,
        onebyr2, onebyr3, onebyr4,
        ddr_lnρT, ddr_S0_by_cp, g, r_cheby, r2_cheby,
        ηρ_by_r, ηρ_by_r2, ηρ2_by_r2, ddr_ηρbyr, ddr_ηρbyr2, ηρ_by_r3,
        ddr_ηρ, d2dr2_ηρ, d3dr3_ηρ, ddr_S0_by_cp_by_r2,
        ddrηρ_by_r, d2dr2ηρ_by_r

    diff_operators = (; DDr, DDr_minus_2byr, rDDr, rddr, ddrDDr, d2dr2DDr,
        ddr, d2dr2, d3dr3, d4dr4, r2d2dr2, ddr_plus_2byr)

    operator_matrices = Dict{Symbol, Matrix{Float64}}();
    @pack! operator_matrices = DDrMCU2,
        ddrMCU4, d2dr2MCU2, d2dr2MCU4,
        ddrDDrMCU4,
        ddr_minus_2byrMCU4, DDr_minus_2byrMCU2,
        d3dr3MCU4, d4dr4MCU4,
        # uniform rotation terms
        κ_∇r2_plus_ddr_lnρT_ddrMCU2,
        # viscosity terms
        ηρ_ddr_minus_2byrMCU2, onebyr2_d2dr2MCU4, onebyr3_ddrMCU4,
        onebyrMCU2, onebyrMCU4, onebyr2MCU2,
        onebyr2MCU4, ddr_S0_by_cp_by_r2MCU2, κ_by_r2MCU2,
        gMCU4, ηρ_by_rMCU4, ηρ2_by_r2MCU4, ηρ_by_r3MCU4,
        onebyr2_IplusrηρMCU4, onebyr4_chebyMCU4,
        rMCU4

    (;
        constants, rad_terms,
        scalings,
        splines,
        diff_operators,
        transforms, coordinates,
        radial_params, identities,
        operator_matrices,
        matCU2, matCU4,
        _stratified,
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

matrix_block(M::AbstractBlockMatrix, rowind, colind, nblocks = 3) = M[Block(rowind, colind)]
function matrix_block(M::StructMatrix{<:Complex}, rowind, colind, nblocks = 3)
    Mr = matrix_block(M.re, rowind, colind, nblocks)
    Mi = matrix_block(M.im, rowind, colind, nblocks)
    StructArray{eltype(M)}((Mr, Mi))
end
function matrix_block(M::AbstractMatrix, rowind, colind, nblocks = 3)
    nparams = div(size(M, 1), nblocks)
    inds1 = (rowind - 1) * nparams .+ (1:nparams)
    inds2 = (colind - 1) * nparams .+ (1:nparams)
    inds = CartesianIndices((inds1, inds2))
    @view M[inds]
end

matrix_block_maximum(f, M::AbstractMatrix, nblocks = 3) = [maximum(f, matrix_block(M, i, j, nblocks)) for i in 1:nblocks, j in 1:nblocks]
function matrix_block_maximum(f, M::BlockBandedMatrix, nblocks = 3)
    maximum(f, (maximum(f, b) for b in blocks(M)))
end
function matrix_block_maximum(f, M::BlockMatrix, nblocks = 3)
    [matrix_block_maximum(f, Mb, nblocks) for Mb in blocks(M)]
end
function matrix_block_maximum(M::StructMatrix{<:Complex}, nblocks = 3)
    R = matrix_block_maximum(abs, M.re, nblocks)
    I = matrix_block_maximum(abs, M.im, nblocks)
    [R I]
end
function matrix_block_maximum(M::AbstractMatrix{<:Complex}, nblocks = 3)
    R = matrix_block_maximum(abs∘real, M, nblocks)
    I = matrix_block_maximum(abs∘imag, M, nblocks)
    [R I]
end
function matrix_block_maximum(M::AbstractMatrix{<:Real}, nblocks = 3)
    matrix_block_maximum(abs∘real, M, nblocks)
end

function computesparse(M::StructMatrix{<:Complex})
    SR = computesparse(M.re)
    SI = computesparse(M.im)
    StructArray{eltype(M)}((SR, SI))
end

hvcatrows(n::Int) = hvcatrows((n, n))
hvcatrows(sz::NTuple{2,Integer}) = ntuple(_ -> sz[2], sz[1])
hvcatrows(M::BlockMatrix) = hvcatrows(blocksize(M))

function computesparse(M::BlockMatrix)
    rows = hvcatrows(M)
    hvcat(rows, [sparse(M[j, i]) for (i,j) in Iterators.product(blockaxes(M)...)]...)
end

computesparse(M::AbstractMatrix) = sparse(M)
computesparse(S::SparseMatrixCSC) = S

blockbandedzero(rows, cols, (l, u)) = BlockBandedMatrix(Zeros(sum(rows), sum(cols)), rows, cols, (l,u))
blockdiagzero(rows, cols) = blockbandedzero(rows, cols, (0, 0))

function allocate_block_matrix(nvariables, bandwidth, rows, cols = rows)
    l,u = bandwidth, bandwidth # block bandwidths
    mortar(reshape(
            [blockbandedzero(rows, cols, (l,u)) for _ in 1:nvariables^2],
                nvariables, nvariables))
end

function allocate_operator_matrix(operators, bandwidth = 2)
    @unpack nr, nℓ = operators.radial_params
    @unpack nvariables = operators.constants
    rows = Fill(nr, nℓ) # block sizes
    R = allocate_block_matrix(nvariables, bandwidth, rows)
    I = allocate_block_matrix(nvariables, 0, rows)
    StructArray{ComplexF64}((R, I))
end

function allocate_mass_matrix(operators)
    @unpack nr, nℓ = operators.radial_params
    @unpack nvariables = operators.constants
    rows = Fill(nr, nℓ) # block sizes
    allocate_block_matrix(nvariables, 0, rows)
end

ℓrange(m, nℓ, symmetric) = range(m + !symmetric, length = nℓ, step = 2)

function mass_matrix(m; operators, kw...)
    B = allocate_mass_matrix(operators)
    mass_matrix!(B, m; operators, kw...)
    return B
end
function mass_matrix!(B, m; operators, V_symmetric = true, kw...)
    @unpack nr, nℓ = operators.radial_params
    @unpack IU2 = operators.identities;
    @unpack nvariables = operators.constants
    @unpack ddrDDrMCU4, onebyr2MCU4 = operators.operator_matrices;
    @unpack Weqglobalscaling = operators.scalings

    B .= 0

    VV = matrix_block(B, 1, 1)
    WW = matrix_block(B, 2, 2)
    if nvariables == 3
        SS = matrix_block(B, 3, 3)
    end

    # V terms
    V_ℓs = ℓrange(m, nℓ, V_symmetric)

    # W, S terms
    W_ℓs = ℓrange(m, nℓ, !V_symmetric)

    @views for ℓind in 1:nℓ
        VV[Block(ℓind, ℓind)] .= IU2
        if nvariables == 3
            SS[Block(ℓind, ℓind)] .= IU2
        end
    end

    ddrDDr_minus_ℓℓp1_by_r2MCU4 = similar(ddrDDrMCU4);
    @views for (ℓind, ℓ) in enumerate(W_ℓs)
        ℓℓp1 = ℓ * (ℓ+1)

        @. ddrDDr_minus_ℓℓp1_by_r2MCU4 = ddrDDrMCU4 - ℓℓp1 * onebyr2MCU4
        WW[Block(ℓind, ℓind)] .= (Weqglobalscaling * Rsun^2) .* ddrDDr_minus_ℓℓp1_by_r2MCU4
    end

    return B
end

function uniform_rotation_matrix(m; operators, kw...)
    A = allocate_operator_matrix(operators)
    uniform_rotation_matrix!(A, m; operators, kw...)
    return A
end

function uniform_rotation_matrix!(A::StructMatrix{<:Complex}, m; operators, V_symmetric = true, kw...)
    @unpack nvariables, Ω0 = operators.constants;
    @unpack nr, nℓ = operators.radial_params
    @unpack Sscaling, Wscaling, Weqglobalscaling, Seqglobalscaling = operators.scalings
    @unpack IU2 = operators.identities;

    @unpack ddrMCU4, DDrMCU2, DDr_minus_2byrMCU2, ddrDDrMCU4, κ_∇r2_plus_ddr_lnρT_ddrMCU2,
        onebyrMCU2, onebyrMCU4, onebyr2MCU4, ηρ_by_rMCU4, ddr_S0_by_cp_by_r2MCU2,
        κ_by_r2MCU2, gMCU4, ddr_minus_2byrMCU4 = operators.operator_matrices;

    A.re .= 0
    A.im .= 0

    VV = matrix_block(A.re, 1, 1)
    VW = matrix_block(A.re, 1, 2)
    WV = matrix_block(A.re, 2, 1)
    WW = matrix_block(A.re, 2, 2)
    # the following are only valid if S is included
    if nvariables == 3
        WS = matrix_block(A.re, 2, 3)
        SW = matrix_block(A.re, 3, 2)
        SS = matrix_block(A.im, 3, 3)
    end

    nCS = 2nℓ+1
    ℓs = range(m, length = nCS)
    cosθ = OffsetArray(costheta_operator(nCS, m), ℓs, ℓs);
    sinθdθ = OffsetArray(sintheta_dtheta_operator(nCS, m), ℓs, ℓs);

    ddrDDr_minus_ℓℓp1_by_r2MCU4 = similar(ddrDDrMCU4);
    T = zeros(nr, nr)

    # V terms
    V_ℓs = ℓrange(m, nℓ, V_symmetric)
    # W, S terms
    W_ℓs = ℓrange(m, nℓ, !V_symmetric)

    @views for (ℓind, ℓ) in enumerate(V_ℓs)
        ℓℓp1 = ℓ * (ℓ + 1)

        twom_by_ℓℓp1 = 2m/ℓℓp1

        VV[Block(ℓind, ℓind)] .= twom_by_ℓℓp1 .* IU2

        for ℓ′ in intersect(W_ℓs, ℓ-1:2:ℓ+1)
            ℓ′ind = findfirst(isequal(ℓ′), W_ℓs)
            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)

            @. T = (-2/ℓℓp1) * (ℓ′ℓ′p1 * DDr_minus_2byrMCU2 * cosθ[ℓ, ℓ′] +
                    (DDrMCU2 - ℓ′ℓ′p1 * onebyrMCU2) * sinθdθ[ℓ, ℓ′]) * Rsun / Wscaling
            VW[Block(ℓind, ℓ′ind)] .= T
        end
    end

    @views for (ℓind, ℓ) in enumerate(W_ℓs)
        ℓℓp1 = ℓ * (ℓ + 1)
        twom_by_ℓℓp1 = 2m/ℓℓp1

        @. ddrDDr_minus_ℓℓp1_by_r2MCU4 = ddrDDrMCU4 - ℓℓp1 * onebyr2MCU4

        WW[Block(ℓind, ℓind)] .= (Weqglobalscaling * twom_by_ℓℓp1 * Rsun^2) .*
                                (ddrDDr_minus_ℓℓp1_by_r2MCU4 .- ℓℓp1 .* ηρ_by_rMCU4)

        if nvariables == 3
            @. T = - gMCU4 / (Ω0^2 * Rsun)  * Wscaling/Sscaling
            WS[Block(ℓind, ℓind)] .= Weqglobalscaling .* T
            @. T = ℓℓp1 * ddr_S0_by_cp_by_r2MCU2 * (Rsun^3 * Sscaling/Wscaling)
            SW[Block(ℓind, ℓind)] .= Seqglobalscaling .* T
            @. T = -(κ_∇r2_plus_ddr_lnρT_ddrMCU2 - ℓℓp1 * κ_by_r2MCU2) * Rsun^2
            SS[Block(ℓind, ℓind)] .= Seqglobalscaling .* T
        end

        for ℓ′ in intersect(V_ℓs, ℓ-1:2:ℓ+1)
            ℓ′ind = findfirst(isequal(ℓ′), V_ℓs)
            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)

            @. T = (-2/ℓℓp1) * (ℓ′ℓ′p1 * ddr_minus_2byrMCU4 * cosθ[ℓ, ℓ′] +
                    (ddrMCU4 - ℓ′ℓ′p1 * onebyrMCU4) * sinθdθ[ℓ, ℓ′]) * Rsun * Wscaling

            WV[Block(ℓind, ℓ′ind)] .= Weqglobalscaling .* T
        end
    end

    viscosity_terms!(A, m; operators, V_symmetric, kw...)

    return A
end

function viscosity_terms!(A::StructMatrix{<:Complex}, m; operators, V_symmetric = true, kw...)
    @unpack nr, nℓ = operators.radial_params;

    @unpack ddrMCU4, d2dr2MCU2, d3dr3MCU4, onebyr2MCU2,
        ηρ_ddr_minus_2byrMCU2, onebyr2_d2dr2MCU4,
        onebyr3_ddrMCU4, onebyr4_chebyMCU4, d4dr4MCU4, ηρ2_by_r2MCU4,
        ηρ_by_r3MCU4 = operators.operator_matrices;

    @unpack ddr, d2dr2, d3dr3, DDr, ddrDDr, d2dr2DDr = operators.diff_operators;

    @unpack ddr_ηρbyr, ηρ, ddr_ηρ, d2dr2_ηρ, d3dr3_ηρ, ddrηρ_by_r, d2dr2ηρ_by_r, ηρ_by_r,
        ηρ_by_r2, ddr_ηρbyr2, onebyr2, onebyr = operators.rad_terms;

    @unpack ν, nvariables = operators.constants;
    @unpack matCU4, matCU2 = operators;
    @unpack Weqglobalscaling = operators.scalings;

    VVim = matrix_block(A.im, 1, 1)
    WWim = matrix_block(A.im, 2, 2)

    # caches for the WW term
    T1_1 = zeros(nr, nr);
    T1_2 = zeros(nr, nr);
    T3_1 = zeros(nr, nr);
    T3_2 = zeros(nr, nr);
    T4 = zeros(nr, nr);
    WWop = zeros(nr, nr);

    Mcache1 = zeros(nr, nr)
    Mcache2 = zeros(nr, nr)
    Mcache3 = zeros(nr, nr)

    ℓs = range(m, length = nℓ);

    # T1_1 terms
    d3dr3ηρMCU4 = matCU4(d3dr3_ηρ);
    ηρ_d3dr3MCU4 = matCU4(ηρ * d3dr3);
    d2dr2ηρ_by_rMCU4 = matCU4(d2dr2ηρ_by_r);
    threeddrηρ_min_4ηρbyr_d2dr2MCU4 = matCU4((3*ddr_ηρ - 4*ηρ_by_r)*d2dr2);
    threed2dr2ηρ_min_8ddrηρ_by_r_plus_ηρ_by_r2_ddrMCU4 = matCU4((3*d2dr2_ηρ - 8*ddrηρ_by_r + 8*ηρ_by_r2)*ddr);
    ddrηρ_by_r2MCU4 = matCU4(ddr_ηρ * onebyr2);
    ηρ_by_r2_ddrMCU4 = matCU4(ηρ_by_r2 * ddr);

    # T1_2 terms
    ηρ_by_r_d2dr2MCU4 = matCU4(ηρ_by_r * d2dr2);
    ddrηρ_by_r_ddr_min_ηρ_by_r2_ddrMCU4 = matCU4(ddrηρ_by_r * ddr) .- matCU4(ηρ_by_r2 * ddr);
    d2dr2ηρ_by_r_min_2ddrηρ_by_r2MCU4 = matCU4(d2dr2_ηρ * onebyr) .- 2 .* matCU4(ddr_ηρ * onebyr2);

    # T3_1 terms
    ddrηρbyr2MCU4 = matCU4(ddr_ηρbyr2);
    ddrηρ_min_2ηρbyr_ddrDDrMCU4 = matCU4((ddr_ηρ - 2ηρ_by_r)*ddrDDr);
    ηρd2dr2DDrMCU4 = matCU4(ηρ * d2dr2DDr);
    ddr_ηρbyr_DDrMCU4 = matCU4(ddr_ηρbyr * DDr);

    # V terms
    V_ℓs = ℓrange(m, nℓ, V_symmetric)

    @views for (ℓind, ℓ) in enumerate(V_ℓs)
        ℓℓp1 = ℓ * (ℓ + 1)
        @. T1_1 = -ν * (d2dr2MCU2 - ℓℓp1 * onebyr2MCU2 + ηρ_ddr_minus_2byrMCU2) * Rsun^2

        VVim[Block(ℓind, ℓind)] .= T1_1
    end

    # W, S terms
    W_ℓs = ℓrange(m, nℓ, !V_symmetric)

    @views for (ℓind, ℓ) in enumerate(W_ℓs)
        ℓℓp1 = ℓ * (ℓ + 1)
        neg2by3_ℓℓp1 = -2ℓℓp1 / 3

        ℓpre = (ℓ-2)*ℓ*(ℓ+1)*(ℓ+3)
        @. T1_1 = ((d3dr3ηρMCU4 -4*d2dr2ηρ_by_rMCU4 + 8*ddrηρ_by_r2MCU4 - 8*ηρ_by_r3MCU4)
                    + threeddrηρ_min_4ηρbyr_d2dr2MCU4
                    + threed2dr2ηρ_min_8ddrηρ_by_r_plus_ηρ_by_r2_ddrMCU4
                    + ηρ_d3dr3MCU4
                    - (ℓℓp1 - 2)*(ηρ_by_r2_ddrMCU4 + ddrηρ_by_r2MCU4 - 4*ηρ_by_r3MCU4)
                    )
        @. T1_2 = (d4dr4MCU4 + ℓpre * onebyr4_chebyMCU4 - 2ℓℓp1*onebyr2_d2dr2MCU4 + 4ℓℓp1*onebyr3_ddrMCU4
                + 4(ηρ_by_r_d2dr2MCU4 + 2 * ddrηρ_by_r_ddr_min_ηρ_by_r2_ddrMCU4 + d2dr2ηρ_by_r_min_2ddrηρ_by_r2MCU4)
                -4(ℓℓp1 -2)*ηρ_by_r3MCU4
                )

        @. T3_1 = (ddrηρ_min_2ηρbyr_ddrDDrMCU4 + ηρd2dr2DDrMCU4 - 2*ddr_ηρbyr_DDrMCU4
            + ℓℓp1 * (ddrηρbyr2MCU4 + ηρ_by_r2_ddrMCU4))
        @. T3_2 = 2ℓℓp1*(ηρ_by_r2_ddrMCU4 -2*ηρ_by_r3MCU4)

        @. T4 = neg2by3_ℓℓp1 * ηρ2_by_r2MCU4

        @. WWop = -ν * (T1_1 + T1_2 + T3_1 + T3_2 + T4) * Rsun^4

        WWim[Block(ℓind, ℓind)] .= Weqglobalscaling .* WWop
    end

    return A
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
    @unpack r = operators.coordinates;
    @unpack r_out = operators.radial_params;

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

function constant_differential_rotation_terms!(M::StructMatrix{<:Complex}, m;
        operators, ΔΩ_frac = 0.02, V_symmetric = true, kw...)

    @unpack nr, nℓ = operators.radial_params;
    @unpack nvariables = operators.constants
    @unpack IU2 = operators.identities;
    @unpack Wscaling, Weqglobalscaling, Seqglobalscaling = operators.scalings

    @unpack ddrMCU4, DDrMCU2, onebyrMCU2, onebyrMCU4,
            DDr_minus_2byrMCU2, ηρ_by_rMCU4, ddrDDrMCU4,
            onebyr2MCU4 = operators.operator_matrices;

    VV = matrix_block(M.re, 1, 1, nvariables)
    VW = matrix_block(M.re, 1, 2, nvariables)
    WV = matrix_block(M.re, 2, 1, nvariables)
    WW = matrix_block(M.re, 2, 2, nvariables)
    if nvariables == 3
        SS = matrix_block(M.re, 3, 3, nvariables)
    end

    nCS = 2nℓ+1
    ℓs = range(m, length = nCS)

    cosθo = OffsetArray(costheta_operator(nCS, m), ℓs, ℓs)
    sinθdθo = OffsetArray(sintheta_dtheta_operator(nCS, m), ℓs, ℓs)
    laplacian_sinθdθo = OffsetArray(Diagonal(@. -ℓs * (ℓs + 1)) * parent(sinθdθo), ℓs, ℓs)

    DDr_minus_2byrMCU2 = @. DDrMCU2 - 2 * onebyrMCU2

    ddr_plus_2byrMCU4 = @. ddrMCU4 + 2 * onebyrMCU4

    ddrDDr_minus_ℓℓp1_by_r2MCU4 = zeros(nr, nr);

    T = zeros(nr, nr);

    # V terms
    V_ℓs = ℓrange(m, nℓ, V_symmetric)
    # W, S terms
    W_ℓs = ℓrange(m, nℓ, !V_symmetric)

    @views for (ℓind, ℓ) in enumerate(V_ℓs)
        # numerical green function
        ℓℓp1 = ℓ * (ℓ + 1)
        two_over_ℓℓp1 = 2 / ℓℓp1

        dopplerterm = -m * ΔΩ_frac
        diagterm = m * two_over_ℓℓp1 * ΔΩ_frac + dopplerterm

        VV[Block(ℓind, ℓind)] .+= diagterm .* IU2

        for ℓ′ in intersect(ℓ-1:2:ℓ+1, W_ℓs)
            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)
            ℓ′ind = findfirst(isequal(ℓ′), W_ℓs)

            @. T = -two_over_ℓℓp1 *
                                    ΔΩ_frac * (
                                        ℓ′ℓ′p1 * cosθo[ℓ, ℓ′] * DDr_minus_2byrMCU2 +
                                        (DDrMCU2 - ℓ′ℓ′p1 * onebyrMCU2) * sinθdθo[ℓ, ℓ′]
                                    ) * Rsun / Wscaling

            VW[Block(ℓind, ℓ′ind)] .+= T
        end
    end

    @views for (ℓind, ℓ) in enumerate(W_ℓs)
        # numerical green function
        ℓℓp1 = ℓ * (ℓ + 1)
        two_over_ℓℓp1 = 2 / ℓℓp1

        dopplerterm = -m * ΔΩ_frac
        diagterm = m * two_over_ℓℓp1 * ΔΩ_frac + dopplerterm

        @. ddrDDr_minus_ℓℓp1_by_r2MCU4 = ddrDDrMCU4 - ℓℓp1 * onebyr2MCU4;

        WW[Block(ℓind, ℓind)] .+= (Weqglobalscaling * Rsun^2) .* (diagterm .* ddrDDr_minus_ℓℓp1_by_r2MCU4 .+ 2dopplerterm .* ηρ_by_rMCU4)

        if nvariables == 3
            SS[Block(ℓind, ℓind)] .+= Seqglobalscaling .* dopplerterm .* IU2
        end

        for ℓ′ in intersect(ℓ-1:2:ℓ+1, V_ℓs)
            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)
            ℓ′ind = findfirst(isequal(ℓ′), V_ℓs)

            @. T = -Rsun * ΔΩ_frac / ℓℓp1 *
                                    ((4ℓ′ℓ′p1 * cosθo[ℓ, ℓ′] + (ℓ′ℓ′p1 + 2) * sinθdθo[ℓ, ℓ′]) * ddrMCU4
                                        +
                                        ddr_plus_2byrMCU4 * laplacian_sinθdθo[ℓ, ℓ′]) * Wscaling

            WV[Block(ℓind, ℓ′ind)] .+= Weqglobalscaling .* T
        end
    end

    return M
end

function equatorial_radial_rotation_profile(operators, thetaGL; smoothing_param = 5e-3)
    @unpack r = operators.coordinates
    ΔΩ_rθ, Ω0 = read_angular_velocity(operators, thetaGL; smoothing_param)
    s = Spline2D(r, thetaGL, ΔΩ_rθ)
    ΔΩ_r = reshape(evalgrid(s, r, [pi/2]), Val(1))
end

function radial_differential_rotation_profile(operators, thetaGL, model = :solar_equator;
    smoothing_param = 5e-3)

    @unpack r = operators.coordinates
    @unpack r_out, nr, r_in = operators.radial_params

    if model == :solar_equator
        Ω0 = equatorial_rotation_angular_velocity(r_out / Rsun)
        ΔΩ_r = equatorial_radial_rotation_profile(operators, thetaGL; smoothing_param)
    elseif model == :linear # for testing
        Ω0 = equatorial_rotation_angular_velocity(r_out / Rsun)
        f = 0.02 / (r_in / Rsun - 1)
        ΔΩ_r = @. Ω0 * f * (r / Rsun - 1)
    elseif model == :constant # for testing
        ΔΩ_frac = 0.02
        Ω0 = equatorial_rotation_angular_velocity(r_out / Rsun)
        ΔΩ_r = fill(ΔΩ_frac * Ω0, nr)
    elseif model == :core
        ΔΩ_frac = 0.3
        Ω0 = equatorial_rotation_angular_velocity(r_out / Rsun)
        ΔΩ_r = @. (Ω0*ΔΩ_frac)*1/2*(1 - tanh((r - 0.6Rsun)/(0.08Rsun)))
    else
        error("$model is not a valid rotation model")
    end
    ΔΩ_r ./= Ω0
    return ΔΩ_r, Ω0
end

function rotationprofile_radialderiv(r, ΔΩ_r, nr, Δr)
    ΔΩ = chop(chebyshevgrid_to_Fun(ΔΩ_r), 1e-2);
    @checkncoeff ΔΩ nr

    ΔΩ_spl = Spline1D(r, ΔΩ_r);
    ddrΔΩ_r = derivative(ΔΩ_spl, r);
    d2dr2ΔΩ_r = derivative(ΔΩ_spl, r, nu=2);

    zchop!(ddrΔΩ_r, 1e-10*(2/Δr))
    zchop!(d2dr2ΔΩ_r, 1e-10*(2/Δr)^2)

    ddrΔΩ = chop(chebyshevgrid_to_Fun(ddrΔΩ_r), 1e-2);
    @checkncoeff ddrΔΩ nr
    d2dr2ΔΩ = chop(chebyshevgrid_to_Fun(d2dr2ΔΩ_r), 5e-2);
    @checkncoeff d2dr2ΔΩ nr

    (ΔΩ, ddrΔΩ, d2dr2ΔΩ)
end

function radial_differential_rotation_profile_derivatives(m; operators,
        rotation_profile = :solar_equator, smoothing_param = 5e-3)
    @unpack r = operators.coordinates;
    @unpack nr, nℓ, Δr = operators.radial_params;

    ntheta = ntheta_ℓmax(nℓ, m);
    @unpack thetaGL = gausslegendre_theta_grid(ntheta);

    ΔΩ_r, Ω0 = radial_differential_rotation_profile(operators, thetaGL, rotation_profile; smoothing_param);

    (ΔΩ, ddrΔΩ, d2dr2ΔΩ) = replaceemptywitheps.(rotationprofile_radialderiv(r, ΔΩ_r, nr, Δr))
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

function radial_differential_rotation_terms!(M::StructMatrix{<:Complex}, m;
        operators, rotation_profile = :radial, V_symmetric = true, kw...)

    @unpack nr, nℓ = operators.radial_params
    @unpack nvariables = operators.constants;
    @unpack DDr, ddr, ddrDDr = operators.diff_operators;
    @unpack ddrMCU4 = operators.operator_matrices;
    @unpack onebyr, g, ηρ_by_r, onebyr2 = operators.rad_terms;
    @unpack Sscaling, Wscaling, Weqglobalscaling, Seqglobalscaling = operators.scalings
    @unpack matCU4, matCU2 = operators;

    VV = matrix_block(M.re, 1, 1)
    VW = matrix_block(M.re, 1, 2)
    WV = matrix_block(M.re, 2, 1)
    WW = matrix_block(M.re, 2, 2)
    if nvariables === 3
        SV = matrix_block(M.re, 3, 1)
        SW = matrix_block(M.re, 3, 2)
        SS = matrix_block(M.re, 3, 3)
    end

    ΔΩprofile_deriv = radial_differential_rotation_profile_derivatives(m; operators, rotation_profile);

    (; ΔΩ, ddrΔΩ, d2dr2ΔΩ, Ω0) = ΔΩprofile_deriv;

    ΔΩMCU2 = matCU2(ΔΩ);
    ddrΔΩMCU2 = matCU2(ddrΔΩ);
    d2dr2ΔΩMCU4 = matCU4(d2dr2ΔΩ);

    ddrΔΩ_over_g = ddrΔΩ / g;
    ddrΔΩ_over_gMCU2 = matCU2(ddrΔΩ_over_g);
    ddrΔΩ_over_g_DDr = (ddrΔΩ_over_g * DDr)::Tmul;
    ddrΔΩ_over_g_DDrMCU2 = matCU2(ddrΔΩ_over_g_DDr);
    ddrΔΩ_plus_ΔΩddr = (ddrΔΩ + (ΔΩ * ddr)::Tmul)::Tplus;

    nCS = 2nℓ+1
    ℓs = range(m, length = nCS)
    cosθ = OffsetArray(costheta_operator(nCS, m), ℓs, ℓs);
    sinθdθ = OffsetArray(sintheta_dtheta_operator(nCS, m), ℓs, ℓs);

    cosθo = OffsetArray(cosθ, ℓs, ℓs);
    sinθdθo = OffsetArray(sinθdθ, ℓs, ℓs);
    cosθsinθdθ = (costheta_operator(nCS + 1, m)*sintheta_dtheta_operator(nCS + 1, m))[1:end-1, 1:end-1];
    cosθsinθdθo = OffsetArray(cosθsinθdθ, ℓs, ℓs);
    ∇²_sinθdθo = OffsetArray(Diagonal(@. -ℓs * (ℓs + 1)) * sinθdθ, ℓs, ℓs);

    DDr_min_2byr = (DDr - 2onebyr)::Tplus;
    ΔΩ_DDr_min_2byr = (ΔΩ * DDr_min_2byr)::Tmul;
    ΔΩ_DDr = (ΔΩ * DDr)::Tmul;
    ΔΩ_by_r = ΔΩ * onebyr;
    ΔΩ_by_r2 = ΔΩ * onebyr2;

    ΔΩ_by_rMCU2, ΔΩ_DDrMCU2, ΔΩ_DDr_min_2byrMCU2 =
        map(matCU2, (ΔΩ_by_r, ΔΩ_DDr, ΔΩ_DDr_min_2byr, ddrΔΩ_plus_ΔΩddr));

    ddrΔΩ_plus_ΔΩddrMCU4, twoΔΩ_by_rMCU4, ddrΔΩ_DDrMCU4, ΔΩ_ddrDDrMCU4,
        ΔΩ_by_r2MCU4, ddrΔΩ_ddr_plus_2byrMCU4, ΔΩ_ηρ_by_rMCU4 =
        map(matCU4, (ddrΔΩ_plus_ΔΩddr, 2ΔΩ_by_r, ddrΔΩ * DDr, ΔΩ * ddrDDr,
            ΔΩ_by_r2, ddrΔΩ * (ddr + 2onebyr), ΔΩ * ηρ_by_r))

    ηρbyr_ΔΩMCU4 = matCU4(ηρ_by_r * ΔΩ)

    ΔΩ_ddrDDr_min_ℓℓp1byr2MCU4 = zeros(nr, nr);
    T = zeros(nr, nr);

    # V terms
    V_ℓs = ℓrange(m, nℓ, V_symmetric)
    # W, S terms
    W_ℓs = ℓrange(m, nℓ, !V_symmetric)

    @views for (ℓind, ℓ) in enumerate(V_ℓs)
        ℓℓp1 = ℓ * (ℓ + 1)
        two_over_ℓℓp1 = 2/ℓℓp1
        two_over_ℓℓp1_min_1 = two_over_ℓℓp1 - 1

        @. T = m * two_over_ℓℓp1_min_1 * ΔΩMCU2

        VV[Block(ℓind, ℓind)] .+= T

        for ℓ′ in intersect(W_ℓs, ℓ-1:2:ℓ+1)
            ℓ′ind = findfirst(isequal(ℓ′), W_ℓs)

            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)

            cosθ_ℓℓ′ = cosθo[ℓ, ℓ′]
            sinθdθ_ℓℓ′ = sinθdθo[ℓ, ℓ′]
            ∇²_sinθdθ_ℓℓ′ = ∇²_sinθdθo[ℓ, ℓ′]

            @. T = -Rsun * two_over_ℓℓp1 *
                    (ℓ′ℓ′p1 * cosθ_ℓℓ′ * (ΔΩ_DDr_min_2byrMCU2 - ddrΔΩMCU2) +
                    sinθdθ_ℓℓ′ * ((ΔΩ_DDrMCU2 - ℓ′ℓ′p1 * ΔΩ_by_rMCU2) - ℓ′ℓ′p1 / 2 * ddrΔΩMCU2)) / Wscaling

            VW[Block(ℓind, ℓ′ind)] .+= T
        end
    end

    @views for (ℓind, ℓ) in enumerate(W_ℓs)
        inds_ℓℓ = blockinds((m, nr), ℓ, ℓ);

        ℓℓp1 = ℓ * (ℓ + 1)
        two_over_ℓℓp1 = 2/ℓℓp1
        two_over_ℓℓp1_min_1 = two_over_ℓℓp1 - 1

        @. ΔΩ_ddrDDr_min_ℓℓp1byr2MCU4 = ΔΩ_ddrDDrMCU4 - ℓℓp1 * ΔΩ_by_r2MCU4

        @. T = m * Rsun^2 * (two_over_ℓℓp1_min_1 * (ddrΔΩ_DDrMCU4 + ΔΩ_ddrDDr_min_ℓℓp1byr2MCU4)
                - 2ΔΩ_ηρ_by_rMCU4 + d2dr2ΔΩMCU4 + ddrΔΩ_ddr_plus_2byrMCU4)

        WW[Block(ℓind, ℓind)] .+= Weqglobalscaling .* T

        if nvariables == 3
            SS[Block(ℓind, ℓind)] .+= Seqglobalscaling .* (-m) .* ΔΩMCU2
        end

        for ℓ′ in intersect(V_ℓs, ℓ-1:2:ℓ+1)
            ℓ′ind = findfirst(isequal(ℓ′), V_ℓs)

            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)

            cosθ_ℓℓ′ = cosθo[ℓ, ℓ′]
            sinθdθ_ℓℓ′ = sinθdθo[ℓ, ℓ′]
            ∇²_sinθdθ_ℓℓ′ = ∇²_sinθdθo[ℓ, ℓ′]

            @. T = -1/ℓℓp1 * Rsun * (
                        (4ℓ′ℓ′p1 * cosθ_ℓℓ′ + (ℓ′ℓ′p1 + 2) * sinθdθ_ℓℓ′ + ∇²_sinθdθ_ℓℓ′) * ddrΔΩ_plus_ΔΩddrMCU4 +
                        ∇²_sinθdθ_ℓℓ′ * twoΔΩ_by_rMCU4
                    ) * Wscaling

            WV[Block(ℓind, ℓ′ind)] .+= Weqglobalscaling .* T

            # if nvariables == 3
            #     @. T = -(Ω0^2 * Rsun^2) * 2m * cosθo[ℓ, ℓ′] * ddrΔΩ_over_gMCU2 * Sscaling;
            #     SV[Block(ℓind, ℓ′ind)] .+= Seqglobalscaling .* T
            # end
        end

        # for ℓ′ in intersect(W_ℓs, ℓ-2:2:ℓ+2)
        #     ℓ′ind = findfirst(isequal(ℓ′), W_ℓs)
        #     if nvariables == 3
        #         @. T = (Ω0^2 * Rsun^3) * 2cosθsinθdθo[ℓ, ℓ′] * ddrΔΩ_over_g_DDrMCU2 * Sscaling/Wscaling;
        #         SW[Block(ℓind, ℓ′ind)] .+= Seqglobalscaling .* T
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
        ΔΩ_frac = 0.02
        @unpack radial_params = operators
        @unpack r_out, nr = radial_params
        Ω0 = equatorial_rotation_angular_velocity(r_out / Rsun)
        return fill(ΔΩ_frac * Ω0, nr, nθ), Ω0
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

function solar_differential_rotation_terms!(M, m;
    operators = radial_operators(nr, nℓ),
    rotation_profile = :constant)

    @unpack nvariables = operators.constants
    @unpack Iℓ, Ir = operators.identities
    @unpack ddr, DDr, d2dr2, rddr, r2d2dr2, DDr_minus_2byr = operators.diff_operators
    @unpack Tcrfwd, Tcrinv = operators.transforms
    @unpack g_cheby, onebyr, onebyr2, r2_cheby, r_cheby = operators.rad_terms
    @unpack r, r_chebyshev = operators.coordinates
    two_over_g = 2g_cheby^-1

    ddr_plus_2byr = ddr + 2onebyr_cheby

    # for the stream functions
    @unpack PLMfwd, PLMinv = associatedlegendretransform_matrices(nℓ, m)
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
    @unpack thetaGL = gausslegendre_theta_grid(ntheta)

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
        reshape((kron(cosθ_Ω, ddr) - kron(sinθdθ_Ω, onebyr)) * vec(ΔΩ_nℓ), nr, nℓ_Ω)
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
    ∇²_by_r2 = kron2(∇², onebyr2)
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

    @unpack Wscaling, Sscaling = operators.constants.scalings

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

function _differential_rotation_matrix!(M, m; rotation_profile, kw...)
    if rotation_profile === :radial
        radial_differential_rotation_terms!(M, m; rotation_profile = :solar_equator, kw...)
    elseif rotation_profile === :radial_constant
        radial_differential_rotation_terms!(M, m; rotation_profile = :constant, kw...)
    elseif rotation_profile === :radial_linear
        radial_differential_rotation_terms!(M, m; rotation_profile = :linear, kw...)
    elseif rotation_profile === :radial_core
        radial_differential_rotation_terms!(M, m; rotation_profile = :core, kw...)
    elseif rotation_profile === :solar
        solar_differential_rotation_terms!(M, m; rotation_profile, kw...)
    elseif rotation_profile === :solar_radial
        solar_differential_rotation_terms!(M, m; rotation_profile = :radial, kw...)
    elseif rotation_profile === :solar_constant
        solar_differential_rotation_terms!(M, m; rotation_profile = :constant, kw...)
    elseif rotation_profile === :constant
        constant_differential_rotation_terms!(M, m; kw...)
    else
        throw(ArgumentError("Invalid rotation profile"))
    end
    return M
end
function differential_rotation_matrix(m; operators, kw...)
    M = allocate_operator_matrix(operators)
    differential_rotation_matrix!(M, m; operators, kw...)
    return M
end
function differential_rotation_matrix!(M, m; kw...)
    uniform_rotation_matrix!(M, m; kw...)
    _differential_rotation_matrix!(M, m; kw...)
    return M
end

function constrained_matmul_cache(constraints)
    @unpack ZC = constraints
    sz_constrained = (size(ZC, 2), size(ZC, 2))
    A_constrained = zeros(ComplexF64, sz_constrained)
    B_constrained = zeros(ComplexF64, sz_constrained)
    return (; A_constrained, B_constrained)
end
function compute_constrained_matrix!(out, constraints, A)
    @unpack ZC = constraints
    out .= ZC' * A * ZC
    return out
end
function compute_constrained_matrix!(out, constraints, A::StructMatrix{<:Complex})
    @unpack ZC = constraints
    out .= (ZC' * A.re * ZC) .+ im .* (ZC' * A.im * ZC)
    return out
end
function compute_constrained_matrix(A::AbstractMatrix{<:Complex}, constraints,
        cache = constrained_matmul_cache(constraints))

    @unpack A_constrained = cache
    compute_constrained_matrix!(A_constrained, constraints, computesparse(A))
    return A_constrained
end

function compute_constrained_matrix(B::AbstractMatrix{<:Real}, constraints,
        cache = constrained_matmul_cache(constraints))

    @unpack B_constrained = cache
    compute_constrained_matrix!(B_constrained, constraints, computesparse(B))
    return B_constrained
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

function constrained_eigensystem_timed(AB; timer = TimerOutput(), kw...)
    X = @timeit timer "eigen" constrained_eigensystem(computesparse.(AB); timer, kw...)
    if get(kw, :print_timer, false)
        println(timer)
    end
    X
end
function constrained_eigensystem((A, B);
    operators,
    constraints = constraintmatrix(operators),
    cache = constrained_matmul_cache(constraints),
    temp_projectback = allocate_projectback_temp_matrices(size(constraints.ZC)),
    timer = TimerOutput(),
    kw...
    )

    @timeit timer "basis" begin
        A_constrained = compute_constrained_matrix(A, constraints, cache)
        B_constrained = compute_constrained_matrix(B, constraints, cache)
    end
    @timeit timer "eigen" λ, w = eigen!(A_constrained, B_constrained)
    @timeit timer "projectback" v = realmatcomplexmatmul(constraints.ZC, w, temp_projectback)
    λ, v, (A, B)
end

function uniform_rotation_spectrum(m; operators, kw...)
    A = allocate_operator_matrix(operators)
    B = allocate_mass_matrix(operators)
    uniform_rotation_spectrum!((A, B), m; operators, kw...)
end
function uniform_rotation_spectrum!((A, B), m; operators, kw...)
    timer = TimerOutput()
    @timeit timer "matrix" begin
        uniform_rotation_matrix!(A, m; operators, kw...)
        mass_matrix!(B, m; operators, kw...)
    end
    constrained_eigensystem_timed((A, B); operators, timer, kw...)
end

uniformrotmatrixfn!(V_symmetric = true) =
    (x...; kw...) -> uniform_rotation_matrix!(x...; V_symmetric, kw...)
uniformrotspectrumfn!(V_symmetric = true) =
    (x...; kw...) -> uniform_rotation_spectrum!(x...; V_symmetric, kw...)

function real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop)
    @unpack PLMfwd, PLMinv = thetaop
    @unpack transforms, radial_params = operators
    @unpack nℓ, nchebyr = radial_params
    @unpack Tcrfwd, Tcrinv = transforms
    pad = nchebyr * nℓ
    PaddedMatrix((PLMfwd ⊗ Tcrfwd) * Diagonal(vec(ΔΩ_r_thetaGL)) * (PLMinv ⊗ Tcrinv), pad)
end

function real_to_r_assocleg(ΔΩ_r_thetaGL, operators, thetaop)
    @unpack PLMfwd, PLMinv = thetaop
    @unpack radial_params = operators
    @unpack nℓ, nchebyr = radial_params
    Ir = I(nchebyr)
    pad = nchebyr * nℓ
    PaddedMatrix((PLMfwd ⊗ Ir) * Diagonal(vec(ΔΩ_r_thetaGL)) * (PLMinv ⊗ Ir), pad)
end

function differential_rotation_spectrum(m; operators, kw...)
    A = allocate_operator_matrix(operators)
    B = allocate_mass_matrix(operators)
    differential_rotation_spectrum!((A, B), m; operators, kw...)
end
function differential_rotation_spectrum!((A, B), m; rotation_profile, operators, kw...)
    timer = TimerOutput()
    @timeit timer "matrix" begin
        differential_rotation_matrix!(A, m; operators, rotation_profile, kw...)
        mass_matrix!(B, m; operators)
    end
    constrained_eigensystem_timed((A, B); operators, timer, kw...)
end

diffrotmatrixfn!(rotation_profile, V_symmetric = true) =
    (x...; kw...) -> differential_rotation_matrix!(x...; rotation_profile, V_symmetric, kw...)
diffrotspectrumfn!(rotation_profile, V_symmetric = true) =
    (x...; kw...) -> differential_rotation_spectrum!(x...; rotation_profile, V_symmetric, kw...)

rossby_ridge(m; ΔΩ_frac = 0) = 2 / (m + 1) * (1 + ΔΩ_frac) - m * ΔΩ_frac

function eigenvalue_filter(x, m;
    eig_imag_unstable_cutoff = -1e-6,
    eig_imag_to_real_ratio_cutoff = 1)

    freq_sectoral = 2 / (m + 1)
    eig_imag_unstable_cutoff <= imag(x) < freq_sectoral * eig_imag_to_real_ratio_cutoff
end
function boundary_condition_filter(v, BC, BCVcache = allocate_BCcache(size(BC,1)), atol = 1e-5)
    mul!(BCVcache.re, BC, v.re)
    mul!(BCVcache.im, BC, v.im)
    norm(BCVcache) < atol
end
function eigensystem_satisfy_filter(λ::Number, v::StructVector{<:Complex},
        AB::Tuple{StructMatrix{<:Complex}, AbstractMatrix{<:Real}},
        MVcache::NTuple{2, StructArray{<:Complex,1}} = allocate_MVcache(size(AB[1], 1)), rtol = 1e-1)

    A, B = computesparse.(AB)
    Av, λBv = MVcache

    mul!(Av.re, A.re, v.re)
    mul!(Av.re, A.im, v.im, -1.0, 1.0)
    mul!(Av.im, A.re, v.im)
    mul!(Av.im, A.im, v.re,  1.0, 1.0)

    mul!(λBv.re, B, v.re)
    mul!(λBv.im, B, v.im)
    λBv .*= λ

    isapprox(Av, λBv; rtol) && return true
    return false
end

function filterfields(coll, v, nparams, nvariables; filterfieldpowercutoff = 1e-4)
    Vpow = sum(abs2, @view v[1:nparams])
    Wpow = sum(abs2, @view v[nparams .+ (1:nparams)])
    Spow = nvariables == 3 ? sum(abs2, @view(v[2nparams .+ (1:nparams)])) : 0.0

    maxpow = max(Vpow, Wpow, Spow)

    filterfields = typeof(coll.V)[]

    if Spow/maxpow > filterfieldpowercutoff
        push!(filterfields, coll.S)
    end
    if Vpow/maxpow > filterfieldpowercutoff
        push!(filterfields, coll.V)
    end

    if Wpow/maxpow > filterfieldpowercutoff
        push!(filterfields, coll.W)
    end
    return filterfields
end

function sphericalharmonic_filter!(VWSinvsh, F, v, operators,
        Δl_cutoff = 7, power_cutoff = 0.9, filterfieldpowercutoff = 1e-4)

    eigenfunction_rad_sh!(VWSinvsh, F, v; operators)
    l_cutoff_ind = 1 + Δl_cutoff÷2

    flag = true

    @unpack nparams = operators.radial_params
    @unpack nvariables = operators.constants
    fields = filterfields(VWSinvsh, v, nparams, nvariables; filterfieldpowercutoff)

    @views for X in fields
        PV_frac = sum(abs2, X[:, 1:l_cutoff_ind]) / sum(abs2, X)
        flag &= PV_frac > power_cutoff
    end

    flag
end

allocate_Pl(m, nℓ) = zeros(range(m, length = 2nℓ + 1))

function chebyshev_filter!(VWSinv, F, v, m, operators, n_cutoff = 7, n_power_cutoff = 0.9;
    nℓ = operators.radial_params.nℓ,
    Plcosθ = allocate_Pl(m, nℓ),
    filterfieldpowercutoff = 1e-4)

    eigenfunction_n_theta!(VWSinv, F, v, m; operators, nℓ, Plcosθ)

    @unpack V = VWSinv
    n_cutoff_ind = 1 + n_cutoff
    # ensure that most of the power at the equator is below the cutoff
    nθ = size(V, 2)
    equator_ind = nθ÷2

    Δθ_scan = div(nθ, 5)
    rangescan = intersect(equator_ind .+ (-Δθ_scan:Δθ_scan), axes(V, 2))

    flag = true

    @unpack nparams = operators.radial_params
    @unpack nvariables = operators.constants
    fields = filterfields(VWSinv, v, nparams, nvariables; filterfieldpowercutoff)

    @views for X in fields
        Xrange = X[:, rangescan]
        PV_frac = sum(abs2, X[1:n_cutoff_ind, rangescan]) / sum(abs2, X[:, rangescan])
        flag &= PV_frac > n_power_cutoff
    end

    flag
end

function spatial_filter!(VWSinv, VWSinvsh, F, v, m, operators,
    θ_cutoff = deg2rad(60), equator_power_cutoff_frac = 0.3;
    nℓ = operators.radial_params.nℓ,
    Plcosθ = allocate_Pl(m, nℓ),
    filterfieldpowercutoff = 1e-4)

    (; θ) = eigenfunction_realspace!(VWSinv, VWSinvsh, F, v, m; operators, nℓ, Plcosθ)

    eqfilter = true

    @unpack nparams = operators.radial_params
    @unpack nvariables = operators.constants
    fields = filterfields(VWSinv, v, nparams, nvariables; filterfieldpowercutoff)

    for X in fields
        peak_latprofile = @view X[end, :]
        θlowind = searchsortedfirst(θ, θ_cutoff)
        θhighind = searchsortedlast(θ, pi - θ_cutoff)
        powfrac = sum(abs2, @view peak_latprofile[θlowind:θhighind]) / sum(abs2, peak_latprofile)
        powflag = powfrac > equator_power_cutoff_frac
        peakflag = maximum(abs2, @view peak_latprofile[θlowind:θhighind]) == maximum(abs2, peak_latprofile)
        eqfilter &= powflag & peakflag
        eqfilter || break
    end

    return eqfilter
end

function nodes_filter(VWSinv, VWSinvsh, F, v, m, operators;
    nℓ = operators.radial_params.nℓ,
    Plcosθ = allocate_Pl(m, nℓ),
    filterfieldpowercutoff = 1e-4,
    nnodesmax = 7)

    nodesfilter = true

    @unpack nparams = operators.radial_params
    @unpack nvariables = operators.constants
    fields = filterfields(VWSinv, v, nparams, nvariables; filterfieldpowercutoff)

    (; θ) = eigenfunction_realspace!(VWSinv, VWSinvsh, F, v, m; operators, nℓ, Plcosθ)
    eqind = argmin(abs.(θ .- pi/2))

    for X in fields
        radprof = @view X[:, eqind]
        nnodes_real = count(Bool.(sign.(abs.(diff(sign.(real(radprof)))))))
        nnodes_imag = count(Bool.(sign.(abs.(diff(sign.(imag(radprof)))))))
        nodesfilter &= nnodes_real <= nnodesmax && nnodes_imag <= nnodesmax
    end
    nodesfilter
end

module Filters
    using BitFlags
    export DefaultFilter
    @bitflag FilterFlag::UInt8 begin
        NONE=0
        EIGVAL
        EIGEN
        SPHARM
        CHEBY
        BC
        SPATIAL
        NODES
    end
    FilterFlag(F::FilterFlag) = F
    Base.:(!)(F::FilterFlag) = FilterFlag(Int(typemax(UInt8) >> 1) - Int(F))
    Base.in(t::FilterFlag, F::FilterFlag) = (t & F) != NONE
    Base.broadcastable(x::FilterFlag) = Ref(x)

    const DefaultFilter = EIGVAL | EIGEN | SPHARM | CHEBY | BC | SPATIAL
end
using .Filters

function filterfn(λ, v, m, M, (operators, constraints, filtercache, kw)::NTuple{4,Any}, filterflags)

    @unpack BC = constraints
    @unpack nℓ = operators.radial_params;

    @unpack eig_imag_unstable_cutoff = kw
    @unpack eig_imag_to_real_ratio_cutoff = kw
    @unpack Δl_cutoff = kw
    @unpack Δl_power_cutoff = kw
    @unpack bc_atol = kw
    @unpack n_cutoff = kw
    @unpack n_power_cutoff = kw
    @unpack θ_cutoff = kw
    @unpack equator_power_cutoff_frac = kw
    @unpack eigen_rtol = kw
    @unpack filterfieldpowercutoff = kw
    @unpack nnodesmax = kw

    (; MVcache, BCVcache, VWSinv, VWSinvsh, Plcosθ, F) = filtercache;

    allfilters = Filters.FilterFlag(filterflags)

    if Filters.EIGVAL in allfilters
        f1 = eigenvalue_filter(λ, m;
        eig_imag_unstable_cutoff, eig_imag_to_real_ratio_cutoff)
        f1 || return false
    end

    if Filters.SPHARM in allfilters
        f2 = sphericalharmonic_filter!(VWSinvsh, F, v, operators,
            Δl_cutoff, Δl_power_cutoff, filterfieldpowercutoff)
        f2 || return false
    end

    if Filters.BC in allfilters
        f3 = boundary_condition_filter(v, BC, BCVcache, bc_atol)
        f3 || return false
    end

    if Filters.CHEBY in allfilters
        f4 = chebyshev_filter!(VWSinv, F, v, m, operators, n_cutoff,
            n_power_cutoff; nℓ, Plcosθ,filterfieldpowercutoff)
        f4 || return false
    end

    if Filters.SPATIAL in allfilters
        f5 = spatial_filter!(VWSinv, VWSinvsh, F, v, m, operators,
            θ_cutoff, equator_power_cutoff_frac; nℓ, Plcosθ,
            filterfieldpowercutoff)
        f5 || return false
    end

    if Filters.EIGEN in allfilters
        f6 = eigensystem_satisfy_filter(λ, v, M, MVcache, eigen_rtol)
        f6 || return false
    end

    if Filters.NODES in allfilters
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

function allocate_MVcache(nrows)
    StructArray{ComplexF64}((zeros(nrows), zeros(nrows))),
        StructArray{ComplexF64}((zeros(nrows), zeros(nrows)))
end

function allocate_BCcache(n_bc)
    StructArray{ComplexF64}((zeros(n_bc), zeros(n_bc)))
end

function allocate_filter_caches(m; operators, constraints = constraintmatrix(operators))
    @unpack BC, nvariables = constraints
    @unpack nr, nℓ, nparams = operators.radial_params
    # temporary cache arrays
    nrows = nvariables * nparams
    MVcache = allocate_MVcache(nrows)

    n_bc = size(BC, 1)
    BCVcache = allocate_BCcache(n_bc)

    nθ = length(spharm_θ_grid_uniform(m, nℓ).θ)

    @unpack VWSinv, VWSinvsh, F = allocate_field_caches(nr, nθ, nℓ)

    Plcosθ = allocate_Pl(m, nℓ)

    return (; MVcache, BCVcache, VWSinv, VWSinvsh, Plcosθ, F)
end

const DefaultFilterParams = Dict(
    :bc_atol => 1e-5,
    :Δl_cutoff => 7,
    :Δl_power_cutoff => 0.9,
    :eigen_rtol => 0.01,
    :n_cutoff => 10,
    :n_power_cutoff => 0.9,
    :eig_imag_unstable_cutoff => -1e-6,
    :eig_imag_to_real_ratio_cutoff => 1,
    :θ_cutoff => deg2rad(60),
    :equator_power_cutoff_frac => 0.3,
    :nnodesmax => 10,
    :filterfieldpowercutoff => 1e-4,
)

function filter_eigenvalues(λ::AbstractVector, v::AbstractMatrix,
    M, m::Integer;
    operators,
    constraints = constraintmatrix(operators),
    filtercache = allocate_filter_caches(m; operators, constraints),
    filterflags = DefaultFilter,
    kw...)

    @unpack nparams = operators.radial_params;
    kw = merge(DefaultFilterParams, kw)
    additional_params = (operators, constraints, filtercache, kw)

    inds_bool = filterfn.(λ, eachcol(v), m, (computesparse.(M),), (additional_params,), filterflags)
    filtinds = axes(λ, 1)[inds_bool]
    λ, v = λ[filtinds], v[:, filtinds]

    # re-apply scalings
    @views if get(kw, :scale_eigenvectors, false)
        V = v[1:nparams, :]
        W = v[nparams .+ (1:nparams), :]
        S = v[2nparams .+ (1:nparams), :]

        V .*= Rsun
        @unpack Wscaling, Sscaling = operators.constants.scalings
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

function filter_eigenvalues(λs::AbstractVector{<:AbstractVector},
    vs::AbstractVector{<:AbstractMatrix}, mr::AbstractVector;
    matrixfn! = uniform_rotation_matrix!,
    operators, constraints = constraintmatrix(operators),kw...)

    @unpack nr, nℓ, nparams = operators.radial_params
    @unpack nvariables = operators.constants
    ABs = [(allocate_operator_matrix(operators), allocate_mass_matrix(operators)) for _ in 1:Threads.nthreads()]
    nthreads = Threads.nthreads();
    c = Channel(nthreads);
    for el in ABs
        put!(c, el)
    end

    λv = @maybe_reduce_blas_threads(Threads.nthreads(),
        Folds.map(zip(λs, vs, mr)) do (λm, vm, m)
            A, B = take!(c)
            matrixfn!(A, m; operators)
            mass_matrix!(B, m; operators)
            Y = filter_eigenvalues(λm, vm, (A,B), m; operators, constraints, kw...)
            put!(c, (A, B))
            Y
        end
    )
    first.(λv), last.(λv)
end

function fmap(spectrumfn!::F, m, c, operators, constraints; kw...) where {F}
    Ctid = take!(c)
    M, cache, temp_projectback = Ctid;
    X = spectrumfn!(M, m; operators, constraints, cache, temp_projectback, kw...);
    Y = filter_eigenvalues(X..., m; operators, constraints, kw...)
    put!(c, Ctid)
    return Y
end

function filter_eigenvalues(spectrumfn!, mr::AbstractVector;
    operators, constraints = constraintmatrix(operators), kw...)

    to = TimerOutput()

    @timeit to "alloc" begin
        nthreads = Threads.nthreads()
        @timeit to "M" ABs = [(allocate_operator_matrix(operators), allocate_mass_matrix(operators)) for _ in 1:nthreads];
        @timeit to "caches" caches = [constrained_matmul_cache(constraints) for _ in 1:nthreads];
        @timeit to "projectback" temp_projectback_mats = [allocate_projectback_temp_matrices(size(constraints.ZC)) for _ in 1:nthreads];
        z = zip(ABs, caches, temp_projectback_mats)
        c = Channel(nthreads);
        for el in z
            put!(c, el)
        end
    end

    @timeit to "spectrum" begin
        nblasthreads = BLAS.get_num_threads()
        nthreads_trailing_elems = rem(length(mr), nthreads)

        if nthreads_trailing_elems > 0 && div(nblasthreads, nthreads_trailing_elems) > max(1, div(nblasthreads, nthreads))
            # in this case the extra elements may be run using a higher number of blas threads
            mr1 = @view mr[1:end-nthreads_trailing_elems]
            λv1 = @maybe_reduce_blas_threads(Threads.nthreads(),
                Folds.map(mr1) do m
                    fmap(spectrumfn!, m, c, operators, constraints; kw...)
                end
            )
            λs, vs = first.(λv1), last.(λv1)

            mr2 = @view mr[end-nthreads_trailing_elems+1:end]

            λv2 = @maybe_reduce_blas_threads(nthreads_trailing_elems,
                Folds.map(mr2) do m
                    fmap(spectrumfn!, m, c, operators, constraints; kw...)
                end
            )

            λs2, vs2 = first.(λv2), last.(λv2)
            append!(λs, λs2)
            append!(vs, vs2)
        else
            λv = @maybe_reduce_blas_threads(Threads.nthreads(),
                Folds.map(mr) do m
                    fmap(spectrumfn!, m, c, operators, constraints; kw...)
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
function save_eigenvalues(f, mr; operators, kw...)
    lam, vec = filter_eigenvalues(f, mr; operators, kw...)
    isdiffrot = get(kw, :diffrot, false)
    filenametag = isdiffrot ? "dr" : "ur"
    posttag = get(kw, :V_symmetric, true) ? "sym" : "asym"
    @unpack nr, nℓ = operators.radial_params;
    fname = datadir(rossbyeigenfilename(nr, nℓ, filenametag, posttag))
    @info "saving to $fname"
    jldsave(fname; lam, vec, mr, nr, nℓ, kw, operators)
end

function eigenfunction_cheby_ℓm_spectrum!(F, v; operators, kw...)
    @unpack radial_params = operators
    @unpack nparams, nr, nℓ = radial_params
    @unpack nvariables = operators.constants

    F.V .= @view v[1:nparams]
    F.W .= @view v[nparams.+(1:nparams)]
    F.S .= nvariables == 3 ? @view(v[2nparams.+(1:nparams)]) : 0.0

    V = reshape(F.V, nr, nℓ)
    W = reshape(F.W, nr, nℓ)
    S = reshape(F.S, nr, nℓ)

    (; V, W, S)
end

function eigenfunction_rad_sh!(VWSinvsh, F, v; operators, n_cutoff = -1, kw...)
    VWS = eigenfunction_cheby_ℓm_spectrum!(F, v; operators, kw...)
    @unpack Tcrinvc = operators.transforms;
    @unpack V, W, S = VWS

    Vinv = VWSinvsh.V
    Winv = VWSinvsh.W
    Sinv = VWSinvsh.S

    if n_cutoff >= 0
        temp = similar(V)
        temp .= V
        temp[n_cutoff+1:end, :] .= 0
        mul!(Vinv, Tcrinvc, temp)
        temp .= W
        temp[n_cutoff+1:end, :] .= 0
        mul!(Winv, Tcrinvc, temp)
        temp .= S
        temp[n_cutoff+1:end, :] .= 0
        mul!(Sinv, Tcrinvc, temp)
    else
        mul!(Vinv, Tcrinvc, V)
        mul!(Winv, Tcrinvc, W)
        mul!(Sinv, Tcrinvc, S)
    end

    return VWSinvsh
end

function spharm_θ_grid_uniform(m, nℓ, ℓmax_mul = 4)
    ℓs = range(m, length = 2nℓ+1)
    ℓmax = maximum(ℓs)

    θ, _ = sph_points(ℓmax_mul * ℓmax)
    return (; ℓs, θ)
end

function invshtransform2!(VWSinv, VWS, m;
    nℓ = size(VWS.V, 2),
    Plcosθ = allocate_Pl(m, nℓ),
    V_symmetric = true,
    Δl_cutoff = lastindex(Plcosθ),
    kw...)

    V_lm = VWS.V
    W_lm = VWS.W
    S_lm = VWS.S

    (; ℓs, θ) = spharm_θ_grid_uniform(m, nℓ)

    V = VWSinv.V
    V .= 0
    W = VWSinv.W
    W .= 0
    S = VWSinv.S
    S .= 0

    V_ℓs = ℓrange(m, nℓ, V_symmetric)

    W_symmetric = !V_symmetric
    W_ℓs = ℓrange(m, nℓ, W_symmetric)

    @views for (θind, θi) in enumerate(θ)
        collectPlm!(Plcosθ, cos(θi); m, norm = Val(:normalized))
        # V
        for (ℓind, ℓ) in enumerate(V_ℓs)
            ℓ > m + Δl_cutoff && continue
            Plmcosθ = Plcosθ[ℓ]
            @. V[:, θind] += V_lm[:, ℓind] * Plmcosθ
        end
        # W, S
        for (ℓind, ℓ) in enumerate(W_ℓs)
            ℓ > m + Δl_cutoff && continue
            Plmcosθ = Plcosθ[ℓ]
            @. W[:, θind] += W_lm[:, ℓind] * Plmcosθ
            @. S[:, θind] += S_lm[:, ℓind] * Plmcosθ
        end
    end

    (; VWSinv, θ)
end

function eigenfunction_realspace!(VWSinv, VWSinvsh, F, v, m;
    operators,
    nℓ = operators.radial_params.nℓ,
    Plcosθ = allocate_Pl(m, nℓ),
    kw...)

    eigenfunction_rad_sh!(VWSinvsh, F, v; operators, kw...)
    invshtransform2!(VWSinv, VWSinvsh, m; nℓ, Plcosθ, kw...)
end

function eigenfunction_realspace(v, m; operators, kw...)
    @unpack nr, nℓ = operators.radial_params
    (; θ) = spharm_θ_grid_uniform(m, nℓ)
    nθ = length(θ)

    @unpack VWSinv, VWSinvsh, F = allocate_field_caches(nr, nθ, nℓ)

    eigenfunction_realspace!(VWSinv, VWSinvsh, F, v, m; operators, kw...)

    return (; VWSinv, θ)
end

function eigenfunction_n_theta!(VWSinv, F, v, m;
    operators,
    nℓ = operators.radial_params.nℓ,
    Plcosθ = allocate_Pl(m, nℓ),
    kw...)

    VW = eigenfunction_cheby_ℓm_spectrum!(F, v; operators, kw...)
    invshtransform2!(VWSinv, VW, m; nℓ, Plcosθ, kw...)
end

# precompile
precompile(_radial_operators, (Int, Int, Float64, Float64, Bool, Int, Float64, NTuple{4,Float64}))

end # module
