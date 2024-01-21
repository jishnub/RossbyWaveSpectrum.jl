module RossbyWaveSpectrum

using MKL

using ApproxFun
using ApproxFunAssociatedLegendre
using ApproxFunBase
using BlockArrays
using BandedMatrices
using BlockBandedMatrices
using DelimitedFiles: readdlm
import Dierckx
using Dierckx: Dierckx, Spline1D, Spline2D, derivative
using DomainSets
using FFTW
using FillArrays
using Folds
using IntervalSets
using JLD2
using LinearAlgebra
using LinearAlgebra: BLAS
using LegendrePolynomials
using OffsetArrays
using SparseArrays
using StructArrays
using TimerOutputs
using Trapz
using UnPack
using ZChop

export radial_operators
export rossby_ridge
export datadir
export rossbyeigenfilename
export Filters
export eigvalspairs
export FilteredEigen
export RotMatrix
export Rsun
export mass_matrix
export mass_matrix!
export uniform_rotation_matrix
export uniform_rotation_matrix!
export differential_rotation_matrix
export differential_rotation_matrix!
export uniform_rotation_spectrum
export uniform_rotation_spectrum!
export differential_rotation_spectrum
export differential_rotation_spectrum!
export filter_eigenvalues
export save_eigenvalues

const SCRATCH = Ref("")
const DATADIR = Ref("")
const SOLARMODELDIR = Ref("")

# cgs units
const G = 6.6743e-8
const Msun = 1.989e+33
const Rsun = 6.959894677e+10

const StructMatrix{T} = StructArray{T,2}

function __init__()
    SCRATCH[] = get(ENV, "SCRATCH", homedir())
    DATADIR[] = get(ENV, "DATADIR", joinpath(SCRATCH[], "RossbyWaves"))
    if !ispath(RossbyWaveSpectrum.DATADIR[])
        mkdir(RossbyWaveSpectrum.DATADIR[])
    end
    SOLARMODELDIR[] = joinpath(@__DIR__, "solarmodel")
    FFTW.set_num_threads(1)
end

struct FilteredEigen
    lams :: Vector{Vector{ComplexF64}}
    vs :: Vector{StructArray{ComplexF64, 2, NamedTuple{(:re, :im), NTuple{2,Matrix{Float64}}}, Int64}}
    mr :: UnitRange{Int}
    kw :: Dict{Symbol, Any}
    operators
    constraints::NamedTuple{(:BC, :ZC), NTuple{2,SparseArrays.SparseMatrixCSC{Float64, Int64}}}
end

Base.show(io::IO, f::FilteredEigen) = print(io, "Filtered eigen with m range = $(f.mr)")

m_index(Feig, m) = findfirst(==(m), Feig.mr)

function FilteredEigen(lams, vs, mr, kw, operators)
    constraints = constraintmatrix(operators)
    FilteredEigen(lams, vs, mr, kw, operators, constraints)
end

function rewrapstruct(vec::Vector{<:NTuple{2,AbstractMatrix{T}}}) where {T<:Real}
    [StructArray{Complex{T}}(t) for t in vec]
end
# for backward compatibility
rewrapstruct(vec::Vector{<:StructArray{<:Complex}}) = vec

function FilteredEigen(fname::String)
    isfile(fname) || throw(ArgumentError("Couldn't find $fname"))
    lam, vec, mr, kw, operatorparams =
        load(fname, "lam", "vec", "mr", "kw", "operatorparams");
    operators = radial_operators(operatorparams...)
    if haskey(kw, :filterflags)
        kw[:filterflags] = Filters.FilterFlag(kw[:filterflags])
    end
    FilteredEigen(lam, rewrapstruct(vec), mr, kw, operators)
end

function Base.getindex(Feig::FilteredEigen, m::Integer)
    mind = m_index(Feig, m)
    m in Feig.mr || throw(ArgumentError("m = $m is not contained in $Feig"))
    FilteredEigenSingleOrder(Feig.lams[mind], Feig.vs[mind], m, Feig.kw, Feig.operators, Feig.constraints)
end

function Base.getindex(Feig::FilteredEigen, mr::UnitRange{<:Integer})
    ((mr ∩ Feig.mr) == mr) || throw(ArgumentError("m range = $mr is not contained in $Feig"))
    minds = m_index(Feig, minimum(mr)):m_index(Feig, maximum(mr))
    FilteredEigen(Feig.lams[minds], Feig.vs[minds], mr, Feig.kw, Feig.operators, Feig.constraints)
end

struct FilteredEigenSingleOrder
    lams::Vector{ComplexF64}
    vs :: StructArray{ComplexF64, 2, NamedTuple{(:re, :im), NTuple{2,Matrix{Float64}}}, Int64}
    m :: Int
    kw :: Dict{Symbol, Any}
    operators
    constraints::NamedTuple{(:BC, :ZC), NTuple{2,SparseArrays.SparseMatrixCSC{Float64, Int64}}}
end

Base.iterate(F::FilteredEigenSingleOrder) = F.lams, Val(:vecs)
Base.iterate(F::FilteredEigenSingleOrder, ::Val{:vecs}) = F.vs, nothing
Base.iterate(F::FilteredEigenSingleOrder, ::Nothing) = nothing

eigvalspairs(f::FilteredEigenSingleOrder) = pairs(f.lams)

Base.show(io::IO, f::FilteredEigenSingleOrder) = print(io, "Filtered eigen with m = $(f.m)")

function Base.getindex(f::FilteredEigenSingleOrder, ind::Integer)
    (; λ = f.lams[ind], v = f.vs[:, ind])
end

struct RotMatrix{TV,TW,F}
    kw :: @NamedTuple{V_symmetric::Bool, rotation_profile::Symbol}
    ΔΩprofile_deriv :: TV
    ωΩ_deriv :: TW
    f :: F
end
function RotMatrix(V_symmetric::Bool, rotation_profile::Symbol, ΔΩprofile_deriv, ωΩ_deriv, matrixfn!)
    RotMatrix((; V_symmetric, rotation_profile), ΔΩprofile_deriv, ωΩ_deriv, matrixfn!)
end
function updaterotatationprofile(d::RotMatrix; operators, timer = TimerOutput(), kw...)
    if d.kw.rotation_profile == :uniform || d.kw.rotation_profile == :constant
        ΔΩprofile_deriv = @timeit timer "velocity" begin nothing end
        ωΩ_deriv = @timeit timer "vorticity" begin nothing end
    elseif startswith(String(d.kw.rotation_profile), "radial")
        ΔΩprofile_deriv = @timeit timer "velocity" begin
            rotation_profile = rotationtag(d.kw.rotation_profile)
            radial_differential_rotation_profile_derivatives_Fun(;
                operators, rotation_profile, kw...)
        end
        ωΩ_deriv = @timeit timer "vorticity" begin nothing end
    elseif startswith(String(d.kw.rotation_profile), "solar")
        rotation_profile = rotationtag(d.kw.rotation_profile)
        ΔΩprofile_deriv = @timeit timer "velocity" begin
            solar_differential_rotation_profile_derivatives_Fun(;
                operators, rotation_profile, kw...)
        end
        ωΩ_deriv = @timeit timer "vorticity" begin
            solar_differential_rotation_vorticity_Fun(; operators, ΔΩprofile_deriv)
        end
    else
        error("unknown rotation profile $(repr(d.kw.rotation_profile)),"*
            " must be one of :uniform, :constant, :radial_* or solar_*")
    end
    return RotMatrix(d.kw, ΔΩprofile_deriv, ωΩ_deriv, d.f)
end
updaterotatationprofile(d; kw...) = d

mergekw(_, kw) = kw
mergekw(f::RotMatrix, kw) = (; f.kw..., kw...)
(d::RotMatrix)(args...; kw...) = d.f(args...;
    ΔΩprofile_deriv = d.ΔΩprofile_deriv,
    ωΩ_deriv = d.ωΩ_deriv,
    d.kw...,
    kw...)

function RotMatrix(::Val{T}, V_symmetric, rotation_profile; operators, kw...) where {T}
    T ∈ (:spectrum, :matrix) || error("unknown code ", T)
    diffrot = rotation_profile != :uniform
    d = RotMatrix(V_symmetric, rotation_profile, nothing, nothing,
            if !diffrot
                T == :matrix ? uniform_rotation_matrix! : uniform_rotation_spectrum!
            else
                T == :matrix ? differential_rotation_matrix! : differential_rotation_spectrum!
            end
    )

    updaterotatationprofile(d; operators, kw...)
end

function updaterotatationprofile(F::FilteredEigen; kw...)
    @unpack operators = F
    @unpack V_symmetric, rotation_profile = F.kw
    RotMatrix(Val(:matrix), V_symmetric, rotation_profile; operators, kw...)
end

datadir(f) = joinpath(DATADIR[], f)

function sph_points(N)
    @assert N > 0
    M = 2 * N - 1
    return π / N * (0.5:(N-0.5)), 2π / M * (0:(M-1))
end

matrix_block(M::AbstractBlockMatrix, rowind, colind, nblocks = 3) = @view M[Block(rowind, colind)]
function matrix_block(M::StructMatrix{<:Complex}, rowind, colind, nblocks = 3)
    Mr = matrix_block(M.re, rowind, colind, nblocks)
    Mi = matrix_block(M.im, rowind, colind, nblocks)
    StructArray{eltype(M)}((Mr, Mi))
end
# function matrix_block(M::AbstractMatrix, rowind, colind, nblocks = 3)
#     nparams = div(size(M, 1), nblocks)
#     inds1 = (rowind - 1) * nparams .+ (1:nparams)
#     inds2 = (colind - 1) * nparams .+ (1:nparams)
#     inds = CartesianIndices((inds1, inds2))
#     @view M[inds]
# end

include("uniqueinterval.jl")

const Tmul = typeof(Multiplication(Fun()) * Derivative())
const Tplusinf = typeof(Multiplication(Fun()) + Derivative())

const BandedMatrixType = typeof(BandedMatrix(0=>Float64[]))

grid_to_fun(v::AbstractVector, space) = Fun(space, transform(space, v))
grid_to_fun(v::AbstractMatrix, space) = Fun(ProductFun(transform(space, v), space))

function V_boundary_op(operators)
    @unpack r = operators.rad_terms;
    @unpack ddr = operators.diff_operators;
    @unpack radialspace = operators.radialspaces

    (r * ddr - 2) : radialspace;
end

function constraintmatrix(operators, ::Val{extramatrices} = Val(false)) where {extramatrices}
    @unpack radial_params = operators;
    @unpack r_in, r_out = radial_params;
    @unpack nr, nℓ, Δr = operators.radial_params;
    @unpack r = operators.rad_terms;
    @unpack ddr = operators.diff_operators;
    @unpack radialspace = operators.radialspaces

    V_BC_op = V_boundary_op(operators);
    CV = [Evaluation(r_in) * V_BC_op; Evaluation(r_out) * V_BC_op];
    nradconstraintsVS = size(CV,1)::Int;
    MVn = Matrix(CV[:, 1:nr])::Matrix{Float64};
    QV = QuotientSpace(radialspace, CV);
    PV = Conversion(QV, radialspace);
    ZMVn = PV[1:nr, 1:nr-nradconstraintsVS]::BandedMatrixType;

    CW = [Dirichlet(radialspace); Dirichlet(radialspace, 1) * Δr/2];
    nradconstraintsW = size(CW,1);
    MWn = Matrix(CW[:, 1:nr]);
    QW = QuotientSpace(radialspace, CW);
    PW = Conversion(QW, radialspace);
    ZMWn = PW[1:nr, 1:nr-nradconstraintsW];

    CS = Dirichlet(radialspace, 1) * Δr/2;
    MSn = Matrix(CS[:, 1:nr]);
    QS = QuotientSpace(radialspace, CS);
    PS = Conversion(QS, radialspace);
    ZMSn = PS[1:nr, 1:nr-nradconstraintsVS];

    rowsB_VS = Fill(nradconstraintsVS, nℓ);
    rowsB_W = Fill(nradconstraintsW, nℓ);
    colsB = Fill(nr, nℓ);
    B_blocks = [blockdiagzero(rowsB_VS, colsB), blockdiagzero(rowsB_W, colsB), blockdiagzero(rowsB_VS, colsB)]

    BC_block = mortar(Diagonal(B_blocks))

    rowsZ = Fill(nr, nℓ)
    colsZ_VS = Fill(nr - nradconstraintsVS, nℓ)
    colsZ_W = Fill(nr - nradconstraintsW, nℓ)
    Z_blocks = [blockdiagzero(rowsZ, colsZ_VS), blockdiagzero(rowsZ, colsZ_W), blockdiagzero(rowsZ, colsZ_VS)]
    ZC_block = mortar(Diagonal(Z_blocks))

    BCmatrices = [MVn, MWn, MSn]

    for (Mind, M) in enumerate(BCmatrices)
        BCi = B_blocks[Mind]
        for ℓind = 1:nℓ
            BCi[Block(ℓind, ℓind)] = M
        end
    end
    nullspacematrices = [ZMVn, ZMWn, ZMSn]
    for (Zind, Z) in enumerate(nullspacematrices)
        ZCi = Z_blocks[Zind]
        for ℓind = 1:nℓ
            ZCi[Block(ℓind, ℓind)] = Z
        end
    end

    BC = computesparse(BC_block)
    ZC = computesparse(ZC_block)

    extramatrices ? (; BC, ZC, nullspacematrices, BCmatrices) : (; BC, ZC)
end

function parameters(nr, nℓ; r_in = 0.7Rsun, r_out = 0.98Rsun)
    nchebyr = nr
    r_mid = (r_in + r_out) / 2
    Δr = r_out - r_in
    nparams = nchebyr * nℓ
    return (; nchebyr, r_in, r_out, Δr, nr, nparams, nℓ, r_mid)
end

const SuperAdiabaticityParamsDefault = pairs((; δcz = 3e-6, δtop = 3e-5, δrad = -1e-3,
                                        dtrans = 0.05Rsun, dtop = 0.05Rsun,
                                        r_sub = 0.8 * Rsun, r_tran = 0.725 * Rsun))

function superadiabaticity(r::Real; r_out = Rsun, kw...)
    kw2 = merge(SuperAdiabaticityParamsDefault, kw)

    @unpack δcz = kw2
    @unpack δtop = kw2
    @unpack δrad = kw2
    @unpack dtrans = kw2
    @unpack dtop = kw2
    @unpack r_sub = kw2
    @unpack r_tran = kw2

    δconv = δtop * exp((r - r_out) / dtop) + δcz * (r - r_sub) / (r_out - r_sub)
    δconv + (δrad - δconv) * 1 / 2 * (1 - tanh((r - r_tran) / dtrans))
end

read_solar_model() = readdlm(joinpath(SOLARMODELDIR[], "ModelS.detailed"))::Matrix{Float64}

function solar_structure_parameter_splines(; r_in = 0.7Rsun, r_out = Rsun, _stratified #= only for tests =# = true)
    ModelS = read_solar_model();
    r_modelS = @view ModelS[:, 1];
    r_inds = min(r_in, 0.5Rsun) .<= r_modelS .<= max(r_out, 0.995Rsun);
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
    sηρ_by_r = smoothed_spline(r_modelS, ddrlogρ_modelS ./ r_modelS, s = 1e-4);
    sηρ_by_r2 = smoothed_spline(r_modelS, ddrlogρ_modelS ./ r_modelS.^2, s = 1e-4);
    ddrsηρ = smoothed_spline(r_modelS, Dierckx.derivative(sηρ, r_modelS), s = 1e-4);
    ddrsηρ_by_r = smoothed_spline(r_modelS, Dierckx.derivative(sηρ_by_r, r_modelS), s = 1e-4);
    ddrsηρ_by_r2 = smoothed_spline(r_modelS, Dierckx.derivative(sηρ_by_r2, r_modelS), s = 1e-4);
    d2dr2sηρ = smoothed_spline(r_modelS, Dierckx.derivative(ddrsηρ, r_modelS), s = 1e-4);
    d3dr3sηρ = smoothed_spline(r_modelS, Dierckx.derivative(ddrsηρ, r_modelS, nu=2), s = 1e-4);

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


iszerofun(v) = ncoefficients(v) == 0 || (ncoefficients(v) == 1 && coefficients(v)[] == 0.0)
function replaceemptywitheps(f::Fun, eps = 1e-100)
    T = eltype(coefficients(f))
    iszerofun(f) ? typeof(f)(space(f), T[eps]) : f
end

function checkncoeff(v, vname, nr)
    if ncoefficients(v) > 2/3*nr
        @debug "number of coefficients in $vname is $(ncoefficients(v)), but nr = $nr"
    end
    return nothing
end
macro checkncoeff(v, nr)
    :(checkncoeff($(esc(v)), $(String(v)), $(esc(nr))))
end

struct OperatorWrap{T}
    x::T
end
Base.show(io::IO, o::OperatorWrap) = print(io, "Operators")
Base.getproperty(y::OperatorWrap, name::Symbol) = getproperty(getfield(y, :x), name)
Base.propertynames(y::OperatorWrap) = Base.propertynames(getfield(y, :x))

const DefaultScalings = (; Wscaling = 1e1, Sscaling = 1e3, Weqglobalscaling = 1e-4, Seqglobalscaling = 1.0, trackingratescaling = 1.0)
function radial_operators(nr, nℓ;
        r_in_frac = 0.6, r_out_frac = 0.985,
        _stratified = true,
        nvariables = 3,
        ν = 5e11,
        trackingrate = :hanson2020,
        scalings = DefaultScalings,
        superadiabaticityparams = (;))

    scalings = merge(DefaultScalings, scalings)
    radial_operators(nr, nℓ, r_in_frac, r_out_frac,
        _stratified, nvariables, ν, trackingrate, Tuple(scalings),
        pairs(superadiabaticityparams),
        )
end
function radial_operators(operatorparams...)
    nr, nℓ, r_in_frac, r_out_frac, _stratified,
        nvariables, ν, trackingrate, _scalings = operatorparams

    superadiabaticityparams = merge(SuperAdiabaticityParamsDefault, get(operatorparams, 10, pairs((;))))

    Wscaling, Sscaling, Weqglobalscaling, Seqglobalscaling, trackingratescaling = _scalings

    r_in = r_in_frac * Rsun;
    r_out = r_out_frac * Rsun;
    radialdomain = UniqueInterval(r_in..r_out)
    radialspace = Space(radialdomain)
    radial_params = parameters(nr, nℓ; r_in, r_out);
    @unpack Δr, nchebyr, r_mid = radial_params;
    rpts = points(radialspace, nr);
    rptsrev = reverse(rpts);

    r = Fun(radialspace);
    r2 = r^2;
    r3 = r^3;
    r4 = r2^2;

    @unpack splines = solar_structure_parameter_splines(; r_in, r_out, _stratified);

    @unpack sg, sρ, sηρ, ddrsηρ, d2dr2sηρ,
        sηρ_by_r, ddrsηρ_by_r, ddrsηρ_by_r2, d3dr3sηρ, sηT = splines;

    ddr = ApproxFun.Derivative();
    rddr = (r * ddr)::Tmul;
    d2dr2 = ApproxFun.Derivative(2);
    d3dr3 = ApproxFun.Derivative(3);
    d4dr4 = ApproxFun.Derivative(4);
    r2d2dr2 = (r2 * d2dr2)::Tmul;

    radialspace_D2 = rangespace(d2dr2 : radialspace)
    radialspace_D4 = rangespace(d4dr4 : radialspace)

    n_uppercutoff = 200

    # density stratification
    ρ = replaceemptywitheps(ApproxFun.chop!(Fun(sρ, radialspace, n_uppercutoff), 1e-3));
    @checkncoeff ρ nr

    ηρ = replaceemptywitheps(ApproxFun.chop!(Fun(sηρ, radialspace, n_uppercutoff), 1e-3));
    @checkncoeff ηρ nr

    ηT = replaceemptywitheps(ApproxFun.chop!(Fun(sηT, radialspace, n_uppercutoff), 1e-3));
    @checkncoeff ηT nr

    ηρT = ηρ + ηT

    DDr = (ddr + ηρ)::Tplusinf
    rDDr = (r * DDr)::Tmul

    onebyr = (\(Multiplication(r), 1, tolerance=1e-8))::typeof(r)
    twobyr = 2onebyr
    onebyr2 = (\(Multiplication(r2), 1, tolerance=1e-8))::typeof(r)
    onebyr3 = (\(Multiplication(r3), 1, tolerance=1e-8))::typeof(r)
    onebyr4 = (\(Multiplication(r4), 1, tolerance=1e-8))::typeof(r)
    DDr_minus_2byr = (DDr - twobyr)::Tplusinf
    ddr_plus_2byr = (ddr + twobyr)::Tplusinf

    # ηρ_by_r = onebyr * ηρ
    ηρ_by_r = ηρ * onebyr;
    @checkncoeff ηρ_by_r nr

    ηρ_by_r2 = ηρ * onebyr2
    @checkncoeff ηρ_by_r2 nr

    ηρ_by_r3 = ηρ_by_r2 * onebyr
    @checkncoeff ηρ_by_r3 nr

    ddr_ηρ = chop!(Fun(ddrsηρ, radialspace, n_uppercutoff), 1e-3);
    @checkncoeff ddr_ηρ nr

    ddr_ηρbyr = chop!(Fun(ddrsηρ_by_r, radialspace, n_uppercutoff), 1e-3);
    @checkncoeff ddr_ηρbyr nr

    # ddr_ηρbyr = ddr * ηρ_by_r
    d2dr2_ηρ = chop!(Fun(d2dr2sηρ, radialspace, n_uppercutoff), 1e-3);
    @checkncoeff d2dr2_ηρ nr

    d3dr3_ηρ = chop!(Fun(d3dr3sηρ, radialspace, n_uppercutoff), 1e-3);
    @checkncoeff d3dr3_ηρ nr

    ddr_ηρbyr2 = chop!(Fun(ddrsηρ_by_r2, radialspace, n_uppercutoff), 1e-3);
    @checkncoeff ddr_ηρbyr2 nr

    # ddr_ηρbyr2 = ddr * ηρ_by_r2
    ηρ2_by_r2 = ApproxFun.chop!(ηρ_by_r2 * ηρ, 1e-3);
    @checkncoeff ηρ2_by_r2 nr

    ddrηρ_by_r = ddr_ηρ * onebyr
    d2dr2ηρ_by_r = d2dr2_ηρ * onebyr

    ddrDDr = (d2dr2 + (ηρ * ddr)::Tmul + ddr_ηρ)::Tplusinf
    d2dr2DDr = (d3dr3 + (ηρ * d2dr2)::Tmul + (2ddr_ηρ * ddr)::Tmul + d2dr2_ηρ)::Tplusinf

    d2dr2_ηρbyr_op::Tplusinf = ηρ_by_r * d2dr2 + 2(ddr_ηρ * onebyr - ηρ * onebyr2)*ddr +
        (d2dr2_ηρ * onebyr - 2ddr_ηρ * onebyr2 + 2ηρ * onebyr3)

    g = Fun(sg, radialspace)

    Ω0::Float64 = if trackingrate isa Number
            trackingrate * 2pi
        elseif trackingrate === :cutoff
            equatorial_rotation_angular_velocity_surface(r_out_frac)
        elseif trackingrate === :surface
            equatorial_rotation_angular_velocity_surface(1.0)
        elseif trackingrate === :carrington
            456e-9 * 2pi
        elseif trackingrate === :hanson2020
            453.1e-9 * 2pi
        else
            throw(ArgumentError("trackingrate must be a nHz value, or one of :carrington, :cutoff, :hanson2020 or :surface"))
        end

    Ω0 *= trackingratescaling

    # viscosity
    ν /= Ω0 * Rsun^2
    κ = ν

    # δ = Fun(r -> superadiabaticity(r; superadiabaticityparams...), radialspace);
    γ = 1.64
    Cp = 1.7e8

    # ddr_S0_by_Cp = γ/Cp * δ * ηρ;
    ddr_S0_by_Cp = chop!(
        Fun(x -> γ / Cp * superadiabaticity(x; superadiabaticityparams...) * ηρ(x),
            radialspace), 1e-3);
    @checkncoeff ddr_S0_by_Cp nr

    ddr_S0_by_Cp_by_r2 = onebyr2 * ddr_S0_by_Cp

    # uniform rotation terms
    ∇r2_plus_ddr_lnρT_ddr = (d2dr2 + (twobyr * ddr)::Tmul + (ηρT * ddr)::Tmul)::Tplusinf;

    # terms for viscosity
    ddr_minus_2byr = (ddr - twobyr)::Tplusinf;

    scalings = Dict{Symbol, Float64}()
    @pack! scalings = Sscaling, Wscaling, Weqglobalscaling, Seqglobalscaling, trackingratescaling

    constants = (; κ, ν, Ω0) |> pairs |> Dict

    rad_terms = Dict{Symbol, typeof(r)}();
    @pack! rad_terms = onebyr, twobyr, ηρ, ηT,
        onebyr2, onebyr3, onebyr4,
        ηρT, ρ, g, r, r2,
        ηρ_by_r, ηρ_by_r2, ηρ2_by_r2, ddr_ηρbyr, ddr_ηρbyr2, ηρ_by_r3,
        ddr_ηρ, d2dr2_ηρ, d3dr3_ηρ, ddr_S0_by_Cp_by_r2, ddr_S0_by_Cp,
        ddrηρ_by_r, d2dr2ηρ_by_r

    diff_operators = (; DDr, DDr_minus_2byr, rDDr, rddr, ddrDDr, d2dr2DDr,
        ddr, d2dr2, d3dr3, d4dr4, r2d2dr2, ddr_plus_2byr, ∇r2_plus_ddr_lnρT_ddr,
        d2dr2_ηρbyr_op)

    radialspaces = (; radialspace, radialspace_D2, radialspace_D4)

    op = (;
        radialdomain,
        radialspaces,
        nvariables,
        constants,
        rad_terms,
        scalings,
        splines,
        diff_operators,
        rpts,
        rptsrev,
        radial_params,
        operatorparams,
    )

    OperatorWrap(op)
end

function listncoefficients(d)
    sort([k=>ncoefficients(v) for (k,v) in Dict(pairs(d))], by=last, rev=true)
end

function computesparse(M::StructMatrix{<:Complex})
    SR = computesparse(M.re)
    SI = computesparse(M.im)
    StructArray{eltype(M)}((SR, SI))
end

function computesparse(M::BlockMatrix)
    v = [sparse(M[j, i]) for (i,j) in Iterators.product(blockaxes(M)...)]
    TS = SparseMatrixCSC{eltype(M),Int}
    if blocksize(M, 2) == 3
        hvcat((3,3,3), v...)::TS
    elseif blocksize(M, 2) == 2
        hvcat((2,2), v...)::TS
    else
        error("unsupported block size")
    end
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

function allocate_operator_matrix(operators, bandwidth = operators.radial_params[:nℓ])
    @unpack nr, nℓ = operators.radial_params
    @unpack nvariables = operators
    rows = Fill(nr, nℓ) # block sizes
    R = allocate_block_matrix(nvariables, bandwidth, rows)
    I = allocate_block_matrix(nvariables, 0, rows)
    StructArray{ComplexF64}((R, I))
end

function allocate_mass_matrix(operators)
    @unpack nr, nℓ = operators.radial_params
    @unpack nvariables = operators
    rows = Fill(nr, nℓ) # block sizes
    allocate_block_matrix(nvariables, 0, rows)
end

allocate_operator_mass_matrices(operators, bw...) = (
    allocate_operator_matrix(operators, bw...),
    allocate_mass_matrix(operators))

function smoothed_spline(rpts, v; s = 0.0)
    if issorted(rpts, rev=true)
        rpts, v = reverse(rpts), reverse(v)
    end
    Spline1D(rpts, v, s = sum(abs2, v) * s)
end
function interp1d(xin, z, xout = xin; s = 0.0)
    p = sortperm(xin)
    spline = smoothed_spline(xin[p], z[p]; s)
    spline(xout)
end

# function interp2d(xin, yin, z, xout = xin, yout = yin; s = 0.0)
#     px = sortperm(xin)
#     py = sortperm(yin)
#     spline = Spline2D(xin[px], yin[py], z[px, py]; s = sum(abs2, z) * s)
#     spline.(xout, yout')
# end

function solar_rotation_profile_angles(Ω_raw, maxcolatitude = pi)
    nθ = size(Ω_raw, 2)
    lats_raw = range(0, maxcolatitude, length=nθ)
end

function solar_rotation_profile_radii(dir = SOLARMODELDIR[])
    r_ΔΩ_raw = vec(readdlm(joinpath(dir, "rmesh.orig")))::Vector{Float64}
    r_ΔΩ_raw[1:4:end]
end

function solar_rotation_profile_raw_hemisphere(dir = SOLARMODELDIR[])
    readdlm(joinpath(dir, "rot2d.hmiv72d.ave"))::Matrix{Float64}
end

function solar_rotation_profile_raw(dir = SOLARMODELDIR[])
    ν_raw = solar_rotation_profile_raw_hemisphere(dir)
    ν_raw = [ν_raw reverse(ν_raw[:, 1:end-1], dims = 2)]
    2pi * 1e-9 * ν_raw
end

findnearest(haystack, needle) = findmin(abs, haystack .- needle)[2]

function equatorial_rotation_angular_velocity_surface(r_frac::Number = 1.0,
        r_ΔΩ_raw = solar_rotation_profile_radii(), Ω_raw = solar_rotation_profile_raw())
    Ω_raw_r_eq = equatorial_rotation_angular_velocity_radial_profile(Ω_raw)
    r_frac_ind = findnearest(r_ΔΩ_raw, r_frac)
    Ω_raw_r_eq[r_frac_ind]
end

function equatorial_rotation_angular_velocity_radial_profile(Ω_raw = solar_rotation_profile_raw())
    lats_raw = solar_rotation_profile_angles(Ω_raw)
    θind_equator = findnearest(lats_raw, pi/2)
    Ω_raw[:, θind_equator]
end

function smooth_poles_latitude(Ω_raw, r_ΔΩ_raw, lats_raw, lat_cutoff_deg = 20, lat_cutoff_scale_deg = 4)
    lat_cutoff_rad = deg2rad(lat_cutoff_deg)
    lat_cutoff_rad_southpole = deg2rad(180 - lat_cutoff_deg)
    lat_cutoff_ind = findfirst(>=(lat_cutoff_rad), lats_raw)
    lat_cutoff_scale_rad = deg2rad(lat_cutoff_scale_deg)
    smoothing_profile = (1 .+ tanh.((lats_raw .- lat_cutoff_rad)./lat_cutoff_scale_rad))./2 .*
                            (1 .+ tanh.((lat_cutoff_rad_southpole .- lats_raw)./lat_cutoff_scale_rad))./2

    Ω_smoothed = similar(Ω_raw)
    for (Ω_lat_profile_smoothed, Ω_lat_profile) in zip(eachrow(Ω_smoothed), eachrow(Ω_raw))
        Ω_cutoff_lat = Ω_lat_profile[lat_cutoff_ind]
        Ω_lat_profile_smoothed .= Ω_cutoff_lat .+ (Ω_lat_profile .- Ω_cutoff_lat).*smoothing_profile
    end
    Ω_smoothed
end

function smooth_poles_radius(Ω_raw, r_ΔΩ_raw, lats_raw, lat_cutoff_deg = 20)
    nr = size(Ω_raw, 1)
    Ω_smoothed = copy(Ω_raw)
    lat_cutoff_rad_northpole = deg2rad(lat_cutoff_deg)
    lat_cutoff_rad_southpole = deg2rad(180 - lat_cutoff_deg)
    lat_cutoff_ind_northpole = findfirst(>=(lat_cutoff_rad_northpole), lats_raw)
    lat_cutoff_ind_southpole = findlast(<(lat_cutoff_rad_southpole), lats_raw)
    lat_cutoff_inds_northpole = 1:lat_cutoff_ind_northpole-1
    lat_cutoff_inds_southpole = lat_cutoff_ind_southpole+1:size(Ω_raw,2)
    r_flatind = findfirst(>(0.8), r_ΔΩ_raw)
    rad_smoothing_inds = r_flatind:nr
    Ω_ref_profile = @view Ω_raw[rad_smoothing_inds, lat_cutoff_ind_northpole+1]
    Ω_ref = Ω_ref_profile[1]
    for lat_inds in (lat_cutoff_inds_northpole, lat_cutoff_inds_southpole)
        Ω_smoothed_section = view(Ω_smoothed, rad_smoothing_inds, lat_inds)
        Ω_section = view(Ω_raw, rad_smoothing_inds, lat_inds)
        for (col_smoothed, col) in zip(eachcol(Ω_smoothed_section), eachcol(Ω_section))
            Ω_smoothed_section .= (col[1]/Ω_ref) .* Ω_ref_profile
        end
    end
    return Ω_smoothed
end

function solar_rotation_profile_smoothed(Ω_raw = solar_rotation_profile_raw(SOLARMODELDIR[]); operators, kw...)
    r_ΔΩ_raw = solar_rotation_profile_radii(SOLARMODELDIR[])
    lats_raw = solar_rotation_profile_angles(Ω_raw)
    if get(kw, :smooth_poles, true)
        Ω_raw = smooth_poles_latitude(Ω_raw, r_ΔΩ_raw, lats_raw)
        Ω_raw = smooth_poles_radius(Ω_raw, r_ΔΩ_raw, lats_raw)
    end
    Ω_raw
end

function solar_diffrot_profile_smoothed(; operators, kw...)
    Ω_raw = solar_rotation_profile_smoothed(; operators, kw...)
    @unpack Ω0 = operators.constants;
    Ω_raw .- Ω0
end

function solar_diffrot_profile_spline(; operators, smoothing_param = 1e-4, kw...)
    r_ΔΩ_raw = solar_rotation_profile_radii(SOLARMODELDIR[])
    ΔΩ = solar_diffrot_profile_smoothed(; operators, kw...)
    lats_raw = solar_rotation_profile_angles(ΔΩ)

    cos_lats_raw = cos.(lats_raw) # decreasing order, must be flipped in Spline2D

    Spline2D(r_ΔΩ_raw * Rsun, reverse(cos_lats_raw), reverse(ΔΩ, dims=2); s = sum(abs2, ΔΩ)*smoothing_param)
end

function solar_rotation_profile_and_derivative_grid(splΔΩ2D, rpts, θpts)
    ΔΩ_fullinterp = splΔΩ2D.(rpts, θpts')
    dr_ΔΩ_fullinterp = derivative.((splΔΩ2D,), rpts, θpts', nux = 1, nuy = 0)
    d2r_ΔΩ_fullinterp = derivative.((splΔΩ2D,), rpts, θpts', nux = 2, nuy = 0)
    ΔΩ_fullinterp, dr_ΔΩ_fullinterp, d2r_ΔΩ_fullinterp
end
function maybe_stretched_radius(; operators, squished = false)
    @unpack rpts = operators
    maybe_stretched_radius(rpts; operators, squished)
end
function maybe_stretched_radius(rpts; operators, squished = false)
    @unpack r_out, r_in = operators.radial_params
    if squished
        r_in .+ (rpts .- r_in) .* ((Rsun - r_in) / (r_out - r_in))
    else
        rpts
    end
end

function maybe_squeeze_radius(rpts; operators, squished = false)
    @unpack r_out, r_in = operators.radial_params
    if squished
        r_in .+ (rpts .- r_in) * ((r_out - r_in)/(Rsun - r_in))
    else
        rpts
    end
end

function solar_rotation_profile_and_derivative_grid(; squished = false, operators, kw...)
    @unpack nℓ = operators.radial_params
    θpts = points(ChebyshevInterval(), nℓ)
    splΔΩ2D = solar_diffrot_profile_spline(; operators, kw...)
    rpts_maybestretched = maybe_stretched_radius(; operators, squished)
    solar_rotation_profile_and_derivative_grid(splΔΩ2D, rpts_maybestretched, θpts)
end

function solar_surface_rotation_profile_and_derivative_grid(; squished = false, operators, kw...)
    @unpack nℓ, r_out = operators.radial_params
    θpts = points(ChebyshevInterval(), nℓ)
    splΔΩ2D = solar_diffrot_profile_spline(; operators, kw...)
    r_out_maybesquished = maybe_stretched_radius(r_out; operators, squished)
    splΔΩ2D.(r_out_maybesquished, θpts')
end

function _equatorial_rotation_profile_and_derivative_grid(splΔΩ2D, rpts)
    equator_coord = 0.0 # cos(θ) for θ = pi/2
    ΔΩ_r = splΔΩ2D.(rpts, equator_coord)
    ddrΔΩ_r = derivative.((splΔΩ2D,), rpts, equator_coord, nux = 1, nuy = 0)
    d2dr2ΔΩ_r = derivative.((splΔΩ2D,), rpts, equator_coord, nux = 2, nuy = 0)
    ΔΩ_r, ddrΔΩ_r, d2dr2ΔΩ_r
end
function equatorial_rotation_profile_and_derivative_grid(; squished = false, operators, kw...)
    @unpack rpts = operators
    splΔΩ2D = solar_diffrot_profile_spline(; operators, kw...)
    rpts_maybestretched = maybe_stretched_radius(; operators, squished)
    _equatorial_rotation_profile_and_derivative_grid(splΔΩ2D, rpts_maybestretched)
end

function radial_differential_rotation_profile_derivatives_grid(;
            operators, rotation_profile = :solar_equator, ΔΩ_frac = 0.01,
            ΔΩ_scale = 1.0,
            kw...)

    @unpack rpts = operators
    @unpack r_out, nr, r_in = operators.radial_params
    @unpack Ω0 = operators.constants

    if rotation_profile == :solar_equator
        ΔΩ_r, ddrΔΩ_r, d2dr2ΔΩ_r =
            equatorial_rotation_profile_and_derivative_grid(; operators, kw...)
    elseif rotation_profile == :solar_equator_squished
        ΔΩ_r, ddrΔΩ_r, d2dr2ΔΩ_r =
            equatorial_rotation_profile_and_derivative_grid(; squished = true, operators, kw...)
    elseif rotation_profile == :linear # for testing
        f = ΔΩ_frac / (r_in / Rsun - 1)
        ΔΩ_r = @. Ω0 * f * (rpts / Rsun - 1)
        ddrΔΩ_r = fill(Ω0 * f / Rsun, nr)
        d2dr2ΔΩ_r = zero(ΔΩ_r)
    elseif rotation_profile == :constant # for testing
        ΔΩ_r = fill(ΔΩ_frac * Ω0, nr)
        ddrΔΩ_r = zero(ΔΩ_r)
        d2dr2ΔΩ_r = zero(ΔΩ_r)
    elseif rotation_profile == :core
        pre = (Ω0*ΔΩ_frac)*5
        σr = 0.08Rsun
        r0 = 0.6Rsun
        ΔΩ_r = @. pre * (1 - tanh((rpts - r0)/σr))
        ddrΔΩ_r = @. pre * (-sech((rpts - r0)/σr)^2 * 1/σr)
        d2dr2ΔΩ_r = @. pre * (2sech((rpts - r0)/σr)^2 * tanh((rpts - r0)/σr) * 1/σr^2)
    elseif rotation_profile == :solar_equator_core
        ΔΩ_r_sun, = equatorial_radial_rotation_profile(; operators, kw...)
        σr = 0.08Rsun
        r0 = 0.6Rsun
        ΔΩ_r_core = maximum(abs, ΔΩ_r_sun)/5 * @. (1 - tanh((rpts - r0)/σr))/2
        r_cutoff = 0.4Rsun
        Δr_cutoff = 0.1Rsun
        r_in_inds = rpts .<= (r_cutoff-Δr_cutoff)
        r_out_inds = rpts .>= (r_cutoff+Δr_cutoff)
        r_in = rpts[r_in_inds]
        r_out = rpts[r_out_inds]
        perminds_in = sortperm(r_in)
        perminds_out = sortperm(r_out)
        r_new = [r_in[perminds_in]; r_out[perminds_out]]
        ΔΩ_r_new = [ΔΩ_r_core[r_in_inds][perminds_in]; ΔΩ_r_sun[r_out_inds][perminds_out]]
        ΔΩ_spl = smoothed_spline(r_new, ΔΩ_r_new; s = get(kw, :smoothing_param, 1e-4))
        ΔΩ_r = ΔΩ_spl(rpts)
        ddrΔΩ_r = derivative.((ΔΩ_spl,), rpts)
        d2dr2ΔΩ_r = derivative.((ΔΩ_spl,), rpts, nu=2)
    else
        error("$rotation_profile is not a valid rotation model")
    end
    ΔΩ_r .*= ΔΩ_scale/Ω0;
    ddrΔΩ_r .*= ΔΩ_scale/Ω0;
    d2dr2ΔΩ_r .*= ΔΩ_scale/Ω0;
    return ΔΩ_r, ddrΔΩ_r, d2dr2ΔΩ_r
end

function radial_differential_rotation_profile_derivatives_Fun(; operators, kw...)
    @unpack rpts, radialspaces = operators;
    @unpack radialspace = radialspaces
    @unpack ddr, d2dr2 = operators.diff_operators;
    ΔΩ_terms = radial_differential_rotation_profile_derivatives_grid(; operators, kw...);
    ΔΩ_r, ddrΔΩ_r, d2dr2ΔΩ_r = ΔΩ_terms
    nr = length(ΔΩ_r)

    ΔΩ = chop(grid_to_fun(ΔΩ_r, radialspace), 1e-2);
    @checkncoeff ΔΩ nr

    # ddrΔΩ = chop(grid_to_fun(interp1d(rpts, ddrΔΩ_r, s = 1e-3), radialspace), 1e-3);
    ddrΔΩ_ = ddr * ΔΩ
    ddrΔΩ = Fun(ddrΔΩ_, radialspace)
    @checkncoeff ddrΔΩ nr

    # d2dr2ΔΩ = chop(grid_to_fun(interp1d(rpts, d2dr2ΔΩ_r, s = 1e-3), radialspace), 1e-3);
    d2dr2ΔΩ_ = d2dr2 * ΔΩ
    d2dr2ΔΩ = Fun(d2dr2ΔΩ_, radialspace)
    @checkncoeff d2dr2ΔΩ nr

    ΔΩ, ddrΔΩ, d2dr2ΔΩ = map(replaceemptywitheps, promote(ΔΩ, ddrΔΩ, d2dr2ΔΩ))
    (; ΔΩ, ddrΔΩ, d2dr2ΔΩ)
end

# This function lets us choose between various different profiles
function solar_differential_rotation_profile_derivatives_grid(;
        operators, rotation_profile = :latrad, ΔΩ_frac = 0.01,
        ΔΩ_scale = 1.0,
        kw...)

    @unpack nℓ, nr = operators.radial_params
    @unpack Ω0 = operators.constants
    θpts = points(ChebyshevInterval(), nℓ)
    nθ = length(θpts)

    if rotation_profile == :constant
        ΔΩ = fill(Ω0 * ΔΩ_frac, nr, nθ)
        dr_ΔΩ, d2r_ΔΩ = (zeros(nr, nθ) for i in 1:2)
    elseif rotation_profile ∈ (:radial_equator, :radial_equator_squished)
        ΔΩ_r_, ddrΔΩ_r_, d2dr2ΔΩ_r_ = radial_differential_rotation_profile_derivatives_grid(;
            operators, rotation_profile = :solar_equator,
            squished = rotation_profile == :radial_equator_squished, kw...)
        for x in (ΔΩ_r_, ddrΔΩ_r_, d2dr2ΔΩ_r_)
            x .*= Ω0
        end
        ΔΩ = repeat(ΔΩ_r_, 1, nℓ)
        dr_ΔΩ = repeat(ddrΔΩ_r_, 1, nℓ)
        d2r_ΔΩ = repeat(d2dr2ΔΩ_r_, 1, nℓ)
    elseif rotation_profile ∈ (:surface, :surface_squished)
        ΔΩ_θ =
            solar_surface_rotation_profile_and_derivative_grid(; operators,
                squished = rotation_profile == :surface_squished,
                kw...)
        ΔΩ = repeat(ΔΩ_θ, nr, 1)
        dr_ΔΩ, d2r_ΔΩ = (zeros(nr, nθ) for i in 1:2)
    elseif rotation_profile ∈ (:latrad, :latrad_squished)
        ΔΩ, dr_ΔΩ, d2r_ΔΩ =
            solar_rotation_profile_and_derivative_grid(; operators,
                squished = rotation_profile == :latrad_squished,
                kw...)
    else
        error("$rotation_profile is not a valid rotation model")
    end
    for v in (ΔΩ, dr_ΔΩ, d2r_ΔΩ)
        v .*= ΔΩ_scale/Ω0
    end
    return ΔΩ, dr_ΔΩ, d2r_ΔΩ
end

function solar_differential_rotation_profile_derivatives_Fun(; operators, kw...)
    @unpack rpts, radialspaces = operators;
    @unpack radialspace = radialspaces
    @unpack onebyr = operators.rad_terms;
    @unpack nℓ = operators.radial_params;
    @unpack ddr, d2dr2 = operators.diff_operators;

    ΔΩ_terms = solar_differential_rotation_profile_derivatives_grid(; operators, kw...);
    ΔΩ_rθ, dr_ΔΩ_rθ, d2r_ΔΩ_rθ = ΔΩ_terms;

    # Symmetrize in θ
    ΔΩ_rθ = (reverse(ΔΩ_rθ, dims=2) .+ ΔΩ_rθ) ./ 2
    dr_ΔΩ_rθ = (reverse(dr_ΔΩ_rθ, dims=2) .+ dr_ΔΩ_rθ) ./ 2
    d2r_ΔΩ_rθ = (reverse(d2r_ΔΩ_rθ, dims=2) .+ d2r_ΔΩ_rθ) ./ 2

    angularsp = Ultraspherical(Legendre())
    space2d = radialspace ⊗ angularsp

    cosθ = Fun(angularsp);
    Ir = I : radialspace;
    dcosθ = Derivative(angularsp)
    sinθdθop = -(1-cosθ^2)*dcosθ

    ΔΩ_smoothing_param = get(kw, :ΔΩ_smoothing_param, 5e-2)
    ΔΩ = chop!(grid_to_fun(ΔΩ_rθ, space2d), ΔΩ_smoothing_param);

    dr_ΔΩ_ = (ddr ⊗ I) * ΔΩ;
    dr_ΔΩ = Fun(dr_ΔΩ_, space2d)
    if ncoefficients(dr_ΔΩ) > 0
        c = coefficients(ProductFun(dr_ΔΩ))
        c[:, 2:2:end] .= 0 # odd terms
        dr_ΔΩ = Fun(ProductFun(c, factors(space2d)...))
    end

    onebyr_sinθdθ = KroneckerOperator(Multiplication(onebyr, radialspace), sinθdθop,
                radialspace ⊗ domainspace(dcosθ), radialspace ⊗ rangespace(dcosθ))
    dz_ΔΩ_ = (Ir ⊗ cosθ) * dr_ΔΩ - onebyr_sinθdθ * ΔΩ;
    dz_ΔΩ = chop!(Fun(dz_ΔΩ_, space2d));

    # d2r_ΔΩ = chop(grid_to_fun(interp2d(rpts, θpts, d2r_ΔΩ_rθ, s = s), space2d), 1e-2);
    d2r_ΔΩ_ = (d2dr2 ⊗ I) * ΔΩ
    d2r_ΔΩ = Fun(d2r_ΔΩ_, space2d);
    if ncoefficients(d2r_ΔΩ) > 0
        c = coefficients(ProductFun(d2r_ΔΩ))
        c[:, 2:2:end] .= 0 # odd terms
        d2r_ΔΩ = Fun(ProductFun(c, factors(space2d)...))
    end

    ΔΩ, dr_ΔΩ, d2r_ΔΩ, dz_ΔΩ =
        map(replaceemptywitheps, promote(ΔΩ, dr_ΔΩ, d2r_ΔΩ, dz_ΔΩ))

    (; ΔΩ, dr_ΔΩ, d2r_ΔΩ, dz_ΔΩ)
end

ℓrange(m, nℓ, symmetric) = range(m + !symmetric, length = nℓ, step = 2)

function mass_matrix(m; operators, kw...)
    B = allocate_mass_matrix(operators)
    mass_matrix!(B, m; operators, kw...)
    return B
end
function mass_matrix!(B, m; operators, V_symmetric, kw...)
    @unpack nr, nℓ = operators.radial_params;
    @unpack Weqglobalscaling = operators.scalings;

    @unpack onebyr2 = operators.rad_terms;
    @unpack ddrDDr = operators.diff_operators;
    @unpack radialspaces = operators;
    @unpack radialspace, radialspace_D2, radialspace_D4 = radialspaces

    latitudinal_space = NormalizedPlm(m);

    Ir = ConstantOperator(I);
    Iℓ = I : latitudinal_space;
    space2d = radialspace ⊗ latitudinal_space;

    ∇² = HorizontalLaplacian(latitudinal_space);
    ℓℓp1op = -∇²;

    B .= 0

    VV = matrix_block(B, 1, 1)
    WW = matrix_block(B, 2, 2)
    SS = matrix_block(B, 3, 3)

    V_ℓinds = ℓrange(1, nℓ, V_symmetric)
    W_ℓinds = ℓrange(1, nℓ, !V_symmetric)

    space2d_D2 = radialspace_D2 ⊗ NormalizedPlm(m)
    I2d_D2 = ((Ir ⊗ Iℓ) : space2d → space2d_D2) |> expand

    VV_ = real(kronmatrix(I2d_D2, nr, V_ℓinds, V_ℓinds));
    VV .= VV_
    SS .= VV_

    space2d_D4 = radialspace_D4 ⊗ NormalizedPlm(m)
    WWop = ((ddrDDr ⊗ Iℓ - onebyr2 ⊗ ℓℓp1op) : space2d → space2d_D4) |> expand
    WW_ = real(kronmatrix(WWop, nr, W_ℓinds, W_ℓinds));
    WW .= (Weqglobalscaling * Rsun^2) .* WW_

    return B
end

function uniform_rotation_matrix(m; operators, kw...)
    A = allocate_operator_matrix(operators, 2)
    uniform_rotation_matrix!(A, m; operators, kw...)
    return A
end

function uniform_rotation_matrix!(A::StructMatrix{<:Complex}, m; operators, V_symmetric, kw...)
    @unpack Ω0 = operators.constants;
    @unpack nr, nℓ = operators.radial_params
    @unpack Sscaling, Wscaling, Weqglobalscaling, Seqglobalscaling = operators.scalings

    @unpack onebyr, onebyr2, r, r2, g, ηρ, ddr_S0_by_Cp_by_r2 = operators.rad_terms;
    @unpack ddr, d2dr2, DDr, ddrDDr, DDr_minus_2byr, ∇r2_plus_ddr_lnρT_ddr = operators.diff_operators;
    @unpack radialspaces = operators;
    @unpack radialspace, radialspace_D2, radialspace_D4 = radialspaces;
    @unpack constants = operators;
    @unpack κ = constants;

    A.re .= 0
    A.im .= 0

    VV = matrix_block(A.re, 1, 1);
    VW = matrix_block(A.re, 1, 2);
    WV = matrix_block(A.re, 2, 1);
    WW = matrix_block(A.re, 2, 2);
    WS = matrix_block(A.re, 2, 3);
    SW = matrix_block(A.re, 3, 2);
    SS = matrix_block(A.im, 3, 3);

    nCS = 2nℓ+1
    ℓs = range(m, length = nCS)

    latitudinal_space = NormalizedPlm(m);

    Ir = ConstantOperator(I);
    Iℓ = I : latitudinal_space;
    space2d = radialspace ⊗ latitudinal_space;

    sinθdθop = sinθdθ_Operator(latitudinal_space);
    cosθop = cosθ_Operator(latitudinal_space);
    ∇² = HorizontalLaplacian(latitudinal_space);
    ℓℓp1op = -∇²;
    inv_ℓℓp1 = inv(ℓℓp1op)
    two_by_ℓℓp1 = 2inv_ℓℓp1
    twom_by_ℓℓp1 = m * two_by_ℓℓp1

    scaled_curl_u_x_ω_r_tmp = OpVector(V = Ir ⊗ twom_by_ℓℓp1,
        iW = (Ir ⊗ two_by_ℓℓp1) * (
            DDr_minus_2byr ⊗ (cosθop * ∇²)
            - DDr ⊗ sinθdθop - onebyr ⊗ (sinθdθop * ∇²))
        )

    space2d_D2 = radialspace_D2 ⊗ NormalizedPlm(m)
    scaled_curl_u_x_ω_r = (scaled_curl_u_x_ω_r_tmp : space2d → space2d_D2) |> expand;

    V_ℓinds = ℓrange(1, nℓ, V_symmetric)
    W_ℓinds = ℓrange(1, nℓ, !V_symmetric)

    temp = zeros(eltype(A), nr * nℓ, nr * nℓ)

    VV_ = kronmatrix!(temp, scaled_curl_u_x_ω_r.V, nr, V_ℓinds, V_ℓinds);
    VV .= real.(VV_);

    VW_ = kronmatrix!(temp, scaled_curl_u_x_ω_r.iW, nr, V_ℓinds, W_ℓinds);
    VW .= (Rsun / Wscaling) .* real.(VW_);

    scaled_curl_curl_u_x_ω_r_tmp = OpVector(
        V = (Ir ⊗ two_by_ℓℓp1) * (
            (ddr - 2*onebyr) ⊗ (cosθop * ∇²)
            - ddr ⊗ sinθdθop - onebyr ⊗ (sinθdθop * ∇²)),
        iW = (Ir ⊗ twom_by_ℓℓp1) * (ddrDDr ⊗ Iℓ - ((1 + ηρ*r)*onebyr2) ⊗ ℓℓp1op),
        )
    space2d_D4 = radialspace_D4 ⊗ NormalizedPlm(m)
    scaled_curl_curl_u_x_ω_r = (scaled_curl_curl_u_x_ω_r_tmp : space2d → space2d_D4) |> expand;

    WV_ = kronmatrix!(temp, scaled_curl_curl_u_x_ω_r.V, nr, W_ℓinds, V_ℓinds);
    WV .= (Weqglobalscaling * Rsun * Wscaling) .* real.(WV_);

    WW_ = kronmatrix!(temp, scaled_curl_curl_u_x_ω_r.iW, nr, W_ℓinds, W_ℓinds);
    WW .= (Weqglobalscaling * Rsun^2) .* real.(WW_);

    WSop = (-g/(Ω0^2 * Rsun) ⊗ Iℓ) : space2d → space2d_D4
    WS_ = kronmatrix!(temp, WSop, nr, W_ℓinds, W_ℓinds);
    WS .= real.(WS_) .* (Wscaling/Sscaling) * Weqglobalscaling

    SWop = (ddr_S0_by_Cp_by_r2 ⊗ ℓℓp1op) : space2d → space2d_D2
    SW_ = kronmatrix!(temp, SWop, nr, W_ℓinds, W_ℓinds);
    SW .= real.(SW_) .* (Rsun^3 * Seqglobalscaling * Sscaling/Wscaling)

    SSop = ((-κ * (∇r2_plus_ddr_lnρT_ddr ⊗ Iℓ - onebyr2 ⊗ ℓℓp1op)) : space2d → space2d_D2) |> expand
    SS_ = kronmatrix!(temp, SSop, nr, W_ℓinds, W_ℓinds);
    SS .= real.(SS_) .* (Rsun^2 * Seqglobalscaling)

    viscosity_terms!(A, m; operators, V_symmetric, kw...)

    return A
end

function viscosity_terms!(A::StructMatrix{<:Complex}, m; operators, V_symmetric, kw...)
    @unpack nr, nℓ = operators.radial_params;
    @unpack ddr, d2dr2, d3dr3, DDr, d2dr2_ηρbyr_op = operators.diff_operators;
    @unpack ηρ, ηρ_by_r, ηρ_by_r2, onebyr2, onebyr, r, ηρ2_by_r2 = operators.rad_terms;
    @unpack radialspaces = operators;
    @unpack radialspace, radialspace_D2, radialspace_D4 = radialspaces;
    @unpack ν = operators.constants;
    @unpack Weqglobalscaling = operators.scalings;

    VVim = matrix_block(A.im, 1, 1)
    WWim = matrix_block(A.im, 2, 2)

    latitudinal_space = NormalizedPlm(m);

    Ir = ConstantOperator(I);
    Iℓ = I : latitudinal_space;
    space2d = radialspace ⊗ latitudinal_space;
    ∇² = HorizontalLaplacian(latitudinal_space);
    ℓℓp1op = -∇²;

    V_ℓinds = ℓrange(1, nℓ, V_symmetric)
    W_ℓinds = ℓrange(1, nℓ, !V_symmetric)

    temp = zeros(eltype(A), nr * nℓ, nr * nℓ)

    space2d_D2 = radialspace_D2 ⊗ NormalizedPlm(m)
    VVop_ = -ν * ((d2dr2 + ηρ * (ddr - 2onebyr)) ⊗ Iℓ - onebyr2 ⊗ ℓℓp1op)
    VVop = (VVop_ : space2d → space2d_D2) |> expand
    VV_ = kronmatrix!(temp, VVop, nr, V_ℓinds, V_ℓinds)
    VVim .= real.(VV_) * Rsun^2

    WWop_ = -ν * (
        ((ddr - 2onebyr) ⊗ Iℓ) * ((r * d2dr2_ηρbyr_op) ⊗ Iℓ - ηρ_by_r2 ⊗ ℓℓp1op)
        + (d2dr2 ⊗ Iℓ - onebyr2 ⊗ ℓℓp1op) * ((d2dr2 + 4ηρ_by_r) ⊗ Iℓ - onebyr2 ⊗ ℓℓp1op)
        + (ddr ⊗ Iℓ) * ((ηρ * (ddr - 2onebyr) * DDr) ⊗ Iℓ + ηρ_by_r2 ⊗ ℓℓp1op)
        - ((ηρ_by_r2 * (ddr - 2onebyr)) ⊗ 2ℓℓp1op)
        - (ηρ2_by_r2 ⊗ (2/3 * ℓℓp1op))
        )
    space2d_D4 = radialspace_D4 ⊗ NormalizedPlm(m)
    WWop = (WWop_ : space2d → space2d_D4) |> expand
    WW_ = kronmatrix!(temp, WWop, nr, W_ℓinds, W_ℓinds)
    WWim .= real.(WW_) .* (Rsun^4 * Weqglobalscaling)

    return A
end

function constant_differential_rotation_terms!(M::StructMatrix{<:Complex}, m;
        operators, ΔΩ_frac = 0.01, V_symmetric, kw...)

    @unpack nr, nℓ = operators.radial_params;
    @unpack Wscaling, Weqglobalscaling, Seqglobalscaling = operators.scalings;

    @unpack ddr, DDr, ddrDDr, DDr_minus_2byr = operators.diff_operators;
    @unpack onebyr, onebyr2, ηρ_by_r = operators.rad_terms;

    @unpack radialspaces = operators;
    @unpack radialspace, radialspace_D2, radialspace_D4 = radialspaces;

    VV = matrix_block(M.re, 1, 1)
    VW = matrix_block(M.re, 1, 2)
    WV = matrix_block(M.re, 2, 1)
    WW = matrix_block(M.re, 2, 2)
    SS = matrix_block(M.re, 3, 3)

    latitudinal_space = NormalizedPlm(m);

    Ir = ConstantOperator(I);
    Iℓ = I : latitudinal_space;
    space2d = radialspace ⊗ latitudinal_space;
    sinθdθop = sinθdθ_Operator(latitudinal_space);
    cosθop = cosθ_Operator(latitudinal_space);
    ∇² = HorizontalLaplacian(latitudinal_space);
    ℓℓp1op = -∇²;
    inv_ℓℓp1 = inv(ℓℓp1op)
    two_by_ℓℓp1 = 2inv_ℓℓp1

    V_ℓinds = ℓrange(1, nℓ, V_symmetric)
    W_ℓinds = ℓrange(1, nℓ, !V_symmetric)

    VVop_ = ΔΩ_frac * m * (Ir ⊗ (4inv_ℓℓp1 - 1))
    VWop_ = ΔΩ_frac * (Ir ⊗ (-4inv_ℓℓp1)) * (DDr_minus_2byr ⊗ (cosθop * ℓℓp1op) +
                    DDr ⊗ sinθdθop - onebyr ⊗ (sinθdθop * ℓℓp1op))

    temp = zeros(eltype(M), nr * nℓ, nr * nℓ)

    space2d_D2 = radialspace_D2 ⊗ NormalizedPlm(m)
    VVop = (VVop_ : space2d → space2d_D2) |> expand
    VWop = (VWop_ : space2d → space2d_D2) |> expand
    VV_ = kronmatrix!(temp, VVop, nr, V_ℓinds, V_ℓinds)
    VV .+= real.(VV_)
    VW_ = kronmatrix!(temp, VWop, nr, V_ℓinds, W_ℓinds)
    VW .+= (Rsun / Wscaling) .* real.(VW_)

    WVop_ = -ΔΩ_frac * (Ir ⊗ inv_ℓℓp1) * (ddr ⊗ (8cosθop * ℓℓp1op + sinθdθop * (ℓℓp1op + 4)) +
        (ddr + 4onebyr) ⊗ (∇² * sinθdθop))
    WWop_ = m * (-ΔΩ_frac) * (4ηρ_by_r ⊗ Iℓ - ddrDDr ⊗ (4inv_ℓℓp1 - 1) + onebyr2 ⊗ (4 - ℓℓp1op))

    space2d_D4 = radialspace_D4 ⊗ NormalizedPlm(m)
    WVop = (WVop_ : space2d → space2d_D4) |> expand
    WWop = (WWop_ : space2d → space2d_D4) |> expand
    WV_ = kronmatrix!(temp, WVop, nr, W_ℓinds, V_ℓinds);
    WV .+= (Weqglobalscaling * Rsun * Wscaling) .* real.(WV_);
    WW_ = kronmatrix!(temp, WWop, nr, W_ℓinds, W_ℓinds);
    WW .+= (Weqglobalscaling * Rsun^2) .* real.(WW_);

    SSop_ = -m * ΔΩ_frac * (Ir ⊗ Iℓ)
    SSop = (SSop_ : space2d → space2d_D2) |> expand
    SS_ = kronmatrix!(temp, SSop, nr, W_ℓinds, W_ℓinds);
    SS .+= Seqglobalscaling .* real.(SS_)

    return M
end

function radial_differential_rotation_terms!(M::StructMatrix{<:Complex}, m;
        operators, rotation_profile = :solar_equator,
        ΔΩ_frac = 0.01, # only used to test the constant case
        ΔΩ_scale = 1.0,
        ΔΩprofile_deriv = radial_differential_rotation_profile_derivatives_Fun(;
                            operators, rotation_profile, ΔΩ_frac, ΔΩ_scale),
        V_symmetric, kw...)

    @unpack nr, nℓ = operators.radial_params;
    @unpack DDr, ddr, ddrDDr, DDr_minus_2byr = operators.diff_operators;
    @unpack onebyr, g, ηρ_by_r, onebyr2, twobyr = operators.rad_terms;
    @unpack Sscaling, Wscaling, Weqglobalscaling, Seqglobalscaling = operators.scalings;
    @unpack Ω0 = operators.constants;

    @unpack radialspaces = operators;
    @unpack radialspace, radialspace_D2, radialspace_D4 = radialspaces;

    VV = matrix_block(M.re, 1, 1);
    VW = matrix_block(M.re, 1, 2);
    WV = matrix_block(M.re, 2, 1);
    WW = matrix_block(M.re, 2, 2);
    SV = matrix_block(M.re, 3, 1);
    SW = matrix_block(M.re, 3, 2);
    SS = matrix_block(M.re, 3, 3);

    @unpack r_out = operators.radial_params;
    (; ΔΩ, ddrΔΩ, d2dr2ΔΩ) = ΔΩprofile_deriv;

    latitudinal_space = NormalizedPlm(m);

    Ir = ConstantOperator(I);
    Iℓ = I : latitudinal_space;
    space2d = radialspace ⊗ latitudinal_space;
    sinθdθop = sinθdθ_Operator(latitudinal_space);
    cosθop = cosθ_Operator(latitudinal_space);
    ∇² = HorizontalLaplacian(latitudinal_space);
    ℓℓp1op = -∇²;
    inv_ℓℓp1 = inv(ℓℓp1op)
    two_by_ℓℓp1 = 2inv_ℓℓp1
    two_by_ℓℓp1_min_1 = two_by_ℓℓp1 - 1

    V_ℓinds = ℓrange(1, nℓ, V_symmetric)
    W_ℓinds = ℓrange(1, nℓ, !V_symmetric)

    VVop_ = m * (ΔΩ ⊗ (4inv_ℓℓp1 - 1))
    VWop_ = -(Ir ⊗ 4inv_ℓℓp1) * (
        (ΔΩ * DDr_minus_2byr - ddrΔΩ) ⊗ (cosθop * ℓℓp1op) +
        (ΔΩ * DDr) ⊗ sinθdθop - (ΔΩ * onebyr + ddrΔΩ/4) ⊗ (sinθdθop * ℓℓp1op)
    )

    temp = zeros(eltype(M), nr * nℓ, nr * nℓ)

    space2d_D2 = radialspace_D2 ⊗ NormalizedPlm(m)
    VVop = (VVop_ : space2d → space2d_D2) |> expand
    VWop = (VWop_ : space2d → space2d_D2) |> expand
    VV_ = kronmatrix!(temp, VVop, nr, V_ℓinds, V_ℓinds)
    VV .+= real.(VV_)
    VW_ = kronmatrix!(temp, VWop, nr, V_ℓinds, W_ℓinds)
    VW .+= (Rsun / Wscaling) .* real.(VW_)

    # V terms
    V_ℓs = ℓrange(m, nℓ, V_symmetric);
    # W, S terms
    W_ℓs = ℓrange(m, nℓ, !V_symmetric);

    WVop_ = -(Ir ⊗ inv_ℓℓp1) * (
        (ΔΩ * ddr + ddrΔΩ) ⊗ (4cosθop * ℓℓp1op + sinθdθop * (ℓℓp1op + 2) + ∇² * sinθdθop) +
        (2ΔΩ *onebyr) ⊗ (∇² * sinθdθop))
    WWop_ = m * (-2ΔΩ * ηρ_by_r ⊗ Iℓ + (ddrΔΩ * DDr) ⊗ (two_by_ℓℓp1 - 1)
        + (ΔΩ * ddrDDr) ⊗ (two_by_ℓℓp1 - 1)
        - (ΔΩ * onebyr2) ⊗ ((two_by_ℓℓp1 - 1) * ℓℓp1op)
        + (d2dr2ΔΩ + ddrΔΩ * (ddr + twobyr)) ⊗ Iℓ
        )

    space2d_D4 = radialspace_D4 ⊗ NormalizedPlm(m)
    WVop = (WVop_ : space2d → space2d_D4) |> expand
    WWop = (WWop_ : space2d → space2d_D4) |> expand
    WV_ = kronmatrix!(temp, WVop, nr, W_ℓinds, V_ℓinds);
    WV .+= (Weqglobalscaling * Rsun * Wscaling) .* real.(WV_);
    WW_ = kronmatrix!(temp, WWop, nr, W_ℓinds, W_ℓinds);
    WW .+= (Weqglobalscaling * Rsun^2) .* real.(WW_);

    # ddrΔΩ_over_g = ddrΔΩ / g;
    # SVop_ = -m * 2ddrΔΩ_over_g ⊗ cosθop
    # SVop = (SVop_ : space2d → space2d_D2) |> expand
    # SV_ = real(kronmatrix!(temp, SVop, nr, W_ℓinds, V_ℓinds));
    # SV .+= (Ω0^2 * Rsun^2 * Seqglobalscaling * Sscaling) .* SV_

    # SWop_ = (2ddrΔΩ_over_g * DDr) ⊗ (cosθop * sinθdθop)
    # SWop = (SWop_ : space2d → space2d_D2) |> expand
    # SW_ = real(kronmatrix!(temp, SWop, nr, W_ℓinds, W_ℓinds));
    # SW .+= (Ω0^2 * Rsun^3 * Seqglobalscaling * Sscaling/Wscaling) .* SW_

    SSop_ = -m * (ΔΩ ⊗ Iℓ)
    SSop = (SSop_ : space2d → space2d_D2) |> expand
    SS_ = kronmatrix!(temp, SSop, nr, W_ℓinds, W_ℓinds);
    SS .+= Seqglobalscaling .* real.(SS_)

    return M
end

struct OpVector{VT,WT}
    V :: VT
    iW :: WT
end
OpVector(t::Tuple) = OpVector(t...)
OpVector(t::NamedTuple{(:V, :iW)}) = OpVector(t.V, t.iW)
OpVector(; V, iW) = OpVector(V, iW)
Base.:(*)(O::Operator, B::OpVector) = OpVector(O * B.V, O * B.iW)
Base.:(*)(O::Union{Function, Number}, B::OpVector) = OpVector(O * B.V, O * B.iW)
Base.:(*)(A::OpVector, O::Operator) = OpVector(A.B * O, A.iW * O)
Base.:(*)(A::OpVector, O::Union{Function, Number}) = OpVector(A.B * O, A.iW * O)
Base.:(+)(A::OpVector) = A
Base.:(-)(A::OpVector) = OpVector(-A.V, -A.iW)
Base.:(+)(A::OpVector, B::OpVector) = OpVector(A.V + B.V, A.iW + B.iW)
Base.:(-)(A::OpVector, B::OpVector) = OpVector(A.V - B.V, A.iW - B.iW)
Base.:(*)(A::OpVector, B::OpVector) = OpVector(A.V * B.V, A.iW * B.iW)

Base.:(:)(A::OpVector, s::Space) = OpVector(A.V : s, A.iW : s)
ApproxFunBase.:(→)(A::OpVector, s::Space) = OpVector(A.V → s, A.iW → s)

ApproxFunAssociatedLegendre.expand(A::OpVector) = OpVector(expand(A.V), expand(A.iW))

function Base.show(io::IO, O::OpVector)
    print(io, "OpVector(V = ")
    show(io, O.V)
    print(io, ", iW = ")
    show(io, O.iW)
    print(io, ")")
end

function solar_differential_rotation_profile_derivatives_Fun(Feig::FilteredEigen; kw...)
    rotation_profile = rotationtag(Feig.kw[:rotation_profile])
    solar_differential_rotation_profile_derivatives_Fun(; Feig.operators,
        Feig.kw..., rotation_profile, kw...)
end

function solar_differential_rotation_vorticity_Fun(; operators, ΔΩprofile_deriv)
    @unpack onebyr, r, onebyr2, twobyr = operators.rad_terms;
    @unpack ddr, d2dr2 = operators.diff_operators;

    (; ΔΩ, dr_ΔΩ, d2r_ΔΩ) = ΔΩprofile_deriv;

    @unpack radialspace = operators.radialspaces;
    # velocity and its derivatives are expanded in Legendre poly
    latitudinal_space = NormalizedPlm(0);
    space2d = radialspace ⊗ latitudinal_space

    angularsp = Ultraspherical(Legendre())
    cosθ = Fun(angularsp);
    Ir = I : radialspace;
    Iℓ = I : latitudinal_space;
    ∇² = HorizontalLaplacian(latitudinal_space);

    sinθdθ_plus_2cosθ = sinθdθ_plus_2cosθ_Operator(latitudinal_space);
    ωΩr = (Ir ⊗ sinθdθ_plus_2cosθ) * ΔΩ;
    ∂rωΩr = (Ir ⊗ sinθdθ_plus_2cosθ) * dr_ΔΩ;
    # cotθddθ = cosθ * 1/sinθ * d/dθ = -cosθ * d/d(cosθ) = -x*d/dx
    cotθdθ = KroneckerOperator(Ir, -cosθ * Derivative(angularsp),
        radialspace * angularsp,
        radialspace * rangespace(Derivative(angularsp)));

    cotθdθΔΩ = Fun(cotθdθ * ΔΩ, space2d);

    ∇²_min_2 = ∇²-2;
    inv_sinθ_∂θωΩr = (Ir ⊗ ∇²_min_2) * ΔΩ + 2cotθdθΔΩ;
    cotθdθdr_ΔΩ = Fun(cotθdθ * dr_ΔΩ, space2d);
    inv_sinθ_∂r∂θωΩr = (Ir ⊗ ∇²_min_2) * dr_ΔΩ + 2cotθdθdr_ΔΩ;
    inv_rsinθ_ωΩθ = Fun(-(dr_ΔΩ + (twobyr ⊗ Iℓ) * ΔΩ), space2d);
    ∂r_inv_rsinθ_ωΩθ = Fun(-(d2r_ΔΩ + (twobyr ⊗ Iℓ) * dr_ΔΩ - (2onebyr2 ⊗ Iℓ) * ΔΩ), space2d);

    ωΩr, ∂rωΩr, inv_sinθ_∂θωΩr, inv_sinθ_∂r∂θωΩr, inv_rsinθ_ωΩθ, ∂r_inv_rsinθ_ωΩθ =
        promote(map(replaceemptywitheps ∘ chop,
            (ωΩr, ∂rωΩr, inv_sinθ_∂θωΩr, inv_sinθ_∂r∂θωΩr, inv_rsinθ_ωΩθ, ∂r_inv_rsinθ_ωΩθ))...)

    # Add the Coriolis force terms, that is ωΩ -> ωΩ + 2ΔΩ
    ωΩr_plus_2ΔΩr = ωΩr + Fun((Ir ⊗ 2cosθ) * ΔΩ, space2d)
    ∂r_ωΩr_plus_2ΔΩr = ∂rωΩr + Fun((Ir ⊗ 2cosθ) * dr_ΔΩ, space2d)
    inv_sinθ_∂θ_ωΩr_plus_2ΔΩr = inv_sinθ_∂θωΩr + Fun(2(cotθdθ * ΔΩ - ΔΩ), space2d)
    inv_sinθ_∂r∂θ_ωΩr_plus_2ΔΩr = inv_sinθ_∂r∂θωΩr + Fun(2(cotθdθ * dr_ΔΩ - dr_ΔΩ), space2d)
    inv_rsinθ_ωΩθ_plus_2ΔΩθ = inv_rsinθ_ωΩθ - Fun(2(onebyr ⊗ Iℓ) * ΔΩ, space2d)
    ∂r_inv_rsinθ_ωΩθ_plus_2ΔΩθ = ∂r_inv_rsinθ_ωΩθ - Fun(2*((onebyr ⊗ Iℓ) * dr_ΔΩ - (onebyr2 ⊗ Iℓ) * ΔΩ), space2d)

    ωΩr_plus_2ΔΩr, ∂r_ωΩr_plus_2ΔΩr, inv_sinθ_∂θ_ωΩr_plus_2ΔΩr,
            inv_sinθ_∂r∂θ_ωΩr_plus_2ΔΩr, inv_rsinθ_ωΩθ_plus_2ΔΩθ, ∂r_inv_rsinθ_ωΩθ_plus_2ΔΩθ =
        map(replaceemptywitheps ∘ chop,
            (ωΩr_plus_2ΔΩr, ∂r_ωΩr_plus_2ΔΩr, inv_sinθ_∂θ_ωΩr_plus_2ΔΩr,
                inv_sinθ_∂r∂θ_ωΩr_plus_2ΔΩr, inv_rsinθ_ωΩθ_plus_2ΔΩθ, ∂r_inv_rsinθ_ωΩθ_plus_2ΔΩθ))

    raw = (; ωΩr, ∂rωΩr, inv_sinθ_∂θωΩr, inv_rsinθ_ωΩθ, inv_sinθ_∂r∂θωΩr, ∂r_inv_rsinθ_ωΩθ)
    coriolis = NamedTuple{keys(raw)}((ωΩr_plus_2ΔΩr, ∂r_ωΩr_plus_2ΔΩr, inv_sinθ_∂θ_ωΩr_plus_2ΔΩr,
        inv_rsinθ_ωΩθ_plus_2ΔΩθ, inv_sinθ_∂r∂θ_ωΩr_plus_2ΔΩr, ∂r_inv_rsinθ_ωΩθ_plus_2ΔΩθ))
    (; raw, coriolis)
end

function solar_differential_rotation_vorticity_Fun(Feig::FilteredEigen; kw...)
    ΔΩprofile_deriv = solar_differential_rotation_profile_derivatives_Fun(Feig; kw...)
    solar_differential_rotation_vorticity_Fun(; Feig.operators, ΔΩprofile_deriv)
end

function solar_differential_rotation_terms!(M::StructMatrix{<:Complex}, m;
        operators, V_symmetric,
        rotation_profile = nothing,
        ΔΩ_frac = 0.01, # only used to test the constant case
        ΔΩ_scale = 1.0, # scale the diff rot profile, for testing
        ΔΩprofile_deriv = solar_differential_rotation_profile_derivatives_Fun(;
                            operators, rotation_profile, ΔΩ_frac, ΔΩ_scale),
        ωΩ_deriv = solar_differential_rotation_vorticity_Fun(; operators, ΔΩprofile_deriv),
        kw...)

    VV = matrix_block(M.re, 1, 1);
    VW = matrix_block(M.re, 1, 2);
    WV = matrix_block(M.re, 2, 1);
    WW = matrix_block(M.re, 2, 2);
    SV = matrix_block(M.re, 3, 1);
    SW = matrix_block(M.re, 3, 2);
    SS = matrix_block(M.re, 3, 3);

    @unpack nr, nℓ = operators.radial_params;
    @unpack onebyr, onebyr2, r2, g = operators.rad_terms;
    @unpack ddr, d2dr2, DDr, ddrDDr = operators.diff_operators;
    @unpack radialspaces = operators;
    @unpack radialspace, radialspace_D2, radialspace_D4 = radialspaces;
    @unpack Wscaling, Sscaling, Weqglobalscaling, Seqglobalscaling = operators.scalings;
    @unpack Ω0 = operators.constants;

    latitudinal_space = NormalizedPlm(m);

    Ir = ConstantOperator(I);
    Iℓ = I : latitudinal_space;
    unsetspace2d = domainspace(Ir) ⊗ latitudinal_space;
    space2d = radialspace ⊗ latitudinal_space;
    I2d_unset = I : unsetspace2d;

    sinθdθop = sinθdθ_Operator(latitudinal_space);
    ∇h² = HorizontalLaplacian(latitudinal_space);
    ℓℓp1op = -∇h²;

    (; ΔΩ, dr_ΔΩ, dz_ΔΩ) = ΔΩprofile_deriv;
    (; ωΩr, ∂rωΩr, inv_sinθ_∂θωΩr, inv_rsinθ_ωΩθ, inv_sinθ_∂r∂θωΩr,
        ∂r_inv_rsinθ_ωΩθ, ∂r_inv_rsinθ_ωΩθ) = ωΩ_deriv.coriolis;

    onebyr2op = Multiplication(onebyr2);
    ufr_W = onebyr2op ⊗ ℓℓp1op;
    ufr = OpVector(V = 0 * I2d_unset, iW = -im * ufr_W);
    ωfr = OpVector(V = ufr_W, iW = 0 * I2d_unset);

    rsinθufθ = OpVector(V = im * m * I2d_unset, iW = -im * DDr ⊗ sinθdθop);
    rsinθωfθ = OpVector(V = ddr ⊗ sinθdθop, iW = m * (onebyr2op ⊗ ℓℓp1op - ddrDDr ⊗ Iℓ));

    rsinθufϕ = OpVector(V = -Ir ⊗ sinθdθop, iW = (m * DDr) ⊗ Iℓ);
    rsinθωfϕ = OpVector(V = im * m * ddr ⊗ Iℓ, iW = -im * (ddrDDr ⊗ Iℓ - onebyr2op ⊗ ℓℓp1op));

    curl_uf_x_ωΩ_r = ((ωΩr * (DDr ⊗ Iℓ) + inv_rsinθ_ωΩθ * (Ir ⊗ sinθdθop) - ∂rωΩr) * ufr +
        ((-onebyr2 ⊗ Iℓ) * inv_sinθ_∂θωΩr) * rsinθufθ);
    curl_uΩ_x_ωf_r = -im * m * ΔΩ * ωfr;
    curl_u_x_ω_r = curl_uf_x_ωΩ_r + curl_uΩ_x_ωf_r;
    scaled_curl_u_x_ω_r_tmp = ((-im * r2) ⊗ inv(ℓℓp1op)) * curl_u_x_ω_r;
    space2d_D2 = radialspace_D2 ⊗ NormalizedPlm(m)
    scaled_curl_u_x_ω_r = (scaled_curl_u_x_ω_r_tmp : space2d → space2d_D2) |> expand;

    V_ℓinds = ℓrange(1, nℓ, V_symmetric)
    W_ℓinds = ℓrange(1, nℓ, !V_symmetric)

    temp = zeros(eltype(M), nr * nℓ, nr * nℓ)

    VV_ = kronmatrix!(temp, scaled_curl_u_x_ω_r.V, nr, V_ℓinds, V_ℓinds)
    VV .+= real.(VV_);

    VW_ = kronmatrix!(temp, scaled_curl_u_x_ω_r.iW, nr, V_ℓinds, W_ℓinds)
    VW .+= (Rsun / Wscaling) .* real.(VW_);

    # we compute the derivative of rdiv_ucrossω_h analytically
    # this lets us use the more accurate representations of ddrDDr instead of using ddr * DDr
    ddr_rdiv_ucrossω_h = OpVector(
        V = ((2ωΩr + ΔΩ * (Ir ⊗ sinθdθop)) * (ddr ⊗ ℓℓp1op) - inv_sinθ_∂θωΩr * (ddr ⊗ sinθdθop)
            + (2∂rωΩr + dr_ΔΩ * (Ir ⊗ sinθdθop)) * (Ir ⊗ ℓℓp1op) - inv_sinθ_∂r∂θωΩr * (Ir ⊗ sinθdθop)
            ),
        iW = m *
            (inv_sinθ_∂θωΩr * (ddrDDr ⊗ Iℓ) + inv_rsinθ_ωΩθ * (ddr ⊗ ℓℓp1op)
            + inv_sinθ_∂r∂θωΩr * (DDr ⊗ Iℓ) + ∂r_inv_rsinθ_ωΩθ * (Ir ⊗ ℓℓp1op)
            )
    );

    uf_x_ωΩ_r = -inv_rsinθ_ωΩθ * rsinθufϕ;
    uΩ_x_ωf_r = -ΔΩ * rsinθωfθ;
    u_x_ω_r = uf_x_ωΩ_r + uΩ_x_ωf_r;
    ∇h²_u_x_ω_r = (Ir ⊗ ∇h²) * u_x_ω_r;

    scaled_curl_curl_u_x_ω_r_tmp = (Ir ⊗ inv(ℓℓp1op)) * (-ddr_rdiv_ucrossω_h + ∇h²_u_x_ω_r);
    space2d_D4 = radialspace_D4 ⊗ NormalizedPlm(m)
    scaled_curl_curl_u_x_ω_r = (scaled_curl_curl_u_x_ω_r_tmp : space2d → space2d_D4) |> expand;

    WV_ = kronmatrix!(temp, scaled_curl_curl_u_x_ω_r.V, nr, W_ℓinds, V_ℓinds)
    WV .+= (Weqglobalscaling * Rsun * Wscaling) .* real.(WV_);

    WW_ = kronmatrix!(temp, scaled_curl_curl_u_x_ω_r.iW, nr, W_ℓinds, W_ℓinds)
    WW .+= (Weqglobalscaling * Rsun^2) .* real.(WW_);

    # entropy terms
    # thermal_wind_term_tmp = im*((2/g) ⊗ Iℓ) * dz_ΔΩ * rsinθufθ
    # thermal_wind_term = expand(thermal_wind_term_tmp : space2d → space2d_D2)
    # SV_ = kronmatrix(thermal_wind_term.V, nr, W_ℓinds, V_ℓinds);
    # SV .+= (Seqglobalscaling * (Ω0^2 * Rsun^2) * Sscaling) .* real.(SV_)
    # SW_ = kronmatrix(thermal_wind_term.iW, nr, W_ℓinds, W_ℓinds);
    # SW .+= (Seqglobalscaling * (Ω0^2 * Rsun^3) * Sscaling/Wscaling) .* real.(SW_)

    SS_doppler_term = expand(-m*ΔΩ * I : space2d → space2d_D2)
    SS_ = kronmatrix!(temp, SS_doppler_term, nr, W_ℓinds, W_ℓinds);
    SS .+= Seqglobalscaling .* real.(SS_);

    return M
end

function rotationtag(rotation_profile)
    rstr = String(rotation_profile)
    if startswith(rstr, "radial")
        return Symbol(split(rstr, "radial_")[2])
    elseif startswith(rstr, "solar")
        return Symbol(split(rstr, "solar_")[2])
    else
        return Symbol(rotation_profile)
    end
end

function _differential_rotation_matrix!(M, m; rotation_profile, kw...)
    rstr = String(rotation_profile)
    tag = rotationtag(rotation_profile)
    if startswith(rstr, "radial")
        radial_differential_rotation_terms!(M, m; rotation_profile = tag, kw...)
    elseif startswith(rstr, "solar")
        solar_differential_rotation_terms!(M, m; rotation_profile = tag, kw...)
    elseif Symbol(rotation_profile) == :constant
        constant_differential_rotation_terms!(M, m; kw...)
    else
        throw(ArgumentError("Invalid rotation profile"))
    end
    return M
end
function differential_rotation_matrix(m; operators, rotation_profile, kw...)
    @unpack nℓ = operators.radial_params;
    rstr = String(rotation_profile)
    bandwidth = (rstr == "constant" || startswith(rstr, "radial")) ? 2 : nℓ
    M = allocate_operator_matrix(operators, bandwidth)
    differential_rotation_matrix!(M, m; operators, rotation_profile, kw...)
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
    v = StructArray{Complex{eltype(vr)}}((vr, vi))
    temp = zeros(sz[2], sz[2])
    v, temp
end

function constrained_eigensystem_timed(AB; timer = TimerOutput(), kw...)
    Y = map(computesparse, AB)
    X = @timeit timer "eigen" constrained_eigensystem(Y; timer, kw...)
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
    @timeit timer "eigen!" λ::Vector{ComplexF64}, w::Matrix{ComplexF64} = eigen!(A_constrained, B_constrained)
    @timeit timer "projectback" v = realmatcomplexmatmul(constraints.ZC, w, temp_projectback)
    # normalize the eigenvectors
    for c in eachcol(v)
        c ./= c[1] # remove extra phase
        normalize!(c)
    end
    λ, v, (A, B)
end

function uniform_rotation_spectrum(m; operators, kw...)
    AB = allocate_operator_mass_matrices(operators, 2)
    uniform_rotation_spectrum!(AB, m; operators, kw...)
end
function uniform_rotation_spectrum!((A, B), m; operators, timer = TimerOutput(), kw...)
    @timeit timer "matrix" begin
        uniform_rotation_matrix!(A, m; operators, kw...)
        mass_matrix!(B, m; operators, kw...)
    end
    constrained_eigensystem_timed((A, B); operators, timer, kw...)
end

function getbw(rotation_profile, nℓ)
    rotation_profile == :uniform && return 2
    rotation_profile == :constant && return 2
    startswith(String(rotation_profile), "radial") && return 2
    nℓ
end

function differential_rotation_spectrum(m::Integer; operators, rotation_profile, kw...)
    (; nℓ) = operators.radial_params
    bw = getbw(rotation_profile, nℓ)
    AB = allocate_operator_mass_matrices(operators, bw)
    differential_rotation_spectrum!(AB, m; operators, rotation_profile, kw...)
end
function differential_rotation_spectrum!((A, B)::Tuple{StructMatrix{<:Complex}, AbstractMatrix{<:Real}},
        m::Integer; rotation_profile, operators, timer = TimerOutput(), kw...)
    @timeit timer "matrix" begin
        differential_rotation_matrix!(A, m; operators, rotation_profile, kw...)
        mass_matrix!(B, m; operators, kw...)
    end
    constrained_eigensystem_timed((A, B); operators, timer, kw...)
end

rossby_ridge(m; ΔΩ_frac = 0) = 2 / (m + 1) * (1 + ΔΩ_frac) - m * ΔΩ_frac

function eigenvalue_filter(λ, m;
    eig_imag_unstable_cutoff = DefaultFilterParams[:eig_imag_unstable_cutoff],
    eig_imag_to_real_ratio_cutoff = DefaultFilterParams[:eig_imag_to_real_ratio_cutoff],
    eig_imag_stable_cutoff = DefaultFilterParams[:eig_imag_stable_cutoff])

    freq_sectoral = 2 / (m + 1)
    eig_imag_unstable_cutoff <= imag(λ) < min(freq_sectoral * eig_imag_to_real_ratio_cutoff, eig_imag_stable_cutoff)
end
function boundary_condition_filter(v::StructVector{<:Complex}, BC::AbstractMatrix{<:Real},
        BCVcache::StructVector{<:Complex} = allocate_BCcache(size(BC,1)), atol = 1e-5)
    mul!(BCVcache.re, BC, v.re)
    mul!(BCVcache.im, BC, v.im)
    norm(BCVcache) < atol
end

include("eigensystem_satisfy_filter.jl")

function filterfields(coll, v, nparams, nvariables; filterfieldpowercutoff = DefaultFilterParams[:filterfieldpowercutoff])
    Vpow = sum(abs2, @view v[1:nparams])
    Wpow = sum(abs2, @view v[nparams .+ (1:nparams)])
    Spow = sum(abs2, @view(v[2nparams .+ (1:nparams)]))

    maxpow = max(Vpow, Wpow, Spow)

    filterfields = Pair{Symbol, typeof(coll.V)}[]

    if Spow/maxpow > filterfieldpowercutoff
        push!(filterfields, :S => coll.S)
    end
    if Vpow/maxpow > filterfieldpowercutoff
        push!(filterfields, :V => coll.V)
    end
    if Wpow/maxpow > filterfieldpowercutoff
        push!(filterfields, :W => coll.W)
    end
    return filterfields
end

include("eigvec_spectrum_filter.jl")

allocate_Pl(m, nℓ) = zeros(range(m, length = 2nℓ + 1))

function peakindabs1(X)
    findmax(v -> sum(abs2, v), eachrow(X))[2]
end

include("spatial_filter.jl")

function angularindex_maximum(Vr::AbstractMatrix{<:Real}, θ)
    equator_ind = angularindex_equator(Vr, θ)
    nθ = length(θ)
    Δθ_scan = div(nθ, 5)
    rangescan = intersect(equator_ind .+ (-Δθ_scan:Δθ_scan), axes(Vr, 2))
    Vr_section = view(Vr, :, rangescan)
    Imax = argmax(Vr_section)
    ind_max = Tuple(Imax)[2]
    ind_max += first(rangescan) - 1
end

angularindex_colatitude(Vr, θ, θi) = findmin(x -> abs(x - θi), θ)[2]
angularindex_equator(Vr, θ) = angularindex_colatitude(Vr, θ, pi/2)
indexof_colatitude(θ, θi) = angularindex_colatitude(nothing, θ, θi)
indexof_equator(θ) = indexof_colatitude(θ, pi/2)

function sign_changes(v)
    isempty(v) && return 0
    n = Int(iszero(v[1]))
    for i in eachindex(v)[1:end-1]
        if sign(v[i]) != sign(v[i+1]) && v[i] != 0
            n += 1
        end
    end
    return n
end

function count_num_nodes!(radprof::AbstractVector{<:Real}, rptsrev;
        nodessmallpowercutoff = DefaultFilterParams[:nodessmallpowercutoff])

    reverse!(radprof)
    radprof .*= argmax(abs.(extrema(radprof))) == 2 ? 1 : -1
    zerocrossings = sign_changes(radprof)
    iszero(zerocrossings) && return 0
    s = smoothed_spline(rptsrev, radprof)
    radroots = Dierckx.roots(s, maxn = 4zerocrossings)
    isempty(radroots) && return 0

    # Discount nodes that appear spurious
    radprof .= abs.(radprof)
    sa = smoothed_spline(rptsrev, radprof)
    unsignedarea = Dierckx.integrate(sa, rptsrev[1], rptsrev[end])

    signed_areas = zeros(Float64, length(radroots)+1)
    signed_areas[1] = Dierckx.integrate(s, rptsrev[1], radroots[1])
    for (ind, (spt, ept)) in enumerate(zip(@view(radroots[1:end-1]), @view(radroots[2:end])))
        signed_areas[ind+1] = Dierckx.integrate(s, spt, ept)
    end
    signed_areas[end] = Dierckx.integrate(s, radroots[end], rptsrev[end])

    signed_areas = filter(area -> abs(area / unsignedarea) > nodessmallpowercutoff, signed_areas)
    ncross = sign_changes(signed_areas)

    min(ncross, zerocrossings)
end
function count_radial_nodes(radprof::AbstractVector{<:Real},
        rptsrev::AbstractVector{<:Real},
        radproftempreal = copy(radprof))
    count_num_nodes!(radproftempreal, rptsrev)
end
function count_radial_nodes(radprof::AbstractVector{<:Complex},
        rptsrev::AbstractVector{<:Real},
        radproftempreal = similar(radprof, real(eltype(radprof))))

    radproftempreal .= real.(radprof)
    nnodes_real = count_radial_nodes(radproftempreal, rptsrev, radproftempreal)
    radproftempreal .= imag.(radprof)
    nnodes_imag = count_radial_nodes(radproftempreal, rptsrev, radproftempreal)
    nnodes_real, nnodes_imag
end
function count_radial_nodes(v::AbstractVector{<:Complex}, m::Integer; operators,
        angularindex_fn = angularindex_equator,
        nr = operators.radial_params[:nr],
        nℓ = operators.radial_params[:nℓ],
        θ = colatitude_grid(m, nℓ),
        field = :V,
        fieldcaches = allocate_field_caches(nr, nℓ, length(θ)),
        realcache = similar(fieldcaches.VWSinv.V, real(eltype(fieldcaches.VWSinv.V))),
        kw...)

    @unpack rptsrev = operators
    @unpack VWSinv, radproftempreal, radproftempcomplex = fieldcaches
    eigenfunction_realspace!(fieldcaches, v, m; operators, kw...)
    X = getproperty(VWSinv, field)
    realcache .= real.(X)
    eqind = angularindex_fn(realcache, θ)
    for (i, row) in enumerate(eachrow(X))
        radproftempcomplex[i] = trapz(θ, row)
    end
    count_radial_nodes(radproftempcomplex, rptsrev, radproftempreal)
end

function count_radial_nodes(Feig::FilteredEigen, m, ind; kw...)
    count_radial_nodes(Feig[m][ind].v, m; Feig.operators, Feig.kw..., kw...)
end

function nodes_filter!(filtercache, v, m, operators;
    nℓ = operators.radial_params.nℓ,
    Plcosθ = allocate_Pl(m, nℓ),
    filterfieldpowercutoff = DefaultFilterParams[:filterfieldpowercutoff],
    nnodesmax = DefaultFilterParams[:nnodesmax],
    compute_invtransform = true,
    radproftempreal = similar(VWSinv.V, real(eltype(VWSinv.V)), size(VWSinv.V,1)),
    radproftempcomplex = similar(VWSinv.V, size(VWSinv.V,1)),
    kw...)

    nodesfilter = true

    (; VWSinv, VWSinvsh, F) = filtercache

    @unpack nparams = operators.radial_params
    @unpack nvariables, rptsrev = operators
    fields = filterfields(VWSinv, v, nparams, nvariables; filterfieldpowercutoff)

    θ = colatitude_grid(m, nℓ)
    lowercutoffind = indexof_colatitude(θ, deg2rad(60))
    eqind = indexof_equator(θ)

    if compute_invtransform
        eigenfunction_realspace!(VWSinv, VWSinvsh, F, v, m; operators, nℓ, Plcosθ, kw...)
    end

    for _X in fields
        f, X = first(_X), last(_X)
        for col in eachcol(@view X[:, lowercutoffind:eqind])
            radproftempcomplex .+= col
        end
        nnodes_real, nnodes_imag = count_radial_nodes(radproftempcomplex, rptsrev, radproftempreal)
        nodesfilter &= nnodes_real <= nnodesmax && nnodes_imag <= nnodesmax
        nodesfilter || break
    end
    return nodesfilter
end

module Filters
    using BitFlags
    export DefaultFilter
    @bitflag FilterFlag::UInt16 begin
        NONE=0
        EIGEN # Eigensystem satisfy
        EIGVAL # Imaginary to real ratio
        EIGVEC # spectrum power cutoff in n and l
        BC # boundary conditions
        SPATIAL_EQUATOR # peak near the equator
        SPATIAL_HIGHLAT # peak near the equator
        SPATIAL_RADIAL # power not concentrated at the top/bottom surface layers
        NODES # number of radial nodes
    end
    FilterFlag(F::FilterFlag) = F
    function Base.:(!)(F::FilterFlag)
        n = length(instances(FilterFlag))
        FilterFlag(2^(n-1)-1 - Int(F))
    end
    Base.in(t::FilterFlag, F::FilterFlag) = (t & F) != NONE
    Base.broadcastable(x::FilterFlag) = Ref(x)

    const SPATIAL = SPATIAL_EQUATOR | SPATIAL_RADIAL
    const DefaultFilter = EIGEN | EIGVAL | EIGVEC | BC | SPATIAL | NODES
end
using .Filters

include("DefaultFilterParams.jl")

include("filterfn.jl")

function allocate_field_vectors(nr, nℓ)
    nparams = nr * nℓ
    (; V = zeros(ComplexF64, nparams), W = zeros(ComplexF64, nparams), S = zeros(ComplexF64, nparams))
end

function allocate_field_caches(nr, nℓ, nθ)
    VWSinv = (; V = zeros(ComplexF64, nr, nθ), W = zeros(ComplexF64, nr, nθ), S = zeros(ComplexF64, nr, nθ))
    VWSinvsh = (; V = zeros(ComplexF64, nr, nℓ), W = zeros(ComplexF64, nr, nℓ), S = zeros(ComplexF64, nr, nℓ))
    F = allocate_field_vectors(nr, nℓ)
    fieldtempreal = similar(VWSinv.V, real(eltype(VWSinv.V)))
    radproftempreal = zeros(real(eltype(VWSinv.V)), size(VWSinv.V,1))
    radproftempcomplex = zeros(eltype(VWSinv.V), size(VWSinv.V,1))
    (; VWSinv, VWSinvsh, F, radproftempreal, radproftempcomplex, fieldtempreal)
end

function allocate_field_caches(Feig::FilteredEigen, m)
    @unpack nr, nℓ = Feig.operators.radial_params
    nθ = length(colatitude_grid(m, nℓ))
    allocate_field_caches(nr, nℓ, nθ)
end

function allocate_MVcache(nrows)
    StructArray{ComplexF64}((zeros(nrows), zeros(nrows))),
        StructArray{ComplexF64}((zeros(nrows), zeros(nrows)))
end

function allocate_BCcache(n_bc)
    StructArray{ComplexF64}((zeros(n_bc), zeros(n_bc)))
end

function allocate_filter_caches(m; operators, constraints = constraintmatrix(operators))
    @unpack BC = constraints
    @unpack nr, nℓ, nparams = operators.radial_params
    # temporary cache arrays
    nrows = 3nparams
    MVcache = allocate_MVcache(nrows)

    n_bc = size(BC, 1)
    BCVcache = allocate_BCcache(n_bc)

    nθ = length(colatitude_grid(m, nℓ))

    fieldcaches = allocate_field_caches(nr, nℓ, nθ)

    Plcosθ = allocate_Pl(m, nℓ)

    return (; MVcache, BCVcache, Plcosθ, fieldcaches...)
end

allocate_filter_caches(Feig::FilteredEigen, m) = allocate_filter_caches(Feig[m])
allocate_filter_caches(Feig::FilteredEigenSingleOrder) = allocate_filter_caches(Feig.m; Feig.operators, Feig.constraints)

function scale_eigenvectors!(v::AbstractMatrix; operators)
    # re-apply scalings
    @unpack nparams = operators.radial_params;
    @views begin
        V = v[1:nparams, :]
        W = v[nparams .+ (1:nparams), :]
        S = v[2nparams .+ (1:nparams), :]

        scale_eigenvectors!((; V, W, S); operators)
    end
    return v
end
function scale_eigenvectors!(VWSinv::NamedTuple; operators)
    @unpack V, W, S = VWSinv

    V .*= Rsun
    @unpack Wscaling, Sscaling = operators.scalings
    W .*= -im * Rsun^2 / Wscaling
    S ./= operators.constants.Ω0 * Rsun * Sscaling

    return VWSinv
end

function filter_eigenvalues(λ::AbstractVector, v::AbstractMatrix, m::Integer;
        operators,
        V_symmetric,
        rotation_profile,
        matrixfn!,
        kw...)

    matrixfn2! = matrixfn! isa RotMatrix ? matrixfn! :
            RotMatrix(Val(:matrix), V_symmetric, rotation_profile; operators)

    @unpack nℓ = operators.radial_params
    M = if Filters.EIGEN in Filters.FilterFlag(get(kw, :filterflags, DefaultFilter))
            bw = matrixfn! isa RotMatrix ? getbw(matrixfn!.kw.rotation_profile, nℓ) : nℓ
            M = allocate_operator_mass_matrices(operators, bw)
            A, B = M
            matrixfn2!(A, m; operators, kw...)
            mass_matrix!(B, m; operators, V_symmetric, kw...)
            M
        end
    filter_eigenvalues(λ, v, M, m; operators, V_symmetric, kw...);
end

function filter_eigenvalues(λ::AbstractVector, v::AbstractMatrix,
    M, m::Integer;
    operators,
    V_symmetric::Bool,
    constraints = constraintmatrix(operators),
    filtercache = allocate_filter_caches(m; operators, constraints),
    filterflags = DefaultFilter,
    scale_eigenvectors = false,
    filterparams...)

    @unpack nparams = operators.radial_params;
    additional_params = (; operators, V_symmetric, constraints, filtercache)
    Ms = M isa NTuple{2,AbstractMatrix} ? map(computesparse, M) : M

    inds_bool = filterfn.(λ, eachcol(v), m, Ref(Ms); additional_params..., filterflags, filterparams...)
    filtinds = axes(λ, 1)[inds_bool]
    λ, v = λ[filtinds], v[:, filtinds]

    # re-apply scalings
    scale_eigenvectors && scale_eigenvectors!(v; operators)

    λ, v
end

function filter_map(λm::AbstractVector, vm::AbstractMatrix, AB, m::Int, matrixfn! = nothing; kw...)
    if AB isa NTuple{2,AbstractMatrix} # standard case, where Filters.EIGEN in kw[:filterflags]
        A, B = AB
        matrixfn!(A, m; kw...)
        mass_matrix!(B, m; kw...)
    end
    filter_eigenvalues(λm, vm, AB, m; kw...)
end

function filter_map_nthreads!(c::Channel, nt::Int, λs::AbstractVector{<:AbstractVector},
        vs::AbstractVector{<:AbstractMatrix}, mr::AbstractVector{<:Integer},
        matrixfn! = nothing; kw...)

    TMapReturnEltype = Tuple{eltype(λs), eltype(vs)}
    nblasthreads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(max(1, div(nblasthreads, nt)))
        z = zip(λs, vs, mr)
        if length(z) > 0
            Folds.map(z) do (λm, vm, m)
                AB = take!(c)
                Y = filter_map(λm, vm, AB, m, matrixfn!; kw...)
                put!(c, AB)
                Y
            end::Vector{TMapReturnEltype}
        else
            TMapReturnEltype[]
        end
    finally
        BLAS.set_num_threads(nblasthreads)
    end
end

function filter_eigenvalues(λs::AbstractVector{<:AbstractVector},
    vs::AbstractVector{<:AbstractMatrix}, mr::AbstractVector{<:Integer};
    matrixfn!,
    operators, constraints = constraintmatrix(operators), kw...)

    @unpack nr, nℓ, nparams = operators.radial_params
    nthreads = Threads.nthreads();
    bw = matrixfn! isa RotMatrix ? getbw(matrixfn!.kw.rotation_profile, nℓ) : nℓ
    ABs = if Filters.EIGEN in Filters.FilterFlag(get(kw, :filterflags, DefaultFilter))
        [allocate_operator_mass_matrices(operators, bw) for _ in 1:nthreads]
    else
        fill(nothing, nthreads)
    end
    c = Channel{eltype(ABs)}(nthreads);
    for el in ABs
        put!(c, el)
    end

    λv = filter_map_nthreads!(c, nthreads, λs, vs, mr, matrixfn!; operators, constraints, kw...)
    map(first, λv), map(last, λv)
end

function eigvec_spectrum_filter_map!(Ctid, spectrumfn!, m, operators, constraints;
        timer = TimerOutput(), kw...)
    M, cache, temp_projectback = Ctid;
    timerlocal = TimerOutput()
    @debug "starting computation for m = $m on tid = $(Threads.threadid()) with $(BLAS.get_num_threads()) BLAS threads"
    X = @timeit timerlocal "m=$m tid=$(Threads.threadid()) spectrum" begin
        spectrumfn!(M, m; operators, constraints, cache, temp_projectback, timer = timerlocal, kw...)
    end;
    @debug "computed eigenvalues for m = $m on tid = $(Threads.threadid()) with $(BLAS.get_num_threads()) BLAS threads"
    F = @timeit timerlocal "m=$m tid=$(Threads.threadid()) filter" begin
        kw2 = mergekw(spectrumfn!, kw)
        filter_eigenvalues(X..., m; operators, constraints, kw2...)
    end;
    @debug "filtered eigenvalues for m = $m on tid = $(Threads.threadid()) with $(BLAS.get_num_threads()) BLAS threads"
    merge!(timer, timerlocal, tree_point = ["spectrum_filter"])
    return F
end

function eigvec_spectrum_filter_map_nthreads!(c::Channel, nt, spectrumfn!, mr, operators, constraints; kw...)
    nblasthreads = BLAS.get_num_threads()

    TMapReturnEltype = Tuple{Vector{ComplexF64},
                    StructArray{ComplexF64, 2, @NamedTuple{re::Matrix{Float64},im::Matrix{Float64}}, Int64}}
    try
        nblasthreads_new = max(1, div(nblasthreads, nt))
        BLAS.set_num_threads(nblasthreads_new)
        @debug "using $nblasthreads_new BLAS threads for m range = $mr"
        if length(mr) > 0
            Folds.map(mr) do m
                Ctid = take!(c)
                Y = eigvec_spectrum_filter_map!(Ctid, spectrumfn!, m, operators, constraints; kw...)
                put!(c, Ctid)
                Y
            end::Vector{TMapReturnEltype}
        else
            TMapReturnEltype[]
        end
    finally
        BLAS.set_num_threads(nblasthreads)
    end
end

function filter_eigenvalues(spectrumfn!, mr::AbstractVector;
    operators, constraints = constraintmatrix(operators), kw...)

    timer = TimerOutput()

    @timeit timer "alloc" begin
        nthreads = Threads.nthreads()
        (; nℓ) = operators.radial_params;
        bw = spectrumfn! isa RotMatrix ? getbw(spectrumfn!.kw.rotation_profile, nℓ) : nℓ
        @timeit timer "M" ABs =
            [allocate_operator_mass_matrices(operators, bw) for _ in 1:nthreads];
        @timeit timer "caches" caches =
            [constrained_matmul_cache(constraints) for _ in 1:nthreads];
        @timeit timer "projectback" temp_projectback_mats =
            [allocate_projectback_temp_matrices(size(constraints.ZC)) for _ in 1:nthreads];
        z = zip(ABs, caches, temp_projectback_mats);
        c = Channel{eltype(z)}(nthreads);
        for el in z
            put!(c, el)
        end
    end

    @timeit timer "spectrum_filter" begin
        nblasthreads = BLAS.get_num_threads()
        nthreads_trailing_elems = rem(length(mr), nthreads)

        if nthreads_trailing_elems > 0 && div(nblasthreads, nthreads_trailing_elems) > max(1, div(nblasthreads, nthreads))
            # in this case the extra elements may be run using a higher number of blas threads
            mr1 = @view mr[1:end-nthreads_trailing_elems]
            mr2 = @view mr[end-nthreads_trailing_elems+1:end]

            @debug "starting first set for m range = $mr1 with $(Threads.nthreads()) threads"
            λv1 = eigvec_spectrum_filter_map_nthreads!(c, Threads.nthreads(), spectrumfn!, mr1, operators, constraints;
                timer, kw...)

            @debug "starting first set for m range = $mr2 with $nthreads_trailing_elems threads"
            λv2 = eigvec_spectrum_filter_map_nthreads!(c, nthreads_trailing_elems, spectrumfn!, mr2, operators, constraints;
                timer, kw...)

            λs, vs = map(first, λv1), map(last, λv1)
            λs2, vs2 = map(first, λv2), map(last, λv2)
            append!(λs, λs2)
            append!(vs, vs2)
        else
            @debug "starting for m range = $mr with $(Threads.nthreads()) threads"
            λv = eigvec_spectrum_filter_map_nthreads!(c, Threads.nthreads(), spectrumfn!, mr, operators, constraints;
                timer, kw...)
            λs, vs = map(first, λv), map(last, λv)
        end
    end
    get(kw, :print_timer, true) && println(timer)
    λs, vs
end

function filter_eigenvalues(Feig::FilteredEigen; kw...)
    @unpack operators = Feig
    kw2 = merge(Feig.kw, kw)
    matrixfn! = updaterotatationprofile(Feig)
    λfs, vfs =
        filter_eigenvalues(Feig.lams, Feig.vs, Feig.mr; operators, matrixfn!, kw2...);
    FilteredEigen(λfs, vfs, Feig.mr, kw2, operators)
end

function filter_eigenvalues(filename::String; kw...)
    Feig = FilteredEigen(filename)
    filter_eigenvalues(Feig; kw...)
end

rossbyeigenfilename(nr, nℓ, rottag, symtag, modeltag = "") =
    datadir("$(rottag)_nr$(nr)_nl$(nℓ)_$(symtag)$((isempty(modeltag) ? "" : "_") * modeltag).jld2")

function rossbyeigenfilename(; operators, V_symmetric, rotation_profile, kw...)
    modeltag = get(kw, :modeltag, "")
    symtag = V_symmetric ? "sym" : "asym"
    @unpack nr, nℓ = operators.radial_params;
    return rossbyeigenfilename(nr, nℓ, rotation_profile, symtag, modeltag)
end

function save_eigenvalues(spectrumfn!, mr; operators, save=true, kw...)
    lam, vec = filter_eigenvalues(spectrumfn!, mr; operators, kw...)
    kw2 = mergekw(spectrumfn!, kw)
    save && save_to_file(lam, vec, mr; operators, kw2...)
end

function save_to_file(lam, vec, mr; operators, kw...)
    fname = rossbyeigenfilename(; operators, kw...)
    @unpack operatorparams = operators
    kw_ = Dict{Symbol,Any}(kw)
    kw_[:filterflags] = UInt16(get(kw, :filterflags, DefaultFilter))
    println("saving to $fname")
    jldsave(fname; lam, vec = unwrapstruct(vec), mr, kw = kw_, operatorparams)
    return fname
end

unwrapstruct(vec::Vector{<:StructArray{<:Complex}}) = [(v.re, v.im) for v in vec]

function differential_rotation_matrix(Feig::FilteredEigen, m; kw...)
    differential_rotation_matrix(m; Feig.operators, Feig.kw..., kw...)
end

function mass_matrix(Feig::FilteredEigen, m; kw...)
    mass_matrix(m; Feig.operators, Feig.kw..., kw...)
end

function operator_matrices(m; operators, kw...)
    diffrot = kw[:rotation_profile] != :uniform
    A = if diffrot
        differential_rotation_matrix(m; operators, kw...)
    else
        uniform_rotation_matrix(m; operators, kw...)
    end
    B = mass_matrix(m; operators, kw...)
    map(computesparse, (A, B))
end

function operator_matrices(Feig::FilteredEigen, m; kw...)
    operator_matrices(m; Feig.operators, Feig.kw..., kw...)
end

function eigenfunction_spectrum_2D!(F, v; operators, kw...)
    @unpack radial_params = operators
    @unpack nparams, nr, nℓ = radial_params

    F.V .= @view v[1:nparams]
    F.W .= @view v[nparams.+(1:nparams)]
    F.S .= @view(v[2nparams.+(1:nparams)])

    V = reshape(F.V, nr, nℓ)
    W = reshape(F.W, nr, nℓ)
    S = reshape(F.S, nr, nℓ)

    (; V, W, S)
end

function eigenfunction_spectrum_2D(v; operators, kw...)
    @unpack nr, nℓ = operators.radial_params
    F = allocate_field_vectors(nr, nℓ)
    eigenfunction_spectrum_2D!(F, v; operators, kw...)
end

function eigenfunction_spectrum_2D(Feig, m, ind; kw...)
    eigenfunction_spectrum_2D(Feig[m][ind].v; Feig.operators, Feig.kw..., kw...)
end

function invtransform1!(radialspace::Space, out::AbstractMatrix, coeffs::AbstractMatrix,
        itransplan! = ApproxFunBase.plan_itransform!(radialspace, @view(out[:, 1])),
        )

    copyto!(out, coeffs)
    for outcol in eachcol(out)
        itransplan! * outcol
    end
    return out
end

function eigenfunction_rad_sh!(VWSinvsh, F, v; operators, n_lowpass_cutoff::Union{Int,Nothing} = nothing, kw...)
    VWS = eigenfunction_spectrum_2D!(F, v; operators, kw...)
    @unpack V, W, S = VWS
    @unpack radialspace = operators.radialspaces;

    Vinv = VWSinvsh.V
    Winv = VWSinvsh.W
    Sinv = VWSinvsh.S

    itransplan! = ApproxFunBase.plan_itransform!(radialspace, @view(Vinv[:, 1]))

    if !isnothing(n_lowpass_cutoff)
        field_lowpass = similar(V)
        field_lowpass .= V
        field_lowpass[n_lowpass_cutoff+1:end, :] .= 0
        invtransform1!(radialspace, Vinv, field_lowpass, itransplan!)
        field_lowpass .= W
        field_lowpass[n_lowpass_cutoff+1:end, :] .= 0
        invtransform1!(radialspace, Winv, field_lowpass, itransplan!)
        field_lowpass .= S
        field_lowpass[n_lowpass_cutoff+1:end, :] .= 0
        invtransform1!(radialspace, Sinv, field_lowpass, itransplan!)
    else
        invtransform1!(radialspace, Vinv, V, itransplan!)
        invtransform1!(radialspace, Winv, W, itransplan!)
        invtransform1!(radialspace, Sinv, S, itransplan!)
    end

    return VWSinvsh
end

function colatitude_grid(m::Integer, operators::OperatorWrap, ℓmax_mul = 4)
    colatitude_grid(m, operators.radial_params.nℓ, ℓmax_mul)
end
function colatitude_grid(m::Integer, nℓ::Integer, ℓmax_mul = 4)
    ℓs = range(m, length = 2nℓ+1)
    ℓmax = maximum(ℓs)

    θ, _ = sph_points(ℓmax_mul * ℓmax)
    return θ
end

function invshtransform2!(VWSinv, VWS, m;
    V_symmetric,
    nℓ = size(VWS.V, 2),
    Plcosθ = allocate_Pl(m, nℓ),
    Δl_lowpass_cutoff = nothing,
    kw...)

    V_lm = VWS.V
    W_lm = VWS.W
    S_lm = VWS.S

    θ = colatitude_grid(m, nℓ)

    V = VWSinv.V
    V .= 0
    W = VWSinv.W
    W .= 0
    S = VWSinv.S
    S .= 0

    V_ℓs = ℓrange(m, nℓ, V_symmetric)

    W_symmetric = !V_symmetric
    W_ℓs = ℓrange(m, nℓ, W_symmetric)

    _Δl_lowpass_cutoff = isnothing(Δl_lowpass_cutoff) ? lastindex(Plcosθ) : Δl_lowpass_cutoff

    @views for (θind, θi) in enumerate(θ)
        collectPlm!(Plcosθ, cos(θi); m, norm = Val(:normalized))
        # V
        for (ℓind, ℓ) in enumerate(V_ℓs)
            ℓ > m + _Δl_lowpass_cutoff && continue
            Plmcosθ = Plcosθ[ℓ]
            @. V[:, θind] += V_lm[:, ℓind] * Plmcosθ
        end
        # W, S
        for (ℓind, ℓ) in enumerate(W_ℓs)
            ℓ > m + _Δl_lowpass_cutoff && continue
            Plmcosθ = Plcosθ[ℓ]
            @. W[:, θind] += W_lm[:, ℓind] * Plmcosθ
            @. S[:, θind] += S_lm[:, ℓind] * Plmcosθ
        end
    end

    return VWSinv
end

function eigenfunction_realspace!(fieldcaches, v, m;
    operators,
    nℓ = operators.radial_params.nℓ,
    Plcosθ = allocate_Pl(m, nℓ),
    kw...)

    @unpack VWSinv, VWSinvsh, F = fieldcaches
    eigenfunction_rad_sh!(VWSinvsh, F, v; operators, kw...)
    invshtransform2!(VWSinv, VWSinvsh, m; nℓ, Plcosθ, kw...)
end

function eigenfunction_realspace(v, m; operators, kw...)
    @unpack nr, nℓ = operators.radial_params
    nθ = length(colatitude_grid(m, nℓ))

    fieldcaches = allocate_field_caches(nr, nℓ, nθ)
    eigenfunction_realspace!(fieldcaches, v, m; operators, kw...)
end

function eigenfunction_realspace(Feig::FilteredEigen, m, ind; kw...)
    eigenfunction_realspace(Feig[m][ind].v, m; Feig.operators, Feig.kw..., kw...)
end

function eigenfunction_n_theta!(VWSinv, F, v, m;
    operators,
    nℓ = operators.radial_params.nℓ,
    Plcosθ = allocate_Pl(m, nℓ),
    kw...)

    VW = eigenfunction_spectrum_2D!(F, v; operators, kw...)
    invshtransform2!(VWSinv, VW, m; nℓ, Plcosθ, kw...)
end

end # module
