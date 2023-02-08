module SolarModel

using Reexport

using ApproxFun
using BandedMatrices
using BlockArrays
using BlockBandedMatrices
using DelimitedFiles: readdlm
using Dierckx: Dierckx, Spline1D, Spline2D, derivative
using LinearAlgebra
using DomainSets
using FillArrays
using IntervalSets
using SparseArrays
using StructArrays
using UnPack

include("uniqueinterval.jl")

# cgs units
const G = 6.6743e-8
const Msun = 1.989e+33
const Rsun = 6.959894677e+10

const StructMatrix{T} = StructArray{T,2}

const SOLARMODELDIR = Ref(joinpath(dirname(@__FILE__), "solarmodel"))

function __init__()
	SOLARMODELDIR[] = joinpath(@__DIR__, "solarmodel")
end

export Rsun, Msun, G
export SOLARMODELDIR
export radial_operators
export constraintmatrix
export allocate_operator_mass_matrix
export allocate_operator_matrix
export allocate_mass_matrix
export radial_differential_rotation_profile_derivatives_Fun
export solar_differential_rotation_profile_derivatives_Fun
export computesparse
export StructMatrix
export interp1d
export interp2d
export smoothed_spline
export solar_structure_parameter_splines
export read_solar_model
export Tmul, Tplusinf, TplusInt
export replaceemptywitheps
export equatorial_rotation_angular_velocity_surface
export listncoefficients

const Tmul = typeof(Multiplication(Fun()) * Derivative())
const Tplusinf = typeof(Multiplication(Fun()) + Derivative())
const TplusInt = typeof(Multiplication(Fun(), Chebyshev()) + Derivative(Chebyshev()))


# const BlockMatrixType = BlockMatrix{Float64, Matrix{BlockBandedMatrix{Float64}},
#     Tuple{BlockedUnitRange{Vector{Int64}}, BlockedUnitRange{Vector{Int64}}}}
const BandedMatrixType = BandedMatrices.BandedMatrix{Float64, Matrix{Float64}, Base.OneTo{Int64}}

grid_to_fun(v::AbstractVector, space) = Fun(space, transform(space, v))
grid_to_fun(v::AbstractMatrix, space) = Fun(ProductFun(transform(space, v), space))

function operatormatrix(f::Fun, nr, spaceconversion::Pair)
    operatormatrix(Multiplication(f), nr, spaceconversion)
end

function operatormatrix(A, nr, spaceconversion::Pair)::Matrix{Float64}
    domain_space, range_space = first(spaceconversion), last(spaceconversion)
    C = A:domain_space → range_space
    C[1:nr, 1:nr]
end

function V_boundary_op(operators)::TplusInt
    @unpack r = operators.rad_terms;
    @unpack ddr = operators.diff_operators;
    @unpack radialspace = operators.radialspaces

    (r * ddr - 2) : radialspace;
end

function constraintmatrix(operators)
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

    (; BC, ZC, nullspacematrices, BCmatrices)
end

function parameters(nr, nℓ; r_in = 0.7Rsun, r_out = 0.98Rsun)
    nchebyr = nr
    r_mid = (r_in + r_out) / 2
    Δr = r_out - r_in
    nparams = nchebyr * nℓ
    return (; nchebyr, r_in, r_out, Δr, nr, nparams, nℓ, r_mid)
end

function superadiabaticity(r::Real; r_out = Rsun)
    δcz = 3e-6
    δtop = 3e-5
    dtrans = dtop = 0.05Rsun
    r_sub = 0.8 * Rsun
    r_tran = 0.725 * Rsun
    δrad = -1e-3
    δconv = δtop * exp((r - r_out) / dtop) + δcz * (r - r_sub) / (r_out - r_sub)
    δconv + (δrad - δconv) * 1 / 2 * (1 - tanh((r - r_tran) / dtrans))
end

read_solar_model() = readdlm(joinpath(SOLARMODELDIR[], "ModelS.detailed"))::Matrix{Float64}

function solar_structure_parameter_splines(; r_in = 0.7Rsun, r_out = Rsun, _stratified #= only for tests =# = true)
    ModelS = read_solar_model()
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
Base.show(io::IO, ::Type{<:OperatorWrap}) = print(io, "Operators")
Base.show(io::IO, o::OperatorWrap) = print(io, "Operators")
Base.getproperty(y::OperatorWrap, name::Symbol) = getproperty(getfield(y, :x), name)
Base.propertynames(y::OperatorWrap) = Base.propertynames(getfield(y, :x))

const DefaultScalings = (; Wscaling = 1e1, Sscaling = 1e6, Weqglobalscaling = 1e-3, Seqglobalscaling = 1.0, trackingratescaling = 1.0)
function radial_operators(nr, nℓ; r_in_frac = 0.6, r_out_frac = 0.985, _stratified = true, nvariables = 3, ν = 1e10,
    trackingrate = :cutoff,
    scalings = DefaultScalings)
    scalings = merge(DefaultScalings, scalings)
    radial_operators(nr, nℓ, r_in_frac, r_out_frac, _stratified, nvariables, ν, trackingrate, Tuple(scalings))
end
function radial_operators(operatorparams...)
    nr, nℓ, r_in_frac, r_out_frac, _stratified, nvariables, ν, trackingrate, _scalings = operatorparams

    Wscaling, Sscaling, Weqglobalscaling, Seqglobalscaling, trackingratescaling = _scalings

    r_in = r_in_frac * Rsun;
    r_out = r_out_frac * Rsun;
    radialdomain = UniqueInterval(r_in..r_out)
    radialspace = Chebyshev(radialdomain)
    radial_params = parameters(nr, nℓ; r_in, r_out);
    @unpack Δr, nchebyr, r_mid = radial_params;
    rpts = points(radialspace, nr);

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

    # density stratification
    ρ = replaceemptywitheps(ApproxFun.chop(Fun(sρ, radialspace), 1e-3));
    @checkncoeff ρ nr

    ηρ = replaceemptywitheps(ApproxFun.chop(Fun(sηρ, radialspace), 1e-3));
    @checkncoeff ηρ nr

    ηT = replaceemptywitheps(ApproxFun.chop(Fun(sηT, radialspace), 1e-3));
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

    ddr_ηρ = chop(Fun(ddrsηρ, radialspace), 1e-2)
    @checkncoeff ddr_ηρ nr

    ddr_ηρbyr = chop(Fun(ddrsηρ_by_r, radialspace), 1e-2)
    @checkncoeff ddr_ηρbyr nr

    # ddr_ηρbyr = ddr * ηρ_by_r
    d2dr2_ηρ = chop!(Fun(d2dr2sηρ, radialspace), 1e-2);
    @checkncoeff d2dr2_ηρ nr

    d3dr3_ηρ = chop(Fun(d3dr3sηρ, radialspace), 1e-2)
    @checkncoeff d3dr3_ηρ nr

    ddr_ηρbyr2 = chop(Fun(ddrsηρ_by_r2, radialspace), 5e-3)
    @checkncoeff ddr_ηρbyr2 nr

    # ddr_ηρbyr2 = ddr * ηρ_by_r2
    ηρ2_by_r2 = ApproxFun.chop(ηρ_by_r2 * ηρ, 1e-3)
    @checkncoeff ηρ2_by_r2 nr

    ddrηρ_by_r = ddr_ηρ * onebyr
    d2dr2ηρ_by_r = d2dr2_ηρ * onebyr

    ddrDDr = (d2dr2 + (ηρ * ddr)::Tmul + ddr_ηρ)::Tplusinf
    d2dr2DDr = (d3dr3 + (ηρ * d2dr2)::Tmul + (2ddr_ηρ * ddr)::Tmul + d2dr2_ηρ)::Tplusinf

    d2dr2_ηρbyr_op::Tplusinf = ηρ_by_r * d2dr2 + 2(ddr_ηρ * onebyr - ηρ * onebyr2)*ddr +
        (d2dr2_ηρ * onebyr - 2ddr_ηρ * onebyr2 + 2ηρ * onebyr3)

    g = Fun(sg, radialspace)

    tracking_rate_rad = if trackingrate === :cutoff
            r_out_frac
        elseif trackingrate === :surface
            1.0
        else
            throw(ArgumentError("trackingrate must be one of :cutoff or :surface"))
        end
    Ω0 = equatorial_rotation_angular_velocity_surface(tracking_rate_rad) * trackingratescaling

    # viscosity
    ν /= Ω0 * Rsun^2
    κ = ν

    δ = Fun(superadiabaticity, radialspace);
    γ = 1.64
    cp = 1.7e8

    # ddr_S0_by_cp = γ/cp * δ * ηρ;
    ddr_S0_by_cp = chop(Fun(x -> γ / cp * superadiabaticity(x) * ηρ(x), radialspace), 1e-3);
    @checkncoeff ddr_S0_by_cp nr

    ddr_S0_by_cp_by_r2 = onebyr2 * ddr_S0_by_cp

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
        ddr_ηρ, d2dr2_ηρ, d3dr3_ηρ, ddr_S0_by_cp_by_r2,
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
        radial_params,
        operatorparams,
    )

    OperatorWrap(op)
end

function listncoefficients(d)
    sort([k=>ncoefficients(v) for (k,v) in Dict(pairs(d))], by=last, rev=true)
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

allocate_operator_mass_matrix(operators, bw...) = (
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

function interp2d(xin, yin, z, xout = xin, yout = yin; s = 0.0)
    px = sortperm(xin)
    py = sortperm(yin)
    spline = Spline2D(xin[px], yin[py], z[px, py]; s = sum(abs2, z) * s)
    spline.(xout, yout')
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


function equatorial_rotation_angular_velocity_surface(r_frac::Number = 1.0,
        r_ΔΩ_raw = solar_rotation_profile_radii(), Ω_raw = solar_rotation_profile_raw())
    Ω_raw_r_eq = equatorial_rotation_angular_velocity_radial_profile(Ω_raw)
    r_frac_ind = findmin(abs, r_ΔΩ_raw .- r_frac)[2]
    Ω_raw_r_eq[r_frac_ind]
end

function equatorial_rotation_angular_velocity_radial_profile(Ω_raw = solar_rotation_profile_raw())
    nθ = size(Ω_raw, 2)
    lats_raw = range(0, pi, length=nθ)
    θind_equator = findmin(abs, lats_raw .- pi/2)[2]
    @view Ω_raw[:, θind_equator]
end

function solar_rotation_profile_spline(; operators, smoothing_param = 1e-5, kw...)
    @unpack r_out = operators.radial_params;
    @unpack Ω0 = operators.constants;

    r_ΔΩ_raw = solar_rotation_profile_radii(SOLARMODELDIR[])
    Ω_raw = solar_rotation_profile_raw(SOLARMODELDIR[])
    ΔΩ_raw = Ω_raw .- Ω0

    nθ = size(ΔΩ_raw, 2)
    lats_raw = LinRange(0, pi, nθ)
    cos_lats_raw = cos.(lats_raw) # decreasing order, must be flipped in Spline2D

    Spline2D(r_ΔΩ_raw * Rsun, reverse(cos_lats_raw), reverse(ΔΩ_raw, dims=2); s = sum(abs2, ΔΩ_raw)*smoothing_param)
end

function solar_rotation_profile_and_derivative_grid(splΔΩ2D, rpts, θpts)
    ΔΩ_fullinterp = splΔΩ2D.(rpts, θpts')
    dr_ΔΩ_fullinterp = derivative.((splΔΩ2D,), rpts, θpts', nux = 1, nuy = 0)
    d2r_ΔΩ_fullinterp = derivative.((splΔΩ2D,), rpts, θpts', nux = 2, nuy = 0)
    ΔΩ_fullinterp, dr_ΔΩ_fullinterp, d2r_ΔΩ_fullinterp
end
function solar_rotation_profile_and_derivative_grid(; operators, kw...)
    @unpack rpts = operators
    @unpack nℓ = operators.radial_params
    θpts = points(ChebyshevInterval(), nℓ)
    splΔΩ2D = solar_rotation_profile_spline(; operators, kw...)
    solar_rotation_profile_and_derivative_grid(splΔΩ2D, rpts, θpts)
end

function _equatorial_rotation_profile_and_derivative_grid(splΔΩ2D, rpts)
    equator_coord = 0.0 # cos(θ) for θ = pi/2
    ΔΩ_r = splΔΩ2D.(rpts, equator_coord)
    ddrΔΩ_r = derivative.((splΔΩ2D,), rpts, equator_coord, nux = 1, nuy = 0)
    d2dr2ΔΩ_r = derivative.((splΔΩ2D,), rpts, equator_coord, nux = 2, nuy = 0)
    ΔΩ_r, ddrΔΩ_r, d2dr2ΔΩ_r
end
function equatorial_rotation_profile_and_derivative_grid(; operators, kw...)
    @unpack rpts = operators
    splΔΩ2D = solar_rotation_profile_spline(; operators, kw...)
    _equatorial_rotation_profile_and_derivative_grid(splΔΩ2D, rpts)
end

function equatorial_rotation_profile_and_derivative_squished_grid(; operators, kw...)
    @unpack rpts = operators;
    @unpack r_out = operators.radial_params;
    splΔΩ2D = solar_rotation_profile_spline(; operators, kw...)
    rpts_stretched = rpts ./ r_out .* Rsun
    _equatorial_rotation_profile_and_derivative_grid(splΔΩ2D, rpts_stretched)
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
            equatorial_rotation_profile_and_derivative_squished_grid(; operators, kw...)
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
        ΔΩ_spl = smoothed_spline(r_new, ΔΩ_r_new; s = get(kw, :smoothing_param, 1e-5))
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
    ΔΩ_terms = radial_differential_rotation_profile_derivatives_grid(; operators, kw...);
    ΔΩ_r, ddrΔΩ_r, d2dr2ΔΩ_r = ΔΩ_terms
    nr = length(ΔΩ_r)

    ΔΩ = chop(grid_to_fun(ΔΩ_r, radialspace), 1e-3);
    @checkncoeff ΔΩ nr

    ddrΔΩ = chop(grid_to_fun(interp1d(rpts, ddrΔΩ_r, s = 1e-3), radialspace), 1e-3);
    @checkncoeff ddrΔΩ nr

    d2dr2ΔΩ = chop(grid_to_fun(interp1d(rpts, d2dr2ΔΩ_r, s = 1e-3), radialspace), 1e-3);
    @checkncoeff d2dr2ΔΩ nr

    ΔΩ, ddrΔΩ, d2dr2ΔΩ = map(replaceemptywitheps, (ΔΩ, ddrΔΩ, d2dr2ΔΩ))
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
    elseif rotation_profile == :radial_equator
        ΔΩ_r_, ddrΔΩ_r_, d2dr2ΔΩ_r_ = radial_differential_rotation_profile_derivatives_grid(;
            operators, rotation_profile = :solar_equator, kw...)
        for x in (ΔΩ_r_, ddrΔΩ_r_, d2dr2ΔΩ_r_)
            x .*= Ω0
        end
        ΔΩ = repeat(ΔΩ_r_, 1, nℓ)
        dr_ΔΩ = repeat(ddrΔΩ_r_, 1, nℓ)
        d2r_ΔΩ = repeat(d2dr2ΔΩ_r_, 1, nℓ)
    elseif rotation_profile == :latrad
        ΔΩ, dr_ΔΩ, d2r_ΔΩ =
            solar_rotation_profile_and_derivative_grid(; operators, kw...)
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
    @unpack nℓ = operators.radial_params
    θpts = points(ChebyshevInterval(), nℓ);

    ΔΩ_terms = solar_differential_rotation_profile_derivatives_grid(; operators, kw...);
    ΔΩ_rθ, dr_ΔΩ_rθ, d2r_ΔΩ_rθ = ΔΩ_terms;

    space = radialspace ⊗ NormalizedLegendre()

    cosθ = Fun(Legendre());
    Ir = I : radialspace;
    sinθdθop = -(1-cosθ^2)*Derivative(Legendre())

    s = get(kw, :smoothing_param, 1e-5)

    ΔΩ = chop(grid_to_fun(ΔΩ_rθ, space), s);

    dr_ΔΩ = chop(grid_to_fun(interp2d(rpts, θpts, dr_ΔΩ_rθ, s = s), space), s);
    dz_ΔΩ_ = (Ir ⊗ cosθ) * dr_ΔΩ - (onebyr ⊗ sinθdθop) * ΔΩ;
    dz_ΔΩ = chop(Fun(dz_ΔΩ_, space))::typeof(ΔΩ);

    d2r_ΔΩ = chop(grid_to_fun(interp2d(rpts, θpts, d2r_ΔΩ_rθ, s = s), space), s);

    ΔΩ, dr_ΔΩ, d2r_ΔΩ, dz_ΔΩ =
        map(replaceemptywitheps, (ΔΩ, dr_ΔΩ, d2r_ΔΩ, dz_ΔΩ))

    (; ΔΩ, dr_ΔΩ, d2r_ΔΩ, dz_ΔΩ)
end

end # module SolarModel
