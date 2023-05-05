module RossbyWaveSpectrum

using MKL

using Reexport

@reexport using ApproxFun
@reexport using ApproxFunAssociatedLegendre
@reexport using BlockArrays
using BlockBandedMatrices
import Dierckx
using Folds
using JLD2
@reexport using LinearAlgebra
using LinearAlgebra: BLAS
using LegendrePolynomials
using OffsetArrays
using SolarModel
using SparseArrays
using StructArrays
using TimerOutputs
@reexport using UnPack
using ZChop

export datadir
export rossbyeigenfilename
export Filters
export filteredeigen
export FilteredEigen
export Rsun
export SolarModel

const SCRATCH = Ref("")
const DATADIR = Ref("")

function __init__()
    SCRATCH[] = get(ENV, "SCRATCH", homedir())
    DATADIR[] = get(ENV, "DATADIR", joinpath(SCRATCH[], "RossbyWaves"))
    if !ispath(RossbyWaveSpectrum.DATADIR[])
        mkdir(RossbyWaveSpectrum.DATADIR[])
    end
end

datadir(f) = joinpath(DATADIR[], f)

function sph_points(N)
    @assert N > 0
    M = 2 * N - 1
    return π / N * (0.5:(N-0.5)), 2π / M * (0:(M-1))
end

# Legedre expansion constants
α⁻ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((ℓ - m) * (ℓ + m) / ((2ℓ - 1) * (2ℓ + 1)))))

β⁻ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((2ℓ + 1) / (2ℓ - 1) * (ℓ^2 - m^2))))
γ⁺ℓm(ℓ, m) = ℓ * α⁻ℓm(ℓ+1, m)
γ⁻ℓm(ℓ, m) = ℓ * α⁻ℓm(ℓ, m) - β⁻ℓm(ℓ, m)

function costheta_operator_matrix(nℓ, m)
    space = NormalizedPlm(m)
    C = cosθ_Operator(space)
    C[1:nℓ, 1:nℓ]
end

function sintheta_dtheta_operator_matrix(nℓ, m)
    space = NormalizedPlm(m)
    S = sinθdθ_Operator(space)
    S[1:nℓ, 1:nℓ]
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

matrix_block_apply(fred, f, M::AbstractMatrix, nblocks = 3) = [fred(f, matrix_block(M, i, j, nblocks)) for i in 1:nblocks, j in 1:nblocks]

matrix_block_maximum(f, M::AbstractMatrix, nblocks = 3) = matrix_block_apply(maximum, f, M, nblocks)
matrix_block_maximum(M::AbstractMatrix, nblocks = 3) = matrix_block_apply(maximum, M, nblocks)

function matrix_block_apply(fred, f, M::BlockBandedMatrix, nblocks = 3)
    fred(f, (fred(f, b) for b in blocks(M)))
end
function matrix_block_apply(fred, f, M::BlockMatrix, nblocks = 3)
    [matrix_block_apply(fred, f, Mb, nblocks) for Mb in blocks(M)]
end
function matrix_block_apply(fred, M::StructMatrix{<:Complex}, nblocks = 3)
    R = matrix_block_apply(fred, abs, M.re, nblocks)
    I = matrix_block_apply(fred, abs, M.im, nblocks)
    [R I]
end
function matrix_block_apply(fred, M::AbstractMatrix{<:Complex}, nblocks = 3)
    R = matrix_block_apply(fred, abs∘real, M, nblocks)
    I = matrix_block_apply(fred, abs∘imag, M, nblocks)
    [R I]
end
function matrix_block_apply(fred, M::AbstractMatrix{<:Real}, nblocks = 3)
    matrix_block_apply(fred, abs∘real, M, nblocks)
end

ℓrange(m, nℓ, symmetric) = range(m + !symmetric, length = nℓ, step = 2)

function mass_matrix(m; operators, kw...)
    B = allocate_mass_matrix(operators)
    mass_matrix!(B, m; operators, kw...)
    return B
end
function mass_matrix!(B, m; operators, V_symmetric = true, kw...)
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
    @unpack ddr, d2dr2, DDr, ddrDDr, ∇r2_plus_ddr_lnρT_ddr = operators.diff_operators;
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
            (DDr - 2*onebyr) ⊗ (cosθop * ∇²)
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

function viscosity_terms!(A::StructMatrix{<:Complex}, m; operators, V_symmetric = true, kw...)
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
        operators, ΔΩ_frac = 0.01, V_symmetric = true, kw...)

    @unpack nr, nℓ = operators.radial_params;
    @unpack Wscaling, Weqglobalscaling, Seqglobalscaling = operators.scalings;

    @unpack ddr, DDr, ddrDDr = operators.diff_operators;
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
    VWop_ = ΔΩ_frac * (Ir ⊗ (-4inv_ℓℓp1)) * ((DDr - 2onebyr) ⊗ (cosθop * ℓℓp1op) +
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
        V_symmetric = true, kw...)

    @unpack nr, nℓ = operators.radial_params;
    @unpack DDr, ddr, ddrDDr = operators.diff_operators;
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
        (ΔΩ * (DDr - 2onebyr) - ddrΔΩ) ⊗ (cosθop * ℓℓp1op) +
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

function solar_differential_rotation_vorticity_Fun(; operators, ΔΩprofile_deriv)
    @unpack onebyr, r, onebyr2, twobyr = operators.rad_terms;
    @unpack ddr, d2dr2 = operators.diff_operators;

    (; ΔΩ, dr_ΔΩ, d2r_ΔΩ) = ΔΩprofile_deriv;

    @unpack radialspace = operators.radialspaces;
    # velocity and its derivatives are expanded in Legendre poly
    latitudinal_space = NormalizedPlm(0, NormalizedLegendre());
    space2d = radialspace ⊗ latitudinal_space

    cosθ = Fun(Legendre());
    Ir = I : radialspace;
    Iℓ = I : latitudinal_space;
    ∇² = HorizontalLaplacian(latitudinal_space);

    sinθdθ_plus_2cosθ = sinθdθ_plus_2cosθ_Operator(latitudinal_space);
    ωΩr = (Ir ⊗ sinθdθ_plus_2cosθ) * ΔΩ;
    ∂rωΩr = (Ir ⊗ sinθdθ_plus_2cosθ) * dr_ΔΩ;
    # cotθddθ = cosθ * 1/sinθ * d/dθ = -cosθ * d/d(cosθ) = -x*d/dx
    cotθdθ = KroneckerOperator(Ir, -cosθ * Derivative(Legendre()),
        radialspace * Legendre(), radialspace * Jacobi(1,1));
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

function solar_differential_rotation_terms!(M::StructMatrix{<:Complex}, m;
        operators, rotation_profile = :latrad,
        ΔΩ_frac = 0.01, # only used to test the constant case
        ΔΩ_scale = 1.0, # scale the diff rot profile, for testing
        ΔΩprofile_deriv = solar_differential_rotation_profile_derivatives_Fun(;
                            operators, rotation_profile, ΔΩ_frac, ΔΩ_scale),
        ωΩ_deriv = solar_differential_rotation_vorticity_Fun(; operators, ΔΩprofile_deriv),
        V_symmetric = true, kw...)

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
    v = StructArray{ComplexF64}((vr, vi))
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
    λ, v, (A, B)
end

function uniform_rotation_spectrum(m; operators, kw...)
    AB = allocate_operator_mass_matrix(operators, 2)
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
    AB = allocate_operator_mass_matrix(operators, bw)
    differential_rotation_spectrum!(AB, m; operators, rotation_profile, kw...)
end
function differential_rotation_spectrum!((A, B)::Tuple{StructMatrix{<:Complex}, AbstractMatrix{<:Real}},
        m::Integer; rotation_profile, operators, timer = TimerOutput(), kw...)
    @timeit timer "matrix" begin
        differential_rotation_matrix!(A, m; operators, rotation_profile, kw...)
        mass_matrix!(B, m; operators)
    end
    constrained_eigensystem_timed((A, B); operators, timer, kw...)
end

struct RotMatrix{TV,TW,F}
    V_symmetric :: Bool
    rotation_profile :: Symbol
    ΔΩprofile_deriv :: TV
    ωΩ_deriv :: TW
    f :: F
end
function updaterotatationprofile(d::RotMatrix, operators; timer = TimerOutput(), kw...)
    if String(d.rotation_profile) == "constant"
        ΔΩprofile_deriv = @timeit timer "velocity" begin nothing end
        ωΩ_deriv = @timeit timer "vorticity" begin nothing end
    elseif startswith(String(d.rotation_profile), "radial")
        ΔΩprofile_deriv = @timeit timer "velocity" begin
            radial_differential_rotation_profile_derivatives_Fun(; operators,
            rotation_profile = rotationtag(d.rotation_profile), kw...)
        end
        ωΩ_deriv = @timeit timer "vorticity" begin nothing end
    elseif startswith(String(d.rotation_profile), "solar")
        ΔΩprofile_deriv = @timeit timer "velocity" begin
            solar_differential_rotation_profile_derivatives_Fun(; operators,
            rotation_profile = rotationtag(d.rotation_profile), kw...)
        end
        ωΩ_deriv = @timeit timer "vorticity" begin
            solar_differential_rotation_vorticity_Fun(; operators, ΔΩprofile_deriv)
        end
    else
        error("unknown rotation profile, must be one of :constant, :radial_* or solar_*")
    end
    return RotMatrix(d.V_symmetric, d.rotation_profile, ΔΩprofile_deriv, ωΩ_deriv, d.f)
end
updaterotatationprofile(d, _) = d

(d::RotMatrix)(args...; kw...) = d.f(args...;
    rotation_profile = d.rotation_profile,
    ΔΩprofile_deriv = d.ΔΩprofile_deriv,
    V_symmetric = d.V_symmetric,
    ωΩ_deriv = d.ωΩ_deriv,
    kw...)

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

function eigensystem_satisfy_filter(λ::Number, v::StructVector{<:Complex},
        AB::Tuple{StructMatrix{<:Complex}, AbstractMatrix{<:Real}}, args...; kw...)

    eigensystem_satisfy_filter(λ, v, map(computesparse, AB), args...; kw...)
end

const TStructSparseComplexMat{T} = @NamedTuple{re::SparseMatrixCSC{T, Int64}, im::SparseMatrixCSC{T, Int64}}
function eigensystem_satisfy_filter(λ::Number, v::StructVector{<:Complex},
        AB::Tuple{StructArray{Complex{T},2,TStructSparseComplexMat{T}}, SparseMatrixCSC{T, Int64}},
        MVcache::NTuple{2, StructArray{<:Complex,1}} = allocate_MVcache(size(AB[1], 1));
        rtol = DefaultFilterParams[:eigen_rtol]) where {T<:Real}

    A, B = AB;
    Av, λBv = MVcache;

    mul!(Av.re, A.re, v.re)
    mul!(Av.re, A.im, v.im, -1.0, 1.0)
    mul!(Av.im, A.re, v.im)
    mul!(Av.im, A.im, v.re,  1.0, 1.0)

    mul!(λBv.re, B, v.re)
    mul!(λBv.im, B, v.im)
    λBv .*= λ

    normAv = norm(Av)
    normλBv = norm(λBv)
    Av .-= λBv
    norm(Av) <= rtol*max(normAv, normλBv)
end

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

function eigvec_spectrum_filter!(F, v, m;
    operators,
    n_cutoff = DefaultFilterParams[:n_cutoff],
    Δl_cutoff = DefaultFilterParams[:Δl_cutoff],
    eigvec_spectrum_power_cutoff = DefaultFilterParams[:eigvec_spectrum_power_cutoff],
    filterfieldpowercutoff = DefaultFilterParams[:filterfieldpowercutoff],
    kw...)

    VW = eigenfunction_spectrum_2D!(F, v; operators, kw...)
    Δl_inds = Δl_cutoff ÷ 2

    @unpack nparams = operators.radial_params
    @unpack nvariables = operators

    flag = true
    fields = filterfields(VW, v, nparams, nvariables; filterfieldpowercutoff)

    @views for _X in fields
        f, X = first(_X), last(_X)
        PV_frac_real = sum(abs2∘real, X[1:n_cutoff, 1:Δl_inds]) / sum(abs2∘real, X)
        PV_frac_imag = sum(abs2∘imag, X[1:n_cutoff, 1:Δl_inds]) / sum(abs2∘imag, X)
        flag &= (PV_frac_real > eigvec_spectrum_power_cutoff) & (PV_frac_imag > eigvec_spectrum_power_cutoff)
        flag || break
    end

    return flag
end

allocate_Pl(m, nℓ) = zeros(range(m, length = 2nℓ + 1))

function peakindabs1(X)
    findmax(v -> sum(abs2, v), eachrow(X))[2]
end

function spatial_filter!(VWSinv, VWSinvsh, F, v, m;
    operators,
    V_symmetric,
    θ_cutoff = DefaultFilterParams[:θ_cutoff],
    equator_power_cutoff_frac = DefaultFilterParams[:equator_power_cutoff_frac],
    filterfieldpowercutoff = DefaultFilterParams[:filterfieldpowercutoff],
    nℓ = operators.radial_params.nℓ,
    Plcosθ = allocate_Pl(m, nℓ),
    angular_filter = true,
    radial_filter = true,
    compute_invtransform = true,
    kw...
    )

    (; θ) = spharm_θ_grid_uniform(m, nℓ)
    eqind = indexof_equator(θ)

    if compute_invtransform
        eigenfunction_realspace!(VWSinv, VWSinvsh, F, v, m;
            operators, nℓ, Plcosθ, V_symmetric)
    end

    eqfilter = true

    radfilter = true

    @unpack nparams, r_out, r_in = operators.radial_params
    @unpack nvariables = operators
    @unpack rpts = operators
    fields = filterfields(VWSinv, v, nparams, nvariables; filterfieldpowercutoff)

    for _X in fields
        f, X = first(_X), last(_X)
        if angular_filter
            r_ind_peak = peakindabs1(X)
            peak_latprofile = @view X[r_ind_peak, :]
            θlowind = searchsortedfirst(θ, θ_cutoff)
            θhighind = searchsortedlast(θ, pi - θ_cutoff)
            powfrac = sum(abs2, @view peak_latprofile[θlowind:θhighind]) / sum(abs2, peak_latprofile)
            powflag = powfrac > equator_power_cutoff_frac
            peak_latprofile_max = maximum(abs2, peak_latprofile)
            peak_latprofile_max_inrange = maximum(abs2, @view peak_latprofile[θlowind:θhighind])
            peakflag = peak_latprofile_max_inrange == peak_latprofile_max
            eqfilter &= powflag & peakflag
            eqfilter || break
        end

        if radial_filter
            # ensure that the radial peak isn't at the bottom of the domain
            maxradpowind = findmax(y -> sum(abs, y), eachslice(X, dims=1))[2]
            radfilter = rpts[maxradpowind] > r_in + (r_out - r_in)/10 # 10% above the lower boundary
            radfilter || break

            # ensure that most of the power isn't concentrated in the top and bottom surface layers
            r_top_cutoff = r_out - (r_out - r_in)*5/100
            r_out_ind = findmin(r -> abs(r - r_out), rpts)[2]
            r_top_ind = findmin(r -> abs(r - r_top_cutoff), rpts)[2]
            r_bot_cutoff = r_in + (r_out - r_in)*5/100
            r_bot_ind = findmin(r -> abs(r - r_bot_cutoff), rpts)[2]
            r_in_ind = findmin(r -> abs(r - r_in), rpts)[2]
            top_power = sum(abs, view(X, :, r_out_ind:r_top_ind))
            bot_power = sum(abs, view(X, :, r_bot_ind:r_in_ind))
            tot_power = sum(abs, X)
            radfilter = (top_power + bot_power)/tot_power < 0.95
            radfilter || break
        end
    end

    return eqfilter & radfilter
end

function spatial_filter(v, m;
    operators,
    filtercache = allocate_filter_caches(m; operators),
    kw...
    )

    (; VWSinv, VWSinvsh, F) = filtercache
    spatial_filter!(VWSinv, VWSinvsh, F, v, m; operators, kw...)
end

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

angularindex_equator(Vr, θ) = findmin(x -> abs(x - pi/2), θ)[2]
indexof_equator(θ) = angularindex_equator(nothing, θ)

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

function count_num_nodes!(radprof::AbstractVector{<:Real}, rpts; smallcutoff = 0.1)
    rpts2, radprof = reverse(rpts), reverse!(radprof)
    radprof .*= argmax(abs.(extrema(radprof))) == 2 ? 1 : -1
    zerocrossings = sign_changes(radprof)
    iszero(zerocrossings) && return 0
    s = smoothed_spline(rpts2, radprof)
    radroots = Dierckx.roots(s, maxn = 2zerocrossings)
    isempty(radroots) && return 0

    # Discount nodes that appear spurious
    radprof .= abs.(radprof)
    sa = smoothed_spline(rpts2, radprof)
    unsignedarea = Dierckx.integrate(sa, rpts2[1], rpts2[end])

    signed_areas = zeros(Float64, length(radroots)+1)
    signed_areas[1] = Dierckx.integrate(s, rpts2[1], radroots[1])
    for (ind, (spt, ept)) in enumerate(zip(@view(radroots[1:end-1]), @view(radroots[2:end])))
        signed_areas[ind+1] = Dierckx.integrate(s, spt, ept)
    end
    signed_areas[end] = Dierckx.integrate(s, radroots[end], rpts2[end])

    largecrossings = abs.(signed_areas ./ unsignedarea) .> smallcutoff
    signed_areas = signed_areas[largecrossings]
    ncross = sign_changes(signed_areas)

    min(ncross, zerocrossings)
end
function count_radial_nodes_equator(V::AbstractMatrix{<:Complex},
        angularindex, rpts,
        temp = similar(V, real(eltype(V)), size(V,1)))

    radprof = @view V[:, angularindex]
    temp .= real.(radprof)
    nnodes_real = count_num_nodes!(temp, rpts)
    temp .= imag.(radprof)
    nnodes_imag = count_num_nodes!(temp, rpts)
    nnodes_real, nnodes_imag
end
function count_V_radial_nodes(v::AbstractVector{<:Complex}, m; operators,
        angularindex_fn = angularindex_equator,
        kw...)
    @unpack rpts = operators
    (; VWSinv, θ) = eigenfunction_realspace(v, m; operators, kw...)
    (; V) = VWSinv
    eqind = angularindex_fn(real(V), θ)
    count_radial_nodes_equator(V, eqind, rpts)
end

function nodes_filter!(VWSinv, VWSinvsh, F, v, m, operators;
    nℓ = operators.radial_params.nℓ,
    Plcosθ = allocate_Pl(m, nℓ),
    filterfieldpowercutoff = DefaultFilterParams[:filterfieldpowercutoff],
    nnodesmax = DefaultFilterParams[:nnodesmax],
    compute_invtransform = true,
    radproftemp = similar(VWSinv.V, real(eltype(VWSinv.V)), size(VWSinv.V,1)),
    kw...)

    nodesfilter = true

    @unpack nparams = operators.radial_params
    @unpack nvariables = operators
    @unpack rpts = operators
    fields = filterfields(VWSinv, v, nparams, nvariables; filterfieldpowercutoff)

    (; θ) = spharm_θ_grid_uniform(m, nℓ)
    eqind = indexof_equator(θ)

    if compute_invtransform
        eigenfunction_realspace!(VWSinv, VWSinvsh, F, v, m; operators, nℓ, Plcosθ, kw...)
    end

    for _X in fields
        f, X = first(_X), last(_X)
        radprof = @view X[:, eqind]
        nnodes_real, nnodes_imag = count_radial_nodes_equator(X, eqind, rpts, radproftemp)
        nodesfilter &= nnodes_real <= nnodesmax && nnodes_imag <= nnodesmax
        nodesfilter || break
    end
    return nodesfilter
end

module Filters
    using BitFlags
    export DefaultFilter
    @bitflag FilterFlag::UInt8 begin
        NONE=0
        EIGEN # Eigensystem satisfy
        EIGVAL # Imaginary to real ratio
        EIGVEC # spectrum power cutoff in n and l
        BC # boundary conditions
        SPATIAL_EQUATOR # peak near the equator
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

    const DefaultFilter = EIGEN | EIGVAL | EIGVEC | BC | SPATIAL_RADIAL | NODES
    const SPATIAL = SPATIAL_EQUATOR | SPATIAL_RADIAL
end
using .Filters

const DefaultFilterParams = Dict(
    # boundary condition filter
    :bc_atol => 1e-5,
    # eigval filter
    :eig_imag_unstable_cutoff => -1e-6,
    :eig_imag_to_real_ratio_cutoff => 3,
    :eig_imag_stable_cutoff => 0.5,
    # eigensystem satisfy filter
    :eigen_rtol => 0.01,
    # smooth eigenvector filter
    :Δl_cutoff => 7,
    :n_cutoff => 10,
    :eigvec_spectrum_power_cutoff => 0.5,
    # spatial localization filter
    :θ_cutoff => deg2rad(45),
    :equator_power_cutoff_frac => 0.4,
    # radial nodes filter
    :nnodesmax => 10,
    # exclude a field from a filter if relative power is below a cutoff
    :filterfieldpowercutoff => 1e-4,
)

function filterfn(λ, v, m, M, filterparams;
        operators,
        constraints = constraintmatrix(operators),
        filtercache = allocate_filter_caches(m; operators, constraints),
        filterflags = DefaultFilter)

    @unpack BC = constraints
    @unpack nℓ = operators.radial_params;

    filterparams = merge(DefaultFilterParams, filterparams);

    @unpack eig_imag_unstable_cutoff = filterparams
    @unpack eig_imag_to_real_ratio_cutoff = filterparams
    @unpack eig_imag_stable_cutoff = filterparams
    @unpack eigvec_spectrum_power_cutoff = filterparams
    @unpack bc_atol = filterparams
    @unpack Δl_cutoff = filterparams
    @unpack n_cutoff = filterparams
    @unpack θ_cutoff = filterparams
    @unpack equator_power_cutoff_frac = filterparams
    @unpack eigen_rtol = filterparams
    @unpack filterfieldpowercutoff = filterparams
    @unpack nnodesmax = filterparams
    @unpack V_symmetric = filterparams

    (; MVcache, BCVcache, VWSinv, VWSinvsh, Plcosθ, F, radproftemp) = filtercache;

    allfilters = Filters.FilterFlag(filterflags)
    compute_invtransform = true

    if Filters.EIGVAL in allfilters
        f = eigenvalue_filter(λ, m;
        eig_imag_unstable_cutoff, eig_imag_to_real_ratio_cutoff, eig_imag_stable_cutoff)
        if !f
            @debug "EIGVAL" λ, eig_imag_unstable_cutoff, eig_imag_to_real_ratio_cutoff, eig_imag_stable_cutoff
            return false
        end
    end

    if Filters.EIGVEC in allfilters
        f = eigvec_spectrum_filter!(F, v, m; operators,
            n_cutoff, Δl_cutoff, eigvec_spectrum_power_cutoff,
            filterfieldpowercutoff)
        if !f
            @debug "EIGVEC" n_cutoff, Δl_cutoff, eigvec_spectrum_power_cutoff, filterfieldpowercutoff
            return false
        end
    end

    if Filters.BC in allfilters
        f = boundary_condition_filter(v, BC, BCVcache, bc_atol)
        if !f
            @debug "BC" bc_atol
            return false
        end
    end

    if Filters.SPATIAL in allfilters
        f = spatial_filter!(VWSinv, VWSinvsh, F, v, m;
            θ_cutoff, equator_power_cutoff_frac, operators, nℓ, Plcosθ,
            filterfieldpowercutoff, V_symmetric,
            angular_filter = Filters.SPATIAL_EQUATOR in allfilters,
            radial_filter = Filters.SPATIAL_RADIAL in allfilters,
            compute_invtransform,
            )
        if !f
            @debug "SPATIAL" θ_cutoff, equator_power_cutoff_frac, filterfieldpowercutoff
            return false
        end
        compute_invtransform = false
    end

    if Filters.NODES in allfilters
        f = nodes_filter!(VWSinv, VWSinvsh, F, v, m, operators;
                nℓ, Plcosθ, filterfieldpowercutoff, nnodesmax, V_symmetric,
                compute_invtransform, radproftemp)
        if !f
            @debug "NODES" filterfieldpowercutoff, nnodesmax
            return false
        end
        compute_invtransform = false
    end

    if Filters.EIGEN in allfilters
        f = eigensystem_satisfy_filter(λ, v, M, MVcache; rtol = eigen_rtol)
        if !f
            @debug "EIGEN" λ eigen_rtol
            return false
        end
    end

    return true
end

function allocate_field_caches(nr, nℓ, nθ)
    VWSinv = (; V = zeros(ComplexF64, nr, nθ), W = zeros(ComplexF64, nr, nθ), S = zeros(ComplexF64, nr, nθ))
    VWSinvsh = (; V = zeros(ComplexF64, nr, nℓ), W = zeros(ComplexF64, nr, nℓ), S = zeros(ComplexF64, nr, nℓ))
    F = (; V = zeros(ComplexF64, nr * nℓ), W = zeros(ComplexF64, nr * nℓ), S = zeros(ComplexF64, nr * nℓ))
    radproftemp = zeros(real(eltype(VWSinv.V)), size(VWSinv.V,1))
    (; VWSinv, VWSinvsh, F, radproftemp)
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

    nθ = length(spharm_θ_grid_uniform(m, nℓ).θ)

    fieldcaches = allocate_field_caches(nr, nℓ, nθ)

    Plcosθ = allocate_Pl(m, nℓ)

    return (; MVcache, BCVcache, Plcosθ, fieldcaches...)
end

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
        matrixfn = differential_rotation_matrix,
        kw...)

    matrixfn2 = updaterotatationprofile(
            RotMatrix(V_symmetric, rotation_profile, nothing, nothing, matrixfn),
            operators)
    A = matrixfn2(m; operators, kw...);
    B = mass_matrix(m; operators, kw...);
    M = (A,B)
    filter_eigenvalues(λ, v, M, m; operators, V_symmetric, kw...);
end

function filter_eigenvalues(λ::AbstractVector, v::AbstractMatrix,
    M, m::Integer;
    operators,
    constraints = constraintmatrix(operators),
    filtercache = allocate_filter_caches(m; operators, constraints),
    filterflags = DefaultFilter,
    filterparams...)

    @unpack nparams = operators.radial_params;
    additional_params = (; operators, constraints, filtercache);
    Ms = map(computesparse, M)

    inds_bool = filterfn.(λ, eachcol(v), m, Ref(Ms), Ref(filterparams); operators, additional_params..., filterflags)
    filtinds = axes(λ, 1)[inds_bool]
    λ, v = λ[filtinds], v[:, filtinds]

    # re-apply scalings
    if get(filterparams, :scale_eigenvectors, false)
        scale_eigenvectors!(v; operators)
    end

    λ, v
end

function filter_map(λm::AbstractVector, vm::AbstractMatrix,
        AB::Tuple{StructMatrix{<:Complex}, AbstractMatrix{<:Real}}, m::Int, matrixfn!::F; kw...) where {F}
    A, B = AB
    matrixfn!(A, m; kw...)
    mass_matrix!(B, m; kw...)
    filter_eigenvalues(λm, vm, AB, m; kw...)
end
function filter_map_nthreads!(c::Channel, nt::Int, λs::AbstractVector{<:AbstractVector},
        vs::AbstractVector{<:AbstractMatrix}, mr::AbstractVector{<:Integer}, matrixfn!; kw...)

    nblasthreads = BLAS.get_num_threads()
    TMapReturnEltype = Tuple{eltype(λs), eltype(vs)}
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
    bw = matrixfn! isa RotMatrix ? getbw(matrixfn!.rotation_profile, nℓ) : nℓ
    ABs = [allocate_operator_mass_matrix(operators, bw) for _ in 1:nthreads]
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
        filter_eigenvalues(X..., m; operators, constraints, kw...)
    end;
    @debug "filtered eigenvalues for m = $m on tid = $(Threads.threadid()) with $(BLAS.get_num_threads()) BLAS threads"
    merge!(timer, timerlocal, tree_point = ["spectrum_filter"])
    return F
end

function eigvec_spectrum_filter_map_nthreads!(c, nt, spectrumfn!, mr, operators, constraints; kw...)
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
        bw = spectrumfn! isa RotMatrix ? getbw(spectrumfn!.rotation_profile, nℓ) : nℓ
        @timeit timer "M" ABs =
            [allocate_operator_mass_matrix(operators, bw) for _ in 1:nthreads];
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

filenamerottag(isdiffrot, rotation_profile) = isdiffrot ? "dr_$rotation_profile" : "ur"
filenamesymtag(Vsym) = Vsym ? "sym" : "asym"
rossbyeigenfilename(nr, nℓ, rottag, symtag, modeltag = "") =
    datadir("$(rottag)_nr$(nr)_nl$(nℓ)_$(symtag)$((isempty(modeltag) ? "" : "_") * modeltag).jld2")

function rossbyeigenfilename(; operators, kw...)
    isdiffrot = get(kw, :diffrot, false)
    rotation_profile = get(kw, :rotation_profile, "")
    Vsym = kw[:V_symmetric]
    modeltag = get(kw, :modeltag, "")
    rottag = filenamerottag(isdiffrot, rotation_profile)
    symtag = filenamesymtag(Vsym)
    @unpack nr, nℓ = operators.radial_params;
    return rossbyeigenfilename(nr, nℓ, rottag, symtag, modeltag)
end
function save_eigenvalues(f, mr; operators, save=true, kw...)
    lam, vec = filter_eigenvalues(f, mr; operators, kw...)
    if save
        fname = rossbyeigenfilename(; operators, kw...)
        @unpack operatorparams = operators
        @info "saving to $fname"
        jldsave(fname; lam, vec, mr, kw, operatorparams)
    end
end

struct FilteredEigen
    lams :: Vector{Vector{ComplexF64}}
    vs :: Vector{StructArray{ComplexF64, 2, NamedTuple{(:re, :im), NTuple{2,Matrix{Float64}}}, Int64}}
    mr :: UnitRange{Int}
    kw :: Dict{Symbol, Any}
    operators
end

function FilteredEigen(fname::String)
    lam, vec, mr, kw, operatorparams =
        load(fname, "lam", "vec", "mr", "kw", "operatorparams");
    operators = radial_operators(operatorparams...)
    FilteredEigen(lam, vec, mr, kw, operators)
end

function Base.getindex(F::FilteredEigen, r::AbstractUnitRange{Int})
    lams = F.lams[r]
    vs = F.vs[r]
    FilteredEigen(lams, vs, r, F.kw, F.operators)
end

function filter_eigenvalues(f::FilteredEigen; kw...)
    @unpack operators = f
    kw2 = merge(f.kw, kw)
    λfs, vfs =
        filter_eigenvalues(f.lams, f.vs, f.mr; operators, kw2...);
    FilteredEigen(λfs, vfs, f.mr, kw2, operators)
end

function RotMatrix(::Val{T}, V_symmetric, diffrot, rotation_profile; operators, kw...) where {T}
    T ∈ (:spectrum, :matrix) || error("unknown code ", T)
    if !diffrot
        RotMatrix(V_symmetric, :uniform, nothing, nothing,
            T == :matrix ? uniform_rotation_matrix! : uniform_rotation_spectrum!)
    else
        d = RotMatrix(V_symmetric, rotation_profile, nothing, nothing,
            T == :matrix ? differential_rotation_matrix! : differential_rotation_spectrum!)
        updaterotatationprofile(d, operators; kw...)
    end
end

function filteredeigen(filename::String; kw...)
    feig = FilteredEigen(filename)
    operators = feig.operators
    fkw = feig.kw
    diffrot::Bool = fkw[:diffrot]
    V_symmetric::Bool = fkw[:V_symmetric]
    rotation_profile::Union{Symbol, Nothing} = fkw[:rotation_profile]
    smoothing_param::Float64 = get(fkw, :smoothing_param, 1e-4)

    matrixfn! = RotMatrix(Val(:matrix), V_symmetric, diffrot, rotation_profile;
                    operators, smoothing_param)
    filter_eigenvalues(feig; matrixfn!, fkw..., kw...)
end

function differential_rotation_matrix(feig::FilteredEigen, m; kw...)
    operators = feig.operators
    fkw = feig.kw
    differential_rotation_matrix(m; operators, fkw..., kw...)
end

function mass_matrix(feig::FilteredEigen, m; kw...)
    operators = feig.operators
    fkw = feig.kw
    mass_matrix(m; operators, fkw..., kw...)
end

function operator_matrices(feig::FilteredEigen, m, kw...)
    A = differential_rotation_matrix(feig, m; kw...)
    B = mass_matrix(feig, m; kw...)
    A, B
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

function eigvec_spectrum_filter(Feig::FilteredEigen, m, ind; kw...)
    @unpack operators = Feig
    constraints = constraintmatrix(operators)
    filtercache = allocate_filter_caches(m; operators, constraints)
    (; F) = filtercache;
    eigvec_spectrum_filter!(F, Feig.vs[m][:, ind], m; operators, Feig.kw..., kw...)
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

function eigenfunction_rad_sh!(VWSinvsh, F, v; operators, n_cutoff = -1, kw...)
    VWS = eigenfunction_spectrum_2D!(F, v; operators, kw...)
    @unpack V, W, S = VWS
    @unpack radialspace = operators.radialspaces;

    Vinv = VWSinvsh.V
    Winv = VWSinvsh.W
    Sinv = VWSinvsh.S

    itransplan! = ApproxFunBase.plan_itransform!(radialspace, @view(Vinv[:, 1]))

    if n_cutoff >= 0
        field_lowpass = similar(V)
        field_lowpass .= V
        field_lowpass[n_cutoff+1:end, :] .= 0
        invtransform1!(radialspace, Vinv, field_lowpass, itransplan!)
        field_lowpass .= W
        field_lowpass[n_cutoff+1:end, :] .= 0
        invtransform1!(radialspace, Winv, field_lowpass, itransplan!)
        field_lowpass .= S
        field_lowpass[n_cutoff+1:end, :] .= 0
        invtransform1!(radialspace, Sinv, field_lowpass, itransplan!)
    else
        invtransform1!(radialspace, Vinv, V, itransplan!)
        invtransform1!(radialspace, Winv, W, itransplan!)
        invtransform1!(radialspace, Sinv, S, itransplan!)
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
    V_symmetric,
    nℓ = size(VWS.V, 2),
    Plcosθ = allocate_Pl(m, nℓ),
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

    @unpack VWSinv, VWSinvsh, F = allocate_field_caches(nr, nℓ, nθ)

    eigenfunction_realspace!(VWSinv, VWSinvsh, F, v, m; operators, kw...)

    return (; VWSinv, θ)
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
