module RossbyWaveSpectrum

using MKL

using Reexport

@reexport using ApproxFun
@reexport using ApproxFunAssociatedLegendre
@reexport using BlockArrays
using BlockBandedMatrices
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
    @unpack nr, nℓ = operators.radial_params
    @unpack nvariables = operators
    @unpack ddrDDrMCU4, onebyr2MCU4, IU2 = operators.operator_matrices;
    @unpack Weqglobalscaling = operators.scalings

    B .= 0

    VV = matrix_block(B, 1, 1)
    WW = matrix_block(B, 2, 2)
    SS = matrix_block(B, 3, 3)

    # V terms
    V_ℓs = ℓrange(m, nℓ, V_symmetric)

    # W, S terms
    W_ℓs = ℓrange(m, nℓ, !V_symmetric)

    @views for ℓind in eachindex(V_ℓs)
        VV[Block(ℓind, ℓind)] .= IU2
        SS[Block(ℓind, ℓind)] .= IU2
    end

    T = similar(ddrDDrMCU4);
    @views for (ℓind, ℓ) in enumerate(W_ℓs)
        ℓℓp1 = ℓ * (ℓ+1)

        @. T = ddrDDrMCU4 - ℓℓp1 * onebyr2MCU4
        WW[Block(ℓind, ℓind)] .= (Weqglobalscaling * Rsun^2) .* T
    end

    return B
end

function uniform_rotation_matrix(m; operators, kw...)
    A = allocate_operator_matrix(operators, 2)
    uniform_rotation_matrix!(A, m; operators, kw...)
    return A
end

function uniform_rotation_matrix!(A::StructMatrix{<:Complex}, m; operators, V_symmetric = true, kw...)
    @unpack Ω0 = operators.constants;
    @unpack nr, nℓ = operators.radial_params
    @unpack Sscaling, Wscaling, Weqglobalscaling, Seqglobalscaling = operators.scalings

    @unpack ddrMCU4, DDrMCU2, DDr_minus_2byrMCU2, ddrDDrMCU4, κ_∇r2_plus_ddr_lnρT_ddrMCU2,
        onebyrMCU2, onebyrMCU4, onebyr2MCU4, ηρ_by_rMCU4, ddr_S0_by_cp_by_r2MCU2,
        κ_by_r2MCU2, gMCU4, ddr_minus_2byrMCU4, IU2 = operators.operator_matrices;

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
    cosθ = OffsetArray(costheta_operator_matrix(nCS, m), ℓs, ℓs);
    sinθdθ = OffsetArray(sintheta_dtheta_operator_matrix(nCS, m), ℓs, ℓs);

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

        @. T = ddrDDrMCU4 - ℓℓp1 * onebyr2MCU4

        WW[Block(ℓind, ℓind)] .= (Weqglobalscaling * twom_by_ℓℓp1 * Rsun^2) .*
                                (T .- ℓℓp1 .* ηρ_by_rMCU4)

        @. T = - gMCU4 / (Ω0^2 * Rsun)  * Wscaling/Sscaling
        WS[Block(ℓind, ℓind)] .= Weqglobalscaling .* T
        @. T = ℓℓp1 * ddr_S0_by_cp_by_r2MCU2 * (Rsun^3 * Sscaling/Wscaling)
        SW[Block(ℓind, ℓind)] .= Seqglobalscaling .* T
        @. T = -(κ_∇r2_plus_ddr_lnρT_ddrMCU2 - ℓℓp1 * κ_by_r2MCU2) * Rsun^2
        SS[Block(ℓind, ℓind)] .= Seqglobalscaling .* T

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

    @unpack ν = operators.constants;
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

function laplacian_operator(nℓ, m)
    ℓs = range(m, length = nℓ)
    Diagonal(@. -ℓs * (ℓs + 1))
end

function constant_differential_rotation_terms!(M::StructMatrix{<:Complex}, m;
        operators, ΔΩ_frac = 0.01, V_symmetric = true, kw...)

    @unpack nr, nℓ = operators.radial_params;
    @unpack Wscaling, Weqglobalscaling, Seqglobalscaling = operators.scalings

    @unpack ddrMCU4, DDrMCU2, onebyrMCU2, onebyrMCU4,
            DDr_minus_2byrMCU2, ηρ_by_rMCU4, ddrDDrMCU4,
            onebyr2MCU4, IU2 = operators.operator_matrices;

    VV = matrix_block(M.re, 1, 1)
    VW = matrix_block(M.re, 1, 2)
    WV = matrix_block(M.re, 2, 1)
    WW = matrix_block(M.re, 2, 2)
    SS = matrix_block(M.re, 3, 3)

    nCS = 2nℓ+1
    ℓs = range(m, length = nCS)

    cosθo = OffsetArray(costheta_operator_matrix(nCS, m), ℓs, ℓs)
    sinθdθo = OffsetArray(sintheta_dtheta_operator_matrix(nCS, m), ℓs, ℓs)
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

        WW[Block(ℓind, ℓind)] .+= (Weqglobalscaling * Rsun^2) .*
            (diagterm .* ddrDDr_minus_ℓℓp1_by_r2MCU4 .+ 2dopplerterm .* ηρ_by_rMCU4)

        SS[Block(ℓind, ℓind)] .+= Seqglobalscaling .* dopplerterm .* IU2

        for ℓ′ in intersect(ℓ-1:2:ℓ+1, V_ℓs)
            ℓ′ℓ′p1 = ℓ′ * (ℓ′ + 1)
            ℓ′ind = findfirst(isequal(ℓ′), V_ℓs)

            @. T = -(Rsun * Wscaling) * ΔΩ_frac / ℓℓp1 *
                                    ((4ℓ′ℓ′p1 * cosθo[ℓ, ℓ′] + (ℓ′ℓ′p1 + 2) * sinθdθo[ℓ, ℓ′]) * ddrMCU4
                                        +
                                        ddr_plus_2byrMCU4 * laplacian_sinθdθo[ℓ, ℓ′])

            WV[Block(ℓind, ℓ′ind)] .+= Weqglobalscaling .* T
        end
    end

    return M
end

function radial_differential_rotation_terms!(M::StructMatrix{<:Complex}, m;
        operators, rotation_profile = :solar_equator,
        ΔΩ_frac = 0.01, # only used to test the constant case
        ΔΩprofile_deriv = radial_differential_rotation_profile_derivatives_Fun(;
                            operators, rotation_profile, ΔΩ_frac),
        V_symmetric = true, kw...)

    @unpack nr, nℓ = operators.radial_params
    @unpack DDr, ddr, ddrDDr = operators.diff_operators;
    @unpack ddrMCU4 = operators.operator_matrices;
    @unpack onebyr, g, ηρ_by_r, onebyr2, twobyr = operators.rad_terms;
    @unpack Sscaling, Wscaling, Weqglobalscaling, Seqglobalscaling = operators.scalings
    @unpack matCU4, matCU2 = operators;

    VV = matrix_block(M.re, 1, 1);
    VW = matrix_block(M.re, 1, 2);
    WV = matrix_block(M.re, 2, 1);
    WW = matrix_block(M.re, 2, 2);
    SV = matrix_block(M.re, 3, 1);
    SW = matrix_block(M.re, 3, 2);
    SS = matrix_block(M.re, 3, 3);

    # @unpack Ω0 = operators.constants;
    @unpack r_out = operators.radial_params;
    (; ΔΩ, ddrΔΩ, d2dr2ΔΩ) = ΔΩprofile_deriv;
    Ω0 = ΔΩ(r_out)

    ΔΩMCU2 = matCU2(ΔΩ);
    ddrΔΩMCU2 = matCU2(ddrΔΩ);
    d2dr2ΔΩMCU4 = matCU4(d2dr2ΔΩ);

    ddrΔΩ_over_g = ddrΔΩ / g;
    ddrΔΩ_over_gMCU2 = matCU2(ddrΔΩ_over_g);
    ddrΔΩ_over_g_DDr = (ddrΔΩ_over_g * DDr)::Tmul;
    ddrΔΩ_over_g_DDrMCU2 = matCU2(ddrΔΩ_over_g_DDr);
    ddrΔΩ_plus_ΔΩddr = (ddrΔΩ + (ΔΩ * ddr)::Tmul)::Tplusinf;

    nCS = 2nℓ+1
    ℓs = range(m, length = nCS)
    cosθ = costheta_operator_matrix(nCS, m);
    sinθdθ = sintheta_dtheta_operator_matrix(nCS, m);
    cosθsinθdθ = (costheta_operator_matrix(nCS + 1, m) * sintheta_dtheta_operator_matrix(nCS + 1, m))[1:end-1, 1:end-1];

    cosθo = OffsetArray(cosθ, ℓs, ℓs);
    sinθdθo = OffsetArray(sinθdθ, ℓs, ℓs);
    cosθsinθdθo = OffsetArray(cosθsinθdθ, ℓs, ℓs);
    ∇²_sinθdθo = OffsetArray(Diagonal(@. -ℓs * (ℓs + 1)) * sinθdθ, ℓs, ℓs);

    DDr_min_2byr = (DDr - twobyr)::Tplusinf;
    ΔΩ_DDr_min_2byr = (ΔΩ * DDr_min_2byr)::Tmul;
    ΔΩ_DDr = (ΔΩ * DDr)::Tmul;
    ΔΩ_by_r = ΔΩ * onebyr;
    ΔΩ_by_r2 = ΔΩ * onebyr2;

    ΔΩ_by_rMCU2, ΔΩ_DDrMCU2, ΔΩ_DDr_min_2byrMCU2 =
        map(matCU2, (ΔΩ_by_r, ΔΩ_DDr, ΔΩ_DDr_min_2byr, ddrΔΩ_plus_ΔΩddr));

    ddrΔΩ_plus_ΔΩddrMCU4, twoΔΩ_by_rMCU4, ddrΔΩ_DDrMCU4, ΔΩ_ddrDDrMCU4,
        ΔΩ_by_r2MCU4, ddrΔΩ_ddr_plus_2byrMCU4, ΔΩ_ηρ_by_rMCU4 =
        map(matCU4, (ddrΔΩ_plus_ΔΩddr, 2ΔΩ_by_r, ddrΔΩ * DDr, ΔΩ * ddrDDr,
            ΔΩ_by_r2, ddrΔΩ * (ddr + twobyr), ΔΩ * ηρ_by_r))

    ΔΩ_ddrDDr_min_ℓℓp1byr2MCU4 = zeros(nr, nr);
    T = zeros(nr, nr);

    # V terms
    V_ℓs = ℓrange(m, nℓ, V_symmetric);
    # W, S terms
    W_ℓs = ℓrange(m, nℓ, !V_symmetric);

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

        SS[Block(ℓind, ℓind)] .+= Seqglobalscaling .* (-m) .* ΔΩMCU2

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

            #     @. T = -(Ω0^2 * Rsun^2) * 2m * cosθo[ℓ, ℓ′] * ddrΔΩ_over_gMCU2 * Sscaling;
            #     SV[Block(ℓind, ℓ′ind)] .+= Seqglobalscaling .* T
        end

        # for ℓ′ in intersect(W_ℓs, ℓ-2:2:ℓ+2)
        #     ℓ′ind = findfirst(isequal(ℓ′), W_ℓs)
        #     @. T = (Ω0^2 * Rsun^3) * 2cosθsinθdθo[ℓ, ℓ′] * ddrΔΩ_over_g_DDrMCU2 * Sscaling/Wscaling;
        #     SW[Block(ℓind, ℓ′ind)] .+= Seqglobalscaling .* T
        # end
    end
    return M
end

struct OpVector{VT,WT}
    V :: VT
    iW :: WT
end
OpVector(t::Tuple) = OpVector(t...)
OpVector(t::NamedTuple{(:V, :iW)}) = OpVector(t.V, t.iW)
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

function solar_differential_rotation_vorticity_Fun(; operators,
        ΔΩprofile_deriv = solar_differential_rotation_profile_derivatives_Fun(; operators)
        )

    @unpack onebyr, r, onebyr2, twobyr = operators.rad_terms;
    @unpack ddr, d2dr2 = operators.diff_operators;

    (; ΔΩ, dr_ΔΩ, d2r_ΔΩ) = ΔΩprofile_deriv;

    @unpack radialspace = operators;
    # velocity and its derivatives are expanded in Legendre poly
    latitudinal_space = NormalizedPlm(0, NormalizedLegendre());

    cosθ = Fun(Legendre());
    Ir = I : radialspace;
    Iℓ = I : latitudinal_space;
    ∇² = HorizontalLaplacian(latitudinal_space);

    sinθ_plus_2cosθ = sinθdθ_plus_2cosθ_Operator(latitudinal_space);
    ωΩr = (Ir ⊗ sinθ_plus_2cosθ) * ΔΩ;
    ∂rωΩr = (Ir ⊗ sinθ_plus_2cosθ) * dr_ΔΩ;
    # cotθddθ = cosθ * 1/sinθ * d/dθ = -cosθ * d/d(cosθ) = -x*d/dx
    cotθdθ = KroneckerOperator(Ir, -cosθ * Derivative(Legendre()),
        radialspace * Legendre(), radialspace * Jacobi(1,1));
    cotθdθΔΩ = Fun(cotθdθ * ΔΩ, radialspace ⊗ latitudinal_space);
    ∇²_min_2 = ∇²-2;
    ∂θωΩr_by_sinθ = (Ir ⊗ ∇²_min_2) * ΔΩ + 2cotθdθΔΩ;
    cotθdθdr_ΔΩ = Fun(cotθdθ * dr_ΔΩ, radialspace ⊗ latitudinal_space);
    ∂r∂θωΩr_by_sinθ = (Ir ⊗ ∇²_min_2) * dr_ΔΩ + 2cotθdθdr_ΔΩ;
    ωΩθ_by_rsinθ = -(dr_ΔΩ + (twobyr ⊗ Iℓ) * ΔΩ);
    ∂rωΩθ_by_rsinθ = -(d2r_ΔΩ + (twobyr ⊗ Iℓ) * dr_ΔΩ - (2onebyr2 ⊗ Iℓ) * ΔΩ);

    ωΩr, ∂rωΩr, ∂θωΩr_by_sinθ, ωΩθ_by_rsinθ, ∂r∂θωΩr_by_sinθ, ∂rωΩθ_by_rsinθ =
        map(replaceemptywitheps,
            (ωΩr, ∂rωΩr, ∂θωΩr_by_sinθ, ωΩθ_by_rsinθ, ∂r∂θωΩr_by_sinθ, ∂rωΩθ_by_rsinθ))

    (; ωΩr, ∂rωΩr, ∂θωΩr_by_sinθ, ωΩθ_by_rsinθ, ∂r∂θωΩr_by_sinθ, ∂rωΩθ_by_rsinθ)
end

function solar_differential_rotation_terms!(M::StructMatrix{<:Complex}, m;
        operators, rotation_profile = :latrad,
        ΔΩ_frac = 0.01, # only used to test the constant case
        ΔΩprofile_deriv = solar_differential_rotation_profile_derivatives_Fun(;
                            operators, rotation_profile, ΔΩ_frac),
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
    @unpack radialspace = operators;
    @unpack Wscaling, Sscaling, Weqglobalscaling, Seqglobalscaling = operators.scalings;
    @unpack Ω0 = operators.constants;

    latitudinal_space = NormalizedPlm(m);

    Ir = ConstantOperator(I);
    Iℓ = I : latitudinal_space;
    unsetspace2d = domainspace(Ir) ⊗ latitudinal_space;
    space2d = radialspace ⊗ latitudinal_space;
    I2d_unset = I : unsetspace2d;

    sinθdθop = sinθdθ_Operator(latitudinal_space);
    ∇² = HorizontalLaplacian(latitudinal_space);
    ℓℓp1op = -∇²;

    (; ΔΩ, dr_ΔΩ, dz_ΔΩ) = ΔΩprofile_deriv;
    (; ωΩr, ∂rωΩr, ∂θωΩr_by_sinθ, ωΩθ_by_rsinθ, ∂r∂θωΩr_by_sinθ,
        ∂rωΩθ_by_rsinθ, ∂rωΩθ_by_rsinθ) = ωΩ_deriv;

    onebyr2op = Multiplication(onebyr2);
    ufr_W = onebyr2op ⊗ ℓℓp1op;
    ufr = OpVector((; V = 0 * I2d_unset, iW = -im * ufr_W));
    ωfr = OpVector((; V = ufr_W, iW = 0 * I2d_unset));

    rsinθufθ = OpVector((; V = im * m * I2d_unset, iW = -im * DDr ⊗ sinθdθop));
    rsinθωfθ = OpVector((; V = ddr ⊗ sinθdθop, iW = m * (onebyr2op ⊗ ℓℓp1op - ddrDDr ⊗ Iℓ)));

    rsinθufϕ = OpVector((; V = -Ir ⊗ sinθdθop, iW = (m * DDr) ⊗ Iℓ));
    rsinθωfϕ = OpVector((; V = im * m * ddr ⊗ Iℓ, iW = -im * (ddrDDr ⊗ Iℓ - onebyr2op ⊗ ℓℓp1op)));

    curl_u_x_ω_r = (ωΩr * (DDr ⊗ Iℓ) + ωΩθ_by_rsinθ * (Ir ⊗ sinθdθop) - ∂rωΩr) * ufr +
        ((-onebyr2 ⊗ Iℓ) * ∂θωΩr_by_sinθ) * rsinθufθ - im * m * ΔΩ * ωfr;
    scaled_curl_u_x_ω_r_tmp = ((-im * r2) ⊗ inv(ℓℓp1op)) * curl_u_x_ω_r;
    space2d_D2 = rangespace(Derivative(radialspace, 2)) ⊗ NormalizedPlm(m)
    scaled_curl_u_x_ω_r = (scaled_curl_u_x_ω_r_tmp : space2d → space2d_D2) |> expand;

    V_ℓinds = ℓrange(1, nℓ, V_symmetric)
    W_ℓinds = ℓrange(1, nℓ, !V_symmetric)

    VV_ = real(kronmatrix(scaled_curl_u_x_ω_r.V, nr, V_ℓinds, V_ℓinds));
    VW_ = real(kronmatrix(scaled_curl_u_x_ω_r.iW, nr, V_ℓinds, W_ℓinds));
    VV .+= VV_;
    VW .+= (Rsun / Wscaling) .* VW_;

    # we compute the derivative of rdiv_ucrossω_h analytically
    # this lets us use the more accurate representations of ddrDDr instead of using ddr * DDr
    ddr_rdiv_ucrossω_h = OpVector((;
        V = ((2ωΩr + ΔΩ * (Ir ⊗ sinθdθop)) * (ddr ⊗ ℓℓp1op) - ∂θωΩr_by_sinθ * (ddr ⊗ sinθdθop)
            + (2∂rωΩr + dr_ΔΩ * (Ir ⊗ sinθdθop)) * (Ir ⊗ ℓℓp1op) - ∂r∂θωΩr_by_sinθ * (Ir ⊗ sinθdθop)
            ),
        iW = m *
            (∂θωΩr_by_sinθ * (ddrDDr ⊗ Iℓ) + ωΩθ_by_rsinθ * (ddr ⊗ ℓℓp1op)
            + ∂r∂θωΩr_by_sinθ * (DDr ⊗ Iℓ) + ∂rωΩθ_by_rsinθ * (Ir ⊗ ℓℓp1op)
            )
    ));

    uf_x_ωΩ_r = -ωΩθ_by_rsinθ * rsinθufϕ;
    uΩ_x_ωf_r = -ΔΩ * rsinθωfθ;
    u_x_ω_r = uf_x_ωΩ_r + uΩ_x_ωf_r;
    ∇²_u_x_ω_r = (Ir ⊗ ∇²) * u_x_ω_r;

    scaled_curl_curl_u_x_ω_r_tmp = (Ir ⊗ inv(ℓℓp1op)) * (-ddr_rdiv_ucrossω_h + ∇²_u_x_ω_r);
    space2d_D4 = rangespace(Derivative(radialspace, 4)) ⊗ NormalizedPlm(m)
    scaled_curl_curl_u_x_ω_r = (scaled_curl_curl_u_x_ω_r_tmp : space2d → space2d_D4) |> expand;

    WV_ = real(kronmatrix(scaled_curl_curl_u_x_ω_r.V, nr, W_ℓinds, V_ℓinds));
    WW_ = real(kronmatrix(scaled_curl_curl_u_x_ω_r.iW, nr, W_ℓinds, W_ℓinds));
    WV .+= (Weqglobalscaling * Rsun * Wscaling) .* WV_;
    WW .+= (Weqglobalscaling * Rsun^2) .* WW_;

    # entropy terms
    # thermal_wind_term_tmp = im*((2/g) ⊗ Iℓ) * dz_ΔΩ * rsinθufθ
    # thermal_wind_term = expand(thermal_wind_term_tmp : space2d → space2d_D2)
    # SV_ = real(kronmatrix(thermal_wind_term.V, nr, W_ℓinds, V_ℓinds));
    # SV .+= (Seqglobalscaling * (Ω0^2 * Rsun^2) * Sscaling) .* SV_
    # SW_ = real(kronmatrix(thermal_wind_term.iW, nr, W_ℓinds, W_ℓinds));
    # SW .+= (Seqglobalscaling * (Ω0^2 * Rsun^3) * Sscaling/Wscaling) .* SW_

    SS_doppler_term = expand(-m*ΔΩ * I : space2d → space2d_D2)
    SS_ = real(kronmatrix(SS_doppler_term, nr, W_ℓinds, W_ℓinds));
    SS .+= Seqglobalscaling * SS_

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

function eigenvalue_filter(x, m;
    eig_imag_unstable_cutoff = -1e-6,
    eig_imag_to_real_ratio_cutoff = 1,
    eig_imag_stable_cutoff = Inf)

    freq_sectoral = 2 / (m + 1)
    eig_imag_unstable_cutoff <= imag(x) < min(freq_sectoral * eig_imag_to_real_ratio_cutoff, eig_imag_stable_cutoff)
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
        MVcache::NTuple{2, StructArray{<:Complex,1}} = allocate_MVcache(size(AB[1], 1)); rtol = 1e-2) where {T<:Real}

    A, B = AB;
    Av, λBv = MVcache;

    mul!(Av.re, A.re, v.re)
    mul!(Av.re, A.im, v.im, -1.0, 1.0)
    mul!(Av.im, A.re, v.im)
    mul!(Av.im, A.im, v.re,  1.0, 1.0)

    mul!(λBv.re, B, v.re)
    mul!(λBv.im, B, v.im)
    λBv .*= λ

    isapprox(Av, λBv; rtol)
end

function filterfields(coll, v, nparams, nvariables; filterfieldpowercutoff = 1e-4)
    Vpow = sum(abs2, @view v[1:nparams])
    Wpow = sum(abs2, @view v[nparams .+ (1:nparams)])
    Spow = sum(abs2, @view(v[2nparams .+ (1:nparams)]))

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

function eigvec_spectrum_filter!(F, v, m, operators;
    n_cutoff = 7, Δl_cutoff = 7, eigvec_spectrum_power_cutoff = 0.9,
    filterfieldpowercutoff = 1e-4,
    kw...)

    VW = eigenfunction_spectrum_2D!(F, v; operators, kw...)
    Δl_inds = Δl_cutoff ÷ 2

    @unpack nparams = operators.radial_params
    @unpack nvariables = operators

    flag = true
    fields = filterfields(VW, v, nparams, nvariables; filterfieldpowercutoff)

    @views for X in fields
        PV_frac = sum(abs2, X[1:n_cutoff, 1:Δl_inds]) / sum(abs2, X)
        flag &= PV_frac > eigvec_spectrum_power_cutoff
        flag || break
    end

    return flag
end

allocate_Pl(m, nℓ) = zeros(range(m, length = 2nℓ + 1))

function peakindabs1(X)
    findmax(ind -> sum(abs2, @view X[ind, :]), axes(X, 1))[2]
end

function spatial_filter!(VWSinv, VWSinvsh, F, v, m, operators,
    θ_cutoff = deg2rad(60), equator_power_cutoff_frac = 0.3;
    nℓ = operators.radial_params.nℓ,
    Plcosθ = allocate_Pl(m, nℓ),
    filterfieldpowercutoff = 1e-4)

    (; θ) = eigenfunction_realspace!(VWSinv, VWSinvsh, F, v, m; operators, nℓ, Plcosθ)

    eqfilter = true

    @unpack nparams = operators.radial_params
    @unpack nvariables = operators
    fields = filterfields(VWSinv, v, nparams, nvariables; filterfieldpowercutoff)

    for X in fields
        r_ind_peak = peakindabs1(X)
        peak_latprofile = @view X[r_ind_peak, :]
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
    @unpack nvariables = operators
    fields = filterfields(VWSinv, v, nparams, nvariables; filterfieldpowercutoff)

    (; θ) = eigenfunction_realspace!(VWSinv, VWSinvsh, F, v, m; operators, nℓ, Plcosθ)
    eqind = argmin(abs.(θ .- pi/2))

    for X in fields
        radprof = @view X[:, eqind]
        nnodes_real = count(Bool.(sign.(abs.(diff(sign.(real(radprof)))))))
        nnodes_imag = count(Bool.(sign.(abs.(diff(sign.(imag(radprof)))))))
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
        EIGEN
        EIGVAL
        EIGVEC
        BC
        SPATIAL
        NODES
    end
    FilterFlag(F::FilterFlag) = F
    function Base.:(!)(F::FilterFlag)
        n = length(instances(FilterFlag))
        FilterFlag(2^(n-1)-1 - Int(F))
    end
    Base.in(t::FilterFlag, F::FilterFlag) = (t & F) != NONE
    Base.broadcastable(x::FilterFlag) = Ref(x)

    const DefaultFilter = EIGEN | EIGVAL | EIGVEC | BC | SPATIAL
end
using .Filters

function filterfn(λ, v, m, M, (operators, constraints, filtercache, kw)::NTuple{4,Any}, filterflags)

    @unpack BC = constraints
    @unpack nℓ = operators.radial_params;

    @unpack eig_imag_unstable_cutoff = kw
    @unpack eig_imag_to_real_ratio_cutoff = kw
    @unpack eig_imag_stable_cutoff = kw
    @unpack eigvec_spectrum_power_cutoff = kw
    @unpack bc_atol = kw
    @unpack Δl_cutoff = kw
    @unpack n_cutoff = kw
    @unpack θ_cutoff = kw
    @unpack equator_power_cutoff_frac = kw
    @unpack eigen_rtol = kw
    @unpack filterfieldpowercutoff = kw
    @unpack nnodesmax = kw

    (; MVcache, BCVcache, VWSinv, VWSinvsh, Plcosθ, F) = filtercache;

    allfilters = Filters.FilterFlag(filterflags)

    if Filters.EIGVAL in allfilters
        f = eigenvalue_filter(λ, m;
        eig_imag_unstable_cutoff, eig_imag_to_real_ratio_cutoff, eig_imag_stable_cutoff)
        f || (@debug "EIGVAL" f; return false)
    end

    if Filters.EIGVEC in allfilters
        f = eigvec_spectrum_filter!(F, v, m, operators;
            n_cutoff, Δl_cutoff, eigvec_spectrum_power_cutoff,
            filterfieldpowercutoff)
        f || (@debug "EIGVEC" f; return false)
    end

    if Filters.BC in allfilters
        f = boundary_condition_filter(v, BC, BCVcache, bc_atol)
        f || (@debug "BC" f; return false)
    end

    if Filters.SPATIAL in allfilters
        f = spatial_filter!(VWSinv, VWSinvsh, F, v, m, operators,
            θ_cutoff, equator_power_cutoff_frac; nℓ, Plcosθ,
            filterfieldpowercutoff)
        f || (@debug "SPATIAL" f; return false)
    end

    if Filters.EIGEN in allfilters
        f = eigensystem_satisfy_filter(λ, v, M, MVcache; rtol = eigen_rtol)
        f || (@debug "EIGEN" f; return false)
    end

    if Filters.NODES in allfilters
        f = nodes_filter(VWSinv, VWSinvsh, F, v, m, operators;
                nℓ, Plcosθ, filterfieldpowercutoff, nnodesmax)
        f || (@debug "NODES" f; return false)
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
    @unpack BC = constraints
    @unpack nr, nℓ, nparams = operators.radial_params
    # temporary cache arrays
    nrows = 3nparams
    MVcache = allocate_MVcache(nrows)

    n_bc = size(BC, 1)
    BCVcache = allocate_BCcache(n_bc)

    nθ = length(spharm_θ_grid_uniform(m, nℓ).θ)

    @unpack VWSinv, VWSinvsh, F = allocate_field_caches(nr, nθ, nℓ)

    Plcosθ = allocate_Pl(m, nℓ)

    return (; MVcache, BCVcache, VWSinv, VWSinvsh, Plcosθ, F)
end

const DefaultFilterParams = Dict(
    # boundary condition filter
    :bc_atol => 1e-5,
    # eigval filter
    :eig_imag_unstable_cutoff => -1e-6,
    :eig_imag_to_real_ratio_cutoff => 1,
    :eig_imag_stable_cutoff => Inf,
    # eigensystem satisfy filter
    :eigen_rtol => 0.01,
    # smooth eigenvector filter
    :Δl_cutoff => 7,
    :n_cutoff => 10,
    :eigvec_spectrum_power_cutoff => 0.9,
    # spatial localization filter
    :θ_cutoff => deg2rad(60),
    :equator_power_cutoff_frac => 0.3,
    # radial nodes filter
    :nnodesmax => 10,
    # exclude a field from a filter if relative power is below a cutoff
    :filterfieldpowercutoff => 1e-4,
)

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

function filter_eigenvalues(λ::AbstractVector, v::AbstractMatrix,
    M, m::Integer;
    operators,
    constraints = constraintmatrix(operators),
    filtercache = allocate_filter_caches(m; operators, constraints),
    filterflags = DefaultFilter,
    kw...)

    @unpack nparams = operators.radial_params;
    kw2 = merge(DefaultFilterParams, kw);
    additional_params = (operators, constraints, filtercache, kw2);

    inds_bool = filterfn.(λ, eachcol(v), m, (map(computesparse, M),), (additional_params,), filterflags)
    filtinds = axes(λ, 1)[inds_bool]
    λ, v = λ[filtinds], v[:, filtinds]

    # re-apply scalings
    if get(kw, :scale_eigenvectors, false)
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
    X = @timeit timerlocal "m=$m tid=$(Threads.threadid()) spectrum" begin
        spectrumfn!(M, m; operators, constraints, cache, temp_projectback, timer = timerlocal, kw...)
    end;
    F = @timeit timerlocal "m=$m tid=$(Threads.threadid()) filter" begin
        filter_eigenvalues(X..., m; operators, constraints, kw...)
    end;
    merge!(timer, timerlocal, tree_point = ["spectrum_filter"])
    return F
end

function eigvec_spectrum_filter_map_nthreads!(c, nt, spectrumfn!, mr, operators, constraints; kw...)
    nblasthreads = BLAS.get_num_threads()
    nt = Threads.nthreads()

    TMapReturnEltype = Tuple{Vector{ComplexF64},
                    StructArray{ComplexF64, 2, @NamedTuple{re::Matrix{Float64},im::Matrix{Float64}}, Int64}}
    try
        BLAS.set_num_threads(max(1, div(nblasthreads, nt)))
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

            λv1 = eigvec_spectrum_filter_map_nthreads!(c, Threads.nthreads(), spectrumfn!, mr1, operators, constraints;
                timer, kw...)

            λv2 = eigvec_spectrum_filter_map_nthreads!(c, nthreads_trailing_elems, spectrumfn!, mr2, operators, constraints;
                timer, kw...)

            λs, vs = map(first, λv1), map(last, λv1)
            λs2, vs2 = map(first, λv2), map(last, λv2)
            append!(λs, λs2)
            append!(vs, vs2)
        else
            λv = eigvec_spectrum_filter_map_nthreads!(c, Threads.nthreads(), spectrumfn!, mr, operators, constraints;
                timer, kw...)
            λs, vs = map(first, λv), map(last, λv)
        end
    end
    get(kw, :print_timer, true) && println(timer)
    λs, vs
end
function filter_eigenvalues(filename::String; kw...)
    λs, vs, mr, nr, nℓ, kwold = load(filename, "lam", "vec", "mr", "nr", "nℓ", "kw");
    kw = merge(kwold, kw)
    operators = radial_operators(nr, nℓ)
    constraints = constraintmatrix(operators)
    filter_eigenvalues(λs, vs, mr; operators, constraints, kw...)
end

filenamerottag(isdiffrot, rotation_profile) = isdiffrot ? "dr_$rotation_profile" : "ur"
filenamesymtag(Vsym) = Vsym ? "sym" : "asym"
rossbyeigenfilename(nr, nℓ, rottag, symtag, modeltag = "") =
    datadir("$(rottag)_nr$(nr)_nl$(nℓ)_$(symtag)$((isempty(modeltag) ? "" : "_") * modeltag).jld2")

function rossbyeigenfilename(; operators, kw...)
    isdiffrot = get(kw, :diffrot, false)
    rotation_profile = get(kw, :rotation_profile, "")
    Vsym = kw[:V_symmetric]
    rottag = filenamerottag(isdiffrot, rotation_profile)
    symtag = filenamesymtag(Vsym)
    @unpack nr, nℓ = operators.radial_params;
    return rossbyeigenfilename(nr, nℓ, rottag, symtag)
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

function filter_eigenvalues(f::FilteredEigen; kw...)
    @unpack operators = f
    kw2 = merge(f.kw, kw)
    λfs, vfs =
        filter_eigenvalues(f.lams, f.vs, f.mr; operators, kw2...);
    FilteredEigen(λfs, vfs, f.mr, kw2, operators)
end

function RotMatrix(::Val{T}, V_symmetric, diffrot, rotation_profile; operators, smoothing_param) where {T}
    T ∈ (:spectrum, :matrix) || error("unknown code ", T)
    if !diffrot
        RotMatrix(V_symmetric, :uniform, nothing, nothing,
            T == :matrix ? uniform_rotation_matrix! : uniform_rotation_spectrum!)
    else
        d = RotMatrix(V_symmetric, rotation_profile, nothing, nothing,
            T == :matrix ? differential_rotation_matrix! : differential_rotation_spectrum!)
        updaterotatationprofile(d, operators; smoothing_param)
    end
end

function filteredeigen(filename::String; kw...)
    feig = FilteredEigen(filename)
    operators = feig.operators
    fkw = feig.kw
    diffrot::Bool = fkw[:diffrot]
    V_symmetric::Bool = fkw[:V_symmetric]
    rotation_profile::Symbol = fkw[:rotation_profile]
    smoothing_param::Float64 = get(fkw, :smoothing_param, 1e-5)

    matrixfn! = RotMatrix(Val(:matrix), V_symmetric, diffrot, rotation_profile;
                    operators, smoothing_param)
    filter_eigenvalues(feig; matrixfn!, fkw..., kw...)
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
    @unpack radialspace = operators;

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

    VW = eigenfunction_spectrum_2D!(F, v; operators, kw...)
    invshtransform2!(VWSinv, VW, m; nℓ, Plcosθ, kw...)
end

end # module
