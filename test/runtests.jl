using RossbyWaveSpectrum
using Test
using LinearAlgebra
using OffsetArrays
using Aqua
using SpecialPolynomials
using ForwardDiff

using RossbyWaveSpectrum: matrix_block, Rsun, kron2

@testset "project quality" begin
    Aqua.test_all(RossbyWaveSpectrum,
        ambiguities = false,
        stale_deps = (; ignore =
        [:PyCall, :PyPlot, :LaTeXStrings]),
    )
end

@testset "differential rotation" begin
    nr, nℓ = 40, 2
    nparams = nr * nℓ
    m = 1
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ)
    (; transforms, diff_operators, rad_terms, coordinates, radial_params, identities) = operators
    (; r, r_chebyshev) = coordinates
    (; r_mid, Δr) = radial_params
    (; Tcrfwd) = transforms
    (; Iℓ) = identities
    (; ddr, d2dr2) = diff_operators
    (; onebyr_cheby, r2_cheby, r_cheby) = rad_terms

    # r̄(r) = (r - r_mid::Float64) / (Δr::Float64 / 2)

    # cosθ = RossbyWaveSpectrum.costheta_operator(nℓ, m)
    # sinθdθ = RossbyWaveSpectrum.sintheta_dtheta_operator(nℓ, m)

    @testset "radial constant and constant" begin
        Mr = RossbyWaveSpectrum.differential_rotation_matrix(nr, nℓ, m,
            rotation_profile = :radial_constant; operators)
        Mc = RossbyWaveSpectrum.differential_rotation_matrix(nr, nℓ, m,
            rotation_profile = :constant; operators)
        @testset for colind in 1:3, rowind in 1:3
            @test matrix_block(Mr, rowind, colind) ≈ matrix_block(Mc, rowind, colind) atol = 1e-10 rtol = 1e-1
        end
    end
end
