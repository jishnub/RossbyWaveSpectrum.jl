using RossbyWaveSpectrum: RossbyWaveSpectrum, matrix_block, Rsun, kron2
using Test
using LinearAlgebra
using OffsetArrays
using Aqua
using SpecialPolynomials
using ForwardDiff

@testset "project quality" begin
    Aqua.test_all(RossbyWaveSpectrum,
        ambiguities = false,
        stale_deps = (; ignore =
        [:PyCall, :PyPlot, :LaTeXStrings]),
    )
end

@testset "operators inferred" begin
    @inferred RossbyWaveSpectrum.radial_operators(5, 2)
end

@testset "viscosity" begin
    nr, nℓ = 40, 2
    nparams = nr * nℓ
    m = 1
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ)
    (; transforms, diff_operators, rad_terms, coordinates, radial_params, identities) = operators
    (; r, r_chebyshev) = coordinates
    (; nfields, ν) = operators.constants
    r_mid = radial_params.r_mid::Float64
    Δr = radial_params.Δr::Float64
    a = 1 / (Δr / 2)
    b = -r_mid / (Δr / 2)
    r̄(r) = clamp(a * r + b, -1.0, 1.0)
    (; Tcrfwd) = transforms
    (; Iℓ) = identities
    (; ddr, d2dr2) = diff_operators
    (; onebyr_cheby, r2_cheby, r_cheby, ηρ_cheby) = rad_terms

    cosθ = RossbyWaveSpectrum.costheta_operator(nℓ, m)
    sinθdθ = RossbyWaveSpectrum.sintheta_dtheta_operator(nℓ, m)

    df(f) = x -> ForwardDiff.derivative(f, x)
    df(f, r) = df(f)(r)
    d2f(f) = df(df(f))
    d2f(f, r) = df(df(f))(r)

    function VVterm1fn(r, n)
        r̄_r = r̄(r)
        T = Chebyshev([zeros(n); 1])
        Tn = T(r̄_r)
        d2Tn = a^2 * d2f(T)(r̄_r)
        -√(2 / 3) * (d2Tn - 2 / r^2 * Tn)
    end

    function VVterm3fn(r, n)
        r̄_r = r̄(r)
        Tn = Chebyshev([zeros(n); 1])(r̄_r)
        Unm1 = ChebyshevU([zeros(n - 1); 1])(r̄_r)
        anr = a * n * r
        -√(2 / 3) * (-2Tn + anr * Unm1) * ηρ_cheby(r̄_r) / r
    end

    VVtermfn(r, n) = VVterm1fn(r, n) + VVterm3fn(r, n)

    M = zeros(ComplexF64, nfields * nparams, nfields * nparams)
    RossbyWaveSpectrum.viscosity_terms!(M, nr, nℓ, m; operators)
    VV = RossbyWaveSpectrum.matrix_block(M, 1, 1, nfields)

    V1_inds = 1:nr

    @testset for n in 1:10
        V_cheby = [zeros(n); -√(2 / 3); zeros(nparams - (n + 1))]
        V_op = real((VV*V_cheby)[V1_inds] ./ (-im * ν))
        V_explicit = Tcrfwd * VVtermfn.(r, n)
        @test V_op ≈ V_explicit rtol = 1e-3
    end

end

@testset "differential rotation" begin
    @testset "compare with constant" begin
        nr, nℓ = 20, 2
        operators = RossbyWaveSpectrum.radial_operators(nr, nℓ)
        (; nfields) = operators.constants

        m = 1

        Mc = RossbyWaveSpectrum.differential_rotation_matrix(nr, nℓ, m,
            rotation_profile = :constant; operators)

        @testset "radial constant and constant" begin
            Mr = RossbyWaveSpectrum.differential_rotation_matrix(nr, nℓ, m,
                rotation_profile = :radial_constant; operators)

            @testset for colind in 1:nfields, rowind in 1:nfields
                @test matrix_block(Mr, rowind, colind, nfields) ≈ matrix_block(Mc, rowind, colind, nfields) atol = 1e-10 rtol = 1e-3
            end
        end

        # @testset "solar constant and constant" begin
        #     Ms = RossbyWaveSpectrum.differential_rotation_matrix(nr, nℓ, m,
        #         rotation_profile = :solar_constant; operators)

        #     @testset for colind in 1:nfields, rowind in 1:nfields
        #         if rowind == 2 && colind ∈ (1, 2)
        #             @test_broken matrix_block(Ms, rowind, colind, nfields) ≈ matrix_block(Mc, rowind, colind, nfields) atol = 1e-10 rtol = 1e-3
        #         else
        #             @test matrix_block(Ms, rowind, colind, nfields) ≈ matrix_block(Mc, rowind, colind, nfields) atol = 1e-10 rtol = 1e-3
        #         end
        #     end
        # end
    end

    # nr, nℓ = 40, 2
    # nparams = nr * nℓ
    # m = 1
    # operators = RossbyWaveSpectrum.radial_operators(nr, nℓ)
    # (; transforms, diff_operators, rad_terms, coordinates, radial_params, identities) = operators
    # (; r, r_chebyshev) = coordinates
    # (; r_mid, Δr) = radial_params
    # (; Tcrfwd) = transforms
    # (; Iℓ) = identities
    # (; ddr, d2dr2) = diff_operators
    # (; onebyr_cheby, r2_cheby, r_cheby) = rad_terms

    # r̄(r) = (r - r_mid::Float64) / (Δr::Float64 / 2)

    # cosθ = RossbyWaveSpectrum.costheta_operator(nℓ, m)
    # sinθdθ = RossbyWaveSpectrum.sintheta_dtheta_operator(nℓ, m)
end
