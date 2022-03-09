using RossbyWaveSpectrum: RossbyWaveSpectrum, matrix_block, Rsun, kron2
using ApproxFun
using Test
using LinearAlgebra
using OffsetArrays
using Aqua
using SpecialPolynomials
using ForwardDiff
using FastTransforms

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

df(f) = x -> ForwardDiff.derivative(f, x)
df(f, r) = df(f)(r)
d2f(f) = df(df(f))
d2f(f, r) = df(df(f))(r)

r̄(r, a, b) = clamp(a * r + b, -1.0, 1.0)
r̄tor(r̄, a, b) = (r̄ - b)/a

function chebyfwd(f, r_in, r_out, nr, scalefactor = 5)
    w = FastTransforms.chebyshevpoints(scalefactor * nr)
    r_mid = (r_in + r_out)/2
    Δr = r_out - r_in
    r_fine = (Δr/2) * w .+ r_mid
    v = FastTransforms.chebyshevtransform(f.(r_fine))
    v[1:nr]
end

@testset "uniform rotation" begin
    nr, nℓ = 50, 2
    nparams = nr * nℓ
    m = 1
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ)
    (; transforms, diff_operators, rad_terms, coordinates, radial_params, identities) = operators
    (; r, r_chebyshev) = coordinates
    (; nfields, ν) = operators.constants
    Wscaling = operators.constants.scalings.Wscaling::Float64
    r_mid = radial_params.r_mid::Float64
    Δr = radial_params.Δr::Float64
    a = 1 / (Δr / 2)
    b = -r_mid / (Δr / 2)
    r̄(r) = r̄(r, a, b)
    (; Tcrfwd) = operators.transforms
    (; ddr, d2dr2, DDr) = operators.diff_operators
    (; DDrM, ddrDDrM, onebyr2_chebyM) = operators.diff_operator_matrices
    (; onebyr_cheby, onebyr2_cheby, r2_cheby, r_cheby, ηρ_cheby) = operators.rad_terms
    (; r_in, r_out) = operators.radial_params

    @test isapprox(onebyr_cheby(-1), 1/r_in, rtol=1e-4)
    @test isapprox(onebyr_cheby(1), 1/r_out, rtol=1e-4)
    @test isapprox(onebyr2_cheby(-1), 1/r_in^2, rtol=1e-4)
    @test isapprox(onebyr2_cheby(1), 1/r_out^2, rtol=1e-4)

    onebyr2_IplusrηρM = mat((1 + ηρ_cheby * r_cheby) * onebyr2_cheby)

    ddrηρ = ddr * ηρ_cheby

    M = RossbyWaveSpectrum.uniform_rotation_matrix(nr, nℓ, m; operators)

    nmax_check = nr - ApproxFun.bandwidths(ApproxFun.Multiplication(ηρ_cheby):ApproxFun.Chebyshev())[1] - 1

    chebyfwdnr(f, n, scalefactor = 5) = chebyfwd(r -> f(r, n), r_in, r_out, nr, scalefactor)

    @testset "V terms" begin
    end

    @testset "W terms" begin
        function Drρterm(r, n)
            r̄_r = r̄(r)
            Tn = SpecialPolynomials.Chebyshev([zeros(n); 1])(r̄_r)
            Unm1 = ChebyshevU([zeros(n-1); 1])(r̄_r)
            a * n * Unm1 + ηρ_cheby(r̄_r) * Tn
        end

        @testset "Drρ" begin
            @testset for n in 1:nr - 1
                Drρ_Tn_analytical = chebyfwdnr(Drρterm, n)
                @test DDrM[:, n+1] ≈ Drρ_Tn_analytical rtol=1e-8
            end
        end

        function VWtermfn(r, n)
            r̄_r = r̄(r)
            Tn = SpecialPolynomials.Chebyshev([zeros(n); 1])(r̄_r)
            -2√(1/15) * (Drρterm(r, n) - 2/r * Tn) * Wscaling
        end

        Wn1 = -2√(1/ 3)

        @testset "VW term" begin
            VW = RossbyWaveSpectrum.matrix_block(M, 1, 2, nfields)

            W1_inds = nr .+ (1:nr)

            @testset for n in 1:nr-1
                VW_op_times_W = Wn1 * real(VW[W1_inds, n+1])
                VW_times_W_explicit = chebyfwdnr(VWtermfn, n)
                @test VW_op_times_W ≈ -VW_times_W_explicit rtol = 1e-6
            end
        end

        function ddrDrρTn_term(r, n)
            r̄_r = r̄(r)
            Tn = SpecialPolynomials.Chebyshev([zeros(n); 1])(r̄_r)
            Unm1 = ChebyshevU([zeros(n-1); 1])(r̄_r)
            ηr = ηρ_cheby(r̄_r)
            η′r = ddrηρ(r̄_r)
            T1 = a * n * Unm1 * ηr + Tn * η′r
            if n > 1
                Unm2 = ChebyshevU([zeros(n-2); 1])(r̄_r)
                return a^2 * n * (-n*Unm2 + (n-1)*Unm1*r̄_r)/(r̄_r^2 - 1) + T1
            elseif n == 1
                return T1
            else
                error("invalid n")
            end
        end

        @testset "ddrDrρTn" begin
            @testset for n in 1:nr - 1
                ddrDrρ_Tn_analytical = chebyfwdnr(ddrDrρTn_term, n)
                @test ddrDDrM[:, n+1] ≈ ddrDrρ_Tn_analytical rtol=1e-8
            end
        end

        function onebyr2Tn_term(r, n)
            r̄_r = r̄(r)
            Tn = SpecialPolynomials.Chebyshev([zeros(n); 1])(r̄_r)
            1/r^2 * Tn
        end

        @testset "onebyr2Tn" begin
            @testset for n in 1:nr - 1
                onebyr2_Tn_analytical = chebyfwdnr(onebyr2Tn_term, n)
                @test onebyr2_chebyM[:, n+1] ≈ onebyr2_Tn_analytical rtol=1e-4
            end
        end

        function onebyr2_Iplusrηρ_Tn_term(r, n)
            r̄_r = r̄(r)
            Tn = SpecialPolynomials.Chebyshev([zeros(n); 1])(r̄_r)
            ηr = ηρ_cheby(r̄_r)
            1/r^2 * (1 + ηr * r) * Tn
        end

        @testset "onebyr2_IplusrηρM" begin
            @testset for n in 1:nr - 1
                onebyr2_Iplusrηρ_Tn_analytical = chebyfwdnr(onebyr2_Iplusrηρ_Tn_term, n)
                @test onebyr2_IplusrηρM[:, n+1] ≈ onebyr2_Iplusrηρ_Tn_analytical rtol=1e-4
            end
        end

        function WWtermfn(r, n)
            Wn1 * (ddrDrρTn_term(r, n) - 2onebyr2_Iplusrηρ_Tn_term(r, n))
        end

        @testset "WW term" begin
            ℓ = 1
            m = 1
            ℓℓp1 = ℓ*(ℓ+1)
            WW = 2m / ℓℓp1 * (ddrDDrM - onebyr2_IplusrηρM * ℓℓp1)

            W1_inds = (1:nr)

            @testset for n in 1:nr-1
                WW_op_times_W = Wn1 * real(WW[W1_inds, n+1])
                WW_times_W_explicit = chebyfwdnr(WWtermfn, n)
                @test WW_op_times_W ≈ WW_times_W_explicit rtol = 1e-4
            end
        end
    end

    @testset "S term" begin
    end

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
