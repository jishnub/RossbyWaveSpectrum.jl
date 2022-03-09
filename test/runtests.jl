using RossbyWaveSpectrum: RossbyWaveSpectrum, matrix_block, Rsun
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

# define an alias to avoid clashes in the REPL with Chebyshev from ApproxFun
const ChebyshevT = SpecialPolynomials.Chebyshev
chebyshevT(n) = ChebyshevT([zeros(n); 1])
chebyshevT(n, x) = chebyshevT(n)(x)
chebyshevU(n) = ChebyshevU([zeros(n); 1])
chebyshevU(n, x) = chebyshevU(n)(x)

function chebyfwd(f, r_in, r_out, nr, scalefactor = 5)
    w = FastTransforms.chebyshevpoints(scalefactor * nr)
    r_mid = (r_in + r_out)/2
    Δr = r_out - r_in
    r_fine = @. (Δr/2) * w + r_mid
    v = FastTransforms.chebyshevtransform(f.(r_fine))
    v[1:nr]
end

const P11norm = -2/√3

@testset "uniform rotation" begin
    nr, nℓ = 30, 2
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
    r̄(r) = clamp(a * r + b, -1.0, 1.0)
    (; Tcrfwd) = operators.transforms
    (; ddr, d2dr2, DDr) = operators.diff_operators
    (; DDrM, ddrDDrM, onebyr2_chebyM, ddrM, onebyr_chebyM) = operators.diff_operator_matrices
    (; onebyr_cheby, onebyr2_cheby, r2_cheby, r_cheby, ηρ_cheby) = operators.rad_terms
    (; r_in, r_out) = operators.radial_params
    (; mat) = operators

    @test isapprox(onebyr_cheby(-1), 1/r_in, rtol=1e-4)
    @test isapprox(onebyr_cheby(1), 1/r_out, rtol=1e-4)
    @test isapprox(onebyr2_cheby(-1), 1/r_in^2, rtol=1e-4)
    @test isapprox(onebyr2_cheby(1), 1/r_out^2, rtol=1e-4)

    onebyr2_IplusrηρM = mat((1 + ηρ_cheby * r_cheby) * onebyr2_cheby)

    ddrηρ = ddr * ηρ_cheby

    M = RossbyWaveSpectrum.uniform_rotation_matrix(nr, nℓ, m; operators)

    chebyfwdnr(f, scalefactor = 5) = chebyfwd(f, r_in, r_out, nr, scalefactor)

    @testset "V terms" begin
        function ddr_term(r, n)
            r̄_r = r̄(r)
            Unm1 = chebyshevU(n-1, r̄_r)
            a * n * Unm1
        end

        @testset "ddr" begin
            @testset for n in 1:nr - 1
                ddr_Tn_analytical = chebyfwdnr(r -> ddr_term(r,n))
                @test ddrM[:, n+1] ≈ ddr_Tn_analytical rtol=1e-8
            end
        end

        function onebyrTn_term(r, n)
            r̄_r = r̄(r)
            Tn = chebyshevT(n, r̄_r)
            1/r * Tn
        end

        @testset "onebyrTn" begin
            @testset for n in 1:nr - 1
                onebyr_Tn_analytical = chebyfwdnr(r -> onebyrTn_term(r,n))
                @test onebyr_chebyM[:, n+1] ≈ onebyr_Tn_analytical rtol=1e-4
            end
        end

        Vn1 = P11norm
        function WVtermfn(r, n)
            r̄_r = r̄(r)
            Tn = chebyshevT(n, r̄_r)
            2√(1/15) * (ddr_term(r, n) - 2onebyrTn_term(r, n)) / Wscaling
        end
        @testset "WV term" begin
            ℓ = 2
            ℓ′ = 1
            m = 1
            ℓℓp1 = ℓ*(ℓ+1)
            ℓ′ℓ′p1 = ℓ′*(ℓ′+1)

            Cℓ′ = ddrM - ℓ′ℓ′p1 * onebyr_chebyM
            C1 = ddrM - 2onebyr_chebyM
            WV = -2 / ℓℓp1 * (ℓ′ℓ′p1 * C1 * (1/√5) + Cℓ′ * (1/√5)) / Wscaling

            @testset for n in 1:nr-1
                WV_op_times_W = Vn1 * real(WV[:, n+1])
                WV_times_W_explicit = chebyfwdnr(r -> WVtermfn(r,n))
                @test WV_op_times_W ≈ WV_times_W_explicit rtol = 1e-5
            end
        end
    end

    @testset "W terms" begin
        function Drρterm(r, n)
            r̄_r = r̄(r)
            Tn = chebyshevT(n, r̄_r)
            Unm1 = chebyshevU(n-1, r̄_r)
            a * n * Unm1 + ηρ_cheby(r̄_r) * Tn
        end

        @testset "Drρ" begin
            @testset for n in 1:nr - 1
                Drρ_Tn_analytical = chebyfwdnr(r -> Drρterm(r,n))
                @test DDrM[:, n+1] ≈ Drρ_Tn_analytical rtol=1e-8
            end
        end

        function VWtermfn(r, n)
            r̄_r = r̄(r)
            Tn = chebyshevT(n, r̄_r)
            -2√(1/15) * (Drρterm(r, n) - 2/r * Tn) * Wscaling
        end

        Wn1 = P11norm

        @testset "VW term" begin
            VW = RossbyWaveSpectrum.matrix_block(M, 1, 2, nfields)

            W1_inds = nr .+ (1:nr)

            @testset for n in 1:nr-1
                VW_op_times_W = Wn1 * real(VW[W1_inds, n+1])
                VW_times_W_explicit = chebyfwdnr(r -> VWtermfn(r, n))
                @test VW_op_times_W ≈ -VW_times_W_explicit rtol = 1e-6
            end
        end

        function ddrDrρTn_term(r, n)
            r̄_r = r̄(r)
            Tn = chebyshevT(n, r̄_r)
            Unm1 = chebyshevU(n-1, r̄_r)
            ηr = ηρ_cheby(r̄_r)
            η′r = ddrηρ(r̄_r)
            T1 = a * n * Unm1 * ηr + Tn * η′r
            if n > 1
                Unm2 = chebyshevU(n-2, r̄_r)
                return a^2 * n * (-n*Unm2 + (n-1)*Unm1*r̄_r)/(r̄_r^2 - 1) + T1
            elseif n == 1
                return T1
            else
                error("invalid n")
            end
        end

        @testset "ddrDrρTn" begin
            @testset for n in 1:nr - 1
                ddrDrρ_Tn_analytical = chebyfwdnr(r -> ddrDrρTn_term(r, n))
                @test ddrDDrM[:, n+1] ≈ ddrDrρ_Tn_analytical rtol=1e-8
            end
        end

        function onebyr2Tn_term(r, n)
            r̄_r = r̄(r)
            Tn = chebyshevT(n, r̄_r)
            1/r^2 * Tn
        end

        @testset "onebyr2Tn" begin
            @testset for n in 1:nr - 1
                onebyr2_Tn_analytical = chebyfwdnr(r -> onebyr2Tn_term(r, n))
                @test onebyr2_chebyM[:, n+1] ≈ onebyr2_Tn_analytical rtol=1e-4
            end
        end

        function onebyr2_Iplusrηρ_Tn_term(r, n)
            r̄_r = r̄(r)
            Tn = chebyshevT(n, r̄_r)
            ηr = ηρ_cheby(r̄_r)
            1/r^2 * (1 + ηr * r) * Tn
        end

        @testset "onebyr2_IplusrηρM" begin
            @testset for n in 1:nr - 1
                onebyr2_Iplusrηρ_Tn_analytical = chebyfwdnr(r -> onebyr2_Iplusrηρ_Tn_term(r, n))
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

            @testset for n in 1:nr-1
                WW_op_times_W = Wn1 * real(WW[:, n+1])
                WW_times_W_explicit = chebyfwdnr(r -> WWtermfn(r, n))
                @test WW_op_times_W ≈ WW_times_W_explicit rtol = 1e-4
            end
        end
    end
end

@testset "viscosity" begin
    nr, nℓ = 30, 2
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
    (; r_in, r_out) = operators.radial_params

    chebyfwdnr(f, scalefactor = 5) = chebyfwd(f, r_in, r_out, nr, scalefactor)

    @testset "V terms" begin
        Vn1 = P11norm

        function VVterm1fn(r, n)
            r̄_r = r̄(r)
            T = chebyshevT(n)
            Tn = T(r̄_r)
            d2Tn = a^2 * d2f(T)(r̄_r)
            Vn1 * (d2Tn - 2 / r^2 * Tn)
        end

        function VVterm3fn(r, n)
            r̄_r = r̄(r)
            Tn = chebyshevT(n, r̄_r)
            Unm1 = chebyshevU(n-1, r̄_r)
            anr = a * n * r
            Vn1 * (-2Tn + anr * Unm1) * ηρ_cheby(r̄_r) / r
        end

        VVtermfn(r, n) = VVterm1fn(r, n) + VVterm3fn(r, n)

        M = zeros(ComplexF64, nfields * nparams, nfields * nparams)
        RossbyWaveSpectrum.viscosity_terms!(M, nr, nℓ, m; operators)

        @testset "VV terms" begin
            VV = RossbyWaveSpectrum.matrix_block(M, 1, 1, nfields)
            V1_inds = 1:nr
            @testset for n in 1:nr-1
                V_op = Vn1 * real(VV[V1_inds, n + 1] ./ (-im * ν))
                V_explicit = chebyfwdnr(r -> VVtermfn(r, n))
                @test V_op ≈ V_explicit rtol = 1e-5
            end
        end
    end
    @testset "W terms" begin
        function WWtermfn(r, n)

        end
    end
end

@testset "differential rotation" begin
    @testset "compare with constant" begin
        nr, nℓ = 30, 2
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
    end

    @testset "radial differential rotation" begin
        nr, nℓ = 30, 2
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
        (; ddr, d2dr2, DDr) = diff_operators
        (; onebyr_cheby, r2_cheby, r_cheby, ηρ_cheby) = rad_terms
        (; r_in, r_out) = operators.radial_params
        (; mat) = operators
        chebyfwdnr(f, scalefactor = 5) = chebyfwd(f, r_in, r_out, nr, scalefactor)

        ℓ = 2
        ℓℓp1 = ℓ*(ℓ+1)
        ℓ′ = 1
        cosθ21 = 1/√5
        sinθdθ21 = 1/√5
        ∇²_sinθdθ21 = -ℓℓp1 * sinθdθ21

        (; ΔΩ, Ω0, ddrΔΩ, d2dr2ΔΩ) =
            RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives(
                    nℓ, m, r; operators, rotation_profile = :linear)

        ΔΩ_by_r = ΔΩ * onebyr_cheby
        ΔΩ_DDr = ΔΩ * DDr
        DDr_min_2byr = DDr - 2onebyr_cheby
        ΔΩ_DDr_min_2byr = ΔΩ * DDr_min_2byr
        ddrΔΩ_plus_ΔΩddr = ddrΔΩ + ΔΩ * ddr
        twoΔΩ_by_r = 2ΔΩ * onebyr_cheby

        (; VWterm, WVterm) = RossbyWaveSpectrum.radial_differential_rotation_terms_inner(
                    (ℓ, ℓ′), (cosθ21, sinθdθ21, ∇²_sinθdθ21),
                    (ΔΩ, ddrΔΩ, Ω0),
                    (ΔΩ_by_r, ΔΩ_DDr, ΔΩ_DDr_min_2byr, ddrΔΩ_plus_ΔΩddr, twoΔΩ_by_r);
                    operators)

        @testset "WV terms" begin
            Vn1 = P11norm

            @testset "ddrΔΩ_plus_ΔΩddr" begin
                function ddrΔΩ_plus_ΔΩddr_term(r, n)
                    r̄_r = r̄(r)
                    Tn = chebyshevT(n, r̄_r)
                    Unm1 = chebyshevU(n-1, r̄_r)
                    ddrΔΩ(r̄_r) * Tn + ΔΩ(r̄_r) * a * n * Unm1
                end
                @testset for n in 1:nr-1
                    ddrΔΩ_plus_ΔΩddr_op = mat(ddrΔΩ_plus_ΔΩddr)[:, n + 1]
                    ddrΔΩ_plus_ΔΩddr_explicit = chebyfwdnr(r -> ddrΔΩ_plus_ΔΩddr_term(r, n))
                    @test ddrΔΩ_plus_ΔΩddr_op ≈ ddrΔΩ_plus_ΔΩddr_explicit rtol = 1e-8
                end
            end

            @testset "twoΔΩ_by_r" begin
                function twoΔΩ_by_r_term(r, n)
                    r̄_r = r̄(r)
                    Tn = chebyshevT(n, r̄_r)
                    2ΔΩ(r̄_r)/r * Tn
                end
                @testset for n in 1:nr-1
                    twoΔΩ_by_r_op = mat(twoΔΩ_by_r)[:, n + 1]
                    twoΔΩ_by_r_explicit = chebyfwdnr(r -> twoΔΩ_by_r_term(r, n))
                    @test twoΔΩ_by_r_op ≈ twoΔΩ_by_r_explicit rtol = 1e-4
                end
            end

            @testset "WVterms" begin
                function WVterms(r, n)
                    r̄_r = r̄(r)
                    Tn = chebyshevT(n, r̄_r)
                    Unm1 = chebyshevU(n-1, r̄_r)
                    2√(1/15) * (ΔΩ(r̄_r) * (a * n * Unm1 - 2/r * Tn)  + ddrΔΩ(r̄_r) * Tn) / Ω0
                end
                @testset for n in 1:nr-1
                    WV_op = Vn1 * mat(WVterm)[:, n + 1]
                    WV_explicit = chebyfwdnr(r -> WVterms(r, n))
                    @test WV_op ≈ WV_explicit rtol = 1e-4
                end
            end
        end
        @testset "WW terms" begin
        end
    end
end
