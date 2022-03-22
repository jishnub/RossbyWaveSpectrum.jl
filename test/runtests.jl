using RossbyWaveSpectrum: RossbyWaveSpectrum, matrix_block, Rsun
using Test
using LinearAlgebra
using OffsetArrays
using Aqua
using SpecialPolynomials
using ForwardDiff
using FastTransforms
using Dierckx
using QuadGK

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

@testset "chebyshev" begin
    n = 10
    Tcf, Tci = RossbyWaveSpectrum.chebyshev_lobatto_forward_inverse(n)
    @test Tcf * Tci ≈ Tci * Tcf ≈ I
    r_chebyshev = RossbyWaveSpectrum.chebyshevnodes_lobatto(n)
    @test Tcf * r_chebyshev ≈ [0; 1; zeros(length(r_chebyshev)-2)]

    r_chebyshev, Tcf, Tci = RossbyWaveSpectrum.chebyshev_forward_inverse(n)
    @test Tcf * Tci ≈ Tci * Tcf ≈ I
    @test Tcf * r_chebyshev ≈ [0; 1; zeros(length(r_chebyshev)-2)]
end

@testset "green fn W" begin
    nr, nℓ = 50, 2

    n_lobatto = 2nr
    r_chebyshev_lobatto = RossbyWaveSpectrum.chebyshevnodes_lobatto(n_lobatto)
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ)
    operators_unstratified = RossbyWaveSpectrum.radial_operators(nr, nℓ, _stratified = false)
    (; Δr, r_mid) = operators.radial_params
    r_lobatto = @. (Δr/2) * r_chebyshev_lobatto .+ r_mid
    (; ηρ_cheby, r_cheby, onebyr_cheby) = operators.rad_terms

    @testset for ℓ in 2:5:42
        (; sρ) = operators.rad_terms

        Hℓ, = RossbyWaveSpectrum.greenfn_radial_lobatto(ℓ, operators, n_lobatto)
        @testset "boundaries" begin
            @test all(x -> isapprox(x/(Rsun * (Δr/2)), 0, atol=1e-14), @view Hℓ[1, :])
            @test all(x -> isapprox(x/(Rsun * (Δr/2)), 0, atol=1e-14), @view Hℓ[end, :])
            @test all(x -> isapprox(x/(Rsun * (Δr/2)), 0, atol=1e-14), @view Hℓ[:, 1])
            @test all(x -> isapprox(x/(Rsun * (Δr/2)), 0, atol=1e-14), @view Hℓ[:, end])
        end

        @testset "symmetry" begin
            H2 = Hℓ .* sρ.(r_lobatto)'
            @test H2 ≈ Symmetric(H2) rtol=1e-1
            @test_broken H2 ≈ Symmetric(H2) rtol=1e-2
        end

        @testset "unstratified" begin
            # in this case there's an analytical solution
            function greenfn_radial_lobatto_unstratified_analytical(ℓ, operators)
                (; r_in, r_out, Δr) = operators.radial_params
                r_in_frac = r_in/Rsun
                r_out_frac = r_out/Rsun
                W = (2ℓ+1)*((r_out_frac)^(2ℓ+1) - (r_in_frac)^(2ℓ+1))
                norm = Rsun/W * (Δr/2)
                H = zeros(n_lobatto+1, n_lobatto+1)

                for rind in axes(r_lobatto, 1)[2:end-1], sind in axes(r_lobatto, 1)[2]:rind
                    ri = r_lobatto[rind]/Rsun
                    sj = r_lobatto[sind]/Rsun
                    H[rind, sind] = (sj^(ℓ+1) - r_in_frac^(2ℓ+1)/sj^ℓ)*(ri^(ℓ+1) - r_out_frac^(2ℓ+1)/ri^ℓ) * norm
                end

                Symmetric(H, :L)
            end
            Hℓ, = RossbyWaveSpectrum.greenfn_radial_lobatto(ℓ, operators_unstratified, n_lobatto)
            Hℓ_exp = greenfn_radial_lobatto_unstratified_analytical(ℓ, operators_unstratified)
            if ℓ < 15
                @test Hℓ ≈ Hℓ_exp rtol=1e-2
            else
                @test Hℓ ≈ Hℓ_exp rtol=1e-1
                @test_broken Hℓ ≈ Hℓ_exp rtol=1e-2
            end
            @testset "symmetry" begin
                if ℓ < 20
                    @test Hℓ ≈ Symmetric(Hℓ) rtol=1e-2
                else
                    @test Hℓ ≈ Symmetric(Hℓ) rtol=1e-1
                    @test_broken Hℓ ≈ Symmetric(Hℓ) rtol=1e-2
                end
            end
        end
    end

    @testset "integrals" begin
        (; r_chebyshev) = operators.coordinates
        (; Tcrfwd) = operators.transforms
        ℓ = 2
        H, = RossbyWaveSpectrum.greenfn_radial_lobatto(ℓ, operators, n_lobatto)
        Tcf, Tci = RossbyWaveSpectrum.chebyshev_lobatto_forward_inverse(n_lobatto)
        Hc = RossbyWaveSpectrum.greenfn_cheby(ℓ, operators)

        @testset for n = 0:5:size(Hc, 2)-1
            f = chebyshevT(n)

            intres1 = [begin
                Hs = Spline1D(r_chebyshev_lobatto, H[r_ind, :])
                pi/nr * sum(x -> √(1-x^2) * Hs(x) * f(x), r_chebyshev)
            end
            for r_ind in axes(H, 1)]
            intres1 = (Tcf * intres1)[1:nr]

            intres2 = [begin
                Hs = Spline1D(r_chebyshev_lobatto, H[r_ind, :])
                quadgk(x -> Hs(x) * f(x), -1, 1)[1]
            end
            for r_ind in axes(H, 1)]
            intres2 = (Tcf * intres2)[1:nr]

            if n <= 40
                @test intres1 ≈ intres2 rtol=1e-2
            else
                @test_broken Hf ≈ intres2 rtol=1e-2
                @test intres1 ≈ intres2 rtol=3e-2
            end

            Hf = Hc[:, n+1]
            @test Hf ≈ intres2 rtol=1e-2
        end
    end
end

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
    (; DDrM, ddrDDrM, onebyr2_chebyM, ddrM, onebyr_chebyM, gM) = operators.diff_operator_matrices
    (; onebyr_cheby, onebyr2_cheby, r2_cheby, r_cheby, ηρ_cheby, ηT_cheby,
            g_cheby, ddr_lnρT, κ, ddr_S0_by_cp) = operators.rad_terms
    (; r_in, r_out, nchebyr) = operators.radial_params
    (; mat) = operators

    @test isapprox(onebyr_cheby(-1), 1/r_in, rtol=1e-4)
    @test isapprox(onebyr_cheby(1), 1/r_out, rtol=1e-4)
    @test isapprox(onebyr2_cheby(-1), 1/r_in^2, rtol=1e-4)
    @test isapprox(onebyr2_cheby(1), 1/r_out^2, rtol=1e-4)

    onebyr2_IplusrηρM = mat((1 + ηρ_cheby * r_cheby) * onebyr2_cheby)

    ∇r2_plus_ddr_lnρT_ddr = d2dr2 + 2onebyr_cheby*ddr + ddr_lnρT * ddr
    κ_∇r2_plus_ddr_lnρT_ddrM = RossbyWaveSpectrum.chebyshevmatrix(κ * ∇r2_plus_ddr_lnρT_ddr, nr, 3)
    κ_by_r2M = mat(κ * onebyr2_cheby)
    onebyr2_cheby_ddr_S0_by_cpM = mat(onebyr2_cheby * ddr_S0_by_cp)

    ddrηρ = ddr * ηρ_cheby

    M = RossbyWaveSpectrum.uniform_rotation_matrix(nr, nℓ, m; operators)

    chebyfwdnr(f, scalefactor = 5) = chebyfwd(f, r_in, r_out, nr, scalefactor)

    Ω0 = RossbyWaveSpectrum.equatorial_rotation_angular_velocity(r_out / Rsun)

    ℓ′ = 1
    # for these terms ℓ = ℓ′ (= 1 in this case)
    WWterm, WSterm, SWterm, SSterm = (zeros(nr, nr) for i in 1:4)
    RossbyWaveSpectrum.uniform_rotation_matrix_terms_outer!((WWterm, WSterm, SWterm, SSterm),
                        (ℓ′, m), nchebyr,
                        (ddrDDrM, onebyr2_IplusrηρM, gM,
                            κ_∇r2_plus_ddr_lnρT_ddrM, κ_by_r2M, onebyr2_cheby_ddr_S0_by_cpM), Ω0)

    @testset "V terms" begin
        @testset "WV term" begin
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
        Wn1 = P11norm

        @testset "VW term" begin
            function Drρ_fn(r, n)
                r̄_r = r̄(r)
                Tn = chebyshevT(n, r̄_r)
                Unm1 = chebyshevU(n-1, r̄_r)
                a * n * Unm1 + ηρ_cheby(r̄_r) * Tn
            end

            @testset "Drρ" begin
                @testset for n in 1:nr - 1
                    Drρ_Tn_analytical = chebyfwdnr(r -> Drρ_fn(r,n))
                    @test DDrM[:, n+1] ≈ Drρ_Tn_analytical rtol=1e-8
                end
            end

            function VWterm_fn(r, n)
                r̄_r = r̄(r)
                Tn = chebyshevT(n, r̄_r)
                -2√(1/15) * (Drρ_fn(r, n) - 2/r * Tn) * Wscaling
            end

            VW = RossbyWaveSpectrum.matrix_block(M, 1, 2, nfields)

            W1_inds = nr .+ (1:nr)

            @testset for n in 1:nr-1
                VW_op_times_W = Wn1 * real(VW[W1_inds, n+1])
                VW_times_W_explicit = chebyfwdnr(r -> VWterm_fn(r, n))
                @test VW_op_times_W ≈ -VW_times_W_explicit rtol = 1e-6
            end
        end

        @testset "WW term" begin
            function ddrDrρTn_fn(r, n)
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
                    ddrDrρ_Tn_analytical = chebyfwdnr(r -> ddrDrρTn_fn(r, n))
                    @test ddrDDrM[:, n+1] ≈ ddrDrρ_Tn_analytical rtol=1e-8
                end
            end

            function onebyr2Tn_fn(r, n)
                r̄_r = r̄(r)
                Tn = chebyshevT(n, r̄_r)
                1/r^2 * Tn
            end

            @testset "onebyr2Tn" begin
                @testset for n in 1:nr - 1
                    onebyr2_Tn_analytical = chebyfwdnr(r -> onebyr2Tn_fn(r, n))
                    @test onebyr2_chebyM[:, n+1] ≈ onebyr2_Tn_analytical rtol=1e-4
                end
            end

            function onebyr2_Iplusrηρ_Tn_fn(r, n)
                r̄_r = r̄(r)
                Tn = chebyshevT(n, r̄_r)
                ηr = ηρ_cheby(r̄_r)
                1/r^2 * (1 + ηr * r) * Tn
            end

            @testset "onebyr2_IplusrηρM" begin
                @testset for n in 1:nr - 1
                    onebyr2_Iplusrηρ_Tn_analytical = chebyfwdnr(r -> onebyr2_Iplusrηρ_Tn_fn(r, n))
                    @test onebyr2_IplusrηρM[:, n+1] ≈ onebyr2_Iplusrηρ_Tn_analytical rtol=1e-4
                end
            end

            function WWterm_fn(r, n)
                ℓℓp1 = 2
                ddrDrρTn_fn(r, n) - ℓℓp1 * onebyr2_Iplusrηρ_Tn_fn(r, n)
            end
            @testset for n in 1:nr-1
                WW_op_times_W = @view WWterm[:, n+1]
                WW_times_W_explicit = chebyfwdnr(r -> WWterm_fn(r, n))
                @test WW_op_times_W ≈ WW_times_W_explicit rtol = 1e-4
            end
        end

        @testset "SW term" begin
            function onebyr2_ddr_S0_by_cp_fn(r, n)
                r̄_r = r̄(r)
                Tn = chebyshevT(n, r̄_r)
                1/r^2 * ddr_S0_by_cp(r̄_r) * Tn
            end
            @testset for n in 1:nr-1
                onebyr2_ddr_S0_by_cp_op = onebyr2_cheby_ddr_S0_by_cpM[:, n+1]
                onebyr2_ddr_S0_by_cp_explicit = chebyfwdnr(r -> onebyr2_ddr_S0_by_cp_fn(r, n))
                @test onebyr2_ddr_S0_by_cp_op ≈ onebyr2_ddr_S0_by_cp_explicit rtol = 1e-4
            end
        end
    end

    @testset "S terms" begin
        @testset "WS term" begin
            function WSterm_fn(r, n)
                r̄_r = r̄(r)
                Tn = chebyshevT(n, r̄_r)
                -g_cheby(r̄_r)/Ω0 * Tn
            end
            @testset for n in 1:nr-1
                WStermM_op = @view WSterm[:, n+1]
                WStermM_analytical = chebyfwdnr(r -> WSterm_fn(r, n))
                @test WStermM_op ≈ WStermM_analytical rtol=1e-8
            end
        end
        @testset "SS term" begin
            function ddr_lnρT_fn(r, n)
                r̄_r = r̄(r)
                Tn = chebyshevT(n, r̄_r)
                ηρr = ηρ_cheby(r̄_r)
                ηTr = ηT_cheby(r̄_r)
                (ηρr + ηTr) * Tn
            end
            ddr_lnρTM = mat(ddr_lnρT)
            @testset for n in 1:nr-1
                ddr_lnρT_op = @view ddr_lnρTM[:, n+1]
                ddr_lnρT_analytical = chebyfwdnr(r -> ddr_lnρT_fn(r, n))
                @test ddr_lnρT_op ≈ ddr_lnρT_analytical rtol=1e-8
            end
            function ddr_lnρT_ddr_fn(r, n)
                r̄_r = r̄(r)
                Unm1 = n >= 1 ? chebyshevU(n-1, r̄_r) : 0.0
                ηρr = ηρ_cheby(r̄_r)
                ηTr = ηT_cheby(r̄_r)
                (ηρr + ηTr) * a * n * Unm1
            end
            ddr_lnρT_ddrM = RossbyWaveSpectrum.chebyshevmatrix(ddr_lnρT * ddr, nr, 3)
            @testset for n in 1:nr-1
                ddr_lnρT_ddr_op = @view ddr_lnρT_ddrM[:, n+1]
                ddr_lnρT_ddr_analytical = chebyfwdnr(r -> ddr_lnρT_ddr_fn(r, n))
                @test ddr_lnρT_ddr_op ≈ ddr_lnρT_ddr_analytical rtol=1e-8
            end

            function onebyr_ddr_fn(r, n)
                r̄_r = r̄(r)
                Unm1 = n >= 1 ? chebyshevU(n-1, r̄_r) : 0.0
                1/r * a * n * Unm1
            end
            onebyr_ddrM = mat(onebyr_cheby * ddr)
            @testset for n in 1:nr-1
                onebyr_ddr_op = @view onebyr_ddrM[:, n+1]
                onebyr_ddr_analytical = chebyfwdnr(r -> onebyr_ddr_fn(r, n))
                @test onebyr_ddr_op ≈ onebyr_ddr_analytical rtol=1e-4
            end

            function SSterm_fn(r, n)
                r̄_r = r̄(r)
                Tn = chebyshevT(n, r̄_r)
                Unm1 = n >= 1 ? chebyshevU(n-1, r̄_r) : 0.0
                Unm2 = n >= 2 ? chebyshevU(n-2, r̄_r) : 0.0
                ηρr = ηρ_cheby(r̄_r)
                ηTr = ηT_cheby(r̄_r)
                f = a^2 * n * ((n-1)*r̄_r*Unm1 - n*Unm2)/(r̄_r^2 - 1) +
                    2/r^2 * (a*n*r*Unm1 - Tn) + a*n*Unm1*(ηρr + ηTr)
                κ/Ω0 * f
            end
            @testset for n in 1:nr-1
                SStermM_op = @view SSterm[:, n+1]
                SStermM_analytical = chebyfwdnr(r -> SSterm_fn(r, n))
                @test SStermM_op ≈ SStermM_analytical rtol=1e-4
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
end

@testset "uniform rotation solution" begin
    nr, nℓ = 30, 15
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ); constraints = RossbyWaveSpectrum.constraintmatrix(operators);
    (; r_in, r_out, Δr) = operators.radial_params
    (; BC) = constraints;
    @testset for m in [1, 10, 20]
        λu, vu, Mu = RossbyWaveSpectrum.uniform_rotation_spectrum(nr, nℓ, m; operators, constraints);
        λuf, vuf = RossbyWaveSpectrum.filter_eigenvalues(λu, vu, Mu, m;
            operators, constraints, Δl_cutoff = 7, n_cutoff = 9,
            eig_imag_damped_cutoff = 1e-3, eig_imag_unstable_cutoff = -1e-3);
        @info "$(length(λuf)) eigenmode$(length(λuf) > 1 ? "s" : "") found for m = $m"
        @testset "ℓ == m" begin
            @test findmin(abs.(real(λuf) .- 2/(m+1)))[1] < 1e-4
        end
        @testset "boundary condition" begin
            @testset for n in axes(vuf, 2)
                vfn = @view vuf[:, n]
                @testset "V" begin
                    @testset for ℓind in 1:nℓ
                        ℓ_skip = (ℓind - 1)*nr
                        inds_ℓ = ℓ_skip .+ (1:nr)
                        v = vfn[inds_ℓ]
                        @test BC[1:2, inds_ℓ] * v ≈ [0, 0] atol=1e-10
                        pv = SpecialPolynomials.Chebyshev(v)
                        drpv = 1/(Δr/2) * SpecialPolynomials.derivative(pv)
                        @testset "inner boundary" begin
                            @test (drpv(-1) - 2pv(-1)/r_in)*Rsun ≈ 0 atol=1e-10
                        end
                        @testset "outer boundary" begin
                            @test (drpv(1) - 2pv(1)/r_out)*Rsun ≈ 0 atol=1e-10
                        end
                    end
                end
                @testset "W" begin
                    @testset for ℓind in 1:nℓ
                        ℓ_skip = (ℓind - 1)*nr
                        inds_ℓ = nr*nℓ + ℓ_skip .+ (1:nr)
                        w = vfn[inds_ℓ]
                        @test BC[3:4, inds_ℓ] * w ≈ [0, 0] atol=1e-10
                        pw = SpecialPolynomials.Chebyshev(w)
                        @testset "inner boundary" begin
                            @test sum(i -> (-1)^i * w[i], axes(w, 1)) ≈ 0 atol=1e-10
                            @test pw(-1) ≈ 0 atol=1e-10
                        end
                        @testset "outer boundary" begin
                            @test sum(w) ≈ 0 atol=1e-10
                            @test pw(1) ≈ 0 atol=1e-10
                        end
                    end
                end
            end
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
        (; ddr, d2dr2, DDr, ddrDDr) = diff_operators
        (; onebyr_cheby, onebyr2_cheby, r2_cheby, r_cheby, ηρ_cheby) = rad_terms
        (; r_in, r_out) = operators.radial_params
        (; mat) = operators
        chebyfwdnr(f, scalefactor = 5) = chebyfwd(f, r_in, r_out, nr, scalefactor)

        ℓ = 2
        ℓℓp1 = ℓ*(ℓ+1)
        ℓ′ = 1
        cosθ21 = 1/√5
        sinθdθ21 = 1/√5
        ∇²_sinθdθ21 = -ℓℓp1 * sinθdθ21

        @testset for rotation_profile in [:constant, :linear, :solar_equator]

            (; ΔΩ, Ω0, ddrΔΩ, d2dr2ΔΩ) =
                RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives(
                        nℓ, m, r; operators, rotation_profile)

            ΔΩ_by_r = ΔΩ * onebyr_cheby
            ΔΩ_DDr = ΔΩ * DDr
            DDr_min_2byr = DDr - 2onebyr_cheby
            ΔΩ_DDr_min_2byr = ΔΩ * DDr_min_2byr
            ddrΔΩ_plus_ΔΩddr = ddrΔΩ + ΔΩ * ddr
            twoΔΩ_by_r = 2ΔΩ * onebyr_cheby

            VWterm, WVterm = zeros(nr,nr), zeros(nr,nr)
            RossbyWaveSpectrum.radial_differential_rotation_terms_inner!((VWterm, WVterm),
                        (ℓ, ℓ′), (cosθ21, sinθdθ21, ∇²_sinθdθ21),
                        (mat(ddrΔΩ), Ω0),
                        map(mat, (ΔΩ_by_r, ΔΩ_DDr, ΔΩ_DDr_min_2byr, ddrΔΩ_plus_ΔΩddr, twoΔΩ_by_r)))

            ΔΩ_ddrDDr = ΔΩ * ddrDDr
            ddrΔΩ_DDr = ddrΔΩ * DDr
            ddrΔΩ_DDr_plus_ΔΩ_ddrDDr = ddrΔΩ_DDr + ΔΩ_ddrDDr
            ΔΩ_by_r2 = ΔΩ * onebyr2_cheby
            two_ΔΩbyr_ηρ = twoΔΩ_by_r * ηρ_cheby
            ddrΔΩ_ddr_plus_2byr = ddrΔΩ * (ddr + 2onebyr_cheby)
            WWfixedterms = -two_ΔΩbyr_ηρ + d2dr2ΔΩ + ddrΔΩ_ddr_plus_2byr

            # for these terms ℓ = ℓ′ (=1 in this case)
            VVterm, WWterm = zeros(nr, nr), zeros(nr, nr)
            RossbyWaveSpectrum.radial_differential_rotation_terms_outer!((VVterm, WWterm),
                                    (ℓ′, m), (mat(ΔΩ), Ω0),
                                    map(mat, (ddrΔΩ_DDr_plus_ΔΩ_ddrDDr, ΔΩ_by_r2, WWfixedterms)))

            @testset "V terms" begin
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
                    twoΔΩ_by_rM = mat(twoΔΩ_by_r)
                    @testset for n in 1:nr-1
                        twoΔΩ_by_r_op = twoΔΩ_by_rM[:, n + 1]
                        twoΔΩ_by_r_explicit = chebyfwdnr(r -> twoΔΩ_by_r_term(r, n))
                        @test twoΔΩ_by_r_op ≈ twoΔΩ_by_r_explicit rtol = 1e-4
                    end
                end

                @testset "WV term" begin
                    function WVterms(r, n)
                        r̄_r = r̄(r)
                        Tn = chebyshevT(n, r̄_r)
                        Unm1 = chebyshevU(n-1, r̄_r)
                        2/√15 * (ΔΩ(r̄_r) * (a * n * Unm1 - 2/r * Tn)  + ddrΔΩ(r̄_r) * Tn) / Ω0
                    end
                    @testset for n in 1:nr-1
                        WV_op = Vn1 * @view WVterm[:, n + 1]
                        WV_explicit = chebyfwdnr(r -> WVterms(r, n))
                        @test WV_op ≈ WV_explicit rtol = 1e-4
                    end
                end
                @testset "VV term" begin
                    @testset for n in 1:nr-1
                        VV_op = @view VVterm[:, n + 1]
                        @test all(iszero, VV_op)
                    end
                end
            end
            @testset "W terms" begin
                Wn1 = P11norm
                @testset "VW term" begin
                    function DDrmin2byr_term(r, n)
                        r̄_r = r̄(r)
                        Tn = chebyshevT(n, r̄_r)
                        Unm1 = chebyshevU(n-1, r̄_r)
                        ηr = ηρ_cheby(r̄_r)
                        a * n * Unm1 - 2/r * Tn + ηr * Tn
                    end
                    DDr_min_2byrM = mat(DDr_min_2byr)
                    @testset for n in 1:nr-1
                        DDrmin2byr_op = @view DDr_min_2byrM[:, n + 1]
                        DDrmin2byr_explicit = chebyfwdnr(r -> DDrmin2byr_term(r, n))
                        @test DDrmin2byr_op ≈ DDrmin2byr_explicit rtol = 1e-4
                    end

                    function ddrΔΩ_term(r, n)
                        r̄_r = r̄(r)
                        Tn = chebyshevT(n, r̄_r)
                        ddrΔΩ(r̄_r) * Tn
                    end
                    ddrΔΩM = mat(ddrΔΩ)
                    @testset for n in 1:nr-1
                        ddrΔΩ_op = @view ddrΔΩM[:, n + 1]
                        ddrΔΩ_explicit = chebyfwdnr(r -> ddrΔΩ_term(r, n))
                        @test ddrΔΩ_op ≈ ddrΔΩ_explicit rtol = 1e-4
                    end

                    function ΔΩDDr_min_2byr_min_ddrΔΩ_term(r, n)
                        r̄_r = r̄(r)
                        Tn = chebyshevT(n, r̄_r)
                        ΔΩ(r̄_r) * DDrmin2byr_term(r, n)  - ddrΔΩ(r̄_r) * Tn
                    end
                    ΔΩDDr_min_2byr_min_ddrΔΩM = mat(ΔΩ * DDr_min_2byr - ddrΔΩ)
                    @testset for n in 1:nr-1
                        ΔΩDDr_min_2byr_min_ddrΔΩ_op = ΔΩDDr_min_2byr_min_ddrΔΩM[:, n + 1]
                        ΔΩDDr_min_2byr_min_ddrΔΩ_explicit = chebyfwdnr(r -> ΔΩDDr_min_2byr_min_ddrΔΩ_term(r, n))
                        @test ΔΩDDr_min_2byr_min_ddrΔΩ_op ≈ ΔΩDDr_min_2byr_min_ddrΔΩ_explicit rtol = 1e-4
                    end

                    VWterms(r, n) = -2/√15 * ΔΩDDr_min_2byr_min_ddrΔΩ_term(r, n) / Ω0

                    @testset for n in 1:nr-1
                        VW_op = Wn1 * @view VWterm[:, n + 1]
                        VW_explicit = chebyfwdnr(r -> VWterms(r, n))
                        @test VW_op ≈ -VW_explicit rtol = 1e-4
                    end
                end
                @testset "WW term" begin
                    function WWterms(r, n)
                        r̄_r = r̄(r)
                        Tn = chebyshevT(n, r̄_r)
                        Unm1 = chebyshevU(n-1, r̄_r)
                        ηr = ηρ_cheby(r̄_r)
                        -2/√3 * (ddrΔΩ(r̄_r) * (a * n * Unm1 + 2/r * Tn)  +
                                (d2dr2ΔΩ(r̄_r) - 2ΔΩ(r̄_r)/r * ηr) * Tn) / Ω0
                    end
                    @testset for n in 1:nr-1
                        WW_op = Wn1 * @view WWterm[:, n + 1]
                        WW_explicit = chebyfwdnr(r -> WWterms(r, n))
                        @test WW_op ≈ WW_explicit rtol = 1e-4
                    end
                end
            end
        end
    end
end
