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
    @testset "u × ω matrix" begin
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

        ddr_plus_2byr = ddr + 2onebyr_cheby

        r_ddr_plus_2 = r_cheby * ddr + 2I
        r_ddr_plus_1 = r_cheby * ddr + I
        d²r_r2 = r2_cheby * d2dr2 + 4r_cheby * ddr + 2I

        rotation_profile = :linear
        (; ΔΩ_r, Ω0, ΔΩ, ΔΩ_spl, drΔΩ_real, ddrΔΩ) = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives(
            nℓ, m, r; operators, rotation_profile)

        ΔΩ1 = ΔΩ_r[1] / (r[1] - Rsun)
        cosθ = RossbyWaveSpectrum.costheta_operator(nℓ, m)
        sinθdθ = RossbyWaveSpectrum.sintheta_dtheta_operator(nℓ, m)

        (; negr²Ω0W_rhs, u_cross_ω_r, ∇²h_u_cross_ω_r,
            div_u_cross_ω, ωΩ_dot_ωf, div_uf_cross_ωΩ, div_uΩ_cross_ωf,
            negcurlωΩ_dot_uf, curlωf_dot_uΩ) =
            RossbyWaveSpectrum.negr²Ω0W_rhs_radial(m, (ΔΩ_r, drΔΩ_real, ΔΩ_spl, ΔΩ),
                (cosθ, sinθdθ); operators)

        r̄(r) = (r - r_mid::Float64) / (Δr::Float64 / 2)
        function ucrossωfn_V(r, n)
            r̄_r = r̄(r)
            Tn = Chebyshev([zeros(n); 1])(r̄_r)
            Unm1 = ChebyshevU([zeros(n - 1); 1])(r̄_r)
            1 / 3 * √(12 / 5) * ΔΩ1::Float64 * ((3 - 2Rsun / r) * Tn + n * (2 / Δr::Float64) * (r - Rsun) * Unm1)
        end
        d²r_fn(f) = r -> d²r_fn(f, r)
        d²r_fn(f, r) = ForwardDiff.derivative(r -> ForwardDiff.derivative(f, r), r)
        d²r_r²ucrossωfn_V(r, n) = d²r_fn(r -> r^2 * ucrossωfn_V(r, n), r)

        function ∇²h_ucrossωfn_V(r, n)
            ℓ = 2
            -ℓ * (ℓ + 1) * ucrossωfn_V(r, n)
        end

        function ωΩ_dot_ωf_fn_V(r, n)
            Tn = Chebyshev([zeros(n); 1])(r̄(r))
            Unm1 = ChebyshevU([zeros(n - 1); 1])(r̄(r))
            a = 2 / Δr::Float64
            b = -r_mid::Float64 * a
            2 / √15 * ΔΩ1::Float64 * 1 / r^2 * (a * n * r * (3r - 2Rsun) * Unm1 - 4(r - Rsun) * Tn)
        end

        function neg_curlωΩ_dot_uf_fn_V(r, n)
            Tn = Chebyshev([zeros(n); 1])(r̄(r))
            8 / √15 * ΔΩ1::Float64 * 1 / r * Tn
        end

        function curlωf_dot_uΩ_fn_V(r, n)
            a = 2 / Δr::Float64
            r̄_r = r̄(r)
            if n > 1
                Tn = Chebyshev([zeros(n); 1])(r̄_r)
                Unm1 = ChebyshevU([zeros(n - 1); 1])(r̄_r)
                Unm2 = ChebyshevU([zeros(n - 2); 1])(r̄_r)
                return -2 / √15 * ΔΩ1::Float64 * (r - Rsun) *
                       (a^2 * n * r^2 * ((n - 1) * r̄_r * Unm1 - n * Unm2) - 2(r̄_r^2 - 1) * Tn) /
                       (r^2 * (r̄_r^2 - 1))
            elseif n == 1
                return 4 / √15 * ΔΩ1::Float64 * (r - Rsun) * r̄_r / r^2
            else
                error("Invalid degree $n")
            end
        end

        function div_ufcrossωΩ_fn_V(r, n)
            Tn = Chebyshev([zeros(n); 1])(r̄(r))
            Unm1 = ChebyshevU([zeros(n - 1); 1])(r̄(r))
            a = 2 / Δr::Float64
            2 / √15 * ΔΩ1::Float64 * 1 / r^2 * (4Rsun * Tn + a * n * r * (3r - 2Rsun) * Unm1)
        end

        function div_uΩcrossωf_fn_V(r, n)
            r̄_r = r̄(r)
            a = 2 / Δr::Float64
            anr = a * n * r
            if n > 1
                Tn = Chebyshev([zeros(n); 1])(r̄_r)::Float64
                Unm1 = ChebyshevU([zeros(n - 1); 1])(r̄_r)::Float64
                Unm2 = ChebyshevU([zeros(n - 2); 1])(r̄_r)::Float64
                return 2 / √15 * ΔΩ1::Float64 * 1 / (r^2 * (r̄_r^2 - 1)) *
                       (anr^2 * (Rsun - r) * Unm2 +
                        anr * Unm1 * (a * (n - 1) * r * r̄_r * (r - Rsun) + (r̄_r^2 - 1) * (3r - 2Rsun)) +
                        6(1 - r̄_r^2) * (r - Rsun) * Tn
                )
            elseif n == 1
                return 2 / √15 * ΔΩ1::Float64 * 1 / r^2 * (a * r * (3r - 2Rsun) - 6(r - Rsun) * r̄_r)
            else
                error("invalid degree $n")
            end
        end

        function div_ucrossω_fn_V(r, n)
            a = 2 / Δr::Float64
            r̄_r = r̄(r)
            anr = a * n * r
            if n > 1
                Tn = Chebyshev([zeros(n); 1])(r̄_r)
                Unm1 = ChebyshevU([zeros(n - 1); 1])(r̄_r)
                Unm2 = ChebyshevU([zeros(n - 2); 1])(r̄_r)
                return 2 / √15 * ΔΩ1::Float64 * 1 / (r^2 * (1 - r̄_r^2)) *
                       (anr^2 * (r - Rsun) * Unm2 +
                        anr * Unm1 * (-a * (n - 1) * r * r̄_r * (r - Rsun) + 2(r̄_r^2 - 1) * (2Rsun - 3r)) -
                        2(1 - r̄_r^2) * (3r - 5Rsun) * Tn
                )
            elseif n == 1
                return 4 / √15 * ΔΩ1::Float64 * 1 / r^2 * (a * r * (3r - 2Rsun) + (-3r + 5Rsun) * r̄_r)
            else
                error("invalid degree $n")
            end
        end

        dr_r²div_ucrossω_fn_V(r, n) = ForwardDiff.derivative(r -> r^2 * div_ucrossω_fn_V(r, n), r)

        function Wterm_fn_V(r, n)
            a = 2 / Δr::Float64
            r̄_r = r̄(r)
            Tn = Chebyshev([zeros(n); 1])(r̄_r)
            Unm1 = ChebyshevU([zeros(n - 1); 1])(r̄_r)
            anr = a * n * r
            return 4 * √(3 / 5) * ΔΩ1::Float64 * 1 / r * (anr * (r - Rsun) * Unm1 - (r - 2Rsun) * Tn)
        end

        @testset for n in 1:nr-6
            V = [zeros(n); -√(4 / 3); zeros((nℓ - 1) * nr + nr - (n + 1))]

            rindsℓ1ℓ2 = nr .+ (1:nr)

            ucrossω_r_real = ucrossωfn_V.(r, n)
            ucrossω_rtoc = Tcrfwd * ucrossω_r_real
            ucrossω_operator = u_cross_ω_r.V[rindsℓ1ℓ2, :] * V
            @test ucrossω_operator ≈ ucrossω_rtoc rtol = 1e-5

            ∇²h_ucrossω_r_real = ∇²h_ucrossωfn_V.(r, n)
            ∇²h_ucrossω_rtoc = Tcrfwd * ∇²h_ucrossω_r_real
            ∇²h_ucrossω_operator = ∇²h_u_cross_ω_r.V[rindsℓ1ℓ2, :] * V
            @test ∇²h_ucrossω_operator ≈ ∇²h_ucrossω_rtoc rtol = 1e-5

            d²r_r²ucrossω_real = d²r_r²ucrossωfn_V.(r, n)
            d²r_r²ucrossω_rtoc = Tcrfwd * d²r_r²ucrossω_real
            d²r_r²ucrossω_operator = d²r_r2 * ucrossω_operator
            @test d²r_r²ucrossω_operator ≈ d²r_r²ucrossω_rtoc rtol = 1e-3

            ωΩ_dot_ωf_real = ωΩ_dot_ωf_fn_V.(r, n)
            ωΩ_dot_ωf_rtoc = Tcrfwd * ωΩ_dot_ωf_real
            ωΩ_dot_ωf_operator = ωΩ_dot_ωf.V[rindsℓ1ℓ2, :] * V
            @test ωΩ_dot_ωf_operator ≈ ωΩ_dot_ωf_rtoc rtol = 1e-5

            negcurlωΩ_dot_uf_real = neg_curlωΩ_dot_uf_fn_V.(r, n)
            negcurlωΩ_dot_uf_rtoc = Tcrfwd * negcurlωΩ_dot_uf_real
            negcurlωΩ_dot_uf_operator = negcurlωΩ_dot_uf.V[rindsℓ1ℓ2, :] * V
            @test negcurlωΩ_dot_uf_operator ≈ negcurlωΩ_dot_uf_rtoc rtol = 1e-5

            @test div_ufcrossωΩ_fn_V.(r, n) ≈ ωΩ_dot_ωf_fn_V.(r, n) + neg_curlωΩ_dot_uf_fn_V.(r, n)
            div_ufcrossωΩ_real = div_ufcrossωΩ_fn_V.(r, n)
            div_ufcrossωΩ_rtoc = Tcrfwd * div_ufcrossωΩ_real
            div_ufcrossωΩ_operator = div_uf_cross_ωΩ.V[rindsℓ1ℓ2, :] * V
            @test div_ufcrossωΩ_operator ≈ div_ufcrossωΩ_rtoc rtol = 1e-5

            curlωf_dot_uΩ_real = curlωf_dot_uΩ_fn_V.(r, n)
            curlωf_dot_uΩ_rtoc = Tcrfwd * curlωf_dot_uΩ_real
            curlωf_dot_uΩ_operator = curlωf_dot_uΩ.V[rindsℓ1ℓ2, :] * V
            @test curlωf_dot_uΩ_operator ≈ curlωf_dot_uΩ_rtoc rtol = 1e-5

            @test div_uΩcrossωf_fn_V.(r, n) ≈ ωΩ_dot_ωf_fn_V.(r, n) - curlωf_dot_uΩ_fn_V.(r, n)
            div_uΩcrossωf_real = div_uΩcrossωf_fn_V.(r, n)
            div_uΩcrossωf_rtoc = Tcrfwd * div_uΩcrossωf_real
            div_uΩcrossωf_operator = div_uΩ_cross_ωf.V[rindsℓ1ℓ2, :] * V
            @test div_uΩcrossωf_operator ≈ div_uΩcrossωf_rtoc rtol = 1e-5

            @test div_ucrossω_fn_V.(r, n) ≈ div_uΩcrossωf_fn_V.(r, n) + div_ufcrossωΩ_fn_V.(r, n)
            div_ucrossω_real = div_ucrossω_fn_V.(r, n)
            div_ucrossω_rtoc = Tcrfwd * div_ucrossω_real
            div_ucrossω_operator = div_u_cross_ω.V[rindsℓ1ℓ2, :] * V
            @test div_ucrossω_operator ≈ div_ucrossω_rtoc rtol = 1e-5

            dr_r²div_ucrossω_real = dr_r²div_ucrossω_fn_V.(r, n)
            dr_r²div_ucrossω_rtoc = Tcrfwd * dr_r²div_ucrossω_real
            dr_r²div_ucrossω_operator = (kron2(Iℓ, r_ddr_plus_1 * r_cheby)*div_u_cross_ω.V)[rindsℓ1ℓ2, :] * V
            @test dr_r²div_ucrossω_operator ≈ dr_r²div_ucrossω_rtoc rtol = 1e-5

            @test Wterm_fn_V.(r, n) ≈ (@. -dr_r²div_ucrossω_fn_V(r, n) + d²r_r²ucrossωfn_V(r, n) + ∇²h_ucrossωfn_V(r, n)) rtol = 1e-7

            Wterm_real = Wterm_fn_V.(r, n)
            Wterm_rtoc = Tcrfwd * Wterm_real
            Wterm_operator = negr²Ω0W_rhs.V[rindsℓ1ℓ2, :] * V
            @test Wterm_operator ≈ -dr_r²div_ucrossω_operator + d²r_r²ucrossω_operator + ∇²h_ucrossω_operator
            @test Wterm_operator ≈ Wterm_rtoc rtol = 1e-2
        end
    end
end
