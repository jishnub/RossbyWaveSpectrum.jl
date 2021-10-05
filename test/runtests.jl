using Test
using LinearAlgebra
using RossbyWaveSpectrum
using RossbyWaveSpectrum: ForwardTransform, InverseTransform, IdentityMatrix, OneHotVector
using UnPack
using Polynomials: ChebyshevT
using Kronecker
using QuadGK
using SphericalHarmonics
using ForwardDiff

nr = 4; nℓ = 6;
operators = RossbyWaveSpectrum.basic_operators(nr, nℓ);
(; coordinates) = operators;
(; r) = coordinates;
(; params) = operators;
(; nchebyr, r_in, r_out, Δr) = params;
nparams = nchebyr * nℓ;
r_mid = (r_out + r_in)/2;

@testset "chebyshev" begin
    r, Tcrfwd, Tcrinv = RossbyWaveSpectrum.chebyshev_forward_inverse(nr, r_in, r_out)
    @testset "forward-inverse in r" begin
        @test Tcrfwd * Tcrinv ≈ Tcrinv * Tcrfwd ≈ I
        @test parent(Tcrfwd) * parent(Tcrinv) ≈ parent(Tcrinv) * parent(Tcrfwd) ≈ I
    end
    @testset "derivatives" begin
        ∂ = RossbyWaveSpectrum.chebyshevderiv(nr);
        @testset "T0" begin
            fn = setindex!(zeros(nr), 1, 1)
            ∂fn_expected = zeros(nr)
            @test ∂ * fn ≈ ∂fn_expected
        end
        @testset "T1" begin
            fn = setindex!(zeros(nr), 1, 2)
            ∂fn_expected = setindex!(zeros(nr), 1, 1)
            @test ∂ * fn ≈ ∂fn_expected
        end
        @testset "T2" begin
            fn = setindex!(zeros(nr), 1, 3)
            ∂fn_expected = setindex!(zeros(nr), 4, 2)
            @test ∂ * fn ≈ ∂fn_expected
        end
        @testset "derivative wrt r" begin
            ∂r = ∂ * (2/Δr)
            Dr = Tcrinv * ∂r * Tcrfwd
            r_nodes, _ = RossbyWaveSpectrum.chebyshevnodes(nr, r_in, r_out)
            @testset "f(r) = r" begin
                f = [vcat(r_mid, Δr/2); zeros(nr - 2)] # chebyshev decomposition of r in [r_in, r_out]
                @test Tcrinv * f ≈ r
                p = ChebyshevT(f)
                for (r_node_i, ri) in zip(r_nodes, r)
                    @test p(r_node_i) ≈ ri
                end
                ∂rf = ∂r * f # chebyshev coefficients of the derivative, in this case d/dr(r) = 1 = T0(x)
                @test ∂rf[1] ≈ 1
                @test all(x -> isapprox(x, 0, atol=1e-10), @view ∂rf[2:end])
                @testset "real space" begin
                    x = Dr * r
                    @test all(isapprox(1), x)
                end
            end
            @testset "f(r) = r^2" begin
                f = [r_mid^2 + 1/2*(Δr/2)^2; r_mid*Δr; 1/2*(Δr/2)^2; zeros(nr - 3)]
                @test Tcrinv * f ≈ r.^2
                p = ChebyshevT(f)
                for (r_node_i, ri) in zip(r_nodes, r)
                    @test p(r_node_i) ≈ ri^2
                end
                ∂rf = ∂r * f # chebyshev coefficients of the derivative, in this case d/dr(r^2) = 2r = 2r_mid*T0(x) + Δr*T1(x)
                @test ∂rf[1] ≈ 2r_mid
                @test ∂rf[2] ≈ Δr
                @test all(x -> isapprox(x, 0, atol=1e-10), @view ∂rf[3:end])
                @testset "real space" begin
                    x = Dr * r.^2
                    @test all(x .≈ 2 .* r)
                end
            end
        end
    end
end

@testset "boundary conditions constraint" begin

    M = RossbyWaveSpectrum.constraintmatrix(operators, nℓ)

    Vrones = kron(rand(nℓ), ones(nr));
    v_Vrones = [Vrones; rand(nparams)];
    @test length(v_Vrones) == size(M,2)

    Vrflip = kron(rand(nℓ), [(-1)^n for n = 0:nr-1]);
    v_Vrflip = [Vrflip; rand(nparams)];
    @test length(v_Vrflip) == size(M,2)

    # ∑_n Wkn = 0
    Wrflip = kron(rand(nℓ), [(-1)^n for n = 0:nr-1]);
    v_Wrflip = [rand(nparams); Wrflip];
    @test length(v_Wrflip) == size(M,2)
    @test (M*v_Wrflip)[3] ≈ 0 atol=1e-14

    # ∑_n (-1)^n * Wkn = 0
    Wrones = kron(rand(nℓ), ones(nr));
    v_Wrones = [rand(nparams); Wrones];
    @test length(v_Wrones) == size(M,2)
    @test (M*v_Wrones)[4] ≈ 0 atol=1e-14
end

@testset "invert B" begin

end

@testset "trig matrices in Legendre basis" begin
    m = 2;
    M1 = RossbyWaveSpectrum.cosθ_operator(nℓ, m)
    M2 = RossbyWaveSpectrum.sinθdθ_operator(nℓ, m)
    M12 = M1^2
    M22 = M2^2
    M1M2 = M1*M2
    Plm(θ, l, m) = SphericalHarmonics.associatedLegendre(θ, l, m, norm = SphericalHarmonics.Orthonormal())
    sinθdθPlm(θ, l, m) = sin(θ) * ForwardDiff.derivative(θ -> Plm(θ, l, m), θ)
    sinθdθsinθdθPlm(θ, l, m) = sin(θ) * ForwardDiff.derivative(θ-> sinθdθPlm(θ, l, m), θ)
    rtol = 5e-2
    @testset "cosθ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*cos(θ)*Plm(θ, l′, m), 0, pi, rtol=rtol)
            @test M1[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
        end
    end
    @testset "cos²θ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*cos(θ)^2*Plm(θ, l′, m), 0, pi, rtol=rtol)
            @test M12[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
        end
    end
    @testset "sinθdθ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*sinθdθPlm(θ, l′, m), 0, pi, rtol=rtol)
            @test M2[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
        end
    end
    @testset "sinθdθ*sinθdθ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*sinθdθsinθdθPlm(θ, l′, m), 0, pi, rtol=rtol)
            @test M22[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
        end
    end
end

@testset "eigenvalues" begin
    C = RossbyWaveSpectrum.constraintmatrix(operators, nℓ);

    m = 5
    M = RossbyWaveSpectrum.twoΩcrossv(nr, nℓ, m, operators);

    M_constrained = [M C'
                     C zeros(size(C, 1), size(M,2) + size(C, 1) - size(C,2))];

    λ, v = RossbyWaveSpectrum.filter_eigenvalues(RossbyWaveSpectrum.uniform_rotation_spectrum, nr, nℓ, m);
    @test M * v ≈ v * Diagonal(λ) rtol=1e-2
    @test all(x -> isapprox(x, 0, atol=1e-5), C * v)
end
