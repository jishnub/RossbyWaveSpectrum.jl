using Test
using ApproxFunAssociatedLegendre
using LinearAlgebra
using ApproxFunOrthogonalPolynomials
using ApproxFunSingularities

using Aqua
@testset "project quality" begin
    Aqua.test_all(ApproxFunAssociatedLegendre, ambiguities = false)
end

α⁻ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((ℓ - m) * (ℓ + m) / ((2ℓ - 1) * (2ℓ + 1)))))

β⁻ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((2ℓ + 1) / (2ℓ - 1) * (ℓ^2 - m^2))))
γ⁺ℓm(ℓ, m) = ℓ * α⁻ℓm(ℓ+1, m)
γ⁻ℓm(ℓ, m) = ℓ * α⁻ℓm(ℓ, m) - β⁻ℓm(ℓ, m)

@testset "comparison" begin
    sp = NormalizedPlm(2)
    @test sp == sp
    @test maxspace(sp, sp) == sp
    @test union(sp, sp) == sp
    @test conversion_type(sp, sp) == sp

    @testset for S in (Legendre(), NormalizedLegendre())
        @test ApproxFunBase.hasconversion(NormalizedPlm(0), S)
        @test ApproxFunBase.hasconversion(S, NormalizedPlm(0))
        @test ApproxFunBase.hasconversion(JacobiWeight(0,0,S), NormalizedPlm(0))
        @test ApproxFunBase.hasconversion(NormalizedPlm(0), JacobiWeight(0,0,S))
    end

    @test ApproxFunBase.spacescompatible(NormalizedLegendre(), NormalizedPlm(0))
    @test ApproxFunBase.spacescompatible(NormalizedPlm(0), NormalizedLegendre())
    @test !ApproxFunBase.spacescompatible(Legendre(), NormalizedPlm(0))
    @test !ApproxFunBase.spacescompatible(NormalizedPlm(0), Legendre())
end

@testset "multiplication" begin
    function costheta_operator(nℓ, m)
        dl = [α⁻ℓm(ℓ, m) for ℓ in m .+ (1:nℓ-1)]
        d = zeros(nℓ)
        SymTridiagonal(d, dl)'
    end

    cosθ = Fun(Legendre())
    A = AbstractMatrix(Multiplication(cosθ, NormalizedPlm(0))[1:10, 1:10])
    B = AbstractMatrix(Multiplication(cosθ, NormalizedLegendre())[1:10, 1:10])
    C = costheta_operator(10, 0)
    @test A ≈ B ≈ C

    for m in 1:10
        M = Multiplication(cosθ, NormalizedPlm(m))
        A = M[1:10, 1:10]
        B = costheta_operator(10, m)
        @test A ≈ B
        Cop = cosθ_Operator(NormalizedPlm(m))
        C = Cop[1:10, 1:10]
        @test B ≈ C
        fn(x,m) = (√(1-x^2))^m * x
        f = Fun(x -> fn(x,m), NormalizedPlm(m))
        g = M * f
        @test g(0.2) ≈ (x -> x * fn(x, m))(0.2)
    end

    x = Fun(NormalizedPlm(0))
    @testset for sp in (ConstantSpace(), Legendre(), NormalizedLegendre(), NormalizedPlm(0))
        twox = Multiplication(Fun(2,sp)) * x
        @test twox(0.4) ≈ 2x(0.4) ≈ 2*0.4
    end
    @testset for sp in (Legendre(), NormalizedLegendre(), NormalizedPlm(0))
        twox = Multiplication(Fun(x->x^3,sp)) * x
        @test twox(0.4) ≈ (x -> x^4)(0.4)
    end

    g = Multiplication(Fun(x->(1-x^2) * x^2, NormalizedPlm(2))) * x
    @test g(0.4) ≈ (x -> (1-x^2) * x^3)(0.4)

    @testset "abs2" begin
        f = Fun(x->(1-x^2)^2 * x^2, NormalizedPlm(4))
        g = abs2(f)
        @test g(0.4) ≈ (x->(1-x^2)^4 * x^4)(0.4)

        f = Fun(x->(1-x^2)^2 * x^2 * (2 + 3im), NormalizedPlm(4))
        @test abs2(f)(0.4) ≈ (x->abs2((1-x^2)^2 * x^2 * (2 + 3im)))(0.4)
    end
end

@testset "sintheta_dtheta_operator" begin
    function sintheta_dtheta_operator(nℓ, m)
        dl = [γ⁻ℓm(ℓ, m) for ℓ in m .+ (1:nℓ-1)]
        d = zeros(nℓ)
        du = [γ⁺ℓm(ℓ, m) for ℓ in m .+ (0:nℓ-2)]
        Tridiagonal(dl, d, du)'
    end

    cosθ = Fun(Legendre())
    sinθdθ = -(1-cosθ^2)*Derivative(Legendre())
    f = Fun(NormalizedLegendre())
    g = sinθdθ * f
    h = Fun(g, NormalizedPlm(0))
    sinθdθ2 = sinθdθ_Operator(NormalizedPlm(0))
    f2 = Fun(NormalizedPlm(0))
    g2 = sinθdθ2 * f2
    @test g2 ≈ h

    for m in 0:10
        A = AbstractMatrix(sinθdθ_Operator(NormalizedPlm(m))[1:10, 1:10])
        B = sintheta_dtheta_operator(10, m)
        @test A ≈ B
    end
end

@testset "Conversion" begin
    nsp = NormalizedPlm(1)
    jsp = nsp.jacobispace
    f = Conversion(jsp, nsp) * Fun(x->√(1-x^2), jsp)
    g = Fun(x->√(1-x^2), nsp)
    @test f ≈ g
    f = Conversion(nsp, jsp) * Fun(x->√(1-x^2), nsp)
    g = Fun(x->√(1-x^2), jsp)
end

@testset "Fun" begin
    sp = NormalizedPlm(3)
    f = @inferred Fun(x->√(1-x^2)^3 * x^2, sp.jacobispace)
    g = @inferred Fun(f, sp)
    @test f ≈ g

    @test Fun(g, NormalizedPlm(1)) ≈ Fun(x->√(1-x^2) * (1-x^2) * x^2, NormalizedPlm(1))

    sp = NormalizedPlm(3)
    f = @inferred Fun(x->√(1-x^2)^3 * x^2, sp)
    g = @inferred Fun(f, sp.jacobispace)
    @test f ≈ g

    sp = NormalizedPlm(0)
    f = Fun(sp.jacobispace)
    g = @inferred Fun(sp)
    @test f ≈ g
end

@testset "Definite Integral" begin
    sp = NormalizedPlm(3)
    f = Fun(x->√(1-x^2)^3 * x^2, sp)
    @test sum(f) ≈ pi/16
    @test DefiniteIntegral() * f ≈ pi/16
end

@testset "Tensor space" begin
    z = (x,y) -> x*(1-y^2)
    f = Fun(z, Chebyshev() ⊗ NormalizedPlm(2))
    @test f(0.3, 0.4) ≈ z(0.3, 0.4)
    @test abs2(f)(0.3, 0.4) ≈ ((x,y) -> abs2(z(x,y)))(0.3, 0.4)
end
