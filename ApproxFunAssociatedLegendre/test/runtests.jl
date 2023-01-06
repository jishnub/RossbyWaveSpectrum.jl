using Test
using ApproxFunAssociatedLegendre
using LinearAlgebra
using ApproxFunOrthogonalPolynomials

using Aqua
@testset "project quality" begin
    Aqua.test_all(ApproxFunAssociatedLegendre, ambiguities = false)
end

α⁻ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((ℓ - m) * (ℓ + m) / ((2ℓ - 1) * (2ℓ + 1)))))

β⁻ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((2ℓ + 1) / (2ℓ - 1) * (ℓ^2 - m^2))))
γ⁺ℓm(ℓ, m) = ℓ * α⁻ℓm(ℓ+1, m)
γ⁻ℓm(ℓ, m) = ℓ * α⁻ℓm(ℓ, m) - β⁻ℓm(ℓ, m)

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
end
