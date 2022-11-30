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
        A = AbstractMatrix(M)[1:10, 1:10])
        B = costheta_operator(10, m)
        @test A ≈ B
        f = Fun(x->(√(1-x^2))^m * x, NormalizedPlm(m))
        g = M * f
        @test g(0.2) ≈ (x -> √(1-x^2))^m * x)(0.2)
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
    sinθdθ2 = sinθ∂θ_Operator(NormalizedPlm(0))
    f2 = Fun(NormalizedPlm(0))
    g2 = sinθdθ2 * f2
    @test g2 ≈ h

    for m in 0:10
        A = AbstractMatrix(sinθ∂θ_Operator(NormalizedPlm(m))[1:10, 1:10])
        B = sintheta_dtheta_operator(10, m)
        @test A ≈ B
    end
end
