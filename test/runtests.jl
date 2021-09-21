using Test
using LinearAlgebra
using RossbyWaveSpectrum
using RossbyWaveSpectrum: ForwardTransform, InverseTransform, IdentityMatrix
using UnPack
using Polynomials: ChebyshevT
using Kronecker

struct OneHotVector{T} <: AbstractVector{T}
    n :: Int
    i :: Int
end
OneHotVector(n, i) = OneHotVector{Int}(n, i)
OneHotVector(i) = OneHotVector{Int}(i, i)
Base.size(v::OneHotVector) = (v.n,)
Base.length(v::OneHotVector) = v.n
Base.getindex(v::OneHotVector{T}, i::Int) where {T} = T(v.i == i)

@testset "chebyshev" begin
    nr = 4; ntheta = 6;
    @unpack nchebytheta, nchebyr, r_in, r_out, Δr = RossbyWaveSpectrum.parameters(nr, ntheta)
    r_mid = (r_out + r_in)/2

    r, Tcrfwd, Tcrinv = RossbyWaveSpectrum.chebyshev_forward_inverse(nr, r_in, r_out)
    costheta, Tcthetafwd, Tcthetainv = RossbyWaveSpectrum.chebyshev_forward_inverse(ntheta)
    costheta2, Tcthetafwd2, Tcthetainv2 = RossbyWaveSpectrum.reverse_theta(costheta, Tcthetafwd, Tcthetainv)
    theta = acos.(costheta)
    sintheta = sin.(theta);

    operators = RossbyWaveSpectrum.basic_operators(nr, ntheta)

    @testset "forward-inverse in r" begin
        @test Tcrfwd * Tcrinv ≈ Tcrinv * Tcrfwd ≈ I
        @test parent(Tcrfwd) * parent(Tcrinv) ≈ parent(Tcrinv) * parent(Tcrfwd) ≈ I
    end
    @testset "forward-inverse in theta" begin
        @test Tcthetafwd * Tcthetainv ≈ Tcthetainv * Tcthetafwd ≈ I
        @test parent(Tcthetafwd) * parent(Tcthetainv) ≈ parent(Tcthetainv) * parent(Tcthetafwd) ≈ I
        @test Tcthetafwd2 * Tcthetainv2 ≈ Tcthetainv2 * Tcthetafwd2 ≈ I
        @test parent(Tcthetafwd2) * parent(Tcthetainv2) ≈ parent(Tcthetainv2) * parent(Tcthetafwd2) ≈ I
    end
    @testset "fullinv & fullinv" begin
        @unpack transforms = operators;
        @unpack fullfwd, fullinv = transforms;
        @test fullfwd * fullinv ≈ I
        @test collect(fullfwd) * collect(fullinv) ≈ I
        @test fullinv * fullfwd ≈ I
        @test collect(fullinv) * collect(fullfwd) ≈ I

        @testset "operators" begin
            @unpack trig_functions = operators;
            @unpack identities = operators;
            @unpack Ir, Itheta = identities;
            @unpack sintheta_mat, sintheta = trig_functions;
            @test sintheta_mat ≈ fullfwd * kronecker(Ir, Diagonal(sintheta)) * fullinv
        end
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

        ∂ = RossbyWaveSpectrum.chebyshevderiv(ntheta); ∂θ = ∂;
        @testset "derivative wrt θ" begin
            f = [0; 1; zeros(ntheta-2)] # Chebyshev coefficients of cosθ
            @test Tcthetainv * f ≈ costheta
            p = ChebyshevT(f)
            for θi in LinRange(0, pi, 20)
                @test p(cos(θi)) ≈ cos(θi) atol=1e-10 rtol=1e-10
            end
            ∂θf = ∂θ * f
            @test ∂θf[1] == 1
            @test all(x -> isapprox(x, 0, atol=1e-10), @view ∂θf[2:end])
        end

        @testset "laplacian" begin
            @testset "theta component" begin
                local n = 8
                M = RossbyWaveSpectrum.laplacianh_theta(n)

                @testset "n=0" begin
                    f = OneHotVector(n, 1)
                    Lf = M * f
                    @test all(iszero(Lf))
                end

                @testset "n=1" begin
                    f = OneHotVector(n, 2)
                    Lf = M * f
                    @test iszero(Lf[1])
                    @test Lf[2] == -2
                    @test all(iszero, @view Lf[3:end])
                end

                @testset "n=2" begin
                    f = OneHotVector(n, 3)
                    Lf = M * f
                    @test Lf[1] == -2
                    @test Lf[2] == 0
                    @test Lf[3] == -6
                    @test all(iszero, @view Lf[4:end])
                end

                @testset "n=3" begin
                    f = OneHotVector(n, 4)
                    Lf = M * f
                    @test Lf[1] == 0
                    @test Lf[2] == -6
                    @test Lf[3] == 0
                    @test Lf[4] == -12
                    @test all(iszero, @view Lf[5:end])
                end

                @testset "n=4" begin
                    f = OneHotVector(n, 5)
                    Lf = M * f
                    @test Lf[1] == -4
                    @test Lf[2] == 0
                    @test Lf[3] == -8
                    @test Lf[4] == 0
                    @test Lf[5] == -20
                    @test all(iszero, @view Lf[6:end])
                end
            end
        end
    end
end
