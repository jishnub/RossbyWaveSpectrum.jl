using RossbyWaveSpectrum
using Test
using Folds

@testset "multiple ms" begin
    nr, nℓ = 25, 8

    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ, r_in_frac = 0.5, r_out_frac = 0.985);
    constraints = RossbyWaveSpectrum.constraintmatrix(operators);

    mr = 1:15

    As = Folds.map(m -> RossbyWaveSpectrum.uniform_rotation_matrix(m; operators), mr)
    for (ind, m) in enumerate(mr)
        Am = RossbyWaveSpectrum.uniform_rotation_matrix(m; operators)
        @test As[m].re ≈ Am.re
        @test As[m].im ≈ Am.im
    end

    Bs = Folds.map(m -> RossbyWaveSpectrum.mass_matrix(m; operators), mr)
    for (ind, m) in enumerate(mr)
        @test Bs[m] ≈ RossbyWaveSpectrum.mass_matrix(m; operators)
    end

    λs, vs = RossbyWaveSpectrum.filter_eigenvalues(RossbyWaveSpectrum.uniform_rotation_spectrum!,
            mr; operators, constraints);

    @testset "all m" begin
        @testset for ind in eachindex(mr)
            m = mr[ind]
            λu, vu, Mu = RossbyWaveSpectrum.uniform_rotation_spectrum(m; operators, constraints);
            λuf, vuf = RossbyWaveSpectrum.filter_eigenvalues(λu, vu, Mu, m; operators, constraints);

            @test length(λuf) > 0

            @testset "λ" begin @test λuf ≈ λs[ind] end
            # TODO: why does this not converge exactly on some platforms?
            @testset "v" begin @test vuf ≈ vs[ind] rtol=2e-3 end
        end
    end
end
