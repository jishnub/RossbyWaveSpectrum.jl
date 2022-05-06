using RossbyWaveSpectrum
using Test
using Folds

@testset "multiple ms" begin
    nr, nℓ = 30, 15

    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ, r_in_frac = 0.5, r_out_frac = 0.985);
    constraints = RossbyWaveSpectrum.constraintmatrix(operators);

    mr = 1:3

    As = Folds.map(m -> RossbyWaveSpectrum.uniform_rotation_matrix(m; operators), mr)
    for (ind, m) in enumerate(mr)
        @test As[m] ≈ RossbyWaveSpectrum.uniform_rotation_matrix(m; operators)
    end

    Bs = Folds.map(m -> RossbyWaveSpectrum.mass_matrix(m; operators), mr)
    for (ind, m) in enumerate(mr)
        @test Bs[m] ≈ RossbyWaveSpectrum.mass_matrix(m; operators)
    end

    λs, vs = RossbyWaveSpectrum.filter_eigenvalues(RossbyWaveSpectrum.uniform_rotation_spectrum!,
            mr; operators, constraints);

    for (ind, m) in enumerate(mr)
        λu, vu, Mu = RossbyWaveSpectrum.uniform_rotation_spectrum(m; operators, constraints);
        λuf, vuf = RossbyWaveSpectrum.filter_eigenvalues(λu, vu, Mu, m; operators, constraints);

        @test λuf ≈ λs[ind] rtol=1e-5
        @test vuf ≈ vs[ind] rtol=1e-5
    end
end
