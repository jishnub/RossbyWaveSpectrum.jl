using RossbyWaveSpectrum
using Test

@testset "multiple ms" begin
    nr, nℓ = 40, 15

    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ, r_in_frac = 0.5, r_out_frac = 0.985);
    constraints = RossbyWaveSpectrum.constraintmatrix(operators);

    mr = 1:3

    λs, vs = RossbyWaveSpectrum.filter_eigenvalues(RossbyWaveSpectrum.uniform_rotation_spectrum!,
            mr; operators, constraints)

    for (ind, m) in enumerate(mr)
        λu, vu, Mu = RossbyWaveSpectrum.uniform_rotation_spectrum(nr, nℓ, m; operators, constraints);
        λuf, vuf = RossbyWaveSpectrum.filter_eigenvalues(λu, vu, Mu, m; operators, constraints);

        @test λuf ≈ λs[ind]
        @test vuf ≈ vs[ind]
    end
end