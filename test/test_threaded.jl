using RossbyWaveSpectrum
using Test
using Folds

@testset "multiple ms" begin
    nr, nℓ = 25, 8

    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ, r_in_frac = 0.5, r_out_frac = 0.985);
    constraints = RossbyWaveSpectrum.constraintmatrix(operators);

    mr = 2:7

    As = Folds.map(m -> RossbyWaveSpectrum.uniform_rotation_matrix(m; operators, V_symmetric = true), mr)
    for (ind, m) in enumerate(mr)
        Am = RossbyWaveSpectrum.uniform_rotation_matrix(m; operators, V_symmetric = true)
        @test As[ind].re ≈ Am.re
        @test As[ind].im ≈ Am.im
    end

    Bs = Folds.map(m -> RossbyWaveSpectrum.mass_matrix(m; operators, V_symmetric = true), mr)
    for (ind, m) in enumerate(mr)
        @test Bs[ind] ≈ RossbyWaveSpectrum.mass_matrix(m; operators, V_symmetric = true)
    end

    λs, vs = RossbyWaveSpectrum.filter_eigenvalues(RossbyWaveSpectrum.uniform_rotation_spectrum!,
            mr; operators, constraints, print_timer = false, V_symmetric = true);

    @testset "all m" begin
        @testset for ind in eachindex(mr)
            m = mr[ind]
            λu, vu, Mu = RossbyWaveSpectrum.uniform_rotation_spectrum(m;
                operators, constraints, V_symmetric = true);
            λuf, vuf = RossbyWaveSpectrum.filter_eigenvalues(λu, vu, Mu, m;
                operators, constraints, V_symmetric = true);

            @test length(λuf) > 0

            @testset "λ" begin @test λuf ≈ λs[ind] end
            @testset "v" begin @test vuf ≈ vs[ind] rtol=1e-7 end
        end
    end
end
