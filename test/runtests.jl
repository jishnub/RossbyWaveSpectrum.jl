using RossbyWaveSpectrum: RossbyWaveSpectrum, matrix_block, Rsun, operatormatrix, Msun, G, StructMatrix
using Test
using LinearAlgebra
using OffsetArrays
using Aqua
using FastTransforms
using Dierckx
using ApproxFun
using PerformanceTestTools
using UnPack
using StructArrays
using BlockArrays
using ApproxFunAssociatedLegendre

using RossbyWaveSpectrum.Filters: NODES, SPATIAL, EIGVEC, EIGVAL, EIGEN, BC

@testset "project quality" begin
    Aqua.test_all(RossbyWaveSpectrum, ambiguities = false)
end

@testset "operators" begin
    @inferred RossbyWaveSpectrum.radial_operators(5, 2)
    # Ensure that the Chebyshev interpolations of the scale heights match the actual
    # functional form, given by the spline interpolations
    @testset "stratification" begin
        nr, nℓ = 50, 25
        @testset "shallow" begin
            operators = RossbyWaveSpectrum.radial_operators(nr, nℓ, r_in_frac = 0.7);
            @unpack rpts = operators;
            @unpack sρ, sg, sηρ, ddrsηρ, d2dr2sηρ, d3dr3sηρ, sT, sηT = operators.splines;
            @unpack ηρ, ddr_ηρ, d2dr2_ηρ, d3dr3_ηρ = operators.rad_terms;

            @test sηρ.(rpts) ≈ ηρ.(rpts) rtol=1e-2
            @test ddrsηρ.(rpts) ≈ ddr_ηρ.(rpts) rtol=1e-2
            @test d2dr2sηρ.(rpts) ≈ d2dr2_ηρ.(rpts) rtol=5e-2
            @test d3dr3sηρ.(rpts) ≈ d3dr3_ηρ.(rpts) rtol=1e-1
        end

        @testset "deep" begin
            operators = RossbyWaveSpectrum.radial_operators(nr, nℓ, r_in_frac = 0.5);
            @unpack rpts = operators;
            @unpack sρ, sg, sηρ, ddrsηρ, d2dr2sηρ, d3dr3sηρ, sT, sηT = operators.splines;
            @unpack ηρ, ddr_ηρ, d2dr2_ηρ, d3dr3_ηρ = operators.rad_terms;

            @test sηρ.(rpts) ≈ ηρ.(rpts) rtol=1e-2
            @test ddrsηρ.(rpts) ≈ ddr_ηρ.(rpts) rtol=1e-2
            @test d2dr2sηρ.(rpts) ≈ d2dr2_ηρ.(rpts) rtol=1e-2
            @test d3dr3sηρ.(rpts) ≈ d3dr3_ηρ.(rpts) rtol=2e-2
        end
    end
end

@testset "constraints" begin
    nr, nℓ = 60, 30

    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ)
    constraints = RossbyWaveSpectrum.constraintmatrix(operators)
    @unpack nullspacematrices = constraints
    ZV, ZW, ZS = nullspacematrices
    @unpack r_in, r_out = operators.radial_params

    @test maximum(abs, constraints.BC * constraints.ZC) < 1e-10

    @testset "boundary conditions basis" begin
        @testset "W" begin
            for c in eachcol(ZW)
                p = Fun(operators.radialspace, c)
                @test p(r_in) ≈ 0 atol=1e-14
                @test p(r_out) ≈ 0 atol=1e-14
                @test p'(r_in) ≈ 0 atol=1e-12
                @test p'(r_out) ≈ 0 atol=1e-12
            end
        end
        @testset "S" begin
            for c in eachcol(ZS)
                p = Fun(operators.radialspace, c)
                @test p'(r_in) ≈ 0 atol=1e-12
                @test p'(r_out) ≈ 0 atol=1e-12
            end
        end
        @testset "V" begin
            CV = RossbyWaveSpectrum.V_boundary_op(operators);
            for c in eachcol(ZV)
                p = Fun(operators.radialspace, c)
                Cp = CV * p
                @test Cp(r_in) ≈ 0 atol=1e-11
                @test Cp(r_out) ≈ 0 atol=1e-11
            end
        end
    end
end

@testset "read solar model" begin
    r_in_frac = 0.5
    r_out_frac = 0.985
    r_in = r_in_frac * Rsun
    r_out = r_out_frac * Rsun
    @unpack sρ, sT, sg, sηρ, sηT = RossbyWaveSpectrum.solar_structure_parameter_splines(; r_in, r_out).splines;

    operators = RossbyWaveSpectrum.radial_operators(50, 2; r_in_frac, r_out_frac);
    @unpack rpts = operators;

    ModelS = RossbyWaveSpectrum.read_solar_model()
    r_modelS = @view ModelS[:, 1];
    r_inds = r_in .<= r_modelS .<= r_out;
    r_modelS = reverse(r_modelS[r_inds]);
    q_modelS = exp.(reverse(ModelS[r_inds, 2]));
    T_modelS = reverse(ModelS[r_inds, 3]);
    ρ_modelS = reverse(ModelS[r_inds, 5]);

    g_modelS = @. G * Msun * q_modelS / r_modelS^2;

    @test sρ.(rpts) ≈ RossbyWaveSpectrum.interp1d(r_modelS, ρ_modelS, rpts) rtol=1e-2
    @test sT.(rpts) ≈ RossbyWaveSpectrum.interp1d(r_modelS, T_modelS, rpts) rtol=1e-2
    @test sg.(rpts) ≈ RossbyWaveSpectrum.interp1d(r_modelS, g_modelS, rpts) rtol=1e-2
end

@testset "filters" begin
    @test !BC == NODES | SPATIAL | EIGVEC | EIGVAL | EIGEN
    @test !EIGVAL == NODES | SPATIAL | BC | EIGVEC | EIGEN
    @test !EIGVEC == NODES | SPATIAL | BC | EIGVAL | EIGEN
    @test !EIGEN == NODES | SPATIAL | BC | EIGVEC | EIGVAL
    @test !SPATIAL == NODES | BC | EIGVEC | EIGVAL | EIGEN
    @test !NODES == SPATIAL | BC | EIGVEC | EIGVAL | EIGEN
    @test !(EIGVAL | EIGVEC) == NODES | SPATIAL | BC | EIGEN
end

@testset "invtransform" begin
    for sp in Any[Chebyshev(), Legendre()]
        n = 100
        for v in Any[rand(n, 20), rand(ComplexF64, n, 20)]
            v2 = copy(v)
            out = similar(v)
            RossbyWaveSpectrum.invtransform1!(sp, out, v)
            @test v == v2
            pts = points(sp, n)
            for (p,c) in zip(eachcol(out), eachcol(v))
                f = Fun(sp, c)
                @test f.(pts) ≈ p
            end
        end
    end
end

function matrix_subsample!(M_subsample::BlockArray{<:Real}, M::BlockArray{<:Real}, nr_M, nr, nℓ, nvariables)
    nparams = nr*nℓ
    @views for colind in 1:nvariables, rowind in 1:nvariables
        Mblock = matrix_block(M, rowind, colind)
        M_subsample_block = matrix_block(M_subsample, rowind, colind)
        for ℓ′ind in 1:nℓ, ℓind in 1:nℓ
            Mblock_ℓℓ′ = Mblock[Block(ℓind, ℓ′ind)]
            M_subsample_block_ℓℓ′ = M_subsample_block[Block(ℓind, ℓ′ind)]
            M_subsample_block_ℓℓ′ .= Mblock_ℓℓ′[axes(M_subsample_block_ℓℓ′)...]
        end
    end
    return M_subsample
end

function matrix_subsample!(M_subsample::StructArray{<:Complex,2}, M::StructArray{<:Complex,2}, args...)
    matrix_subsample!(M_subsample.re, M.re, args...)
    matrix_subsample!(M_subsample.im, M.im, args...)
    return M_subsample
end

blockwise_cmp(f, x, y) = f(x,y)
function blockwise_cmp(f, S1::StructMatrix{<:Complex}, S2::StructMatrix{<:Complex})
    blockwise_cmp(f, S1.re, S2.re) && blockwise_cmp(f, S1.im, S2.im)
end
function blockwise_cmp(f, B1::BlockMatrix{<:Real}, B2::BlockMatrix{<:Real})
    all(zip(blocks(B1), blocks(B2))) do (x, y)
        blockwise_cmp(f, x, y)
    end
end

blockwise_isequal(x, y) = blockwise_cmp(==, x, y)
blockwise_isapprox(x, y; kw...) = blockwise_cmp((x, y) -> isapprox(x, y; kw...), x, y)

@testset "matrix convergence with resolution: uniform rotation" begin
    nr, nℓ = 50, 2
    m = 5
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
    @unpack nvariables = operators;
    M1 = RossbyWaveSpectrum.uniform_rotation_matrix(m; operators);
    operators2 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ);
    M2 = RossbyWaveSpectrum.uniform_rotation_matrix(m; operators = operators2);
    operators3 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ+5);
    M3 = RossbyWaveSpectrum.uniform_rotation_matrix(m; operators = operators3);

    M_subsample = RossbyWaveSpectrum.allocate_operator_matrix(operators);

    @testset "same size" begin
        matrix_subsample!(M_subsample, M1, nr, nr, nℓ, nvariables)
        @test blockwise_isequal(M_subsample, M1);
    end

    matrix_subsample!(M_subsample, M2, nr+5, nr, nℓ, nvariables);
    @testset for rowind in 1:nvariables, colind in 1:nvariables
        @testset "real" begin
            M2_ssv = matrix_block(M_subsample.re, rowind, colind);
            M1v = matrix_block(M1.re, rowind, colind);

            rtol = rowind == 3 && colind == 2 ? 5e-3 : 2e-3

            @test blockwise_isapprox(M2_ssv, M1v; rtol)
        end

        @testset "imag" begin
            M2_ssv = matrix_block(M_subsample.im, rowind, colind);
            M1v = matrix_block(M1.im, rowind, colind);

            rtol = rowind == colind == 2 ? 2e-3 : 1e-4

            @test blockwise_isapprox(M2_ssv, M1v; rtol)
        end
    end

    matrix_subsample!(M_subsample, M3, nr+5, nr, nℓ, nvariables);
    @testset for rowind in 1:nvariables, colind in 1:nvariables
        @testset "real" begin
            M3_ssv = matrix_block(M_subsample.re, rowind, colind, nvariables)
            M1v = matrix_block(M1.re, rowind, colind, nvariables)

            rtol = rowind == 3 && colind == 2 ? 5e-3 : 2e-3

            @test blockwise_isapprox(M3_ssv, M1v; rtol)
        end
        @testset "imag" begin
            M3_ssv = matrix_block(M_subsample.im, rowind, colind, nvariables)
            M1v = matrix_block(M1.im, rowind, colind, nvariables)

            rtol = rowind == colind == 2 ? 2e-3 : 1e-4

            @test blockwise_isapprox(M3_ssv, M1v; rtol)
        end
    end
end

function rossby_ridge_eignorm(λ, v, (A, B), m, nparams; ΔΩ_frac = 0)
    matchind = argmin(abs.(real(λ) .- RossbyWaveSpectrum.rossby_ridge(m; ΔΩ_frac)))
    vi = v[:, matchind];
    λi = λ[matchind]
    normsden = [norm(λi * @view(vi[i*nparams .+ (1:nparams)])) for i in 0:2]
    normsnum = [norm((A[i*nparams .+ (1:nparams), :] - λi * B[i*nparams .+ (1:nparams), :]) * vi) for i in 0:2]
    [(d/norm(λi) > 1e-10 ? n/d : 0) for (n, d) in zip(normsnum, normsden)]
end

@testset "uniform rotation solution" begin
    nr, nℓ = 40, 8
    nparams = nr * nℓ
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
    constraints = RossbyWaveSpectrum.constraintmatrix(operators);
    @unpack r_in, r_out, Δr = operators.radial_params
    @unpack BCmatrices = constraints;
    MVn, MWn, MSn = BCmatrices;
    @testset for m in [1, 5, 10]
        λu, vu, Mu = RossbyWaveSpectrum.uniform_rotation_spectrum(m; operators, constraints);
        λuf, vuf = RossbyWaveSpectrum.filter_eigenvalues(λu, vu, Mu, m;
            operators, constraints, Δl_cutoff = 7, n_cutoff = 9,
            eig_imag_damped_cutoff = 1e-3, eig_imag_unstable_cutoff = -1e-3,
            scale_eigenvectors = false);
        @info "uniform rot: $(length(λuf)) eigenmode$(length(λuf) > 1 ? "s" : "") found for m = $m"
        @testset "ℓ == m" begin
            res, ind = findmin(abs.(real(λuf) .- 2/(m+1)))
            @testset "eigenvalue" begin
                @test res < 1e-4
            end
        end
        vfn = zeros(eltype(vuf), size(vuf, 1));
        @testset "boundary condition" begin
            Vop = RossbyWaveSpectrum.V_boundary_op(operators);
            @testset for n in axes(vuf, 2)
                vfn .= @view vuf[:, n];
                @testset "V" begin
                    @testset for ℓind in 1:nℓ
                        ℓ_skip = (ℓind - 1)*nr
                        inds_ℓ = ℓ_skip .+ (1:nr)
                        v = @view vfn[inds_ℓ]
                        @test MVn * v ≈ zeros(size(MVn,1)) atol=1e-10
                        pv = Fun(operators.radialspace, v)
                        Cp = Vop * pv;
                        @testset "inner boundary" begin
                            @test Cp(r_in) ≈ 0 atol=1e-10
                        end
                        @testset "outer boundary" begin
                            @test Cp(r_out) ≈ 0 atol=1e-10
                        end
                    end
                end
                @testset "W" begin
                    @testset for ℓind in 1:nℓ
                        ℓ_skip = (ℓind - 1)*nr
                        inds_ℓ = nr*nℓ + ℓ_skip .+ (1:nr)
                        w = @view vfn[inds_ℓ]
                        @test MWn * w ≈ zeros(size(MWn,1)) atol=1e-10
                        pw = Fun(operators.radialspace, w)
                        drpw = pw'
                        @testset "inner boundary" begin
                            @test pw(r_in) ≈ 0 atol=1e-10
                            @test drpw(r_in) ≈ 0 atol=1e-10
                        end
                        @testset "outer boundary" begin
                            @test pw(r_out) ≈ 0 atol=1e-10
                            @test drpw(r_out) ≈ 0 atol=1e-10
                        end
                    end
                end
                @testset "S" begin
                    @testset for ℓind in 1:nℓ
                        ℓ_skip = (ℓind - 1)*nr
                        inds_ℓ = 2nr*nℓ + ℓ_skip .+ (1:nr)
                        S = @view vfn[inds_ℓ]
                        @test MSn * S ≈ zeros(size(MSn,1)) atol=1e-10
                        pS = Fun(operators.radialspace, S)
                        drpS = pS'
                        @testset "inner boundary" begin
                            @test drpS(r_in) ≈ 0 atol=1e-10
                        end
                        @testset "outer boundary" begin
                            @test drpS(r_out) ≈ 0 atol=1e-10
                        end
                    end
                end
            end
        end
        @testset "eigen rtol" begin
            v = rossby_ridge_eignorm(λuf, vuf, Mu, m, nparams)
            # these improve with resolution
            @test v[1] < 1e-2
            @test v[2] < 0.5
            @test v[3] < 0.8
        end
    end
end

@testset "constant differential rotation and uniform rotation" begin
    nr, nℓ = 20, 8
    m = 3
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ)
    Mu = RossbyWaveSpectrum.uniform_rotation_matrix(m; operators);
    Mc = RossbyWaveSpectrum.differential_rotation_matrix(m;
            rotation_profile = :constant, operators, ΔΩ_frac = 0);
    @test blockwise_isapprox(Mu, Mc)
end

@testset "radial differential rotation" begin
    @testset "compare with constant" begin
        nr, nℓ = 30, 2
        operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
        @unpack nvariables = operators;

        Mc = RossbyWaveSpectrum.allocate_operator_matrix(operators);
        Mr = RossbyWaveSpectrum.allocate_operator_matrix(operators);

        @testset for m in [1, 4, 10]
            @testset "radial constant and constant" begin
                RossbyWaveSpectrum.differential_rotation_matrix!(Mc, m;
                    rotation_profile = :constant, operators);
                RossbyWaveSpectrum.differential_rotation_matrix!(Mr, m;
                    rotation_profile = :radial_constant, operators);

                @testset for colind in 1:nvariables, rowind in 1:nvariables
                    Rc = matrix_block(Mr, rowind, colind, nvariables)
                    C = matrix_block(Mc, rowind, colind, nvariables)
                    @testset "real" begin
                        @test blockwise_isapprox(Rc.re, C.re, atol = 1e-10, rtol = 1e-3)
                    end
                    @testset "imag" begin
                        @test blockwise_isapprox(Rc.im, C.im, atol = 1e-10, rtol = 1e-3)
                    end
                end
            end
        end
    end

    @testset "radial differential rotation" begin
        nr, nℓ = 50, 10
        nparams = nr * nℓ
        m = 1
        operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
        @unpack diff_operators, rad_terms, rpts, radial_params = operators;
        @unpack nvariables = operators;
        @unpack ν = operators.constants;
        r_mid = radial_params.r_mid::Float64;
        Δr = radial_params.Δr::Float64;
        a = 1 / (Δr / 2);
        b = -r_mid / (Δr / 2);
        @unpack ddr, d2dr2, DDr = diff_operators;
        @unpack r2, r, ηρ = rad_terms;
        @unpack r_in, r_out = operators.radial_params;

        ℓ = 2
        ℓℓp1 = ℓ*(ℓ+1)
        ℓ′ = 1
        cosθ21 = 1/√5
        sinθdθ21 = 1/√5
        ∇²_sinθdθ21 = -ℓℓp1 * sinθdθ21

        @testset "convergence of diff rot profile" begin
            local nr, nℓ = 50, 10
            local operators1 = RossbyWaveSpectrum.radial_operators(nr, nℓ);
            local r1 = operators1.rpts;
            local operators2 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ);
            local operators3 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ+5);

            T = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives_Fun(
                operators = operators1, rotation_profile = :solar_equator);
            ΔΩ1, ddrΔΩ1, d2dr2ΔΩ1 = T.ΔΩ, T.ddrΔΩ, T.d2dr2ΔΩ;

            T = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives_Fun(
                operators = operators2, rotation_profile = :solar_equator);
            ΔΩ2, ddrΔΩ2, d2dr2ΔΩ2 = T.ΔΩ, T.ddrΔΩ, T.d2dr2ΔΩ;

            T = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives_Fun(
                operators = operators3, rotation_profile = :solar_equator);
            ΔΩ3, ddrΔΩ3, d2dr2ΔΩ3 = T.ΔΩ, T.ddrΔΩ, T.d2dr2ΔΩ;

            @testset "nr+5, nℓ" begin
                # make sure that nothing nonsensical is happening, and the points are scaled
                @test !isapprox(ΔΩ1.(r1), zeros(nr), atol=1e-12)
                @test ΔΩ1.(r1) ≈ ΔΩ2.(r1) rtol=1e-2
                @test ddrΔΩ1.(r1) ≈ ddrΔΩ2.(r1) rtol=8e-2
                @test d2dr2ΔΩ1.(r1) ≈ d2dr2ΔΩ2.(r1) rtol=2e-1
            end

            @testset "nr+5, nℓ+5" begin
                @test ΔΩ1.(r1) ≈ ΔΩ3.(r1) rtol=1e-2
                @test ddrΔΩ1.(r1) ≈ ddrΔΩ3.(r1) rtol=8e-2
                @test d2dr2ΔΩ1.(r1) ≈ d2dr2ΔΩ3.(r1) rtol=2e-1
            end
        end

        @testset "matrix convergence with resolution" begin
            nr, nℓ = 50, 10
            m = 5
            operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
            @unpack nvariables = operators;
            operators2 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ);
            operators3 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ+5);
            M_subsample = RossbyWaveSpectrum.allocate_operator_matrix(operators);
            M1 = RossbyWaveSpectrum.allocate_operator_matrix(operators);
            M2 = RossbyWaveSpectrum.allocate_operator_matrix(operators2);
            M3 = RossbyWaveSpectrum.allocate_operator_matrix(operators3);

            @testset for rotation_profile in [:radial_linear, :radial_solar_equator]
                RossbyWaveSpectrum.differential_rotation_matrix!(M1, m; operators, rotation_profile);
                RossbyWaveSpectrum.differential_rotation_matrix!(M2, m; operators = operators2, rotation_profile);
                RossbyWaveSpectrum.differential_rotation_matrix!(M3, m; operators = operators3, rotation_profile);

                @testset "same size" begin
                    matrix_subsample!(M_subsample, M1, nr, nr, nℓ, nvariables)
                    @test blockwise_isequal(M_subsample, M1);
                end

                matrix_subsample!(M_subsample, M2, nr+5, nr, nℓ, nvariables);
                @testset for rowind in 1:nvariables, colind in 1:nvariables
                    @testset "real" begin
                        M2_ssv = matrix_block(M_subsample.re, rowind, colind);
                        M1v = matrix_block(M1.re, rowind, colind);

                        rtol = rowind == colind == 2 ? 8e-2 : 1e-2

                        @test blockwise_isapprox(M2_ssv, M1v; rtol)
                    end

                    @testset "imag" begin
                        M2_ssv = matrix_block(M_subsample.im, rowind, colind);
                        M1v = matrix_block(M1.im, rowind, colind);

                        @test blockwise_isapprox(M2_ssv, M1v; rtol = 1e-2)
                    end
                end

                matrix_subsample!(M_subsample, M3, nr+5, nr, nℓ, nvariables);
                @testset for rowind in 1:nvariables, colind in 1:nvariables
                    @testset "real" begin
                        M3_ssv = matrix_block(M_subsample.re, rowind, colind)
                        M1v = matrix_block(M1.re, rowind, colind)

                        rtol = rowind == colind == 2 ? 8e-2 : 1e-2

                        @test blockwise_isapprox(M3_ssv, M1v; rtol)
                    end
                    @testset "imag" begin
                        M3_ssv = matrix_block(M_subsample.im, rowind, colind)
                        M1v = matrix_block(M1.im, rowind, colind)

                        @test blockwise_isapprox(M3_ssv, M1v; rtol = 1e-2)
                    end
                end
            end
        end
    end
end

@testset "solar differential rotation" begin
    nr, nℓ = 30, 10
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
    @unpack nvariables = operators;
    @unpack Wscaling, Weqglobalscaling, Seqglobalscaling = operators.scalings;
    @unpack ddr, DDr, ddrDDr = operators.diff_operators;
    @unpack onebyr, ηρ, onebyr2 = operators.rad_terms;
    @unpack radialspace = operators;

    Mc = RossbyWaveSpectrum.allocate_operator_matrix(operators);
    Mr = RossbyWaveSpectrum.allocate_operator_matrix(operators);
    Ms = RossbyWaveSpectrum.allocate_operator_matrix(operators);

    cosθ = Fun(Legendre());

    @testset "compare with constant" begin
        ΔΩ_frac = 0.01
        ΔΩprofile_deriv = RossbyWaveSpectrum.solar_differential_rotation_profile_derivatives_Fun(;
            operators, rotation_profile = :constant, smoothing_param=1e-3, ΔΩ_frac);
        (; ΔΩ, ∂r_ΔΩ, ∂θ_ΔΩ, ∂2r_ΔΩ, ∂r∂θ_ΔΩ, ∂2θ_ΔΩ) = ΔΩprofile_deriv;

        ωΩ_deriv = RossbyWaveSpectrum.solar_differential_rotation_vorticity_Fun(;
            operators, ΔΩprofile_deriv);
        (; ωΩr, ∂rωΩr, ∂θωΩr_by_sinθ, ωΩθ_by_rsinθ) = ωΩ_deriv;

        @testset "compare with analytical" begin
            cosθop = Multiplication(cosθ)
            @test ωΩr ≈ (I ⊗ 2cosθop) * ΔΩ
            @test ∂rωΩr ≈ zero(∂rωΩr) atol=1e-14
            @test ∂θωΩr_by_sinθ ≈ -2ΔΩ
            @test ωΩθ_by_rsinθ ≈ (Multiplication(-2*onebyr) ⊗ I) * ΔΩ

            @testset for m in [1, 4, 10], V_symmetric in [true, false]
                Ms.re .= 0;
                RossbyWaveSpectrum.solar_differential_rotation_terms!(Ms, m;
                    operators, ΔΩprofile_deriv, ωΩ_deriv, V_symmetric);

                VVre = Ms.re[Block(1,1)];
                VWre = Ms.re[Block(1,2)];
                WVre = Ms.re[Block(2,1)];
                WWre = Ms.re[Block(2,2)];

                latitudinal_space = NormalizedPlm(m);
                cosθop = Multiplication(cosθ, latitudinal_space);
                sinθdθop = sinθ∂θ_Operator(latitudinal_space);
                ∇² = HorizontalLaplacian(latitudinal_space);
                ℓℓp1op = -∇²;

                space2d = radialspace ⊗ latitudinal_space;
                space2d_D2 = rangespace(Derivative(radialspace, 2)) ⊗ latitudinal_space;
                space2d_D4 = rangespace(Derivative(radialspace, 4)) ⊗ latitudinal_space;

                V_ℓinds = RossbyWaveSpectrum.ℓrange(1, nℓ, V_symmetric);
                W_ℓinds = RossbyWaveSpectrum.ℓrange(1, nℓ, !V_symmetric);

                twobyℓℓp1 = 2*inv(ℓℓp1op);

                @testset "VV" begin
                    O = m*ΔΩ*(I ⊗ (twobyℓℓp1 - I)) : space2d → space2d_D2;
                    A = real(kronmatrix(expand(O), nr, V_ℓinds, V_ℓinds));
                    if m == 1
                        @test VVre[Block(1,1)] ≈ A[Block(1,1)] atol=1e-10
                    end
                    @test VVre ≈ A
                end

                @testset "VW" begin
                    O = ΔΩ * (-I ⊗ twobyℓℓp1) * ( ((DDr - 2*onebyr) ⊗ (cosθop * ℓℓp1op))
                        + (DDr ⊗ sinθdθop)
                        - (Multiplication(onebyr) ⊗ (sinθdθop * ℓℓp1op))
                        ) : space2d → space2d_D2;
                    A = real(kronmatrix(expand(O), nr, V_ℓinds, W_ℓinds));
                    A .*= Rsun / Wscaling;
                    @test VWre ≈ A
                end

                @testset "WV" begin
                    O = ΔΩ * (-I ⊗ inv(ℓℓp1op)) *(
                            ddr ⊗ ((4cosθop * ℓℓp1op + sinθdθop * (ℓℓp1op + 2)))
                            + (ddr + 2onebyr) ⊗ (∇² * sinθdθop)
                        ) : space2d → space2d_D4;
                    A = real(kronmatrix(expand(O), nr, W_ℓinds, V_ℓinds));
                    A .*= Rsun * Wscaling * Weqglobalscaling;
                    @test WVre ≈ A
                end

                @testset "WW" begin
                    O = m * ΔΩ * (ddrDDr ⊗ (twobyℓℓp1 - 1)
                        - Multiplication(onebyr2) ⊗ ((twobyℓℓp1 - 1) * ℓℓp1op)
                        - (Multiplication(2*onebyr*ηρ) ⊗ I)
                        ) : space2d → space2d_D4;
                    A = real(kronmatrix(expand(O), nr, W_ℓinds, W_ℓinds));
                    A .*= Weqglobalscaling * Rsun^2;
                    @test WWre ≈ A
                end
            end
        end

        @testset "compare with matrix" begin
            @testset for m in [1, 4, 10]
                Mc.re .= 0; Mc.im .= 0;
                Ms.re .= 0; Ms.im .= 0;
                @testset "radial constant and constant" begin
                    RossbyWaveSpectrum.constant_differential_rotation_terms!(Mc, m;
                        operators, ΔΩ_frac);
                    RossbyWaveSpectrum.solar_differential_rotation_terms!(Ms, m;
                        operators, ΔΩprofile_deriv, ωΩ_deriv);

                    @testset for colind in 1:2, rowind in 1:2
                        Sc = matrix_block(Ms, rowind, colind, nvariables)
                        C = matrix_block(Mc, rowind, colind, nvariables)
                        @testset "real" begin
                            @test blockwise_isapprox(Sc.re, C.re, atol = 1e-10, rtol = 5e-4)
                        end
                        @testset "imag" begin
                            @test blockwise_isapprox(Sc.im, C.im, atol = 1e-10, rtol = 1e-3)
                        end
                    end
                end
            end
        end
    end
    @testset "compare with radial" begin
        @testset "compare with analytical" begin
            ΔΩprofile_deriv = RossbyWaveSpectrum.solar_differential_rotation_profile_derivatives_Fun(;
                operators, rotation_profile = :radial_equator, smoothing_param=1e-3);
            (; ΔΩ, ∂r_ΔΩ, ∂θ_ΔΩ, ∂2r_ΔΩ, ∂r∂θ_ΔΩ, ∂2θ_ΔΩ) = ΔΩprofile_deriv;

            ωΩ_deriv = RossbyWaveSpectrum.solar_differential_rotation_vorticity_Fun(;
                operators, ΔΩprofile_deriv);
            (; ωΩr, ∂rωΩr, ∂θωΩr_by_sinθ, ωΩθ_by_rsinθ) = ωΩ_deriv;

            cosθop = Multiplication(cosθ)
            @test ωΩr ≈ ((I ⊗ 2cosθop) * ΔΩ)
            @test ∂rωΩr ≈ ((I ⊗ 2cosθop) * ∂r_ΔΩ)
            @test ∂θωΩr_by_sinθ ≈ -2ΔΩ

            @testset for m in [1, 4, 10], V_symmetric in [true, false]
                Ms.re .= 0;
                RossbyWaveSpectrum.solar_differential_rotation_terms!(Ms, m;
                    operators, ΔΩprofile_deriv, ωΩ_deriv, V_symmetric);

                VVre = Ms.re[Block(1,1)];
                VWre = Ms.re[Block(1,2)];
                WVre = Ms.re[Block(2,1)];
                WWre = Ms.re[Block(2,2)];

                latitudinal_space = NormalizedPlm(m);
                cosθop = Multiplication(cosθ, latitudinal_space);
                sinθdθop = sinθ∂θ_Operator(latitudinal_space);
                ∇² = HorizontalLaplacian(latitudinal_space);
                ℓℓp1op = -∇²;

                space2d = radialspace ⊗ latitudinal_space;
                space2d_D2 = rangespace(Derivative(radialspace, 2)) ⊗ latitudinal_space;
                space2d_D4 = rangespace(Derivative(radialspace, 4)) ⊗ latitudinal_space;

                V_ℓinds = RossbyWaveSpectrum.ℓrange(1, nℓ, V_symmetric);
                W_ℓinds = RossbyWaveSpectrum.ℓrange(1, nℓ, !V_symmetric);

                @testset "VV" begin
                    O = m*ΔΩ*(I ⊗ (2*inv(ℓℓp1op) - I)) : space2d → space2d_D2;
                    A = real(kronmatrix(expand(O), nr, V_ℓinds, V_ℓinds));
                    @test VVre ≈ A
                end

                @testset "VW" begin
                    O = (-I ⊗ (2*inv(ℓℓp1op))) * ( ΔΩ * ((DDr - 2*onebyr) ⊗ (cosθop * ℓℓp1op))
                        - ∂r_ΔΩ * (I ⊗ (cosθop * ℓℓp1op))
                        + ΔΩ * (DDr ⊗ sinθdθop)
                        - ΔΩ * (Multiplication(onebyr) ⊗ (sinθdθop * ℓℓp1op))
                        - ∂r_ΔΩ * (I ⊗ (sinθdθop * ℓℓp1op/2))
                        )  : space2d → space2d_D2;
                    A = real(kronmatrix(expand(O), nr, V_ℓinds, W_ℓinds));
                    A .*= Rsun / Wscaling
                    @test VWre ≈ A
                end

                @testset "WV" begin
                end

                @testset "WW" begin
                end
            end
        end

        # @testset "compare with matrix" begin
        #     @testset for m in [1, 4, 10]
        #         Mr.re .= 0
        #         Mr.im .= 0
        #         Ms.re .= 0
        #         Ms.im .= 0
        #         @testset "radial constant and constant" begin
        #             RossbyWaveSpectrum.differential_rotation_matrix!(Mr, m;
        #                 rotation_profile = :radial_solar_equator, operators);
        #             RossbyWaveSpectrum.differential_rotation_matrix!(Ms, m;
        #                 rotation_profile = :solar_radial_equator, operators);

        #             @testset for colind in 1:1, rowind in 1:1
        #                 Sr = matrix_block(Ms, rowind, colind, nvariables)
        #                 R = matrix_block(Mr, rowind, colind, nvariables)
        #                 @testset "real" begin
        #                     @test blockwise_isapprox(Sr.re, R.re, atol = 1e-10, rtol = 5e-4)
        #                 end
        #                 @testset "imag" begin
        #                     @test blockwise_isapprox(Sr.im, R.im, atol = 1e-10, rtol = 1e-3)
        #                 end
        #             end
        #         end
        #     end
        # end
    end
end

@testset "constant differential rotation solution" begin
    nr, nℓ = 45, 8
    nparams = nr * nℓ
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ); constraints = RossbyWaveSpectrum.constraintmatrix(operators);
    @unpack r_in, r_out, Δr = operators.radial_params
    @unpack BCmatrices = constraints;
    MVn, MWn, MSn = BCmatrices;
    ΔΩ_frac = 0.02
    @testset for m in [1, 5, 10]
        @testset "constant" begin
            λr, vr, Mr = RossbyWaveSpectrum.differential_rotation_spectrum(m; operators, constraints,
                rotation_profile = :constant, ΔΩ_frac);
            λrf, vrf = RossbyWaveSpectrum.filter_eigenvalues(λr, vr, Mr, m;
                operators, constraints, Δl_cutoff = 7, n_cutoff = 9,
                eig_imag_unstable_cutoff = -1e-3,
                scale_eigenvectors = false);
            @info "constant diff rot: $(length(λrf)) eigenmode$(length(λrf) > 1 ? "s" : "") found for m = $m"
            @testset "ℓ == m" begin
                ω0 = RossbyWaveSpectrum.rossby_ridge(m; ΔΩ_frac)
                @test findmin(abs.(real(λrf) .- ω0))[1] < 1e-4
            end
            vfn = zeros(eltype(vrf), size(vrf, 1))
            @testset "boundary condition" begin
                Vop = RossbyWaveSpectrum.V_boundary_op(operators);
                @testset for n in axes(vrf, 2)
                    vfn .= @view vrf[:, n]
                    @testset "V" begin
                        @testset for ℓind in 1:nℓ
                            ℓ_skip = (ℓind - 1)*nr
                            inds_ℓ = ℓ_skip .+ (1:nr)
                            v = @view vfn[inds_ℓ]
                            @test MVn * v ≈ zeros(size(MVn,1)) atol=1e-10
                            pv = Fun(operators.radialspace, v)
                            Cp = Vop * pv;
                            @testset "inner boundary" begin
                                @test Cp(r_in) ≈ 0 atol=1e-10
                            end
                            @testset "outer boundary" begin
                                @test Cp(r_out) ≈ 0 atol=1e-10
                            end
                        end
                    end
                    @testset "W" begin
                        @testset for ℓind in 1:nℓ
                            ℓ_skip = (ℓind - 1)*nr
                            inds_ℓ = nr*nℓ + ℓ_skip .+ (1:nr)
                            w = @view vfn[inds_ℓ]
                            @test MWn * w ≈ zeros(size(MWn,1)) atol=1e-10
                            pw = Fun(operators.radialspace, w)
                            drpw = pw'
                            @testset "inner boundary" begin
                                @test pw(r_in) ≈ 0 atol=1e-10
                                @test drpw(r_in) ≈ 0 atol=1e-10
                            end
                            @testset "outer boundary" begin
                                @test pw(r_out) ≈ 0 atol=1e-10
                                @test drpw(r_out) ≈ 0 atol=1e-10
                            end
                        end
                    end
                    @testset "S" begin
                        @testset for ℓind in 1:nℓ
                            ℓ_skip = (ℓind - 1)*nr
                            inds_ℓ = 2nr*nℓ + ℓ_skip .+ (1:nr)
                            S = @view vfn[inds_ℓ]
                            @test MSn * S ≈ zeros(size(MSn,1)) atol=1e-10
                            pS = Fun(operators.radialspace, S)
                            drpS = pS'
                            @testset "inner boundary" begin
                                @test drpS(r_in) ≈ 0 atol=1e-10
                            end
                            @testset "outer boundary" begin
                                @test drpS(r_out) ≈ 0 atol=1e-10
                            end
                        end
                    end
                end
            end
            @testset "full eigenvalue problem solution" begin
                v = rossby_ridge_eignorm(λrf, vrf, Mr, m, nparams)
                # these improve with resolution
                @test v[1] < 2e-2
                @test v[2] < 0.5
                @test v[3] < 0.8
            end
        end
        @testset "radial constant" begin
            λr, vr, Mr = RossbyWaveSpectrum.differential_rotation_spectrum(m; operators, constraints,
                rotation_profile = :radial_constant, ΔΩ_frac);
            λrf, vrf = RossbyWaveSpectrum.filter_eigenvalues(λr, vr, Mr, m;
                operators, constraints, Δl_cutoff = 7, n_cutoff = 9, eig_imag_unstable_cutoff = -1e-3,
                scale_eigenvectors = false);
            @info "radial_constant diff rot: $(length(λrf)) eigenmode$(length(λrf) > 1 ? "s" : "") found for m = $m"
            @testset "ℓ == m" begin
                ω0 = RossbyWaveSpectrum.rossby_ridge(m; ΔΩ_frac)
                @test findmin(abs.(real(λrf) .- ω0))[1] < 1e-4
            end
            vfn = zeros(eltype(vrf), size(vrf, 1))
            @testset "boundary condition" begin
                Vop = RossbyWaveSpectrum.V_boundary_op(operators);
                @testset for n in axes(vrf, 2)
                    vfn .= @view vrf[:, n]
                    @testset "V" begin
                        @testset for ℓind in 1:nℓ
                            ℓ_skip = (ℓind - 1)*nr
                            inds_ℓ = ℓ_skip .+ (1:nr)
                            v = @view vfn[inds_ℓ]
                            @test MVn * v ≈ zeros(size(MVn,1)) atol=1e-10
                            pv = Fun(operators.radialspace, v)
                            Cp = Vop * pv;
                            @testset "inner boundary" begin
                                @test Cp(r_in) ≈ 0 atol=1e-10
                            end
                            @testset "outer boundary" begin
                                @test Cp(r_out) ≈ 0 atol=1e-10
                            end
                        end
                    end
                    @testset "W" begin
                        @testset for ℓind in 1:nℓ
                            ℓ_skip = (ℓind - 1)*nr
                            inds_ℓ = nr*nℓ + ℓ_skip .+ (1:nr)
                            w = @view vfn[inds_ℓ]
                            @test MWn * w ≈ zeros(size(MWn,1)) atol=1e-10
                            pw = Fun(operators.radialspace, w)
                            drpw = pw'
                            @testset "inner boundary" begin
                                @test pw(r_in) ≈ 0 atol=1e-10
                                @test drpw(r_in) ≈ 0 atol=1e-10
                            end
                            @testset "outer boundary" begin
                                @test pw(r_out) ≈ 0 atol=1e-10
                                @test drpw(r_out) ≈ 0 atol=1e-10
                            end
                        end
                    end
                    @testset "S" begin
                        @testset for ℓind in 1:nℓ
                            ℓ_skip = (ℓind - 1)*nr
                            inds_ℓ = 2nr*nℓ + ℓ_skip .+ (1:nr)
                            S = @view vfn[inds_ℓ]
                            @test MSn * S ≈ zeros(size(MSn,1)) atol=1e-10
                            pS = Fun(operators.radialspace, S)
                            drpS = pS'
                            @testset "inner boundary" begin
                                @test drpS(r_in) ≈ 0 atol=1e-10
                            end
                            @testset "outer boundary" begin
                                @test drpS(r_out) ≈ 0 atol=1e-10
                            end
                        end
                    end
                end
            end
            @testset "full eigenvalue problem solution" begin
                v = rossby_ridge_eignorm(λrf, vrf, Mr, m, nparams)
                # these improve with resolution
                @test v[1] < 2e-2
                @test v[2] < 0.5
                @test v[3] < 0.8
            end
        end
    end
end

include("run_threadedtests.jl")

@testset "compute_rossby_spectrum.jl" begin
    include(joinpath(dirname(dirname(pathof(RossbyWaveSpectrum))), "compute_rossby_spectrum.jl"))
    for V_symmetric in (true, false), diffrot in (false,)
        ComputeRossbySpectrum.computespectrum(8, 6, 1:1, V_symmetric, diffrot, :radial_solar_equator, save = false)
    end
    @testset "filteredeigen" begin
        nr, nℓ = 8,6
        V_symmetric = true
        diffrotprof = :radial_solar_equator
        @testset for diffrot in (false,)
            ComputeRossbySpectrum.computespectrum(nr, nℓ, 1:1, V_symmetric, diffrot, diffrotprof)
            filename = RossbyWaveSpectrum.rossbyeigenfilename(nr, nℓ,
                RossbyWaveSpectrum.filenamerottag(diffrot, diffrotprof),
                RossbyWaveSpectrum.filenamesymtag(V_symmetric))
            f = filteredeigen(filename)
            @test f.operators.radial_params[:nr] == nr
            @test f.operators.radial_params[:nℓ] == nℓ
            rm(filename)
        end
    end
end
