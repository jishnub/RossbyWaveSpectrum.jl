module TestMod

using RossbyWaveSpectrum
using RossbyWaveSpectrum: Msun, G
using Test
using LinearAlgebra
using OffsetArrays
using Aqua
using FillArrays
using Dierckx
using ApproxFun
using UnPack
using ApproxFunAssociatedLegendre
using StructArrays

using RossbyWaveSpectrum.Filters: NODES, SPATIAL, EIGVEC, EIGVAL, EIGEN, BC, SPATIAL_HIGHLAT

include("testutils.jl")

@testset "project quality" begin
    Aqua.test_all(RossbyWaveSpectrum, ambiguities = false, undefined_exports=false)
end

@testset "operators" begin
    operators = @inferred RossbyWaveSpectrum.radial_operators(5, 2)
    @test repr(operators) == "Operators"
    # test for getfield
    @test operators.radial_params[:nr] == 5
    @test RossbyWaveSpectrum.listncoefficients(operators.rad_terms) isa Vector{Pair{Symbol,Int}}
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
    constraints = RossbyWaveSpectrum.constraintmatrix(operators, Val(true))
    @unpack nullspacematrices = constraints
    ZV, ZW, ZS = nullspacematrices
    @unpack r_in, r_out = operators.radial_params
    @unpack radialspace = operators.radialspaces

    @test maximum(abs, constraints.BC * constraints.ZC) < 1e-10

    @testset "boundary conditions basis" begin
        @testset "W" begin
            for c in eachcol(ZW)
                p = Fun(radialspace, c)
                @test p(r_in) ≈ 0 atol=1e-14
                @test p(r_out) ≈ 0 atol=1e-14
                @test p'(r_in) ≈ 0 atol=1e-12
                @test p'(r_out) ≈ 0 atol=1e-12
            end
        end
        @testset "S" begin
            for c in eachcol(ZS)
                p = Fun(radialspace, c)
                @test p'(r_in) ≈ 0 atol=1e-12
                @test p'(r_out) ≈ 0 atol=1e-12
            end
        end
        @testset "V" begin
            CV = RossbyWaveSpectrum.V_boundary_op(operators);
            for c in eachcol(ZV)
                p = Fun(radialspace, c)
                Cp = CV * p
                @test Cp(r_in) ≈ 0 atol=1e-11
                @test Cp(r_out) ≈ 0 atol=1e-11
            end
        end
    end
end

@testset "FilteredEigen" begin
    operators = RossbyWaveSpectrum.radial_operators(5, 2)
    (; nparams) = operators.radial_params
    mr = 2:3
    nsols = 2
    V_symmetric = true
    lams = [rand(ComplexF64, nsols) for i in axes(mr,1)]
    vs = [StructArray{ComplexF64}((rand(nparams,nsols),rand(nparams,nsols))) for i in axes(mr,1)]
    rotation_profile = :constant
    kw = Dict{Symbol,Any}(:rotation_profile=>rotation_profile, :V_symmetric=>V_symmetric)
    f = FilteredEigen(lams, vs, mr, kw, operators)
    @test repr(f) == "Filtered eigen with m range = $mr"
    f1 = f[2]
    @test f1.m == 2
    @test f1.kw === f.kw
    @test f1.operators === f.operators
    @test f1.constraints === f.constraints
    @test repr(f1) == "Filtered eigen with m = 2"
    @test f1[1] == (λ = lams[1][1], v = vs[1][:,1])
    λs, vs = f1
    @test f1[1] == (λ = λs[1], v = vs[:,1])
    @test_throws BoundsError λs, vs, _ = f1

    @test_throws ArgumentError("m = 1 is not contained in $f") f[1]
    @test_throws ArgumentError("m range = 1:1 is not contained in $f") f[1:1]

    cs = RossbyWaveSpectrum.allocate_filter_caches(f, 2)
    @test all(in(keys(cs)), (:MVcache, :BCVcache, :Plcosθ, :VWSinv, :VWSinvsh, :F))

    f23 = f[2:3]
    @test f23.mr == 2:3
    @test f23.kw === f.kw
    @test f23.operators === f.operators
    @test f23.constraints === f.constraints
    @test collect(eigvalspairs(f1)) == (=>).(1:2, lams[1])

    @testset "operator matrices" begin
        m = 2
        A, B = RossbyWaveSpectrum.operator_matrices(f, m)
        @test A == RossbyWaveSpectrum.differential_rotation_matrix(m;
                        operators, V_symmetric, rotation_profile)
        @test B == RossbyWaveSpectrum.mass_matrix(m;
                        operators, V_symmetric)
    end
end

@testset "RotMatrix" begin
    nr, nℓ = 20, 10
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ)
    V_symmetric = true
    for rotation_profile in (:uniform, :constant, :solar_constant, :solar_latrad, :radial_constant)
        matrixfn! = RossbyWaveSpectrum.RotMatrix(Val(:matrix),
                        V_symmetric, rotation_profile; operators)
        @test R.kw == (; V_symmetric, rotation_profile)
        if rotation_profile == :uniform
            @test R.f == uniform_rotation_matrix!
            @test R isa RotMatrix{Nothing, Nothing}
        else
            @test R.f == differential_rotation_matrix!
            if rotation_profile == :constant
                @test R isa RotMatrix{Nothing, Nothing}
            end
        end
        spectrumfn! = RossbyWaveSpectrum.RotMatrix(Val(:spectrum),
                        V_symmetric, rotation_profile; operators)
        @test R.kw == (; V_symmetric, rotation_profile)
        if rotation_profile == :uniform
            @test R.f == uniform_rotation_spectrum!
        else
            @test R.f == differential_rotation_spectrum!
        end
    end
    @test_throws "unknown rotation profile :unknown" RossbyWaveSpectrum.RotMatrix(Val(:matrix),
                        V_symmetric, :unknown; operators)
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
    @test !BC == NODES | SPATIAL | EIGVEC | EIGVAL | EIGEN | SPATIAL_HIGHLAT
    @test !EIGVAL == NODES | SPATIAL | BC | EIGVEC | EIGEN | SPATIAL_HIGHLAT
    @test !EIGVEC == NODES | SPATIAL | BC | EIGVAL | EIGEN | SPATIAL_HIGHLAT
    @test !EIGEN == NODES | SPATIAL | BC | EIGVEC | EIGVAL | SPATIAL_HIGHLAT
    @test !SPATIAL == NODES | BC | EIGVEC | EIGVAL | EIGEN | SPATIAL_HIGHLAT
    @test !NODES == SPATIAL | BC | EIGVEC | EIGVAL | EIGEN | SPATIAL_HIGHLAT
    @test !(EIGVAL | EIGVEC) == NODES | SPATIAL | BC | EIGEN | SPATIAL_HIGHLAT
end

@testset "invtransform" begin
    for sp in (Chebyshev(), Legendre())
        n = 100
        for v in (rand(n, 20), rand(ComplexF64, n, 20))
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

@testset "matrix convergence with resolution: uniform rotation" begin
    nr, nℓ = 50, 2
    m = 5
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
    @unpack nvariables = operators;
    for V_symmetric in [true, false]
        M1 = RossbyWaveSpectrum.uniform_rotation_matrix(m; operators, V_symmetric);
        operators2 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ);
        M2 = RossbyWaveSpectrum.uniform_rotation_matrix(m; operators = operators2, V_symmetric);
        operators3 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ+5);
        M3 = RossbyWaveSpectrum.uniform_rotation_matrix(m; operators = operators3, V_symmetric);

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
end

@testset "uniform rotation solution" begin
    nr, nℓ = 40, 8
    nparams = nr * nℓ
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ, ν=1e10);
    constraints = RossbyWaveSpectrum.constraintmatrix(operators, Val(true));
    @unpack r_in, r_out, Δr = operators.radial_params
    @unpack BCmatrices = constraints;
    @unpack radialspace = operators.radialspaces;
    MVn, MWn, MSn = BCmatrices;
    @testset for m in (2, 5, 10), V_symmetric in [true, false]
        λu, vu, Mu = RossbyWaveSpectrum.uniform_rotation_spectrum(m;
            operators, constraints, V_symmetric);
        λuf, vuf = RossbyWaveSpectrum.filter_eigenvalues(λu, vu, Mu, m;
            operators, constraints, V_symmetric, Δl_cutoff = 7, n_cutoff = 9,
            eig_imag_damped_cutoff = 1e-3, eig_imag_unstable_cutoff = -1e-3,
            scale_eigenvectors = false);
        @info "uniform rot $V_symmetric: $(length(λuf)) eigenmode$(length(λuf) > 1 ? "s" : "") found for m = $m"
        if V_symmetric
            @testset "ℓ == m" begin
                res, ind = findmin(abs.(real(λuf) .- 2/(m+1)))
                @testset "eigenvalue" begin
                    @test res < 1e-4
                end
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
                        pv = Fun(radialspace, v)
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
                        pw = Fun(radialspace, w)
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
                        pS = Fun(radialspace, S)
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

@testset "zero constant differential rotation and uniform rotation" begin
    nr, nℓ = 20, 8
    m = 3
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ)
    @testset for V_symmetric in [true, false]
        Mu = RossbyWaveSpectrum.uniform_rotation_matrix(m; operators, V_symmetric);
        Mc = RossbyWaveSpectrum.differential_rotation_matrix(m;
                rotation_profile = :constant, operators, ΔΩ_frac = 0, V_symmetric);
        @test blockwise_isapprox(Mu, Mc)
    end
end

# @testset "radial differential rotation" begin
#     @testset "compare with constant" begin
#         nr, nℓ = 30, 10
#         operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
#         @unpack nvariables = operators;

#         Mc = RossbyWaveSpectrum.allocate_operator_matrix(operators);
#         Mr = RossbyWaveSpectrum.allocate_operator_matrix(operators);

#         ΔΩ_frac = 0.01
#         ΔΩprofile_deriv = @inferred RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives_Fun(;
#             operators, rotation_profile = :constant, smoothing_param=1e-3, ΔΩ_frac);

#         @testset for m in [1, 4, 10], V_symmetric in [true, false]
#             Mc.re .= 0; Mc.im .= 0;
#             Mr.re .= 0; Mr.im .= 0;
#             @testset "radial constant and constant" begin
#                 RossbyWaveSpectrum.constant_differential_rotation_terms!(Mc, m;
#                         operators, ΔΩ_frac, V_symmetric);
#                 RossbyWaveSpectrum.radial_differential_rotation_terms!(Mr, m;
#                         operators, ΔΩprofile_deriv, V_symmetric);

#                 @testset for colind in 1:nvariables, rowind in 1:nvariables
#                     Rc = matrix_block(Mr, rowind, colind, nvariables)
#                     C = matrix_block(Mc, rowind, colind, nvariables)
#                     @testset "real" begin
#                         @test blockwise_isapprox(Rc.re, C.re, atol = 1e-9, rtol = 1e-3)
#                     end
#                     @testset "imag" begin
#                         @test blockwise_isapprox(Rc.im, C.im, atol = 1e-9, rtol = 1e-3)
#                     end
#                 end
#             end
#         end
#     end

#     @testset "radial differential rotation" begin
#         nr, nℓ = 50, 10
#         nparams = nr * nℓ
#         m = 1
#         operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
#         @unpack diff_operators, rad_terms, rpts, radial_params = operators;
#         @unpack nvariables = operators;
#         @unpack ν = operators.constants;
#         r_mid = radial_params.r_mid::Float64;
#         Δr = radial_params.Δr::Float64;
#         a = 1 / (Δr / 2);
#         b = -r_mid / (Δr / 2);
#         @unpack ddr, d2dr2, DDr = diff_operators;
#         @unpack r2, r, ηρ = rad_terms;
#         @unpack r_in, r_out = operators.radial_params;

#         ℓ = 2
#         ℓℓp1 = ℓ*(ℓ+1)
#         ℓ′ = 1
#         cosθ21 = 1/√5
#         sinθdθ21 = 1/√5
#         ∇²_sinθdθ21 = -ℓℓp1 * sinθdθ21

#         @testset "convergence of diff rot profile" begin
#             local nr, nℓ = 50, 10
#             local operators1 = RossbyWaveSpectrum.radial_operators(nr, nℓ);
#             local r1 = operators1.rpts;
#             local operators2 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ);
#             local operators3 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ+5);

#             T = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives_Fun(
#                 operators = operators1, rotation_profile = :solar_equator);
#             ΔΩ1, ddrΔΩ1, d2dr2ΔΩ1 = T.ΔΩ, T.ddrΔΩ, T.d2dr2ΔΩ;

#             T = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives_Fun(
#                 operators = operators2, rotation_profile = :solar_equator);
#             ΔΩ2, ddrΔΩ2, d2dr2ΔΩ2 = T.ΔΩ, T.ddrΔΩ, T.d2dr2ΔΩ;

#             T = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives_Fun(
#                 operators = operators3, rotation_profile = :solar_equator);
#             ΔΩ3, ddrΔΩ3, d2dr2ΔΩ3 = T.ΔΩ, T.ddrΔΩ, T.d2dr2ΔΩ;

#             @testset "nr+5, nℓ" begin
#                 # make sure that nothing nonsensical is happening, and the points are scaled
#                 @test !isapprox(ΔΩ1.(r1), zeros(nr), atol=1e-12)
#                 @test ΔΩ1.(r1) ≈ ΔΩ2.(r1) rtol=1e-2
#                 @test ddrΔΩ1.(r1) ≈ ddrΔΩ2.(r1) rtol=8e-2
#                 @test d2dr2ΔΩ1.(r1) ≈ d2dr2ΔΩ2.(r1) rtol=2e-1
#             end

#             @testset "nr+5, nℓ+5" begin
#                 @test ΔΩ1.(r1) ≈ ΔΩ3.(r1) rtol=1e-2
#                 @test ddrΔΩ1.(r1) ≈ ddrΔΩ3.(r1) rtol=8e-2
#                 @test d2dr2ΔΩ1.(r1) ≈ d2dr2ΔΩ3.(r1) rtol=2e-1
#             end
#         end

#         @testset "matrix convergence with resolution" begin
#             nr, nℓ = 50, 10
#             m = 5
#             operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
#             @unpack nvariables = operators;
#             operators2 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ);
#             operators3 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ+5);
#             M_subsample = RossbyWaveSpectrum.allocate_operator_matrix(operators);
#             M1 = RossbyWaveSpectrum.allocate_operator_matrix(operators);
#             M2 = RossbyWaveSpectrum.allocate_operator_matrix(operators2);
#             M3 = RossbyWaveSpectrum.allocate_operator_matrix(operators3);

#             @testset for rotation_profile in [:radial_linear, :radial_solar_equator], V_symmetric in [true, false]
#                 RossbyWaveSpectrum.differential_rotation_matrix!(M1, m; operators, rotation_profile, V_symmetric);
#                 RossbyWaveSpectrum.differential_rotation_matrix!(M2, m; operators = operators2, rotation_profile, V_symmetric);
#                 RossbyWaveSpectrum.differential_rotation_matrix!(M3, m; operators = operators3, rotation_profile, V_symmetric);

#                 @testset "same size" begin
#                     matrix_subsample!(M_subsample, M1, nr, nr, nℓ, nvariables)
#                     @test blockwise_isequal(M_subsample, M1);
#                 end

#                 matrix_subsample!(M_subsample, M2, nr+5, nr, nℓ, nvariables);
#                 @testset for rowind in 1:nvariables, colind in 1:nvariables
#                     @testset "real" begin
#                         M2_ssv = matrix_block(M_subsample.re, rowind, colind);
#                         M1v = matrix_block(M1.re, rowind, colind);

#                         rtol = rowind == colind == 2 ? 8e-2 : 1e-2

#                         @test blockwise_isapprox(M2_ssv, M1v; rtol)
#                     end

#                     @testset "imag" begin
#                         M2_ssv = matrix_block(M_subsample.im, rowind, colind);
#                         M1v = matrix_block(M1.im, rowind, colind);

#                         @test blockwise_isapprox(M2_ssv, M1v; rtol = 1e-2)
#                     end
#                 end

#                 matrix_subsample!(M_subsample, M3, nr+5, nr, nℓ, nvariables);
#                 @testset for rowind in 1:nvariables, colind in 1:nvariables
#                     @testset "real" begin
#                         M3_ssv = matrix_block(M_subsample.re, rowind, colind)
#                         M1v = matrix_block(M1.re, rowind, colind)

#                         rtol = rowind == colind == 2 ? 8e-2 : 1e-2

#                         @test blockwise_isapprox(M3_ssv, M1v; rtol)
#                     end
#                     @testset "imag" begin
#                         M3_ssv = matrix_block(M_subsample.im, rowind, colind)
#                         M1v = matrix_block(M1.im, rowind, colind)

#                         @test blockwise_isapprox(M3_ssv, M1v; rtol = 1e-2)
#                     end
#                 end
#             end
#         end
#     end
# end

@testset "solar differential rotation" begin
    nr, nℓ = 30, 10
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
    @unpack nvariables = operators;
    @unpack Wscaling, Weqglobalscaling, Seqglobalscaling = operators.scalings;
    @unpack ddr, DDr, ddrDDr = operators.diff_operators;
    @unpack onebyr, ηρ, onebyr2, r = operators.rad_terms;
    @unpack radialspaces = operators;
    @unpack radialspace, radialspace_D2, radialspace_D4 = radialspaces

    Mc = RossbyWaveSpectrum.allocate_operator_matrix(operators);
    Mr = RossbyWaveSpectrum.allocate_operator_matrix(operators);
    Ms = RossbyWaveSpectrum.allocate_operator_matrix(operators);

    cosθ = Fun(Legendre());
    cosθop = Multiplication(cosθ)

    @testset "compare with constant" begin
        ΔΩ_frac = 0.01
        ΔΩprofile_deriv = @inferred RossbyWaveSpectrum.solar_differential_rotation_profile_derivatives_Fun(;
            operators, rotation_profile = :constant, smoothing_param=1e-3, ΔΩ_frac);
        (; ΔΩ, dr_ΔΩ, d2r_ΔΩ) = ΔΩprofile_deriv;

        ωΩ_deriv = @inferred RossbyWaveSpectrum.solar_differential_rotation_vorticity_Fun(;
            operators, ΔΩprofile_deriv);
        ωΩterms_raw = ωΩ_deriv.raw;
        ωΩterms_coriolis = ωΩ_deriv.coriolis;

        @testset "compare with analytical" begin
            @test ωΩterms_raw.ωΩr ≈ (I ⊗ 2cosθop) * ΔΩ
            @test ωΩterms_raw.∂rωΩr ≈ zero(ωΩterms_raw.∂rωΩr) atol=1e-14
            @test ωΩterms_raw.inv_sinθ_∂θωΩr ≈ -2ΔΩ
            @test ωΩterms_raw.inv_rsinθ_ωΩθ ≈ (Multiplication(-2*onebyr) ⊗ I) * ΔΩ
            @test ωΩterms_raw.inv_sinθ_∂r∂θωΩr ≈ zero(ωΩterms_raw.inv_sinθ_∂r∂θωΩr) atol=1e-14
            @test ωΩterms_raw.∂r_inv_rsinθ_ωΩθ ≈ (Multiplication(2*onebyr2) ⊗ I) * ΔΩ

            @test ωΩterms_coriolis.ωΩr ≈ (I ⊗ 4cosθop) * ΔΩ
            @test ωΩterms_coriolis.∂rωΩr ≈ zero(ωΩterms_coriolis.∂rωΩr) atol=1e-14
            @test ωΩterms_coriolis.inv_sinθ_∂θωΩr ≈ -4ΔΩ
            @test ωΩterms_coriolis.inv_sinθ_∂r∂θωΩr ≈ zero(ωΩterms_coriolis.inv_sinθ_∂r∂θωΩr) atol=1e-14
            @test ωΩterms_coriolis.inv_rsinθ_ωΩθ ≈ (Multiplication(-4*onebyr) ⊗ I) * ΔΩ
            @test ωΩterms_coriolis.∂r_inv_rsinθ_ωΩθ ≈ (Multiplication(4*onebyr2) ⊗ I) * ΔΩ
        end

        @testset "compare with matrix" begin
            @testset for m in [1, 4, 10], V_symmetric in [true, false]
                Mc.re .= 0; Mc.im .= 0;
                Ms.re .= 0; Ms.im .= 0;
                @testset "solar constant and constant" begin
                    RossbyWaveSpectrum.constant_differential_rotation_terms!(Mc, m;
                        operators, ΔΩ_frac, V_symmetric);
                    RossbyWaveSpectrum.solar_differential_rotation_terms!(Ms, m;
                        operators, ΔΩprofile_deriv, ωΩ_deriv, V_symmetric);

                    @testset for colind in 1:3, rowind in 1:3
                        Sc = matrix_block(Ms, rowind, colind, nvariables)
                        C = matrix_block(Mc, rowind, colind, nvariables)
                        @testset "real" begin
                            @test blockwise_isapprox(Sc.re, C.re, atol = 1e-9, rtol = 5e-4)
                        end
                        @testset "imag" begin
                            @test blockwise_isapprox(Sc.im, C.im, atol = 1e-9, rtol = 1e-3)
                        end
                    end
                end
            end
        end
    end
    @testset "compare with radial" begin
        ΔΩprofile_deriv = @inferred RossbyWaveSpectrum.solar_differential_rotation_profile_derivatives_Fun(;
            operators, rotation_profile = :radial_equator, smoothing_param=1e-4);
        (; ΔΩ, dr_ΔΩ, d2r_ΔΩ) = ΔΩprofile_deriv;

        ωΩ_deriv = @inferred RossbyWaveSpectrum.solar_differential_rotation_vorticity_Fun(;
            operators, ΔΩprofile_deriv);
        ωΩterms_raw = ωΩ_deriv.raw;
        ωΩterms_coriolis = ωΩ_deriv.coriolis;

        @testset "compare with analytical" begin
            @test ωΩterms_raw.ωΩr ≈ (I ⊗ 2cosθop) * ΔΩ
            @test ωΩterms_raw.∂rωΩr ≈ (I ⊗ 2cosθop) * dr_ΔΩ
            @test ωΩterms_raw.inv_sinθ_∂θωΩr ≈ -2ΔΩ
            @test ωΩterms_raw.inv_rsinθ_ωΩθ ≈ -(dr_ΔΩ + (Multiplication(2onebyr) ⊗ I) * ΔΩ)
            @test ωΩterms_raw.inv_sinθ_∂r∂θωΩr ≈ -2dr_ΔΩ
            @test ωΩterms_raw.∂r_inv_rsinθ_ωΩθ ≈ -(d2r_ΔΩ + (Multiplication(2onebyr) ⊗ I) * dr_ΔΩ
                                        - (Multiplication(2onebyr2) ⊗ I) * ΔΩ)

            @test ωΩterms_coriolis.ωΩr ≈ (I ⊗ 4cosθop) * ΔΩ
            @test ωΩterms_coriolis.∂rωΩr ≈ (I ⊗ 4cosθop) * dr_ΔΩ
            @test ωΩterms_coriolis.inv_sinθ_∂θωΩr ≈ -4ΔΩ
            @test ωΩterms_coriolis.inv_rsinθ_ωΩθ ≈ ωΩterms_raw.inv_rsinθ_ωΩθ - (Multiplication(2onebyr) ⊗ I) * ΔΩ
            @test ωΩterms_coriolis.inv_sinθ_∂r∂θωΩr ≈ -4dr_ΔΩ
            @test ωΩterms_coriolis.∂r_inv_rsinθ_ωΩθ ≈ (ωΩterms_raw.∂r_inv_rsinθ_ωΩθ -
                        ((Multiplication(2onebyr) ⊗ I) * dr_ΔΩ - (Multiplication(2onebyr2) ⊗ I) * ΔΩ))
        end

        # @testset "compare with matrix" begin
        #     ΔΩprofile_deriv_rad = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives_Fun(;
        #                     operators, rotation_profile = :solar_equator, smoothing_param=1e-4);

        #     @testset for m in [1, 4, 10], V_symmetric in [true, false]
        #         Mr.re .= 0; Mr.im .= 0;
        #         Ms.re .= 0; Ms.im .= 0;
        #         @testset "radial constant and constant" begin
        #             RossbyWaveSpectrum.radial_differential_rotation_terms!(Mr, m;
        #                 operators, ΔΩprofile_deriv = ΔΩprofile_deriv_rad, V_symmetric);
        #             RossbyWaveSpectrum.solar_differential_rotation_terms!(Ms, m;
        #                 operators, ΔΩprofile_deriv, ωΩ_deriv, V_symmetric);

        #             @testset for (colind, rowind) in ((1,1), (3,3))
        #                 Sr = matrix_block(Ms, rowind, colind, nvariables);
        #                 R = matrix_block(Mr, rowind, colind, nvariables);
        #                 @testset "real" begin
        #                     @test blockwise_isapprox(Sr.re, R.re, atol = 1e-10, rtol = 2e-2)
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
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
    constraints = RossbyWaveSpectrum.constraintmatrix(operators, Val(true));
    @unpack r_in, r_out, Δr = operators.radial_params
    @unpack BCmatrices = constraints;
    @unpack radialspace = operators.radialspaces
    MVn, MWn, MSn = BCmatrices;
    ΔΩ_frac = 0.02
    @testset for m in (2, 5, 10), V_symmetric in [true, false]
        @testset "constant" begin
            λr, vr, Mr = RossbyWaveSpectrum.differential_rotation_spectrum(m; operators, constraints,
                rotation_profile = :constant, ΔΩ_frac, V_symmetric);
            λrf, vrf = RossbyWaveSpectrum.filter_eigenvalues(λr, vr, Mr, m;
                operators, constraints, V_symmetric, Δl_cutoff = 7, n_cutoff = 9,
                eig_imag_unstable_cutoff = -1e-3,
                scale_eigenvectors = false);
            @info "constant diff rot $V_symmetric: $(length(λrf)) eigenmode$(length(λrf) > 1 ? "s" : "") found for m = $m"
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
                            pv = Fun(radialspace, v)
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
                            pw = Fun(radialspace, w)
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
                            pS = Fun(radialspace, S)
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
                rotation_profile = :radial_constant, ΔΩ_frac, V_symmetric);
            λrf, vrf = RossbyWaveSpectrum.filter_eigenvalues(λr, vr, Mr, m;
                operators, constraints, V_symmetric,
                Δl_cutoff = 7, n_cutoff = 9, eig_imag_unstable_cutoff = -1e-3,
                scale_eigenvectors = false);
            @info "radial_constant diff rot: $(length(λrf)) eigenmode$(length(λrf) > 1 ? "s" : "") found for m = $m"
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
                            pv = Fun(radialspace, v)
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
                            pw = Fun(radialspace, w)
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
                            pS = Fun(radialspace, S)
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

@testset "wrap and unwrap structarrays" begin
    A = rand(3,4)
    v = [StructArray{ComplexF64}((A,A))]
    @test RossbyWaveSpectrum.rewrapstruct(v) == v
    vunwrap = RossbyWaveSpectrum.unwrapstruct(v)
    @test vunwrap == [(A,A)]
    vrewrap = RossbyWaveSpectrum.rewrapstruct(vunwrap)
    @test vrewrap == v
end

@testset "saving and loading" begin
    lam, vec = [[1.0+1.0im]], [StructArray{ComplexF64}((rand(4,1), rand(4,1)))]
    nr, nℓ = 8,6
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ)
    kw = (V_symmetric=true,)
    mr = 1:1
    rotation_profile = :uniform
    fname = RossbyWaveSpectrum.save_to_file(lam, vec, mr; operators, rotation_profile, kw...)
    Feig = FilteredEigen(fname)
    @test Feig.operators.operatorparams == operators.operatorparams
    @test Feig.operators.constants == operators.constants
    @test Feig.operators.scalings == operators.scalings
    @test Feig.operators.radial_params == operators.radial_params
    @test Feig.operators.radialdomain == operators.radialdomain
    @test Feig.kw[:V_symmetric] == kw[:V_symmetric]
    @test UInt16(Feig.kw[:filterflags]) == UInt16(Filters.DefaultFilter)
    @test Feig.lams == lam
    @test Feig.vs == vec
    rm(fname)
end

include("run_threadedtests.jl")

@testset "compute_rossby_spectrum.jl" begin
    include(joinpath(dirname(dirname(pathof(RossbyWaveSpectrum))), "compute_rossby_spectrum.jl"))
    for V_symmetric in [true, false]
        ComputeRossbySpectrum.computespectrum(8, 6, 1:1, V_symmetric, :radial_solar_equator,
            save = false, print_timer = false, print_parameters = false)
    end
    @testset "filteredeigen" begin
        nr, nℓ = 20,10
        V_symmetric = true
        mr = 1:1
        for rotation_profile in (:uniform, :constant)
            filename = ComputeRossbySpectrum.computespectrum(nr, nℓ, mr,
                V_symmetric, rotation_profile,
                print_timer = false, print_parameters = false)
            f = RossbyWaveSpectrum.filter_eigenvalues(filename)
            @test f.operators.radial_params[:nr] == nr
            @test f.operators.radial_params[:nℓ] == nℓ

            λs, vs = f[1]
            operators = RossbyWaveSpectrum.radial_operators(nr, nℓ)
            matrixfn! = RossbyWaveSpectrum.RotMatrix(Val(:matrix), V_symmetric, rotation_profile;
                            operators)

            # check that this runs without errors
            RossbyWaveSpectrum.filter_eigenvalues(λs, vs, mr[1];
                operators, V_symmetric, rotation_profile, matrixfn!,
                filterflags = Filters.EIGEN)

            rm(filename)
        end
    end
end

end
