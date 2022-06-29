using RossbyWaveSpectrum: RossbyWaveSpectrum, matrix_block, Rsun, operatormatrix, Msun, G, StructMatrix
using Test
using LinearAlgebra
using OffsetArrays
using Aqua
import SpecialPolynomials
using ForwardDiff
using FastTransforms
using Dierckx
using FastGaussQuadrature
import ApproxFun
using DelimitedFiles
using PerformanceTestTools
using UnPack
using StructArrays
using BlockArrays

using RossbyWaveSpectrum.Filters: NODES, SPATIAL, EIGVEC, EIGVAL, EIGEN, BC

@testset "project quality" begin
    Aqua.test_all(RossbyWaveSpectrum,
        ambiguities = false,
        stale_deps = (; ignore =
        [:PyCall, :PyPlot, :LaTeXStrings, :OrderedCollections]),
    )
end

@testset "operators" begin
    @inferred RossbyWaveSpectrum.radial_operators(5, 2)
    # Ensure that the Chebyshev interpolations of the scale heights match the actual
    # functional form, given by the spline interpolations
    @testset "stratification" begin
        nr, nℓ = 50, 25
        @testset "shallow" begin
            operators = RossbyWaveSpectrum.radial_operators(nr, nℓ, r_in_frac = 0.7);
            @unpack r, r_chebyshev = operators.coordinates;
            @unpack sρ, sg, sηρ, ddrsηρ, d2dr2sηρ,
                sηρ_by_r, ddrsηρ_by_r, ddrsηρ_by_r2, d3dr3sηρ, sT, sηT = operators.splines;
            @unpack ηρ, ddr_ηρbyr, ddr_ηρbyr2,
                ddr_ηρ, d2dr2_ηρ, d3dr3_ηρ = operators.rad_terms;

            @test sηρ.(r) ≈ ηρ.(r_chebyshev) rtol=1e-2
            @test ddrsηρ.(r) ≈ ddr_ηρ.(r_chebyshev) rtol=1e-2
            @test d2dr2sηρ.(r) ≈ d2dr2_ηρ.(r_chebyshev) rtol=1e-2
            @test d3dr3sηρ.(r) ≈ d3dr3_ηρ.(r_chebyshev) rtol=2e-2
            @test ddrsηρ_by_r2.(r) ≈ ddr_ηρbyr2.(r_chebyshev) rtol=1e-2
            @test ddrsηρ_by_r.(r) ≈ ddr_ηρbyr.(r_chebyshev) rtol=1e-2
        end

        @testset "deep" begin
            operators = RossbyWaveSpectrum.radial_operators(nr, nℓ, r_in_frac = 0.5);
            @unpack r, r_chebyshev = operators.coordinates;
            @unpack sρ, sg, sηρ, ddrsηρ, d2dr2sηρ,
                sηρ_by_r, ddrsηρ_by_r, ddrsηρ_by_r2, d3dr3sηρ, sT, sηT = operators.splines;
            @unpack ηρ, ddr_ηρbyr, ddr_ηρbyr2,
                ddr_ηρ, d2dr2_ηρ, d3dr3_ηρ = operators.rad_terms;

            @test sηρ.(r) ≈ ηρ.(r_chebyshev) rtol=1e-2
            @test ddrsηρ.(r) ≈ ddr_ηρ.(r_chebyshev) rtol=1e-2
            @test d2dr2sηρ.(r) ≈ d2dr2_ηρ.(r_chebyshev) rtol=1e-2
            @test d3dr3sηρ.(r) ≈ d3dr3_ηρ.(r_chebyshev) rtol=2e-2
            @test ddrsηρ_by_r2.(r) ≈ ddr_ηρbyr2.(r_chebyshev) rtol=2e-2
            @test ddrsηρ_by_r.(r) ≈ ddr_ηρbyr.(r_chebyshev) rtol=2e-2
        end
    end
end

df(f) = x -> ForwardDiff.derivative(f, x)
df(f, r) = df(f)(r)
d2f(f) = df(df(f))
d2f(f, r) = df(df(f))(r)
d3f(f) = df(d2f(f))
d3f(f, r) = df(d2f(f))(r)
d4f(f) = df(d3f(f))
d4f(f, r) = df(d3f(f))(r)

# define an alias to avoid clashes in the REPL with Chebyshev from ApproxFun
const _ChebyshevT = SpecialPolynomials.Chebyshev
chebyshevT(n) = n < 0 ? _ChebyshevT([0.0]) : _ChebyshevT([zeros(n); 1])
chebyshevT(n, x) = chebyshevT(n)(x)
chebyshevU(n) = n < 0 ? SpecialPolynomials.ChebyshevU([0.0]) : SpecialPolynomials.ChebyshevU([zeros(n); 1])
chebyshevU(n, x) = chebyshevU(n)(x)

function chebyfwd(f, r_in, r_out, nr, scalefactor = 5)
    w = FastTransforms.chebyshevpoints(scalefactor * nr)
    r_mid = (r_in + r_out)/2
    Δr = r_out - r_in
    r_fine = @. (Δr/2) * w + r_mid
    v = FastTransforms.chebyshevtransform(f.(r_fine))
    v[1:nr]
end

function chebyfwd(f, nr, scalefactor = 5)
    w = FastTransforms.chebyshevpoints(scalefactor * nr)
    v = FastTransforms.chebyshevtransform(f.(w))
    v[1:nr]
end

const P11norm = -2/√3

@testset "chebyshev" begin
    n = 10

    r_chebyshev, Tcf, Tci = RossbyWaveSpectrum.chebyshev_forward_inverse(n)
    @test Tcf * Tci ≈ Tci * Tcf ≈ I
    @test Tcf * r_chebyshev ≈ [0; 1; zeros(length(r_chebyshev)-2)]

    @testset "boundary conditions basis" begin
        @testset "W" begin
            MW = RossbyWaveSpectrum.dirichlet_neumann_chebyshev_matrix(n)
            @testset for i in 1:size(MW, 2)
                p = SpecialPolynomials.Chebyshev(MW[:, i])
                @test p(-1) ≈ 0 atol=1e-14
                @test p(1) ≈ 0 atol=1e-14
                @test df(p, -1) ≈ 0 atol=1e-12
                @test df(p, 1) ≈ 0 atol=1e-12
            end
            MW = nullspace(RossbyWaveSpectrum.Wboundary(4, n))
            @testset for i in 1:size(MW, 2)
                p = SpecialPolynomials.Chebyshev(MW[:, i])
                @test p(-1) ≈ 0 atol=1e-14
                @test p(1) ≈ 0 atol=1e-14
                @test df(p, -1) ≈ 0 atol=1e-12
                @test df(p, 1) ≈ 0 atol=1e-12
            end
        end
        @testset "S" begin
            MS = RossbyWaveSpectrum.neumann_chebyshev_matrix(n)
            @testset for i in 1:size(MS, 2)
                p = SpecialPolynomials.Chebyshev(MS[:, i])
                @test df(p, -1) ≈ 0 atol=1e-12
                @test df(p, 1) ≈ 0 atol=1e-12
            end
            MS = nullspace(RossbyWaveSpectrum.Sboundary(2, n))
            @testset for i in 1:size(MS, 2)
                p = SpecialPolynomials.Chebyshev(MS[:, i])
                @test df(p, -1) ≈ 0 atol=1e-12
                @test df(p, 1) ≈ 0 atol=1e-12
            end
        end
        @testset "V" begin
            radial_params = RossbyWaveSpectrum.parameters(2, 2)
            MV = RossbyWaveSpectrum.r2neumann_chebyshev_matrix(n, radial_params)
            @unpack Δr, r_mid = radial_params
            r = x -> r_mid + (Δr/2)*x
            @testset for i in 1:size(MV, 2)
                p = SpecialPolynomials.Chebyshev(MV[:, i])
                pbyr² = x -> p(x) / r(x)^2
                ddx_pbyr² = df(pbyr²)
                @test ddx_pbyr²(-1) ≈ 0 atol=1e-12
                @test ddx_pbyr²(1) ≈ 0 atol=1e-12
            end
            MV = nullspace(RossbyWaveSpectrum.Vboundary(2, n, radial_params))
            @testset for i in 1:size(MV, 2)
                p = SpecialPolynomials.Chebyshev(MV[:, i])
                pbyr² = x -> p(x) / r(x)^2
                ddx_pbyr² = df(pbyr²)
                @test ddx_pbyr²(-1) ≈ 0 atol=1e-12
                @test ddx_pbyr²(1) ≈ 0 atol=1e-12
            end
        end
    end
end

@testset "constraints" begin
    nr, nℓ = 30, 8
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ, r_in_frac = 0.5, r_out_frac = 0.985);
    constraints = RossbyWaveSpectrum.constraintmatrix(operators);
    @test maximum(abs, constraints.BC * constraints.ZC) < 1e-10
end

@testset "read solar model" begin
    r_in_frac = 0.5
    r_out_frac = 0.985
    r_in = r_in_frac * Rsun
    r_out = r_out_frac * Rsun
    @unpack sρ, sT, sg, sηρ, sηT = RossbyWaveSpectrum.read_solar_model(; r_in, r_out).splines;

    operators = RossbyWaveSpectrum.radial_operators(50, 2; r_in_frac, r_out_frac);
    @unpack r, r_chebyshev = operators.coordinates;

    ModelS = readdlm(joinpath(@__DIR__,"../src", "ModelS.detailed"))
    r_modelS = @view ModelS[:, 1];
    r_inds = r_in .<= r_modelS .<= r_out;
    r_modelS = reverse(r_modelS[r_inds]);
    q_modelS = exp.(reverse(ModelS[r_inds, 2]));
    T_modelS = reverse(ModelS[r_inds, 3]);
    ρ_modelS = reverse(ModelS[r_inds, 5]);

    g_modelS = @. G * Msun * q_modelS / r_modelS^2;

    @test sρ.(r) ≈ RossbyWaveSpectrum.interp1d(r_modelS, ρ_modelS, r) rtol=1e-2
    @test sT.(r) ≈ RossbyWaveSpectrum.interp1d(r_modelS, T_modelS, r) rtol=1e-2
    @test sg.(r) ≈ RossbyWaveSpectrum.interp1d(r_modelS, g_modelS, r) rtol=1e-2
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

# @testset "uniform rotation" begin
#     nr, nℓ = 50, 2
#     nparams = nr * nℓ
#     m = 1
#     operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
#     @unpack transforms, diff_operators, rad_terms, coordinates, radial_params, identities = operators;
#     @unpack r, r_chebyshev = coordinates;
#     (; nvariables, ν, Ω0) = operators.constants;
#     r_mid = radial_params.r_mid::Float64;
#     Δr = radial_params.Δr::Float64;
#     a = 1 / (Δr / 2);
#     b = -r_mid / (Δr / 2);
#     r̄(r) = clamp(a * r + b, -1.0, 1.0);

#     @unpack Tcrfwd = operators.transforms;

#     @unpack ddr, d2dr2, DDr = operators.diff_operators;

#     (; DDrM, onebyr2M, ddrM, onebyrM, gM,
#         κ_∇r2_plus_ddr_lnρT_ddrM, onebyr2_ddr_S0_by_cpM,
#         onebyr2_IplusrηρM) = operators.diff_operator_matrices;

#     (; onebyr, onebyr2, r2_cheby, r_cheby, ηρ, ηT,
#             g_cheby, ddr_lnρT, ddr_S0_by_cp) = operators.rad_terms;

#     @unpack r_in, r_out, nchebyr = operators.radial_params;

#     @unpack mat = operators;
#     @unpack Wscaling, Sscaling = operators.constants.scalings;

#     @test isapprox(onebyr(-1), 1/r_in, rtol=1e-4)
#     @test isapprox(onebyr(1), 1/r_out, rtol=1e-4)
#     @test isapprox(onebyr2(-1), 1/r_in^2, rtol=1e-4)
#     @test isapprox(onebyr2(1), 1/r_out^2, rtol=1e-4)

#     κ_by_r2M = κ * onebyr2M;

#     ddrηρ = ddr * ηρ;

#     M = RossbyWaveSpectrum.uniform_rotation_matrix(m; operators);

#     chebyfwdnr(f, scalefactor = 5) = chebyfwd(f, r_in, r_out, nr, scalefactor)

#     ℓ′ = 1
#     # for these terms ℓ = ℓ′ (= 1 in this case)
#     SWterm, SSterm = (zeros(nr, nr) for i in 1:2);
#     RossbyWaveSpectrum.uniform_rotation_matrix_terms_outer!((SWterm, SSterm),
#                         (ℓ′, m),
#                         (κ_∇r2_plus_ddr_lnρT_ddrM, κ_by_r2M, onebyr2_ddr_S0_by_cpM));

#     @testset "V terms" begin
#         @testset "WV term" begin
#             function ddr_term(r, n)
#                 r̄_r = r̄(r)
#                 Unm1 = chebyshevU(n-1, r̄_r)
#                 a * n * Unm1
#             end

#             @testset "ddr" begin
#                 @testset for n in 1:nr - 1
#                     ddr_Tn_analytical = chebyfwdnr(r -> ddr_term(r,n))
#                     @test ddrM[:, n+1] ≈ ddr_Tn_analytical rtol=1e-8
#                 end
#             end

#             function onebyrTn_term(r, n)
#                 r̄_r = r̄(r)
#                 Tn = chebyshevT(n, r̄_r)
#                 1/r * Tn
#             end

#             @testset "onebyrTn" begin
#                 @testset for n in 1:nr - 1
#                     onebyr_Tn_analytical = chebyfwdnr(r -> onebyrTn_term(r,n))
#                     @test onebyrM[:, n+1] ≈ onebyr_Tn_analytical rtol=1e-4
#                 end
#             end

#             Vn1 = P11norm
#             function WVtermfn(r, n)
#                 r̄_r = r̄(r)
#                 Tn = chebyshevT(n, r̄_r)
#                 2√(1/15) * (ddr_term(r, n) - 2onebyrTn_term(r, n)) / Rsun
#             end

#             ℓ = 2
#             ℓ′ = 1
#             m = 1
#             ℓℓp1 = ℓ*(ℓ+1)
#             ℓ′ℓ′p1 = ℓ′*(ℓ′+1)

#             Cℓ′ = ddrM - ℓ′ℓ′p1 * onebyrM
#             C1 = ddrM - 2onebyrM
#             WV = -2 / ℓℓp1 * (ℓ′ℓ′p1 * C1 * (1/√5) + Cℓ′ * (1/√5)) / Rsun

#             @testset for n in 1:nr-1
#                 WV_op_times_W = Vn1 * real(@view WV[:, n+1])
#                 WV_times_W_explicit = chebyfwdnr(r -> WVtermfn(r,n))
#                 @test WV_op_times_W ≈ WV_times_W_explicit rtol = 1e-5
#             end
#         end
#     end

#     @testset "W terms" begin
#         Wn1 = P11norm

#         @testset "VW term" begin
#             function Drρ_fn(r, n)
#                 r̄_r = r̄(r)
#                 Tn = chebyshevT(n, r̄_r)
#                 Unm1 = chebyshevU(n-1, r̄_r)
#                 a * n * Unm1 + ηρ(r̄_r) * Tn
#             end

#             @testset "Drρ" begin
#                 @testset for n in 0:nr - 1
#                     Drρ_Tn_analytical = chebyfwdnr(r -> Drρ_fn(r,n))
#                     @test DDrM[:, n+1] ≈ Drρ_Tn_analytical rtol=1e-8
#                 end
#             end

#             function VWterm_fn(r, n)
#                 r̄_r = r̄(r)
#                 Tn = chebyshevT(n, r̄_r)
#                 -2√(1/15) * (Drρ_fn(r, n) - 2/r * Tn) * Rsun / Wscaling
#             end

#             VW = RossbyWaveSpectrum.matrix_block(M, 1, 2, nvariables)

#             W1_inds = nr .+ (1:nr)

#             @testset for n in 1:nr-1
#                 VW_op_times_W = Wn1 * real(@view VW[W1_inds, n+1])
#                 VW_times_W_explicit = chebyfwdnr(r -> VWterm_fn(r, n))
#                 @test VW_op_times_W ≈ -VW_times_W_explicit rtol = 1e-6
#             end
#         end

#         @testset "WW term" begin
#             function onebyr2_Iplusrηρ_Tn_fn(r, n)
#                 r̄_r = r̄(r)
#                 Tn = chebyshevT(n, r̄_r)
#                 ηr = ηρ(r̄_r)
#                 1/r^2 * (1 + ηr * r) * Tn
#             end

#             @testset "onebyr2_IplusrηρM" begin
#                 @testset for n in 1:nr - 1
#                     onebyr2_Iplusrηρ_Tn_analytical = chebyfwdnr(r -> onebyr2_Iplusrηρ_Tn_fn(r, n))
#                     @test onebyr2_IplusrηρM[:, n+1] ≈ onebyr2_Iplusrηρ_Tn_analytical rtol=1e-4
#                 end
#             end
#         end

#         @testset "SW term" begin
#             function onebyr2_ddr_S0_by_cp_fn(r, n)
#                 r̄_r = r̄(r)
#                 Tn = chebyshevT(n, r̄_r)
#                 1/r^2 * ddr_S0_by_cp(r̄_r) * Tn
#             end
#             @testset for n in 1:nr-1
#                 onebyr2_ddr_S0_by_cp_op = real(@view onebyr2_ddr_S0_by_cpM[:, n+1])
#                 onebyr2_ddr_S0_by_cp_explicit = chebyfwdnr(r -> onebyr2_ddr_S0_by_cp_fn(r, n))
#                 @test onebyr2_ddr_S0_by_cp_op ≈ onebyr2_ddr_S0_by_cp_explicit rtol = 1e-4
#             end
#         end
#     end

#     @testset "S terms" begin
#         @testset "SS term" begin
#             @testset "individual terms" begin
#                 function ddr_lnρT_fn(r, n)
#                     r̄_r = r̄(r)
#                     Tn = chebyshevT(n, r̄_r)
#                     ηρr = ηρ(r̄_r)
#                     ηTr = ηT(r̄_r)
#                     (ηρr + ηTr) * Tn
#                 end
#                 ddr_lnρTM = mat(ddr_lnρT)
#                 @testset for n in 1:nr-1
#                     ddr_lnρT_op = @view ddr_lnρTM[:, n+1]
#                     ddr_lnρT_analytical = chebyfwdnr(r -> ddr_lnρT_fn(r, n))
#                     @test ddr_lnρT_op ≈ ddr_lnρT_analytical rtol=1e-8
#                 end
#                 function ddr_lnρT_ddr_fn(r, n)
#                     r̄_r = r̄(r)
#                     Unm1 = chebyshevU(n-1, r̄_r)
#                     ηρr = ηρ(r̄_r)
#                     ηTr = ηT(r̄_r)
#                     (ηρr + ηTr) * a * n * Unm1
#                 end
#                 ddr_lnρT_ddrM = RossbyWaveSpectrum.operatormatrix(ddr_lnρT * ddr, nr, 4);
#                 @testset for n in 1:nr-1
#                     ddr_lnρT_ddr_op = @view ddr_lnρT_ddrM[:, n+1]
#                     ddr_lnρT_ddr_analytical = chebyfwdnr(r -> ddr_lnρT_ddr_fn(r, n))
#                     @test ddr_lnρT_ddr_op ≈ ddr_lnρT_ddr_analytical rtol=1e-8
#                 end

#                 function onebyr_ddr_fn(r, n)
#                     r̄_r = r̄(r)
#                     Unm1 = n >= 1 ? chebyshevU(n-1, r̄_r) : 0.0
#                     1/r * a * n * Unm1
#                 end
#                 onebyr_ddrM = mat(onebyr * ddr)
#                 @testset for n in 1:nr-1
#                     onebyr_ddr_op = @view onebyr_ddrM[:, n+1]
#                     onebyr_ddr_analytical = chebyfwdnr(r -> onebyr_ddr_fn(r, n))
#                     @test onebyr_ddr_op ≈ onebyr_ddr_analytical rtol=1e-4
#                 end
#             end

#             @testset "SS term" begin
#                 function SSterm_fn(r, n)
#                     r̄_r = r̄(r)
#                     Tn = chebyshevT(n)
#                     d2Tnr = d2f(Tn)(r̄_r) * (2/Δr)^2
#                     dTnr = df(Tn)(r̄_r) * (2/Δr)
#                     Tnr = Tn(r̄_r)
#                     ηρr = ηρ(r̄_r)
#                     ηTr = ηT(r̄_r)
#                     f = d2Tnr + 2/r * dTnr - 2/r^2 * Tnr + (ηρr + ηTr)*dTnr
#                     κ * Rsun^2 * f
#                 end
#                 @testset for n in 1:nr-1
#                     SStermM_op = @view SSterm[:, n+1]
#                     SStermM_analytical = chebyfwdnr(r -> SSterm_fn(r, n))
#                     @test SStermM_op ≈ SStermM_analytical rtol=1e-4
#                 end
#             end
#         end
#     end
# end

# @testset "viscosity" begin
#     nr, nℓ = 50, 2
#     nparams = nr * nℓ
#     m = 1
#     operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
#     @unpack transforms, diff_operators, rad_terms, coordinates, radial_params, identities = operators;
#     @unpack r, r_chebyshev = coordinates;
#     @unpack nvariables, ν = operators.constants;
#     r_mid = radial_params.r_mid::Float64;
#     Δr = radial_params.Δr::Float64;
#     a = 1 / (Δr / 2);
#     b = -r_mid / (Δr / 2);
#     r̄(r) = clamp(a * r + b, -1.0, 1.0);
#     @unpack Tcrfwd = transforms;
#     @unpack Iℓ = identities;
#     @unpack ddr, d2dr2, DDr = diff_operators;
#     (; onebyr, onebyr2, r2_cheby, r_cheby, ηρ, ηρ_by_r, ηρ_by_r2, ηρ2_by_r2) = rad_terms;
#     @unpack r_in, r_out = operators.radial_params;
#     @unpack mat = operators;

#     chebyfwdnr(f, scalefactor = 5) = chebyfwd(f, r_in, r_out, nr, scalefactor)

#     @testset "V terms" begin
#         Vn1 = P11norm

#         function VVterm1fn(r, n)
#             r̄_r = r̄(r)
#             T = chebyshevT(n)
#             Tn = T(r̄_r)
#             d2Tn = a^2 * d2f(T)(r̄_r)
#             Vn1 * (d2Tn - 2 / r^2 * Tn)  * Rsun^2
#         end

#         function VVterm3fn(r, n)
#             r̄_r = r̄(r)
#             Tn = chebyshevT(n, r̄_r)
#             Unm1 = chebyshevU(n-1, r̄_r)
#             anr = a * n * r
#             Vn1 * (-2Tn + anr * Unm1) * ηρ(r̄_r) / r  * Rsun^2
#         end

#         VVtermfn(r, n) = VVterm1fn(r, n) + VVterm3fn(r, n)

#         M = RossbyWaveSpectrum.allocate_matrix(operators)
#         RossbyWaveSpectrum.viscosity_terms!(M, m; operators)

#         @testset "VV terms" begin
#             VV = RossbyWaveSpectrum.matrix_block(M, 1, 1, nvariables)
#             V1_inds = 1:nr
#             @testset for n in 1:nr-1
#                 V_op = Vn1 * real(VV[V1_inds, n + 1] ./ (-im * ν))
#                 V_explicit = chebyfwdnr(r -> VVtermfn(r, n))
#                 @test V_op ≈ V_explicit rtol = 1e-5
#             end
#         end
#     end

#     @testset "W terms" begin
#         # first term
#         r_d2dr2_ηρ_by_r = r_cheby * d2dr2[ηρ_by_r];
#         r_d2dr2_ηρ_by_rM = mat(r_d2dr2_ηρ_by_r);

#         ddr_minus_2byr = ddr - 2onebyr;
#         ddr_minus_2byr_r_d2dr2_ηρ_by_r = ddr_minus_2byr * r_d2dr2_ηρ_by_r;
#         ddr_minus_2byr_r_d2dr2_ηρ_by_rM = mat(ddr_minus_2byr_r_d2dr2_ηρ_by_r);

#         ddr_minus_2byr_ηρ_by_r2 = ddr_minus_2byr[ηρ_by_r2];
#         ddr_minus_2byr_ηρ_by_r2M = mat(ddr_minus_2byr_ηρ_by_r2);

#         four_ηρ_by_r = 4ηρ_by_r;
#         d2dr2_plus_4ηρ_by_rM = mat(d2dr2[four_ηρ_by_r]);

#         d2dr2_plus_4ηρ_by_r = d2dr2 + 4ηρ_by_r;
#         d2dr2_d2dr2_plus_4ηρ_by_rM = mat(d2dr2 * d2dr2_plus_4ηρ_by_r);

#         one_by_r2_d2dr2_plus_4ηρ_by_rM = mat(onebyr2 * d2dr2_plus_4ηρ_by_r);

#         d2dr2_one_by_r2M = mat(d2dr2[onebyr2]);
#         d2dr2_one_by_r2M_2 = mat(onebyr2 * (d2dr2 - 4onebyr*ddr + 6onebyr2));

#         onebyr4_cheby = onebyr2*onebyr2;
#         onebyr4_chebyM = mat(onebyr4_cheby);

#         # third term
#         ddr_minus_2byr_DDr = ddr_minus_2byr * DDr
#         ηρ_ddr_minus_2byr_DDr = ηρ * ddr_minus_2byr_DDr
#         ddr_ηρ_ddr_minus_2byr_DDr = ddr * ηρ_ddr_minus_2byr_DDr
#         ηρ_ddr_minus_2byr_DDrM = operatormatrix(ηρ_ddr_minus_2byr_DDr, nr, 3);
#         ddr_ηρ_ddr_minus_2byr_DDrM = operatormatrix(ddr_ηρ_ddr_minus_2byr_DDr, nr, 3);

#         ddr_ηρ_by_r2 = ddr[ηρ_by_r2];
#         ηρ_by_r2_ddr_minus_2byr = ηρ_by_r2 * ddr_minus_2byr;
#         ddr_ηρ_by_r2_minus_2ηρ_by_r2_ddr_minus_2byr = ddr_ηρ_by_r2 - 2*ηρ_by_r2_ddr_minus_2byr;
#         ddr_ηρ_by_r2_minus_2ηρ_by_r2_ddr_minus_2byrM = mat(ddr_ηρ_by_r2_minus_2ηρ_by_r2_ddr_minus_2byr);

#         # fourth term
#         ηρ2_by_r2M = mat(ηρ2_by_r2);

#         # @testset "first term" begin
#             # function WWterm11_1fn(r, n)
#             #     T = chebyshevT(n)
#             #     dT = x -> df(T, x) * (2/Δr)
#             #     d2T = x -> d2f(T, x) * (2/Δr)^2
#             #     r̄_r = r̄(r)
#             #     # F1 = r -> 2dT(r̄_r)*ddr_ηρ(r̄_r) + ηρ(r̄_r)*(-2dT(r̄_r)/r + d2T(r̄_r)) +
#             #     #         T(r̄_r) * (2 * ηρ_by_r2(r̄_r) - 2 * ddr_ηρ(r̄_r)/r + d2dr2_ηρ(r̄_r))
#             #     F1 = r -> r * d2f(r -> ηρ(r̄(r))/r * T(r̄(r)), r)
#             #     F1(r)
#             # end

#             # @testset "WWterm11_1fn" begin
#             #     @testset for n in 0:nr-1
#             #         WWterm11_1_op = @view r_d2dr2_ηρ_by_rM[:, n+1];
#             #         WWterm11_1_analytical = chebyfwdnr(r -> WWterm11_1fn(r, n));
#             #         @test WWterm11_1_op ≈ WWterm11_1_analytical rtol=5e-4
#             #     end
#             # end

#         #     function WWterm11fn(r, n)
#         #         T = chebyshevT(n)
#         #         F1 = r -> WWterm11_1fn(r, n)
#         #         F2 = r -> df(F1, r) - 2/r * F1(r)
#         #         F2(r)
#         #     end

#         #     @testset "WWterm11fn" begin
#         #         @testset for n in 0:nr-1
#         #             WWterm11_op = @view ddr_minus_2byr_r_d2dr2_ηρ_by_rM[:, n+1]
#         #             WWterm11_analytical = chebyfwdnr(r -> WWterm11fn(r, n))
#         #             @test WWterm11_op ≈ WWterm11_analytical rtol=5e-4
#         #         end
#         #     end

#         #     function WWterm12fn(r, n)
#         #         T = chebyshevT(n)
#         #         F1 = r -> ηρ(r̄(r))/r^2 * T(r̄(r))
#         #         F2 = r -> df(F1, r) - 2/r * F1(r)
#         #         F2(r)
#         #     end

#         #     @testset "WWterm12fn" begin
#         #         @testset for n in 0:nr-1
#         #             WWterm12_op = @view ddr_minus_2byr_ηρ_by_r2M[:, n+1]
#         #             WWterm12_analytical = chebyfwdnr(r -> WWterm12fn(r, n))
#         #             @test WWterm12_op ≈ WWterm12_analytical rtol=5e-4
#         #         end
#         #     end

#         #     function WWterm13fn(r, n)
#         #         T = chebyshevT(n)
#         #         F1 = r -> 4ηρ(r̄(r))/r * T(r̄(r))
#         #         F2 = r -> d2f(F1, r)
#         #         F2(r)
#         #     end

#         #     @testset "WWterm13fn" begin
#         #         @testset for n in 0:nr-1
#         #             WWterm13_op = @view d2dr2_plus_4ηρ_by_rM[:, n+1]
#         #             WWterm13_analytical = chebyfwdnr(r -> WWterm13fn(r, n))
#         #             @test WWterm13_op ≈ WWterm13_analytical rtol=5e-4
#         #         end
#         #     end

#         #     function WWterm13_fullfn(r, n)
#         #         T = chebyshevT(n)
#         #         F1 = r -> d2f(T ∘ r̄, r) + 4ηρ(r̄(r))/r * T(r̄(r))
#         #         F2 = r -> d2f(F1, r)
#         #         F2(r)
#         #     end

#         #     @testset "WWterm13_fullfn" begin
#         #         @testset for n in 0:nr-1
#         #             WWterm13_op = @view d2dr2_d2dr2_plus_4ηρ_by_rM[:, n+1]
#         #             WWterm13_analytical = chebyfwdnr(r -> WWterm13_fullfn(r, n))
#         #             @test WWterm13_op ≈ WWterm13_analytical rtol=5e-4
#         #         end
#         #     end

#         #     function WWterm14_fullfn(r, n)
#         #         T = chebyshevT(n)
#         #         F1 = r -> d2f(T ∘ r̄, r) + 4ηρ(r̄(r))/r * T(r̄(r))
#         #         F2 = r -> F1(r) / r^2
#         #         F2(r)
#         #     end

#         #     @testset "WWterm14_fullfn" begin
#         #         @testset for n in 0:nr-1
#         #             WWterm14_op = @view one_by_r2_d2dr2_plus_4ηρ_by_rM[:, n+1]
#         #             WWterm14_analytical = chebyfwdnr(r -> WWterm14_fullfn(r, n))
#         #             @test WWterm14_op ≈ WWterm14_analytical rtol=5e-4
#         #         end
#         #     end

#         #     function WWterm15fn(r, n)
#         #         T = chebyshevT(n)
#         #         F1 = r -> 1/r^2 * T(r̄(r))
#         #         d2f(F1, r)
#         #     end

#         #     @testset "WWterm15fn" begin
#         #         @testset for n in 0:nr-1
#         #             WWterm15_analytical = chebyfwdnr(r -> WWterm15fn(r, n))
#         #             WWterm15_op_2 = @view d2dr2_one_by_r2M_2[:, n+1]
#         #             @test WWterm15_op_2 ≈ WWterm15_analytical rtol=5e-4
#         #         end
#         #     end

#         #     function WWterm16fn(r, n)
#         #         T = chebyshevT(n)
#         #         1/r^4 * T(r̄(r))
#         #     end

#         #     @testset "WWterm16fn" begin
#         #         @testset for n in 0:nr-1
#         #             WWterm16_op = @view onebyr4_chebyM[:, n+1]
#         #             WWterm16_analytical = chebyfwdnr(r -> WWterm16fn(r, n))
#         #             @test WWterm16_op ≈ WWterm16_analytical rtol=1e-4
#         #         end
#         #     end
#         # end
#         @testset "third term" begin
#             function WWterm31_1fn(r, n)
#                 T = chebyshevT(n) ∘ r̄
#                 # F1 = DDr
#                 F1 = r -> df(T, r) + (ηρ ∘ r̄)(r) * T(r)
#                 F2 = r -> (ηρ ∘ r̄)(r) * (df(F1, r) - 2/r * F1(r))
#                 F2(r)
#             end

#             @testset "WWterm31_1fn" begin
#                 @testset for n in 0:nr-1
#                     WWterm31_1_op = @view ηρ_ddr_minus_2byr_DDrM[:, n+1]
#                     WWterm31_1_analytical = chebyfwdnr(r -> WWterm31_1fn(r, n))
#                     @test WWterm31_1_op ≈ WWterm31_1_analytical rtol=1e-4
#                 end
#             end

#             function WWterm31_2fn(r, n)
#                 F1 = r -> WWterm31_1fn(r, n)
#                 df(F1, r)
#             end

#             @testset "WWterm31_2fn" begin
#                 @testset for n in 0:nr-1
#                     WWterm31_2_op = @view ddr_ηρ_ddr_minus_2byr_DDrM[:, n+1]
#                     WWterm31_2_analytical = chebyfwdnr(r -> WWterm31_2fn(r, n))
#                     @test WWterm31_2_op ≈ WWterm31_2_analytical rtol=1e-4
#                 end
#             end

#             function WWterm32fn(r, n)
#                 T = chebyshevT(n) ∘ r̄
#                 # F1 = DDr
#                 F1 = r -> df(r -> (ηρ ∘ r̄)(r)/r^2 * T(r), r)
#                 F2 = r -> -2/r^2*(ηρ ∘ r̄)(r) * (df(T, r) - 2/r * T(r))
#                 F1(r) + F2(r)
#             end

#             @testset for n in 0:nr-1
#                 WWterm32_op = @view ddr_ηρ_by_r2_minus_2ηρ_by_r2_ddr_minus_2byrM[:, n+1]
#                 WWterm32_analytical = chebyfwdnr(r -> WWterm32fn(r, n))
#                 @test WWterm32_op ≈ WWterm32_analytical rtol=1e-3
#             end
#         end
#         @testset "fourth term" begin
#             function WWterm4fn(r, n)
#                 T = chebyshevT(n) ∘ r̄
#                 # F1 = DDr
#                 F2 = r -> (ηρ2_by_r2 ∘ r̄)(r) * T(r)
#                 F2(r)
#             end

#             @testset for n in 0:nr-1
#                 WWterm4op = @view ηρ2_by_r2M[:, n+1]
#                 WWterm4analytical = chebyfwdnr(r -> WWterm4fn(r, n))
#                 @test WWterm4op ≈ WWterm4analytical rtol=1e-7
#             end
#         end
#     end
# end

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
    @unpack BC = constraints;
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
            # @testset "eigenvector" begin
            #     vi = vuf[:, ind];
            #     (; VWSinv, θ) = RossbyWaveSpectrum.eigenfunction_realspace(vi, m, operators);
            #     nθ = length(θ)
            #     @unpack V = VWSinv;
            #     Vr = real(V);
            #     equator_ind = argmin(abs.(θ .- pi/2))
            #     Δθ_scan = div(nθ, 5)
            #     rangescan = intersect(equator_ind .+ (-Δθ_scan:Δθ_scan), axes(V, 2))
            #     ind_max = findmax(col -> maximum(real, col), eachcol(view(Vr, :, rangescan)))[2]
            #     ind_max += first(rangescan) - 1
            #     V_peak_depthprofile = @view Vr[:, ind_max];
            #     r_max_ind = argmax(abs.(V_peak_depthprofile))
            #     V_surf = @view Vr[r_max_ind, :];
            #     V_surf ./= maximum(abs, V_surf);
            #     @test abs.(V_surf) ≈ sin.(θ).^m rtol=0.15
            # end
        end
        vfn = zeros(eltype(vuf), size(vuf, 1));
        @testset "boundary condition" begin
            @testset for n in axes(vuf, 2)
                vfn .= @view vuf[:, n]
                @testset "V" begin
                    @testset for ℓind in 1:nℓ
                        ℓ_skip = (ℓind - 1)*nr
                        inds_ℓ = ℓ_skip .+ (1:nr)
                        v = vfn[inds_ℓ]
                        @test BC[1:2, inds_ℓ] * v ≈ [0, 0] atol=1e-10
                        pv = SpecialPolynomials.Chebyshev(v)
                        drpv = 1/(Δr/2) * SpecialPolynomials.derivative(pv)
                        @testset "inner boundary" begin
                            @test (drpv(-1) - 2pv(-1)/r_in)*Rsun ≈ 0 atol=1e-10
                        end
                        @testset "outer boundary" begin
                            @test (drpv(1) - 2pv(1)/r_out)*Rsun ≈ 0 atol=1e-10
                        end
                    end
                end
                @testset "W" begin
                    @testset for ℓind in 1:nℓ
                        ℓ_skip = (ℓind - 1)*nr
                        inds_ℓ = nr*nℓ + ℓ_skip .+ (1:nr)
                        w = vfn[inds_ℓ]
                        @test BC[3:4, inds_ℓ] * w ≈ [0, 0] atol=1e-10
                        pw = SpecialPolynomials.Chebyshev(w)
                        @testset "inner boundary" begin
                            @test sum(i -> (-1)^i * w[i], axes(w, 1)) ≈ 0 atol=1e-10
                            @test pw(-1) ≈ 0 atol=1e-10
                        end
                        @testset "outer boundary" begin
                            @test sum(w) ≈ 0 atol=1e-10
                            @test pw(1) ≈ 0 atol=1e-10
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
        @unpack transforms, diff_operators, rad_terms, coordinates, radial_params, identities = operators;
        @unpack r, r_chebyshev = coordinates;
        @unpack nvariables = operators;
        @unpack ν = operators.constants;
        r_mid = radial_params.r_mid::Float64;
        Δr = radial_params.Δr::Float64;
        a = 1 / (Δr / 2);
        b = -r_mid / (Δr / 2);
        r̄(r) = clamp(a * r + b, -1.0, 1.0)
        @unpack Tcrfwd = transforms;
        @unpack Iℓ = identities;
        @unpack ddr, d2dr2, DDr = diff_operators;
        @unpack onebyr, onebyr2, r2_cheby, r_cheby, ηρ = rad_terms;
        @unpack r_in, r_out = operators.radial_params;
        chebyfwdnr(f, scalefactor = 5) = chebyfwd(f, r_in, r_out, nr, scalefactor)

        ℓ = 2
        ℓℓp1 = ℓ*(ℓ+1)
        ℓ′ = 1
        cosθ21 = 1/√5
        sinθdθ21 = 1/√5
        ∇²_sinθdθ21 = -ℓℓp1 * sinθdθ21

        # # used to check the VV term
        # M = RossbyWaveSpectrum.allocate_matrix(operators);
        # RossbyWaveSpectrum._differential_rotation_matrix!(M, m, :constant; operators);

        # @testset for rotation_profile in [:constant, :linear, :solar_equator]

        #     (; ΔΩ, Ω0, ddrΔΩ, d2dr2ΔΩ) =
        #         RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives(
        #             m; operators, rotation_profile);

        #     ΔΩ_by_r = ΔΩ * onebyr
        #     ΔΩ_DDr = ΔΩ * DDr
        #     DDr_min_2byr = DDr - 2onebyr
        #     ΔΩ_DDr_min_2byr = ΔΩ * DDr_min_2byr
        #     ddrΔΩ_plus_ΔΩddr = ddrΔΩ + ΔΩ * ddr
        #     twoΔΩ_by_r = 2ΔΩ * onebyr

        #     VWterm, WVterm = zeros(nr,nr), zeros(nr,nr)
        #     RossbyWaveSpectrum.radial_differential_rotation_terms_inner!((VWterm, WVterm),
        #                 (ℓ, ℓ′), (cosθ21, sinθdθ21, ∇²_sinθdθ21),
        #                 (mat(ddrΔΩ), Ω0),
        #                 map(mat, (ΔΩ_by_r, ΔΩ_DDr, ΔΩ_DDr_min_2byr, ddrΔΩ_plus_ΔΩddr)))

        #     ΔΩ_by_r2 = ΔΩ * onebyr2
        #     two_ΔΩbyr_ηρ = twoΔΩ_by_r * ηρ
        #     ddrΔΩ_ddr_plus_2byr = ddrΔΩ * (ddr + 2onebyr)
        #     WWfixedterms = -two_ΔΩbyr_ηρ + d2dr2ΔΩ + ddrΔΩ_ddr_plus_2byr

        #     @testset "V terms" begin
        #         Vn1 = P11norm

        #         @testset "ddrΔΩ_plus_ΔΩddr" begin
        #             function ddrΔΩ_plus_ΔΩddr_term(r, n)
        #                 r̄_r = r̄(r)
        #                 Tn = chebyshevT(n, r̄_r)
        #                 Unm1 = chebyshevU(n-1, r̄_r)
        #                 ddrΔΩ(r̄_r) * Tn + ΔΩ(r̄_r) * a * n * Unm1
        #             end
        #             @testset for n in 1:nr-1
        #                 ddrΔΩ_plus_ΔΩddr_op = mat(ddrΔΩ_plus_ΔΩddr)[:, n + 1]
        #                 ddrΔΩ_plus_ΔΩddr_explicit = chebyfwdnr(r -> ddrΔΩ_plus_ΔΩddr_term(r, n))
        #                 @test ddrΔΩ_plus_ΔΩddr_op ≈ ddrΔΩ_plus_ΔΩddr_explicit rtol = 1e-8
        #             end
        #         end

        #         @testset "twoΔΩ_by_r" begin
        #             function twoΔΩ_by_r_term(r, n)
        #                 r̄_r = r̄(r)
        #                 Tn = chebyshevT(n, r̄_r)
        #                 2ΔΩ(r̄_r)/r * Tn
        #             end
        #             twoΔΩ_by_rM = mat(twoΔΩ_by_r)
        #             @testset for n in 1:nr-1
        #                 twoΔΩ_by_r_op = twoΔΩ_by_rM[:, n + 1]
        #                 twoΔΩ_by_r_explicit = chebyfwdnr(r -> twoΔΩ_by_r_term(r, n))
        #                 @test twoΔΩ_by_r_op ≈ twoΔΩ_by_r_explicit rtol = 1e-4
        #             end
        #         end

        #         @testset "WV term" begin
        #             function WVterms(r, n)
        #                 r̄_r = r̄(r)
        #                 Tn = chebyshevT(n, r̄_r)
        #                 Unm1 = chebyshevU(n-1, r̄_r)
        #                 2/√15 * (ΔΩ(r̄_r) * (a * n * Unm1 - 2/r * Tn)  + ddrΔΩ(r̄_r) * Tn)
        #             end
        #             @testset for n in 1:nr-1
        #                 WV_op = Vn1 * @view WVterm[:, n + 1]
        #                 WV_explicit = chebyfwdnr(r -> WVterms(r, n))
        #                 @test WV_op ≈ WV_explicit rtol = 1e-4
        #             end
        #         end
        #         @testset "VV term" begin
        #             VV11term = @view M[1:nr, 1:nr]
        #             @testset for n in 1:nr-1
        #                 VV_op = @view VV11term[:, n + 1]
        #                 @test all(iszero, VV_op)
        #             end
        #         end
        #     end
        #     @testset "W terms" begin
        #         Wn1 = P11norm
        #         @testset "VW term" begin
        #             function DDrmin2byr_term(r, n)
        #                 r̄_r = r̄(r)
        #                 Tn = chebyshevT(n, r̄_r)
        #                 Unm1 = chebyshevU(n-1, r̄_r)
        #                 ηr = ηρ(r̄_r)
        #                 a * n * Unm1 - 2/r * Tn + ηr * Tn
        #             end
        #             DDr_min_2byrM = mat(DDr_min_2byr)
        #             @testset for n in 1:nr-1
        #                 DDrmin2byr_op = @view DDr_min_2byrM[:, n + 1]
        #                 DDrmin2byr_explicit = chebyfwdnr(r -> DDrmin2byr_term(r, n))
        #                 @test DDrmin2byr_op ≈ DDrmin2byr_explicit rtol = 1e-4
        #             end

        #             function ddrΔΩ_term(r, n)
        #                 r̄_r = r̄(r)
        #                 Tn = chebyshevT(n, r̄_r)
        #                 ddrΔΩ(r̄_r) * Tn
        #             end
        #             ddrΔΩM = mat(ddrΔΩ)
        #             @testset for n in 1:nr-1
        #                 ddrΔΩ_op = @view ddrΔΩM[:, n + 1]
        #                 ddrΔΩ_explicit = chebyfwdnr(r -> ddrΔΩ_term(r, n))
        #                 @test ddrΔΩ_op ≈ ddrΔΩ_explicit rtol = 1e-4
        #             end

        #             function ΔΩDDr_min_2byr_min_ddrΔΩ_term(r, n)
        #                 r̄_r = r̄(r)
        #                 Tn = chebyshevT(n, r̄_r)
        #                 ΔΩ(r̄_r) * DDrmin2byr_term(r, n)  - ddrΔΩ(r̄_r) * Tn
        #             end
        #             ΔΩDDr_min_2byr_min_ddrΔΩM = mat(ΔΩ * DDr_min_2byr - ddrΔΩ)
        #             @testset for n in 1:nr-1
        #                 ΔΩDDr_min_2byr_min_ddrΔΩ_op = ΔΩDDr_min_2byr_min_ddrΔΩM[:, n + 1]
        #                 ΔΩDDr_min_2byr_min_ddrΔΩ_explicit = chebyfwdnr(r -> ΔΩDDr_min_2byr_min_ddrΔΩ_term(r, n))
        #                 @test ΔΩDDr_min_2byr_min_ddrΔΩ_op ≈ ΔΩDDr_min_2byr_min_ddrΔΩ_explicit rtol = 1e-4
        #             end

        #             VWterms(r, n) = -2/√15 * ΔΩDDr_min_2byr_min_ddrΔΩ_term(r, n)

        #             @testset for n in 1:nr-1
        #                 VW_op = Wn1 * @view VWterm[:, n + 1]
        #                 VW_explicit = chebyfwdnr(r -> VWterms(r, n))
        #                 @test VW_op ≈ -VW_explicit rtol = 1e-4
        #             end
        #         end
        #     end
        # end

        @testset "convergence of diff rot profile" begin
            nr, nℓ = 50, 10
            operators1 = RossbyWaveSpectrum.radial_operators(nr, nℓ);
            r_chebyshev1 = operators1.coordinates[:r_chebyshev];
            operators2 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ);
            operators3 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ+5);

            T = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives(
                operators = operators1, rotation_profile = :solar_equator);
            ΔΩ1, ddrΔΩ1, d2dr2ΔΩ1 = T.ΔΩ, T.ddrΔΩ, T.d2dr2ΔΩ;

            T = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives(
                operators = operators2, rotation_profile = :solar_equator);
            ΔΩ2, ddrΔΩ2, d2dr2ΔΩ2 = T.ΔΩ, T.ddrΔΩ, T.d2dr2ΔΩ;

            T = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives(
                operators = operators3, rotation_profile = :solar_equator);
            ΔΩ3, ddrΔΩ3, d2dr2ΔΩ3 = T.ΔΩ, T.ddrΔΩ, T.d2dr2ΔΩ;

            @testset "nr+5, nℓ" begin
                @test ΔΩ1.(r_chebyshev1) ≈ ΔΩ2.(r_chebyshev1) rtol=1e-2
                @test ddrΔΩ1.(r_chebyshev1) ≈ ddrΔΩ2.(r_chebyshev1) rtol=5e-2
                @test d2dr2ΔΩ1.(r_chebyshev1) ≈ d2dr2ΔΩ2.(r_chebyshev1) rtol=1e-1
            end

            @testset "nr+5, nℓ+5" begin
                @test ΔΩ1.(r_chebyshev1) ≈ ΔΩ3.(r_chebyshev1) rtol=1e-2
                @test ddrΔΩ1.(r_chebyshev1) ≈ ddrΔΩ3.(r_chebyshev1) rtol=5e-2
                @test d2dr2ΔΩ1.(r_chebyshev1) ≈ d2dr2ΔΩ3.(r_chebyshev1) rtol=1e-1
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

                        rtol = rowind == colind == 2 ? 5e-2 : 1e-2

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

                        rtol = rowind == colind == 2 ? 5e-2 : 1e-2

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

        # @testset "S terms" begin
        #     nr, nℓ = 50, 10
        #     m = 5

        #     operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
        #     @unpack Δr = operators.radial_params;
        #     @unpack r_chebyshev = operators.coordinates;
        #     (; g, ηρ) = operators.rad_terms;
        #     @unpack DDr = operators.diff_operators;
        #     @unpack matCU2 = operators;

        #     @testset for rotation_profile in [:linear, :solar_equator]
        #         ΔΩprofile_deriv =
        #             RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives(
        #                 m; operators, rotation_profile);

        #         (; ΔΩ, Ω0, ddrΔΩ, d2dr2ΔΩ) = ΔΩprofile_deriv;

        #         ddrΔΩ_over_g = ddrΔΩ / g;
        #         ddrΔΩ_over_gMCU2 = matCU2(ddrΔΩ_over_g);
        #         ddrΔΩ_over_g_DDr = (ddrΔΩ_over_g * DDr);
        #         ddrΔΩ_over_g_DDrMCU2 = matCU2(ddrΔΩ_over_g_DDr);

        #         @testset "ddrΔΩ_over_g" begin
        #             @testset for n in 0:5:nr-1
        #                 T = chebyshevT(n)
        #                 f = x -> ddrΔΩ(x)/g(x) * T(x)

        #                 fc = chebyfwd(f, nr);

        #                 @test fc ≈ @view(ddrΔΩ_over_gMCU2[:, n+1]) rtol=1e-3
        #             end
        #         end

        #         @testset "ddrΔΩ_over_g * DDr" begin
        #             @testset for n in 0:5:nr-1
        #                 T = chebyshevT(n)
        #                 f = x -> ddrΔΩ_over_g(x) * (df(T, x) * (2/Δr) + ηρ(x) * T(x))

        #                 fc = chebyfwd(f, nr);

        #                 @test fc ≈ @view(ddrΔΩ_over_g_DDrMCU2[:, n+1]) rtol=1e-3
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
    @unpack BC = constraints;
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
                @testset for n in axes(vrf, 2)
                    vfn .= @view vrf[:, n]
                    @testset "V" begin
                        @testset for ℓind in 1:nℓ
                            ℓ_skip = (ℓind - 1)*nr
                            inds_ℓ = ℓ_skip .+ (1:nr)
                            v = vfn[inds_ℓ]
                            @test BC[1:2, inds_ℓ] * v ≈ [0, 0] atol=1e-10
                            pv = SpecialPolynomials.Chebyshev(v)
                            drpv = 1/(Δr/2) * SpecialPolynomials.derivative(pv)
                            @testset "inner boundary" begin
                                @test (drpv(-1) - 2pv(-1)/r_in)*Rsun ≈ 0 atol=1e-10
                            end
                            @testset "outer boundary" begin
                                @test (drpv(1) - 2pv(1)/r_out)*Rsun ≈ 0 atol=1e-10
                            end
                        end
                    end
                    @testset "W" begin
                        @testset for ℓind in 1:nℓ
                            ℓ_skip = (ℓind - 1)*nr
                            inds_ℓ = nr*nℓ + ℓ_skip .+ (1:nr)
                            w = vfn[inds_ℓ]
                            @test BC[3:4, inds_ℓ] * w ≈ [0, 0] atol=1e-10
                            pw = SpecialPolynomials.Chebyshev(w)
                            @testset "inner boundary" begin
                                @test sum(i -> (-1)^i * w[i], axes(w, 1)) ≈ 0 atol=1e-10
                                @test pw(-1) ≈ 0 atol=1e-10
                            end
                            @testset "outer boundary" begin
                                @test sum(w) ≈ 0 atol=1e-10
                                @test pw(1) ≈ 0 atol=1e-10
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
                @testset for n in axes(vrf, 2)
                    vfn .= @view vrf[:, n]
                    @testset "V" begin
                        @testset for ℓind in 1:nℓ
                            ℓ_skip = (ℓind - 1)*nr
                            inds_ℓ = ℓ_skip .+ (1:nr)
                            v = vfn[inds_ℓ]
                            @test BC[1:2, inds_ℓ] * v ≈ [0, 0] atol=1e-10
                            pv = SpecialPolynomials.Chebyshev(v)
                            drpv = 1/(Δr/2) * SpecialPolynomials.derivative(pv)
                            @testset "inner boundary" begin
                                @test (drpv(-1) - 2pv(-1)/r_in)*Rsun ≈ 0 atol=1e-10
                            end
                            @testset "outer boundary" begin
                                @test (drpv(1) - 2pv(1)/r_out)*Rsun ≈ 0 atol=1e-10
                            end
                        end
                    end
                    @testset "W" begin
                        @testset for ℓind in 1:nℓ
                            ℓ_skip = (ℓind - 1)*nr
                            inds_ℓ = nr*nℓ + ℓ_skip .+ (1:nr)
                            w = vfn[inds_ℓ]
                            @test BC[3:4, inds_ℓ] * w ≈ [0, 0] atol=1e-10
                            pw = SpecialPolynomials.Chebyshev(w)
                            @testset "inner boundary" begin
                                @test sum(i -> (-1)^i * w[i], axes(w, 1)) ≈ 0 atol=1e-10
                                @test pw(-1) ≈ 0 atol=1e-10
                            end
                            @testset "outer boundary" begin
                                @test sum(w) ≈ 0 atol=1e-10
                                @test pw(1) ≈ 0 atol=1e-10
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
    for V_symmetric in (true, false), diffrot in (true, false)
        ComputeRossbySpectrum.computespectrum(8, 6, 1:1, V_symmetric, diffrot, :radial_solar_equator, save = false)
    end
    @testset "filteredeigen" begin
        nr, nℓ = 8,6
        V_symmetric = true
        diffrotprof = :radial_solar_equator
        @testset for diffrot in (true, false)
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
