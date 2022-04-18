using RossbyWaveSpectrum: RossbyWaveSpectrum, matrix_block, Rsun, chebyshevmatrix
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

@testset "project quality" begin
    Aqua.test_all(RossbyWaveSpectrum,
        ambiguities = false,
        stale_deps = (; ignore =
        [:PyCall, :PyPlot, :LaTeXStrings]),
    )
end

@testset "operators inferred" begin
    @inferred RossbyWaveSpectrum.radial_operators(5, 2)
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
const ChebyshevT = SpecialPolynomials.Chebyshev
chebyshevT(n) = n < 0 ? ChebyshevT([0.0]) : ChebyshevT([zeros(n); 1])
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
    Tcf, Tci = RossbyWaveSpectrum.chebyshev_lobatto_forward_inverse(n)
    @test Tcf * Tci ≈ Tci * Tcf ≈ I
    r_chebyshev = RossbyWaveSpectrum.chebyshevnodes_lobatto(n)
    @test Tcf * r_chebyshev ≈ [0; 1; zeros(length(r_chebyshev)-2)]

    r_chebyshev, Tcf, Tci = RossbyWaveSpectrum.chebyshev_forward_inverse(n)
    @test Tcf * Tci ≈ Tci * Tcf ≈ I
    @test Tcf * r_chebyshev ≈ [0; 1; zeros(length(r_chebyshev)-2)]
end

@testset "green fn W" begin
    nr, nℓ = 50, 2

    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
    operators_unstratified = RossbyWaveSpectrum.radial_operators(nr, nℓ, _stratified = false);
    (; r_chebyshev_lobatto, r_lobatto, r_chebyshev) = operators.coordinates;
    (; Δr, r_mid) = operators.radial_params;
    (; d2dr2, ddr, DDr, d4dr4) = operators.diff_operators;
    (; ddr_lobatto, ddrM, DDrM, d2dr2M, d3dr3M, d4dr4M) = operators.diff_operator_matrices;
    (; deltafn_matrix_radial) = operators;
    (; ηρ_cheby, r_cheby, ηρ_by_r, ηρ2_by_r2, ηρ_by_r2,
        onebyr2_cheby, onebyr_cheby, onebyr3_cheby, onebyr4_cheby,
        ddr_ηρ, d2dr2_ηρ) = operators.rad_terms;
    (; g_cheby, ddr_ηρbyr2, ddr_ηρbyr, ηρ_by_r3, d3dr3_ηρ) = operators.rad_terms;
    (; TfGL_nr, TiGL_nr) = operators.transforms;
    (; mat) = operators;
    d2dr2_one_by_r2_2M = mat(d2dr2[onebyr2_cheby]);
    onebyr_ddrM = mat(onebyr_cheby * ddr);
    ddr_ηρM = mat(ddr[ηρ_cheby]);
    onebyr3_chebyM = mat(onebyr3_cheby);
    onebyr4_chebyM = mat(onebyr4_cheby);
    onebyr3_ddrM = mat(onebyr3_cheby*ddr);
    ηρ_by_r3M = mat(ηρ_by_r * onebyr2_cheby);

    function greenfn_radial_lobatto_unstratified_analytical(ℓ, operators)
        (; n_lobatto) = operators.transforms;
        (; r_in, r_out, Δr) = operators.radial_params;
        (; r_lobatto) = operators.coordinates;
        r_in_frac = r_in/Rsun;
        r_out_frac = r_out/Rsun;
        W = (2ℓ+1)*((r_out_frac)^(2ℓ+1) - (r_in_frac)^(2ℓ+1))
        norm = 1/W * (Δr/2) / Rsun
        H = zeros(n_lobatto+1, n_lobatto+1);

        for rind in axes(r_lobatto, 1)[2:end-1]
            ri = r_lobatto[rind]/Rsun
            for sind in axes(r_lobatto, 1)[2]:rind
                sj = r_lobatto[sind]/Rsun
                H[rind, sind] = (sj^(ℓ+1) - r_in_frac^(2ℓ+1)/sj^ℓ)*(ri^(ℓ+1) - r_out_frac^(2ℓ+1)/ri^ℓ) * norm
            end
            for sind in rind + 1:axes(r_lobatto, 1)[end-1]
                sj = r_lobatto[sind]/Rsun
                H[rind, sind] = (sj^(ℓ+1) - r_out_frac^(2ℓ+1)/sj^ℓ)*(ri^(ℓ+1) - r_in_frac^(2ℓ+1)/ri^ℓ) * norm
            end
        end

        @test H ≈ Symmetric(H)
        Symmetric(H, :L)
    end

    @testset for ℓ in 2:5:42
        (; sρ) = operators.rad_terms;

        Hℓ = RossbyWaveSpectrum.greenfn_radial_lobatto(ℓ, operators)
        @testset "boundaries" begin
            @test all(x -> isapprox(x/((Δr/2)/Rsun), 0, atol=1e-14), @view Hℓ[1, :])
            @test all(x -> isapprox(x/((Δr/2)/Rsun), 0, atol=1e-14), @view Hℓ[end, :])
            @test all(x -> isapprox(x/((Δr/2)/Rsun), 0, atol=1e-14), @view Hℓ[:, 1])
            @test all(x -> isapprox(x/((Δr/2)/Rsun), 0, atol=1e-14), @view Hℓ[:, end])
        end

        @testset "symmetry" begin
            H2 = Hℓ .* sρ.(r_lobatto)
            @test H2 ≈ Symmetric(H2) rtol=1e-2
        end

        @testset "unstratified" begin
            # in this case there's an analytical solution
            Hℓ = RossbyWaveSpectrum.greenfn_radial_lobatto(ℓ, operators_unstratified)
            J_exp = greenfn_radial_lobatto_unstratified_analytical(ℓ, operators_unstratified)
            @test Hℓ ≈ J_exp rtol=1e-2
            @testset "symmetry" begin
                @test Hℓ ≈ Symmetric(Hℓ) rtol=1e-2
            end
            @testset "differential equation" begin
                B, scale = RossbyWaveSpectrum.Bℓ(ℓ, operators_unstratified)
                B .*= scale
                H2 = Hℓ / (Δr/2)
                @testset "in first coordinate" begin
                    @test isapprox(B * H2, deltafn_matrix_radial, rtol=1e-2)
                end
            end
        end
    end

    Nquad = 1000;
    nodesquad = gausschebyshev(Nquad)[1];

    integral_cache = zeros(length(r_chebyshev_lobatto));

    J_c1 = zeros(nr, nr);
    J_c2_1 = zeros(nr, nr);
    J_c2_2 = zeros(nr, nr);
    J_a1_1 = zeros(nr, nr);
    J_a1_2 = zeros(nr, nr);
    J_a1_3 = zeros(nr, nr);
    J_4ηρbyr3 = zeros(nr, nr);
    J_ηρ = zeros(nr, nr);
    J_ηρ²_min_2ηρbyr = zeros(nr, nr);
    J_ηρ_min_2byr_ddrηρ = zeros(nr, nr);
    J_ddrηρ = zeros(nr, nr);
    J_ddrηρbyr2_plus_4ηρbyr3 = zeros(nr, nr);
    J_ηρ2byr2 = zeros(nr, nr);
    J_ηρbyr2 = zeros(nr, nr);
    J_d2dr2ηρ_by_r_min_2ddrηρ_by_r2 = zeros(nr, nr);
    J_ddrηρ_by_r = zeros(nr, nr);
    J_ddrηρ_by_r_min_ηρbyr2 = zeros(nr, nr);
    J_ddrηρ_by_r2_min_4ηρbyr3 = zeros(nr, nr);
    J_ddrηρ_by_r2 = zeros(nr, nr);
    J_d2dr2ηρ_by_r = zeros(nr, nr);

    viscosity_terms = (;
        J_c1,
        J_c2_1,
        J_c2_2,
        J_a1_1,
        J_a1_2,
        J_a1_3,
        J_4ηρbyr3,
        J_ηρ,
        J_ηρ²_min_2ηρbyr,
        J_ηρ_min_2byr_ddrηρ,
        J_ddrηρ,
        J_ddrηρbyr2_plus_4ηρbyr3,
        J_ηρ2byr2,
        J_ηρbyr2,
        J_d2dr2ηρ_by_r_min_2ddrηρ_by_r2,
        J_ddrηρ_by_r,
        J_ddrηρ_by_r_min_ηρbyr2,
        J_ddrηρ_by_r2_min_4ηρbyr3,
        J_ddrηρ_by_r2,
        J_d2dr2ηρ_by_r,
    );

    funs = RossbyWaveSpectrum.viscosity_functions(operators);

    @testset "integrals" begin
        @testset for ℓ in 2:10:52
            ℓℓp1 = ℓ*(ℓ+1)

            Jrr′ = RossbyWaveSpectrum.greenfn_radial_lobatto(ℓ, operators);
            Tu = RossbyWaveSpectrum.greenfn_cheby(RossbyWaveSpectrum.UniformRotGfn(), ℓ, operators);
            (; J, J_ηρbyr, J_by_r) = Tu.unirot_terms;
            Tv = RossbyWaveSpectrum.greenfn_cheby!(RossbyWaveSpectrum.ViscosityGfn(), ℓ,
                    operators, viscosity_terms, funs, Tu);

            J_splines = [Spline1D(r_chebyshev_lobatto, @view Jrr′[r_ind, :]) for r_ind in axes(Jrr′, 1)];
            intJ_rp_r = [pi/Nquad .* sqrt.(1 .- nodesquad.^2) .* Jsp(nodesquad) for Jsp in J_splines];
            function intJ_quadgc(f)
                integral_cache .= 0
                for (rind, Ji) in enumerate(intJ_rp_r)
                    integral_cache[rind] = sum(((Jinode,node), ) -> Jinode * f(node), zip(Ji, nodesquad))
                end
                integral_cache
            end

            @testset "J" begin
                @testset for n = 0:5:nr-1
                    f = chebyshevT(n);

                    intres = intJ_quadgc(f)
                    intres = TfGL_nr * intres;

                    Jf = @view J[:, n+1];
                    @test Jf ≈ intres rtol=1e-3
                end
            end

            @testset "J/r" begin
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> T(x)/r_cheby(x)

                    intres = intJ_quadgc(f)
                    intresc = TfGL_nr * intres

                    Xn = @view J_by_r[:, n+1]
                    @test Xn ≈ intresc rtol=1e-3
                end
            end

            @testset "J/r^4" begin
                J_by_r4_c = J * onebyr4_chebyM
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> T(x)/r_cheby(x)^4

                    intres = intJ_quadgc(f)
                    intresc = TfGL_nr * intres

                    Xn = @view J_by_r4_c[:, n+1]
                    @test Xn ≈ intresc rtol=1e-3
                end
            end

            @testset "J*4ηρ/r^3" begin
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> T(x)*4ηρ_cheby(x)/r_cheby(x)^3

                    intres = intJ_quadgc(f)
                    intresc = TfGL_nr * intres

                    Xn = @view J_4ηρbyr3[:, n+1]
                    @test Xn ≈ intresc rtol=1e-3
                end
            end

            @testset "J*(∂2rηρ/r - 2∂rηρ/r^2)" begin
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> T(x)*(d2dr2_ηρ(x)/r_cheby(x) - 2ddr_ηρ(x)/r_cheby(x)^2)

                    intres = intJ_quadgc(f)
                    intresc = TfGL_nr * intres

                    Xn = @view J_d2dr2ηρ_by_r_min_2ddrηρ_by_r2[:, n+1]
                    @test Xn ≈ intresc rtol=1e-3
                end
            end

            @testset "J/r^3*ddr" begin
                J_by_r3_ddr_c = J * onebyr3_ddrM;
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> df(T, x) * (2/Δr) / r_cheby(x)^3

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres;

                    Xn = @view J_by_r3_ddr_c[:, n+1];
                    @test Xn ≈ intresc rtol=2e-3
                end
            end

            @testset "J/r^2*d2dr2" begin
                J_by_r2_d2dr2_c = J * mat(onebyr2_cheby*d2dr2)
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> d2f(T, x) * (2/Δr)^2 / r_cheby(x)^2

                    intres = intJ_quadgc(f)
                    intresc = TfGL_nr * intres;

                    Xn = @view J_by_r2_d2dr2_c[:, n+1]
                    @test Xn ≈ intresc rtol=2e-3
                end
            end

            @testset "J(d²dr² - ℓℓp1/r²)²" begin
                ℓpre = (ℓ-2)*ℓ*(ℓ+1)*(ℓ+3)
                Jopc2 = J * mat(d4dr4 + ℓpre*onebyr4_cheby - 2ℓℓp1*onebyr2_cheby*d2dr2 + 4ℓℓp1*onebyr3_cheby*ddr);
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    op = x -> d2f(T, x) * (2/Δr)^2  - ℓℓp1/r_cheby(x)^2 * T(x)
                    f = x -> d2f(op, x) * (2/Δr)^2  - ℓℓp1/r_cheby(x)^2 * op(x)

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres;

                    Xn = @view Jopc2[:, n+1];
                    @test Xn ≈ intresc rtol=1e-3 atol=1e-12/Rsun^4
                end
            end

            @testset "J((d²dr² - ℓℓp1/r²)² - d⁴dr⁴)" begin
                ℓpre = (ℓ-2)*ℓ*(ℓ+1)*(ℓ+3)
                Jopc2 = J * mat(ℓpre*onebyr4_cheby - 2ℓℓp1*onebyr2_cheby*d2dr2 + 4ℓℓp1*onebyr3_cheby*ddr);
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> - 2ℓℓp1* d2f(T, x) * (2/Δr)^2 / r_cheby(x)^2  + ℓpre/r_cheby(x)^4 * T(x) +
                        4ℓℓp1* df(T, x) * (2/Δr) / r_cheby(x)^3

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres;

                    Xn = @view Jopc2[:, n+1];
                    if n <= 35
                        @test Xn ≈ intresc rtol=1e-3 atol=1e-12/Rsun^4
                    else
                        @test Xn ≈ intresc rtol=2e-3 atol=1e-12/Rsun^4
                    end
                end
            end

            @testset "J*ηρ/r" begin
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> T(x) *  ηρ_by_r(x)

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres

                    Xn = @view J_ηρbyr[:, n+1]
                    @test Xn ≈ intresc rtol=1e-2
                end
            end

            @testset "J*ηρ²/r²" begin
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> T(x) * ηρ2_by_r2(x)

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres;

                    Xn = @view J_ηρ2byr2[:, n+1]
                    if n <= 40
                        @test Xn ≈ intresc rtol=1e-2
                    else
                        @test Xn ≈ intresc rtol=5e-2
                    end
                end
            end

            @testset "J*ddr" begin
                Jddrc = J * ddrM
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> df(T, x) * (2/Δr)

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres

                    Jddrc_n = @view Jddrc[:, n+1]
                    if n <= 20
                        @test Jddrc_n ≈ intresc rtol=1e-4
                    elseif n <= 40
                        @test Jddrc_n ≈ intresc rtol=1e-3
                    else
                        @test Jddrc_n ≈ intresc rtol=5e-3
                    end
                end
            end

            @testset "J*1/r*ddr" begin
                Jddr_by_rc = J * onebyr_ddrM;
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> (df(T, x) * (2/Δr))/r_cheby(x)

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres

                    Xn = @view Jddr_by_rc[:, n+1]
                    if n <= 20
                        @test Xn ≈ intresc rtol=1e-4
                    elseif n <= 40
                        @test Xn ≈ intresc rtol=1e-3
                    else
                        @test Xn ≈ intresc rtol=5e-3
                    end
                end
            end

            @testset "J*DDr" begin
                JDDrc = J * DDrM
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> df(T, x) * (2/Δr) + ηρ_cheby(x) * T(x)

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres

                    Xn = @view JDDrc[:, n+1]
                    if n <= 35
                        @test Xn ≈ intresc rtol=1e-2
                    else
                        @test Xn ≈ intresc rtol=2e-1
                    end
                end
            end

            @testset "J * d2dr2" begin
                Jd2dr2c = J * d2dr2M;
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> d2f(T, x) * (2/Δr)^2

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres

                    Xn = @view Jd2dr2c[:, n+1]
                    if n < 20
                        @test Xn ≈ intresc rtol=1e-4
                    elseif n <= 40
                        @test Xn ≈ intresc rtol=1e-3
                    else
                        @test Xn ≈ intresc rtol=5e-3
                    end
                end
            end

            @testset "J * d3dr3" begin
                Jd3dr3c = J * d3dr3M;
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> d3f(T, x) * (2/Δr)^3

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres

                    Xn = @view Jd3dr3c[:, n+1]
                    if n < 20
                        @test Xn ≈ intresc rtol=1e-4
                    elseif n <= 40
                        @test Xn ≈ intresc rtol=1e-3
                    else
                        @test Xn ≈ intresc rtol=5e-3
                    end
                end
            end

            @testset "J * d4dr4" begin
                Jd4dr4c = J * d4dr4M;
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> d4f(T, x) * (2/Δr)^4

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres

                    Xn = @view Jd4dr4c[:, n+1]
                    if n < 20
                        @test Xn ≈ intresc rtol=1e-4
                    elseif n <= 40
                        @test Xn ≈ intresc rtol=1e-3
                    else
                        @test Xn ≈ intresc rtol=5e-3
                    end
                end
            end

            @testset "J * ddrηρ * d2dr2" begin
                J_ddrηρ_d2dr2c = J_ddrηρ * d2dr2M;
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> ddr_ηρ(x) * d2f(T, x) * (2/Δr)^2

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres

                    Xn = @view J_ddrηρ_d2dr2c[:, n+1]
                    if n < 20
                        @test Xn ≈ intresc rtol=1e-4
                    elseif n <= 40
                        @test Xn ≈ intresc rtol=1e-3
                    else
                        @test Xn ≈ intresc rtol=5e-3
                    end
                end
            end

            @testset "J * ηρ/r * d2dr2" begin
                J_ηρbyr_d2dr2c = J_ηρbyr * d2dr2M;
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> ηρ_by_r(x) * d2f(T, x) * (2/Δr)^2

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres

                    Xn = @view J_ηρbyr_d2dr2c[:, n+1]
                    if n < 20
                        @test Xn ≈ intresc rtol=1e-4
                    elseif n <= 35
                        @test Xn ≈ intresc rtol=1e-3
                    else
                        @test Xn ≈ intresc rtol=5e-3
                    end
                end
            end

            @testset "J * ηρ * d3dr3" begin
                J_ηρ_d3dr3 = J_ηρ * d3dr3M;
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> ηρ_cheby(x) * d3f(T, x) * (2/Δr)^3

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres

                    Xn = @view J_ηρ_d3dr3[:, n+1]
                    if n < 20
                        @test Xn ≈ intresc rtol=1e-4
                    else
                        @test Xn ≈ intresc rtol=1e-3
                    end
                end
            end

            @testset "J * ηρ/r^2 * ddr" begin
                J_ηρbyr2_ddr = J_ηρbyr2 * ddrM;
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> ηρ_by_r2(x) * df(T, x) * (2/Δr)

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres

                    Xn = @view J_ηρbyr2_ddr[:, n+1]
                    if n <= 25
                        @test Xn ≈ intresc rtol=1e-3
                    else
                        @test Xn ≈ intresc rtol=5e-3
                    end
                end
            end

            @testset "J * (∂rηρ/r - ηρ/r^2) * ddr" begin
                J_ddrηρ_by_r_min_ηρbyr2_ddr = J_ddrηρ_by_r_min_ηρbyr2 * ddrM;
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> (ddr_ηρ(x)/r_cheby(x) - ηρ_by_r(x)/r_cheby(x)^2) * df(T, x) * (2/Δr)

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres;

                    Xn = @view J_ddrηρ_by_r_min_ηρbyr2_ddr[:, n+1]
                    @test Xn ≈ intresc rtol=2e-2
                end
            end

            @testset "J * (∂rηρ/r^2 - 4ηρ/r^3)" begin
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> (ddr_ηρ(x)/r_cheby(x)^2 - 4ηρ_cheby(x)/r_cheby(x)^3) * T(x)

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres;

                    Xn = @view J_ddrηρ_by_r2_min_4ηρbyr3[:, n+1]
                    @test Xn ≈ intresc rtol=1e-3
                end
            end

            @testset "J * ηρ * d3dr3" begin
                J_ηρ_d3dr3c = J_ηρ * d3dr3M;
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> ηρ_cheby(x) * d3f(T, x) * (2/Δr)^3

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres

                    Xn = @view J_ηρ_d3dr3c[:, n+1]
                    if n < 20
                        @test Xn ≈ intresc rtol=1e-4
                    elseif n <= 40
                        @test Xn ≈ intresc rtol=1e-3
                    else
                        @test Xn ≈ intresc rtol=5e-3
                    end
                end
            end

            @testset "J * d2dr2_onebyr2" begin
                # Jd2dr2byr2c = J * d2dr2_one_by_r2M;
                Jd2dr2_one_by_r2_2 = J * d2dr2_one_by_r2_2M;
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> d2f(x -> T(x) * onebyr2_cheby(x), x) * (2/Δr)^2

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres;

                    Xn = @view Jd2dr2_one_by_r2_2[:, n+1];
                    @test Xn ≈ intresc rtol=1e-2
                end
            end

            @testset "J * ddr_ηρ_by_r2_minus_2ηρ_by_r2_ddr_minus_2byr" begin
                J_ddr_ηρ_by_r2_minus_2ηρ_by_r2_ddr_minus_2byr_c = -J_ηρbyr2 * ddrM .+ J_ddrηρbyr2_plus_4ηρbyr3;
                @testset for n = 0:5:nr-1
                    T = chebyshevT(n)
                    f = x -> df(x -> ηρ_by_r2(x) * T(x), x) * (2/Δr) -
                        2ηρ_cheby(x)/r_cheby(x)^2 * (df(T, x) * (2/Δr) -2T(x)/r_cheby(x))

                    intres = intJ_quadgc(f);
                    intresc = TfGL_nr * intres;

                    Xn = @view J_ddr_ηρ_by_r2_minus_2ηρ_by_r2_ddr_minus_2byr_c[:, n+1];
                    if n <= 40
                        @test Xn ≈ intresc rtol=3e-3
                    else
                        @test Xn ≈ intresc rtol=6e-3
                    end
                end
            end
        end
    end
end

@testset "uniform rotation" begin
    nr, nℓ = 50, 2
    nparams = nr * nℓ
    m = 1
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
    (; transforms, diff_operators, rad_terms, coordinates, radial_params, identities) = operators;
    (; r, r_chebyshev) = coordinates;
    (; nvariables, ν, Ω0) = operators.constants;
    r_mid = radial_params.r_mid::Float64;
    Δr = radial_params.Δr::Float64;
    a = 1 / (Δr / 2);
    b = -r_mid / (Δr / 2);
    r̄(r) = clamp(a * r + b, -1.0, 1.0);

    (; Tcrfwd) = operators.transforms;

    (; ddr, d2dr2, DDr) = operators.diff_operators;

    (; DDrM, ddrDDrM, onebyr2_chebyM, ddrM, onebyr_chebyM, gM,
        κ_∇r2_plus_ddr_lnρT_ddrM, onebyr2_cheby_ddr_S0_by_cpM,
        onebyr2_IplusrηρM) = operators.diff_operator_matrices;

    (; onebyr_cheby, onebyr2_cheby, r2_cheby, r_cheby, ηρ_cheby, ηT_cheby,
            g_cheby, ddr_lnρT, κ, ddr_S0_by_cp) = operators.rad_terms;

    (; r_in, r_out, nchebyr) = operators.radial_params;

    (; mat) = operators;
    (; Wscaling, Sscaling) = operators.constants.scalings;

    @test isapprox(onebyr_cheby(-1), 1/r_in, rtol=1e-4)
    @test isapprox(onebyr_cheby(1), 1/r_out, rtol=1e-4)
    @test isapprox(onebyr2_cheby(-1), 1/r_in^2, rtol=1e-4)
    @test isapprox(onebyr2_cheby(1), 1/r_out^2, rtol=1e-4)

    κ_by_r2M = κ * onebyr2_chebyM;

    ddrηρ = ddr * ηρ_cheby;

    M = RossbyWaveSpectrum.uniform_rotation_matrix(nr, nℓ, m; operators);

    chebyfwdnr(f, scalefactor = 5) = chebyfwd(f, r_in, r_out, nr, scalefactor)

    ℓ′ = 1
    # for these terms ℓ = ℓ′ (= 1 in this case)
    SWterm, SSterm = (zeros(nr, nr) for i in 1:3);
    RossbyWaveSpectrum.uniform_rotation_matrix_terms_outer!((SWterm, SSterm),
                        (ℓ′, m), nchebyr,
                        (ddrDDrM, onebyr2_IplusrηρM, gM,
                            κ_∇r2_plus_ddr_lnρT_ddrM, κ_by_r2M, onebyr2_cheby_ddr_S0_by_cpM), Ω0);

    @testset "V terms" begin
        @testset "WV term" begin
            function ddr_term(r, n)
                r̄_r = r̄(r)
                Unm1 = chebyshevU(n-1, r̄_r)
                a * n * Unm1
            end

            @testset "ddr" begin
                @testset for n in 1:nr - 1
                    ddr_Tn_analytical = chebyfwdnr(r -> ddr_term(r,n))
                    @test ddrM[:, n+1] ≈ ddr_Tn_analytical rtol=1e-8
                end
            end

            function onebyrTn_term(r, n)
                r̄_r = r̄(r)
                Tn = chebyshevT(n, r̄_r)
                1/r * Tn
            end

            @testset "onebyrTn" begin
                @testset for n in 1:nr - 1
                    onebyr_Tn_analytical = chebyfwdnr(r -> onebyrTn_term(r,n))
                    @test onebyr_chebyM[:, n+1] ≈ onebyr_Tn_analytical rtol=1e-4
                end
            end

            Vn1 = P11norm
            function WVtermfn(r, n)
                r̄_r = r̄(r)
                Tn = chebyshevT(n, r̄_r)
                2√(1/15) * (ddr_term(r, n) - 2onebyrTn_term(r, n)) / Rsun
            end

            ℓ = 2
            ℓ′ = 1
            m = 1
            ℓℓp1 = ℓ*(ℓ+1)
            ℓ′ℓ′p1 = ℓ′*(ℓ′+1)

            Cℓ′ = ddrM - ℓ′ℓ′p1 * onebyr_chebyM
            C1 = ddrM - 2onebyr_chebyM
            WV = -2 / ℓℓp1 * (ℓ′ℓ′p1 * C1 * (1/√5) + Cℓ′ * (1/√5)) / Rsun

            @testset for n in 1:nr-1
                WV_op_times_W = Vn1 * real(@view WV[:, n+1])
                WV_times_W_explicit = chebyfwdnr(r -> WVtermfn(r,n))
                @test WV_op_times_W ≈ WV_times_W_explicit rtol = 1e-5
            end
        end
    end

    @testset "W terms" begin
        Wn1 = P11norm

        @testset "VW term" begin
            function Drρ_fn(r, n)
                r̄_r = r̄(r)
                Tn = chebyshevT(n, r̄_r)
                Unm1 = chebyshevU(n-1, r̄_r)
                a * n * Unm1 + ηρ_cheby(r̄_r) * Tn
            end

            @testset "Drρ" begin
                @testset for n in 1:nr - 1
                    Drρ_Tn_analytical = chebyfwdnr(r -> Drρ_fn(r,n))
                    @test DDrM[:, n+1] ≈ Drρ_Tn_analytical rtol=1e-8
                end
            end

            function VWterm_fn(r, n)
                r̄_r = r̄(r)
                Tn = chebyshevT(n, r̄_r)
                -2√(1/15) * (Drρ_fn(r, n) - 2/r * Tn) * Rsun / Wscaling
            end

            VW = RossbyWaveSpectrum.matrix_block(M, 1, 2, nvariables)

            W1_inds = nr .+ (1:nr)

            @testset for n in 1:nr-1
                VW_op_times_W = Wn1 * real(@view VW[W1_inds, n+1])
                VW_times_W_explicit = chebyfwdnr(r -> VWterm_fn(r, n))
                @test VW_op_times_W ≈ -VW_times_W_explicit rtol = 1e-6
            end
        end

        @testset "WW term" begin
            function ddrDrρTn_fn(r, n)
                r̄_r = r̄(r)
                Tn = chebyshevT(n, r̄_r)
                Unm1 = chebyshevU(n-1, r̄_r)
                ηr = ηρ_cheby(r̄_r)
                η′r = ddrηρ(r̄_r)
                T1 = a * n * Unm1 * ηr + Tn * η′r
                if n > 1
                    Unm2 = chebyshevU(n-2, r̄_r)
                    return a^2 * n * (-n*Unm2 + (n-1)*Unm1*r̄_r)/(r̄_r^2 - 1) + T1
                elseif n == 1
                    return T1
                else
                    error("invalid n")
                end
            end

            @testset "ddrDrρTn" begin
                @testset for n in 1:nr - 1
                    ddrDrρ_Tn_analytical = chebyfwdnr(r -> ddrDrρTn_fn(r, n))
                    @test ddrDDrM[:, n+1] ≈ ddrDrρ_Tn_analytical rtol=1e-8
                end
            end

            function onebyr2Tn_fn(r, n)
                r̄_r = r̄(r)
                Tn = chebyshevT(n, r̄_r)
                1/r^2 * Tn
            end

            @testset "onebyr2Tn" begin
                @testset for n in 1:nr - 1
                    onebyr2_Tn_analytical = chebyfwdnr(r -> onebyr2Tn_fn(r, n))
                    @test onebyr2_chebyM[:, n+1] ≈ onebyr2_Tn_analytical rtol=1e-4
                end
            end

            function onebyr2_Iplusrηρ_Tn_fn(r, n)
                r̄_r = r̄(r)
                Tn = chebyshevT(n, r̄_r)
                ηr = ηρ_cheby(r̄_r)
                1/r^2 * (1 + ηr * r) * Tn
            end

            @testset "onebyr2_IplusrηρM" begin
                @testset for n in 1:nr - 1
                    onebyr2_Iplusrηρ_Tn_analytical = chebyfwdnr(r -> onebyr2_Iplusrηρ_Tn_fn(r, n))
                    @test onebyr2_IplusrηρM[:, n+1] ≈ onebyr2_Iplusrηρ_Tn_analytical rtol=1e-4
                end
            end
        end

        @testset "SW term" begin
            function onebyr2_ddr_S0_by_cp_fn(r, n)
                r̄_r = r̄(r)
                Tn = chebyshevT(n, r̄_r)
                1/r^2 * ddr_S0_by_cp(r̄_r) * Tn
            end
            @testset for n in 1:nr-1
                onebyr2_ddr_S0_by_cp_op = real(@view onebyr2_cheby_ddr_S0_by_cpM[:, n+1])
                onebyr2_ddr_S0_by_cp_explicit = chebyfwdnr(r -> onebyr2_ddr_S0_by_cp_fn(r, n))
                @test onebyr2_ddr_S0_by_cp_op ≈ onebyr2_ddr_S0_by_cp_explicit rtol = 1e-4
            end
        end
    end

    @testset "S terms" begin
        @testset "SS term" begin
            @testset "individual terms" begin
                function ddr_lnρT_fn(r, n)
                    r̄_r = r̄(r)
                    Tn = chebyshevT(n, r̄_r)
                    ηρr = ηρ_cheby(r̄_r)
                    ηTr = ηT_cheby(r̄_r)
                    (ηρr + ηTr) * Tn
                end
                ddr_lnρTM = mat(ddr_lnρT)
                @testset for n in 1:nr-1
                    ddr_lnρT_op = @view ddr_lnρTM[:, n+1]
                    ddr_lnρT_analytical = chebyfwdnr(r -> ddr_lnρT_fn(r, n))
                    @test ddr_lnρT_op ≈ ddr_lnρT_analytical rtol=1e-8
                end
                function ddr_lnρT_ddr_fn(r, n)
                    r̄_r = r̄(r)
                    Unm1 = chebyshevU(n-1, r̄_r)
                    ηρr = ηρ_cheby(r̄_r)
                    ηTr = ηT_cheby(r̄_r)
                    (ηρr + ηTr) * a * n * Unm1
                end
                ddr_lnρT_ddrM = RossbyWaveSpectrum.chebyshevmatrix(ddr_lnρT * ddr, nr, 4);
                @testset for n in 1:nr-1
                    ddr_lnρT_ddr_op = @view ddr_lnρT_ddrM[:, n+1]
                    ddr_lnρT_ddr_analytical = chebyfwdnr(r -> ddr_lnρT_ddr_fn(r, n))
                    @test ddr_lnρT_ddr_op ≈ ddr_lnρT_ddr_analytical rtol=1e-8
                end

                function onebyr_ddr_fn(r, n)
                    r̄_r = r̄(r)
                    Unm1 = n >= 1 ? chebyshevU(n-1, r̄_r) : 0.0
                    1/r * a * n * Unm1
                end
                onebyr_ddrM = mat(onebyr_cheby * ddr)
                @testset for n in 1:nr-1
                    onebyr_ddr_op = @view onebyr_ddrM[:, n+1]
                    onebyr_ddr_analytical = chebyfwdnr(r -> onebyr_ddr_fn(r, n))
                    @test onebyr_ddr_op ≈ onebyr_ddr_analytical rtol=1e-4
                end
            end

            @testset "SS term" begin
                function SSterm_fn(r, n)
                    r̄_r = r̄(r)
                    Tn = chebyshevT(n)
                    d2Tnr = d2f(Tn)(r̄_r) * (2/Δr)^2
                    dTnr = df(Tn)(r̄_r) * (2/Δr)
                    Tnr = Tn(r̄_r)
                    ηρr = ηρ_cheby(r̄_r)
                    ηTr = ηT_cheby(r̄_r)
                    f = d2Tnr + 2/r * dTnr - 2/r^2 * Tnr + (ηρr + ηTr)*dTnr
                    κ * Rsun^2 * f
                end
                @testset for n in 1:nr-1
                    SStermM_op = @view SSterm[:, n+1]
                    SStermM_analytical = chebyfwdnr(r -> SSterm_fn(r, n))
                    @test SStermM_op ≈ SStermM_analytical rtol=1e-4
                end
            end
        end
    end
end

@testset "viscosity" begin
    nr, nℓ = 50, 2
    nparams = nr * nℓ
    m = 1
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
    (; transforms, diff_operators, rad_terms, coordinates, radial_params, identities) = operators;
    (; r, r_chebyshev) = coordinates;
    (; nvariables, ν) = operators.constants;
    r_mid = radial_params.r_mid::Float64;
    Δr = radial_params.Δr::Float64;
    a = 1 / (Δr / 2);
    b = -r_mid / (Δr / 2);
    r̄(r) = clamp(a * r + b, -1.0, 1.0);
    (; Tcrfwd) = transforms;
    (; Iℓ) = identities;
    (; ddr, d2dr2, DDr) = diff_operators;
    (; onebyr_cheby, onebyr2_cheby, r2_cheby, r_cheby, ηρ_cheby, ηρ_by_r, ηρ_by_r2, ηρ2_by_r2) = rad_terms;
    (; r_in, r_out) = operators.radial_params;
    (; mat) = operators;

    chebyfwdnr(f, scalefactor = 5) = chebyfwd(f, r_in, r_out, nr, scalefactor)

    @testset "V terms" begin
        Vn1 = P11norm

        function VVterm1fn(r, n)
            r̄_r = r̄(r)
            T = chebyshevT(n)
            Tn = T(r̄_r)
            d2Tn = a^2 * d2f(T)(r̄_r)
            Vn1 * (d2Tn - 2 / r^2 * Tn)  * Rsun^2
        end

        function VVterm3fn(r, n)
            r̄_r = r̄(r)
            Tn = chebyshevT(n, r̄_r)
            Unm1 = chebyshevU(n-1, r̄_r)
            anr = a * n * r
            Vn1 * (-2Tn + anr * Unm1) * ηρ_cheby(r̄_r) / r  * Rsun^2
        end

        VVtermfn(r, n) = VVterm1fn(r, n) + VVterm3fn(r, n)

        M = zeros(ComplexF64, nvariables * nparams, nvariables * nparams)
        RossbyWaveSpectrum.viscosity_terms!(M, nr, nℓ, m; operators)

        @testset "VV terms" begin
            VV = RossbyWaveSpectrum.matrix_block(M, 1, 1, nvariables)
            V1_inds = 1:nr
            @testset for n in 1:nr-1
                V_op = Vn1 * real(VV[V1_inds, n + 1] ./ (-im * ν))
                V_explicit = chebyfwdnr(r -> VVtermfn(r, n))
                @test V_op ≈ V_explicit rtol = 1e-5
            end
        end
    end

    @testset "W terms" begin
        # first term
        r_d2dr2_ηρ_by_r = r_cheby * d2dr2[ηρ_by_r];
        r_d2dr2_ηρ_by_rM = mat(r_d2dr2_ηρ_by_r);

        ddr_minus_2byr = ddr - 2onebyr_cheby;
        ddr_minus_2byr_r_d2dr2_ηρ_by_r = ddr_minus_2byr * r_d2dr2_ηρ_by_r;
        ddr_minus_2byr_r_d2dr2_ηρ_by_rM = mat(ddr_minus_2byr_r_d2dr2_ηρ_by_r);

        ddr_minus_2byr_ηρ_by_r2 = ddr_minus_2byr[ηρ_by_r2];
        ddr_minus_2byr_ηρ_by_r2M = mat(ddr_minus_2byr_ηρ_by_r2);

        four_ηρ_by_r = 4ηρ_by_r;
        d2dr2_plus_4ηρ_by_rM = mat(d2dr2[four_ηρ_by_r]);

        d2dr2_plus_4ηρ_by_r = d2dr2 + 4ηρ_by_r;
        d2dr2_d2dr2_plus_4ηρ_by_rM = mat(d2dr2 * d2dr2_plus_4ηρ_by_r);

        one_by_r2_d2dr2_plus_4ηρ_by_rM = mat(onebyr2_cheby * d2dr2_plus_4ηρ_by_r);

        d2dr2_one_by_r2M = mat(d2dr2[onebyr2_cheby]);
        d2dr2_one_by_r2M_2 = mat(onebyr2_cheby * (d2dr2 - 4onebyr_cheby*ddr + 6onebyr2_cheby));

        onebyr4_cheby = onebyr2_cheby*onebyr2_cheby;
        onebyr4_chebyM = mat(onebyr4_cheby);

        # third term
        ddr_minus_2byr_DDr = ddr_minus_2byr * DDr
        ηρ_cheby_ddr_minus_2byr_DDr = ηρ_cheby * ddr_minus_2byr_DDr
        ddr_ηρ_cheby_ddr_minus_2byr_DDr = ddr * ηρ_cheby_ddr_minus_2byr_DDr
        ηρ_cheby_ddr_minus_2byr_DDrM = mat(ηρ_cheby_ddr_minus_2byr_DDr);
        ddr_ηρ_cheby_ddr_minus_2byr_DDrM = mat(ddr_ηρ_cheby_ddr_minus_2byr_DDr);
        ηρ_cheby_ddr_minus_2byr_DDrM_2 = chebyshevmatrix(ηρ_cheby_ddr_minus_2byr_DDr, nr, 3);
        ddr_ηρ_cheby_ddr_minus_2byr_DDrM_2 = chebyshevmatrix(ddr_ηρ_cheby_ddr_minus_2byr_DDr, nr, 3);

        ddr_ηρ_by_r2 = ddr[ηρ_by_r2];
        ηρ_by_r2_ddr_minus_2byr = ηρ_by_r2 * ddr_minus_2byr;
        ddr_ηρ_by_r2_minus_2ηρ_by_r2_ddr_minus_2byr = ddr_ηρ_by_r2 - 2*ηρ_by_r2_ddr_minus_2byr;
        ddr_ηρ_by_r2_minus_2ηρ_by_r2_ddr_minus_2byrM = mat(ddr_ηρ_by_r2_minus_2ηρ_by_r2_ddr_minus_2byr);

        # fourth term
        ηρ2_by_r2M = mat(ηρ2_by_r2);

        @testset "first term" begin
            function WWterm11_1fn(r, n)
                T = chebyshevT(n)
                F1 = r -> r * d2f(r -> ηρ_cheby(r̄(r))/r * T(r̄(r)), r)
                F1(r)
            end

            @testset for n in 0:nr-1
                WWterm11_1_op = @view r_d2dr2_ηρ_by_rM[:, n+1]
                WWterm11_1_analytical = chebyfwdnr(r -> WWterm11_1fn(r, n))
                @test WWterm11_1_op ≈ WWterm11_1_analytical rtol=1e-4
            end

            function WWterm11fn(r, n)
                T = chebyshevT(n)
                F1 = r -> WWterm11_1fn(r, n)
                F2 = r -> df(F1, r) - 2/r * F1(r)
                F2(r)
            end

            @testset for n in 0:nr-1
                WWterm11_op = @view ddr_minus_2byr_r_d2dr2_ηρ_by_rM[:, n+1]
                WWterm11_analytical = chebyfwdnr(r -> WWterm11fn(r, n))
                @test WWterm11_op ≈ WWterm11_analytical rtol=2e-4
            end

            function WWterm12fn(r, n)
                T = chebyshevT(n)
                F1 = r -> ηρ_cheby(r̄(r))/r^2 * T(r̄(r))
                F2 = r -> df(F1, r) - 2/r * F1(r)
                F2(r)
            end

            @testset for n in 0:nr-1
                WWterm12_op = @view ddr_minus_2byr_ηρ_by_r2M[:, n+1]
                WWterm12_analytical = chebyfwdnr(r -> WWterm12fn(r, n))
                @test WWterm12_op ≈ WWterm12_analytical rtol=1e-4
            end

            function WWterm13fn(r, n)
                T = chebyshevT(n)
                F1 = r -> 4ηρ_cheby(r̄(r))/r * T(r̄(r))
                F2 = r -> d2f(F1, r)
                F2(r)
            end

            @testset for n in 0:nr-1
                WWterm13_op = @view d2dr2_plus_4ηρ_by_rM[:, n+1]
                WWterm13_analytical = chebyfwdnr(r -> WWterm13fn(r, n))
                @test WWterm13_op ≈ WWterm13_analytical rtol=1e-4
            end

            function WWterm13_fullfn(r, n)
                T = chebyshevT(n)
                F1 = r -> d2f(T ∘ r̄, r) + 4ηρ_cheby(r̄(r))/r * T(r̄(r))
                F2 = r -> d2f(F1, r)
                F2(r)
            end

            @testset for n in 0:nr-1
                WWterm13_op = @view d2dr2_d2dr2_plus_4ηρ_by_rM[:, n+1]
                WWterm13_analytical = chebyfwdnr(r -> WWterm13_fullfn(r, n))
                @test WWterm13_op ≈ WWterm13_analytical rtol=1e-4
            end

            function WWterm14_fullfn(r, n)
                T = chebyshevT(n)
                F1 = r -> d2f(T ∘ r̄, r) + 4ηρ_cheby(r̄(r))/r * T(r̄(r))
                F2 = r -> F1(r) / r^2
                F2(r)
            end

            @testset for n in 0:nr-1
                WWterm14_op = @view one_by_r2_d2dr2_plus_4ηρ_by_rM[:, n+1]
                WWterm14_analytical = chebyfwdnr(r -> WWterm14_fullfn(r, n))
                @test WWterm14_op ≈ WWterm14_analytical rtol=1e-4
            end

            function WWterm15fn(r, n)
                T = chebyshevT(n)
                F1 = r -> 1/r^2 * T(r̄(r))
                d2f(F1, r)
            end

            @testset for n in 0:nr-1
                WWterm15_op = @view d2dr2_one_by_r2M[:, n+1]
                WWterm15_analytical = chebyfwdnr(r -> WWterm15fn(r, n))
                @test WWterm15_op ≈ WWterm15_analytical rtol=3e-2
                WWterm15_op_2 = @view d2dr2_one_by_r2M_2[:, n+1]
                @test WWterm15_op_2 ≈ WWterm15_analytical rtol=1e-4
            end

            function WWterm16fn(r, n)
                T = chebyshevT(n)
                1/r^4 * T(r̄(r))
            end

            @testset for n in 0:nr-1
                WWterm16_op = @view onebyr4_chebyM[:, n+1]
                WWterm16_analytical = chebyfwdnr(r -> WWterm16fn(r, n))
                @test WWterm16_op ≈ WWterm16_analytical rtol=1e-4
            end
        end
        @testset "third term" begin
            function WWterm31_1fn(r, n)
                T = chebyshevT(n) ∘ r̄
                # F1 = DDr
                F1 = r -> df(T, r) + (ηρ_cheby ∘ r̄)(r) * T(r)
                F2 = r -> (ηρ_cheby ∘ r̄)(r) * (df(F1, r) - 2/r * F1(r))
                F2(r)
            end

            @testset for n in 0:nr-1
                WWterm31_1_op = @view ηρ_cheby_ddr_minus_2byr_DDrM[:, n+1]
                WWterm31_1_analytical = chebyfwdnr(r -> WWterm31_1fn(r, n))
                @test WWterm31_1_op ≈ WWterm31_1_analytical rtol=5e-2
                WWterm31_1_op = @view ηρ_cheby_ddr_minus_2byr_DDrM_2[:, n+1]
                WWterm31_1_analytical = chebyfwdnr(r -> WWterm31_1fn(r, n))
                @test WWterm31_1_op ≈ WWterm31_1_analytical rtol=1e-4
            end

            function WWterm31_2fn(r, n)
                F1 = r -> WWterm31_1fn(r, n)
                df(F1, r)
            end

            @testset for n in 0:nr-1
                WWterm31_2_op = @view ddr_ηρ_cheby_ddr_minus_2byr_DDrM[:, n+1]
                WWterm31_2_analytical = chebyfwdnr(r -> WWterm31_2fn(r, n))
                @test WWterm31_2_op ≈ WWterm31_2_analytical rtol=5e-2
                WWterm31_2_op = @view ddr_ηρ_cheby_ddr_minus_2byr_DDrM_2[:, n+1]
                WWterm31_2_analytical = chebyfwdnr(r -> WWterm31_2fn(r, n))
                @test WWterm31_2_op ≈ WWterm31_2_analytical rtol=1e-4
            end

            function WWterm32fn(r, n)
                T = chebyshevT(n) ∘ r̄
                # F1 = DDr
                F1 = r -> df(r -> (ηρ_cheby ∘ r̄)(r)/r^2 * T(r), r)
                F2 = r -> -2/r^2*(ηρ_cheby ∘ r̄)(r) * (df(T, r) - 2/r * T(r))
                F1(r) + F2(r)
            end

            @testset for n in 0:nr-1
                WWterm32_op = @view ddr_ηρ_by_r2_minus_2ηρ_by_r2_ddr_minus_2byrM[:, n+1]
                WWterm32_analytical = chebyfwdnr(r -> WWterm32fn(r, n))
                @test WWterm32_op ≈ WWterm32_analytical rtol=1e-3
            end
        end
        @testset "fourth term" begin
            function WWterm4fn(r, n)
                T = chebyshevT(n) ∘ r̄
                # F1 = DDr
                F2 = r -> (ηρ2_by_r2 ∘ r̄)(r) * T(r)
                F2(r)
            end

            @testset for n in 0:nr-1
                WWterm4op = @view ηρ2_by_r2M[:, n+1]
                WWterm4analytical = chebyfwdnr(r -> WWterm4fn(r, n))
                @test WWterm4op ≈ WWterm4analytical rtol=1e-7
            end
        end
    end
end

@testset "matrix convergence with resolution: uniform rotation" begin
    nr, nℓ = 50, 2
    m = 5
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
    (; nvariables) = operators.constants;
    M1 = RossbyWaveSpectrum.uniform_rotation_matrix(nr, nℓ, m; operators);
    operators2 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ);
    M2 = RossbyWaveSpectrum.uniform_rotation_matrix(nr+5, nℓ, m; operators = operators2);
    operators3 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ+5);
    M3 = RossbyWaveSpectrum.uniform_rotation_matrix(nr+5, nℓ+5, m; operators = operators3);

    function matrix_subsample(M, nr_M, nr, nℓ, nvariables)
        nparams = nr*nℓ
        M_subsample = zeros(eltype(M), nvariables*nparams, nvariables*nparams)
        for colind in 1:nvariables, rowind in 1:nvariables
            Mv = matrix_block(M, rowind, colind, nvariables)
            M_subsample_v = matrix_block(M_subsample, rowind, colind, nvariables)
            for ℓ′ind in 1:nℓ, ℓind in 1:nℓ
                indscheb_M = CartesianIndices(((ℓind - 1)*nr_M .+ (1:nr), (ℓ′ind - 1)*nr_M .+ (1:nr)))
                indscheb_Mss = CartesianIndices(((ℓind - 1)*nr .+ (1:nr), (ℓ′ind - 1)*nr .+ (1:nr)))
                @views M_subsample_v[indscheb_Mss] = Mv[indscheb_M]
            end
        end
        return M_subsample
    end

    @test matrix_subsample(M1, nr, nr, nℓ, nvariables) == M1;
    M2_subsampled = matrix_subsample(M2, nr+5, nr, nℓ, nvariables);
    @testset for rowind in 1:nvariables, colind in 1:nvariables
        M2_ssv = matrix_block(M2_subsampled, rowind, colind, nvariables);
        M1v = matrix_block(M1, rowind, colind, nvariables);
        @testset "real" begin
            if rowind == 3 && colind == 2
                @test real(M2_ssv) ≈ real(M1v) rtol=3e-3
            else
                @test real(M2_ssv) ≈ real(M1v) rtol=2e-3
            end
        end
        @testset "imag" begin
            if rowind == colind == 2
                @test imag(M2_ssv) ≈ imag(M1v) rtol=2e-3
            elseif rowind == colind == 3
                @test imag(M2_ssv) ≈ imag(M1v) rtol=1e-4
            end
        end
    end

    M3_subsampled = matrix_subsample(M3, nr+5, nr, nℓ, nvariables);
    @testset for rowind in 1:nvariables, colind in 1:nvariables
        M3_ssv = matrix_block(M3_subsampled, rowind, colind, nvariables)
        M1v = matrix_block(M1, rowind, colind, nvariables)
        @testset "real" begin
            if rowind == 3 && colind == 2
                @test real(M3_ssv) ≈ real(M1v) rtol=3e-3
            else
                @test real(M3_ssv) ≈ real(M1v) rtol=2e-3
            end
        end
        @testset "imag" begin
            if rowind == colind == 2
                @test imag(M3_ssv) ≈ imag(M1v) rtol=2e-3
            else
                @test imag(M3_ssv) ≈ imag(M1v) rtol=1e-4
            end
        end
    end
end

function rossby_ridge_eignorm(λ, v, M, m, nparams; ΔΩ_by_Ω = 0)
    matchind = argmin(abs.(real(λ) .- RossbyWaveSpectrum.rossby_ridge(m; ΔΩ_by_Ω)))
    vi = v[:, matchind];
    λi = λ[matchind]
    normsden = [norm(λi * vi[i*nparams .+ (1:nparams)]) for i in 0:2]
    normsnum = [norm(M[i*nparams .+ (1:nparams), :] * vi - λi * vi[i*nparams .+ (1:nparams)]) for i in 0:2]
    [(d/norm(λi) > 1e-10 ? n/d : 0) for (n, d) in zip(normsnum, normsden)]
end

@testset "uniform rotation solution" begin
    nr, nℓ = 45, 15
    nparams = nr * nℓ
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ); constraints = RossbyWaveSpectrum.constraintmatrix(operators);
    (; r_in, r_out, Δr) = operators.radial_params
    (; BC) = constraints;
    @testset for m in [1, 10, 20]
        λu, vu, Mu = RossbyWaveSpectrum.uniform_rotation_spectrum(nr, nℓ, m; operators, constraints);
        λuf, vuf = RossbyWaveSpectrum.filter_eigenvalues(λu, vu, Mu, m;
            operators, constraints, Δl_cutoff = 7, n_cutoff = 9,
            eig_imag_damped_cutoff = 1e-3, eig_imag_unstable_cutoff = -1e-3,
            scale_eigenvectors = false);
        @info "$(length(λuf)) eigenmode$(length(λuf) > 1 ? "s" : "") found for m = $m"
        @testset "ℓ == m" begin
            res, ind = findmin(abs.(real(λuf) .- 2/(m+1)))
            @testset "eigenvalue" begin
                @test res < 1e-4
            end
            @testset "eigenvector" begin
                vi = vuf[:, ind];
                (; VWSinv, θ) = RossbyWaveSpectrum.eigenfunction_realspace(vi, m, operators);
                nθ = length(θ)
                (; V) = VWSinv;
                Vr = real(V);
                equator_ind = argmin(abs.(θ .- pi/2))
                Δθ_scan = div(nθ, 5)
                rangescan = intersect(equator_ind .+ (-Δθ_scan:Δθ_scan), axes(V, 2))
                ind_max = findmax(col -> maximum(real, col), eachcol(view(Vr, :, rangescan)))[2]
                ind_max += first(rangescan) - 1
                V_peak_depthprofile = @view Vr[:, ind_max];
                r_max_ind = argmax(abs.(V_peak_depthprofile))
                V_surf = @view Vr[r_max_ind, :];
                V_surf ./= maximum(abs, V_surf);
                @test abs.(V_surf) ≈ sin.(θ).^m rtol=0.15
            end
        end
        vfn = zeros(eltype(vuf), size(vuf, 1))
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

@testset "differential rotation" begin
    @testset "compare with constant" begin
        nr, nℓ = 30, 2
        operators = RossbyWaveSpectrum.radial_operators(nr, nℓ)
        (; nvariables) = operators.constants

        m = 1

        Mc = RossbyWaveSpectrum.differential_rotation_matrix(nr, nℓ, m,
            rotation_profile = :constant; operators)

        @testset "radial constant and constant" begin
            Mr = RossbyWaveSpectrum.differential_rotation_matrix(nr, nℓ, m,
                rotation_profile = :radial_constant; operators)

            @testset for colind in 1:nvariables, rowind in 1:nvariables
                @test matrix_block(Mr, rowind, colind, nvariables) ≈ matrix_block(Mc, rowind, colind, nvariables) atol = 1e-10 rtol = 1e-3
            end
        end
    end

    @testset "radial differential rotation" begin
        nr, nℓ = 50, 10
        nparams = nr * nℓ
        m = 1
        operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
        (; transforms, diff_operators, rad_terms, coordinates, radial_params, identities) = operators;
        (; r, r_chebyshev) = coordinates;
        (; nvariables, ν) = operators.constants;
        r_mid = radial_params.r_mid::Float64;
        Δr = radial_params.Δr::Float64;
        a = 1 / (Δr / 2);
        b = -r_mid / (Δr / 2);
        r̄(r) = clamp(a * r + b, -1.0, 1.0)
        (; Tcrfwd) = transforms;
        (; Iℓ) = identities;
        (; ddr, d2dr2, DDr, ddrDDr) = diff_operators;
        (; onebyr_cheby, onebyr2_cheby, r2_cheby, r_cheby, ηρ_cheby) = rad_terms;
        (; r_in, r_out) = operators.radial_params;
        (; mat) = operators;
        chebyfwdnr(f, scalefactor = 5) = chebyfwd(f, r_in, r_out, nr, scalefactor)

        ℓ = 2
        ℓℓp1 = ℓ*(ℓ+1)
        ℓ′ = 1
        cosθ21 = 1/√5
        sinθdθ21 = 1/√5
        ∇²_sinθdθ21 = -ℓℓp1 * sinθdθ21

        # used to check the VV term
        M = zeros(3nr*nℓ, 3nr*nℓ);
        RossbyWaveSpectrum._differential_rotation_matrix!(M, nr, nℓ, m, :constant; operators);

        @testset for rotation_profile in [:constant, :linear, :solar_equator]

            (; ΔΩ, Ω0, ddrΔΩ, d2dr2ΔΩ) =
                RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives(
                    m; operators, rotation_profile);

            ΔΩ_by_r = ΔΩ * onebyr_cheby
            ΔΩ_DDr = ΔΩ * DDr
            DDr_min_2byr = DDr - 2onebyr_cheby
            ΔΩ_DDr_min_2byr = ΔΩ * DDr_min_2byr
            ddrΔΩ_plus_ΔΩddr = ddrΔΩ + ΔΩ * ddr
            twoΔΩ_by_r = 2ΔΩ * onebyr_cheby

            VWterm, WVterm = zeros(nr,nr), zeros(nr,nr)
            RossbyWaveSpectrum.radial_differential_rotation_terms_inner!((VWterm, WVterm),
                        (ℓ, ℓ′), (cosθ21, sinθdθ21, ∇²_sinθdθ21),
                        (mat(ddrΔΩ), Ω0),
                        map(mat, (ΔΩ_by_r, ΔΩ_DDr, ΔΩ_DDr_min_2byr, ddrΔΩ_plus_ΔΩddr)))

            ΔΩ_ddrDDr = ΔΩ * ddrDDr
            ddrΔΩ_DDr = ddrΔΩ * DDr
            ddrΔΩ_DDr_plus_ΔΩ_ddrDDr = ddrΔΩ_DDr + ΔΩ_ddrDDr
            ΔΩ_by_r2 = ΔΩ * onebyr2_cheby
            two_ΔΩbyr_ηρ = twoΔΩ_by_r * ηρ_cheby
            ddrΔΩ_ddr_plus_2byr = ddrΔΩ * (ddr + 2onebyr_cheby)
            WWfixedterms = -two_ΔΩbyr_ηρ + d2dr2ΔΩ + ddrΔΩ_ddr_plus_2byr

            @testset "V terms" begin
                Vn1 = P11norm

                @testset "ddrΔΩ_plus_ΔΩddr" begin
                    function ddrΔΩ_plus_ΔΩddr_term(r, n)
                        r̄_r = r̄(r)
                        Tn = chebyshevT(n, r̄_r)
                        Unm1 = chebyshevU(n-1, r̄_r)
                        ddrΔΩ(r̄_r) * Tn + ΔΩ(r̄_r) * a * n * Unm1
                    end
                    @testset for n in 1:nr-1
                        ddrΔΩ_plus_ΔΩddr_op = mat(ddrΔΩ_plus_ΔΩddr)[:, n + 1]
                        ddrΔΩ_plus_ΔΩddr_explicit = chebyfwdnr(r -> ddrΔΩ_plus_ΔΩddr_term(r, n))
                        @test ddrΔΩ_plus_ΔΩddr_op ≈ ddrΔΩ_plus_ΔΩddr_explicit rtol = 1e-8
                    end
                end

                @testset "twoΔΩ_by_r" begin
                    function twoΔΩ_by_r_term(r, n)
                        r̄_r = r̄(r)
                        Tn = chebyshevT(n, r̄_r)
                        2ΔΩ(r̄_r)/r * Tn
                    end
                    twoΔΩ_by_rM = mat(twoΔΩ_by_r)
                    @testset for n in 1:nr-1
                        twoΔΩ_by_r_op = twoΔΩ_by_rM[:, n + 1]
                        twoΔΩ_by_r_explicit = chebyfwdnr(r -> twoΔΩ_by_r_term(r, n))
                        @test twoΔΩ_by_r_op ≈ twoΔΩ_by_r_explicit rtol = 1e-4
                    end
                end

                @testset "WV term" begin
                    function WVterms(r, n)
                        r̄_r = r̄(r)
                        Tn = chebyshevT(n, r̄_r)
                        Unm1 = chebyshevU(n-1, r̄_r)
                        2/√15 * (ΔΩ(r̄_r) * (a * n * Unm1 - 2/r * Tn)  + ddrΔΩ(r̄_r) * Tn)
                    end
                    @testset for n in 1:nr-1
                        WV_op = Vn1 * @view WVterm[:, n + 1]
                        WV_explicit = chebyfwdnr(r -> WVterms(r, n))
                        @test WV_op ≈ WV_explicit rtol = 1e-4
                    end
                end
                @testset "VV term" begin
                    VV11term = @view M[1:nr, 1:nr]
                    @testset for n in 1:nr-1
                        VV_op = @view VV11term[:, n + 1]
                        @test all(iszero, VV_op)
                    end
                end
            end
            @testset "W terms" begin
                Wn1 = P11norm
                @testset "VW term" begin
                    function DDrmin2byr_term(r, n)
                        r̄_r = r̄(r)
                        Tn = chebyshevT(n, r̄_r)
                        Unm1 = chebyshevU(n-1, r̄_r)
                        ηr = ηρ_cheby(r̄_r)
                        a * n * Unm1 - 2/r * Tn + ηr * Tn
                    end
                    DDr_min_2byrM = mat(DDr_min_2byr)
                    @testset for n in 1:nr-1
                        DDrmin2byr_op = @view DDr_min_2byrM[:, n + 1]
                        DDrmin2byr_explicit = chebyfwdnr(r -> DDrmin2byr_term(r, n))
                        @test DDrmin2byr_op ≈ DDrmin2byr_explicit rtol = 1e-4
                    end

                    function ddrΔΩ_term(r, n)
                        r̄_r = r̄(r)
                        Tn = chebyshevT(n, r̄_r)
                        ddrΔΩ(r̄_r) * Tn
                    end
                    ddrΔΩM = mat(ddrΔΩ)
                    @testset for n in 1:nr-1
                        ddrΔΩ_op = @view ddrΔΩM[:, n + 1]
                        ddrΔΩ_explicit = chebyfwdnr(r -> ddrΔΩ_term(r, n))
                        @test ddrΔΩ_op ≈ ddrΔΩ_explicit rtol = 1e-4
                    end

                    function ΔΩDDr_min_2byr_min_ddrΔΩ_term(r, n)
                        r̄_r = r̄(r)
                        Tn = chebyshevT(n, r̄_r)
                        ΔΩ(r̄_r) * DDrmin2byr_term(r, n)  - ddrΔΩ(r̄_r) * Tn
                    end
                    ΔΩDDr_min_2byr_min_ddrΔΩM = mat(ΔΩ * DDr_min_2byr - ddrΔΩ)
                    @testset for n in 1:nr-1
                        ΔΩDDr_min_2byr_min_ddrΔΩ_op = ΔΩDDr_min_2byr_min_ddrΔΩM[:, n + 1]
                        ΔΩDDr_min_2byr_min_ddrΔΩ_explicit = chebyfwdnr(r -> ΔΩDDr_min_2byr_min_ddrΔΩ_term(r, n))
                        @test ΔΩDDr_min_2byr_min_ddrΔΩ_op ≈ ΔΩDDr_min_2byr_min_ddrΔΩ_explicit rtol = 1e-4
                    end

                    VWterms(r, n) = -2/√15 * ΔΩDDr_min_2byr_min_ddrΔΩ_term(r, n)

                    @testset for n in 1:nr-1
                        VW_op = Wn1 * @view VWterm[:, n + 1]
                        VW_explicit = chebyfwdnr(r -> VWterms(r, n))
                        @test VW_op ≈ -VW_explicit rtol = 1e-4
                    end
                end
            end
        end

        @testset "convergence of diff rot profile" begin
            nr, nℓ = 50, 10
            m = 5
            operators1 = RossbyWaveSpectrum.radial_operators(nr, nℓ);
            operators2 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ);
            operators3 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ+5);

            T = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives(
                m, operators = operators1, rotation_profile = :solar_equator);
            ΔΩ1, ddrΔΩ1, d2dr2ΔΩ1 = T.ΔΩ, T.ddrΔΩ, T.d2dr2ΔΩ;

            T = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives(
                m, operators = operators2, rotation_profile = :solar_equator);
            ΔΩ2, ddrΔΩ2, d2dr2ΔΩ2 = T.ΔΩ, T.ddrΔΩ, T.d2dr2ΔΩ;

            T = RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives(
                m, operators = operators3, rotation_profile = :solar_equator);
            ΔΩ3, ddrΔΩ3, d2dr2ΔΩ3 = T.ΔΩ, T.ddrΔΩ, T.d2dr2ΔΩ;

            @testset "nr+5, nℓ" begin
                @test ApproxFun.ncoefficients(ΔΩ1) ≈ ApproxFun.ncoefficients(ΔΩ2)
                @test ApproxFun.coefficients(ΔΩ1) ≈ ApproxFun.coefficients(ΔΩ2) rtol=1e-3
                @test ApproxFun.ncoefficients(ddrΔΩ1) ≈ ApproxFun.ncoefficients(ddrΔΩ2)
                @test ApproxFun.coefficients(ddrΔΩ1) ≈ ApproxFun.coefficients(ddrΔΩ2) rtol=1e-3
                @test ApproxFun.ncoefficients(d2dr2ΔΩ1) ≈ ApproxFun.ncoefficients(d2dr2ΔΩ2)
                @test ApproxFun.coefficients(d2dr2ΔΩ1) ≈ ApproxFun.coefficients(d2dr2ΔΩ2) rtol=1e-3
            end

            @testset "nr+5, nℓ+5" begin
                @test ApproxFun.ncoefficients(ΔΩ1) ≈ ApproxFun.ncoefficients(ΔΩ3)
                @test ApproxFun.coefficients(ΔΩ1) ≈ ApproxFun.coefficients(ΔΩ3) rtol=5e-3
                @test ApproxFun.ncoefficients(ddrΔΩ1) ≈ ApproxFun.ncoefficients(ddrΔΩ3)
                @test ApproxFun.coefficients(ddrΔΩ1) ≈ ApproxFun.coefficients(ddrΔΩ3) rtol=5e-3
                @test ApproxFun.ncoefficients(d2dr2ΔΩ1) ≈ ApproxFun.ncoefficients(d2dr2ΔΩ3)
                @test ApproxFun.coefficients(d2dr2ΔΩ1) ≈ ApproxFun.coefficients(d2dr2ΔΩ3) rtol=5e-3
            end
        end

        @testset "matrix convergence with resolution" begin
            nr, nℓ = 50, 10
            m = 5
            operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
            (; nvariables) = operators.constants;
            operators2 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ);
            operators3 = RossbyWaveSpectrum.radial_operators(nr+5, nℓ+5);
            @testset for rotation_profile in [:radial_linear, :radial]
                M1 = RossbyWaveSpectrum.differential_rotation_matrix(nr, nℓ, m; operators, rotation_profile);
                M2 = RossbyWaveSpectrum.differential_rotation_matrix(nr+5, nℓ, m; operators = operators2, rotation_profile);
                M3 = RossbyWaveSpectrum.differential_rotation_matrix(nr+5, nℓ+5, m; operators = operators3, rotation_profile);

                function matrix_subsample(M, nr_M, nr, nℓ, nvariables)
                    nparams = nr*nℓ
                    M_subsample = zeros(eltype(M), nvariables*nparams, nvariables*nparams)
                    for colind in 1:nvariables, rowind in 1:nvariables
                        Mv = matrix_block(M, rowind, colind, nvariables)
                        M_subsample_v = matrix_block(M_subsample, rowind, colind, nvariables)
                        for ℓ′ind in 1:nℓ, ℓind in 1:nℓ
                            indscheb_M = CartesianIndices(((ℓind - 1)*nr_M .+ (1:nr), (ℓ′ind - 1)*nr_M .+ (1:nr)))
                            indscheb_Mss = CartesianIndices(((ℓind - 1)*nr .+ (1:nr), (ℓ′ind - 1)*nr .+ (1:nr)))
                            @views M_subsample_v[indscheb_Mss] = Mv[indscheb_M]
                        end
                    end
                    return M_subsample
                end

                @test matrix_subsample(M1, nr, nr, nℓ, nvariables) == M1;
                @testset "nr+5, nℓ" begin
                    M2_subsampled = matrix_subsample(M2, nr+5, nr, nℓ, nvariables);
                    @testset for rowind in 1:nvariables, colind in 1:nvariables
                        M2_ssv = matrix_block(M2_subsampled, rowind, colind, nvariables);
                        M1v = matrix_block(M1, rowind, colind, nvariables);
                        @testset "real" begin
                            if rowind == 3 && colind == 2
                                @test real(M2_ssv) ≈ real(M1v) rtol=3e-3
                            else
                                @test real(M2_ssv) ≈ real(M1v) rtol=2e-3
                            end
                        end
                        @testset "imag" begin
                            if rowind == colind == 2
                                @test imag(M2_ssv) ≈ imag(M1v) rtol=2e-3
                            else
                                @test imag(M2_ssv) ≈ imag(M1v) rtol=1e-4
                            end
                        end
                    end
                end

                @testset "nr+5, nℓ+5" begin
                    M3_subsampled = matrix_subsample(M3, nr+5, nr, nℓ, nvariables);
                    @testset for rowind in 1:nvariables, colind in 1:nvariables
                        M3_ssv = matrix_block(M3_subsampled, rowind, colind, nvariables)
                        M1v = matrix_block(M1, rowind, colind, nvariables)
                        @testset "real" begin
                            if rowind == 3 && colind == 2
                                @test real(M3_ssv) ≈ real(M1v) rtol=3e-3
                            else
                                @test real(M3_ssv) ≈ real(M1v) rtol=2e-3
                            end
                        end
                        @testset "imag" begin
                            if rowind == colind == 2
                                @test imag(M3_ssv) ≈ imag(M1v) rtol=2e-3
                            else
                                @test imag(M3_ssv) ≈ imag(M1v) rtol=1e-4
                            end
                        end
                    end
                end
            end
        end

        @testset "S terms" begin
            nr, nℓ = 50, 10
            m = 5

            operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
            (; Δr) = operators.radial_params;
            (; r_chebyshev) = operators.coordinates;
            (; g_cheby) = operators.rad_terms;
            (; Tcrfwd) = operators.transforms;
            (; mat) = operators;

            @testset for rotation_profile in [:linear, :solar_equator]
                ΔΩprofile_deriv =
                    RossbyWaveSpectrum.radial_differential_rotation_profile_derivatives(
                        m; operators, rotation_profile);

                (; ΔΩ, Ω0, ddrΔΩ, d2dr2ΔΩ) = ΔΩprofile_deriv;

                ΔΩM = mat(ΔΩ);
                ddrΔΩM = mat(ddrΔΩ);

                ddrΔΩ_over_g = ddrΔΩ / g_cheby;
                ddrΔΩ_over_gM = mat(ddrΔΩ_over_g);
                ddrΔΩ_over_g_DDr = (ddrΔΩ_over_g * DDr);
                ddrΔΩ_over_g_DDrM = mat(ddrΔΩ_over_g_DDr);

                @testset "ddrΔΩ_over_g" begin
                    @testset for n in 0:5:nr-1
                        T = chebyshevT(n)
                        f = x -> ddrΔΩ(x)/g_cheby(x) * T(x)

                        fc = chebyfwd(f, nr);

                        @test fc ≈ @view(ddrΔΩ_over_gM[:, n+1]) rtol=5e-4
                    end
                end

                @testset "ddrΔΩ_over_g * DDr" begin
                    @testset for n in 0:5:nr-1
                        T = chebyshevT(n)
                        f = x -> ddrΔΩ_over_g(x) * (df(T, x) * (2/Δr) + ηρ_cheby(x) * T(x))

                        fc = chebyfwd(f, nr);

                        @test fc ≈ @view(ddrΔΩ_over_g_DDrM[:, n+1]) rtol=1e-4
                    end
                end
            end
        end
    end
end

@testset "constant differential rotation solution" begin
    nr, nℓ = 50, 15
    nparams = nr * nℓ
    operators = RossbyWaveSpectrum.radial_operators(nr, nℓ); constraints = RossbyWaveSpectrum.constraintmatrix(operators);
    (; r_in, r_out, Δr) = operators.radial_params
    (; BC) = constraints;
    @testset for m in [1, 10, 20]
        @testset "constant" begin
            λu, vu, Mu = RossbyWaveSpectrum.differential_rotation_spectrum(nr, nℓ, m; operators, constraints,
                rotation_profile = :constant, ΔΩ_by_Ω = 0.02);
            λuf, vuf = RossbyWaveSpectrum.filter_eigenvalues(λu, vu, Mu, m;
                operators, constraints, Δl_cutoff = 7, n_cutoff = 9, ΔΩ_by_Ω_low = -0.01,
                ΔΩ_by_Ω_high = 0.03, eig_imag_damped_cutoff = 1e-3, eig_imag_unstable_cutoff = -1e-3,
                scale_eigenvectors = false);
            @info "$(length(λuf)) eigenmode$(length(λuf) > 1 ? "s" : "") found for m = $m"
            @testset "ℓ == m" begin
                ω0 = RossbyWaveSpectrum.rossby_ridge(m, ΔΩ_by_Ω = 0.02)
                @test findmin(abs.(real(λuf) .- ω0))[1] < 1e-4
            end
            vfn = zeros(eltype(vuf), size(vuf, 1))
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
            @testset "full eigenvalue problem solution" begin
                v = rossby_ridge_eignorm(λuf, vuf, Mu, m, nparams)
                # these improve with resolution
                @test v[1] < 2e-2
                @test v[2] < 0.5
                @test v[3] < 0.8
            end
        end
        @testset "radial constant" begin
            λu, vu, Mu = RossbyWaveSpectrum.differential_rotation_spectrum(nr, nℓ, m; operators, constraints,
                rotation_profile = :radial_constant);
            λuf, vuf = RossbyWaveSpectrum.filter_eigenvalues(λu, vu, Mu, m;
                operators, constraints, Δl_cutoff = 7, n_cutoff = 9, ΔΩ_by_Ω_low = -0.01,
                ΔΩ_by_Ω_high = 0.03, eig_imag_damped_cutoff = 1e-3, eig_imag_unstable_cutoff = -1e-3,
                scale_eigenvectors = false);
            @info "$(length(λuf)) eigenmode$(length(λuf) > 1 ? "s" : "") found for m = $m"
            @testset "ℓ == m" begin
                ω0 = RossbyWaveSpectrum.rossby_ridge(m, ΔΩ_by_Ω = 0.02)
                @test findmin(abs.(real(λuf) .- ω0))[1] < 1e-4
            end
            vfn = zeros(eltype(vuf), size(vuf, 1))
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
            @testset "full eigenvalue problem solution" begin
                v = rossby_ridge_eignorm(λuf, vuf, Mu, m, nparams)
                # these improve with resolution
                @test v[1] < 2e-2
                @test v[2] < 0.5
                @test v[3] < 0.8
            end
        end
    end
end
