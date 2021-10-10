using RossbyWaveSpectrum
using RossbyWaveSpectrum: ForwardTransform, InverseTransform, IdentityMatrix, OneHotVector, PaddedMatrix
using RossbyWaveSpectrum: intPl1mPl20Pl3m, intPl1mP′l20Pl3m
using Test
using LinearAlgebra
using UnPack
using Polynomials: ChebyshevT
using Kronecker
using QuadGK
using SphericalHarmonics
using ForwardDiff
using WignerSymbols
using OffsetArrays

nr = 4; nℓ = 6;
operators = RossbyWaveSpectrum.radial_operators(nr, nℓ);
(; coordinates, rad_terms, transforms, diff_operators) = operators;
(; Tcrfwd, Tcrinv) = transforms;
(; r, r_cheby) = coordinates;
(; params) = operators;
(; nchebyr, r_in, r_out, Δr) = params;
(; rDDr, DDr, rddr, ddr) = diff_operators;
(; ηρ, ηρ_cheby) = rad_terms;
nparams = nchebyr * nℓ;
r_mid = (r_out + r_in)/2;
rηρ_cheby = r_cheby * ηρ_cheby;
Ir = IdentityMatrix(nchebyr);
Iℓ = PaddedMatrix(IdentityMatrix(2nℓ), nℓ);

@testset "chebyshev" begin
    r, Tcrfwd, Tcrinv = RossbyWaveSpectrum.chebyshev_forward_inverse(nr, r_in, r_out)
    @testset "forward-inverse in r" begin
        @test Tcrfwd * Tcrinv ≈ Tcrinv * Tcrfwd ≈ I
        @test parent(Tcrfwd) * parent(Tcrinv) ≈ parent(Tcrinv) * parent(Tcrfwd) ≈ I
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
    end
end

@testset "boundary conditions constraint" begin
    (; C) = RossbyWaveSpectrum.constraintmatrix(operators)

    Vrones = kron(rand(nℓ), ones(nr));
    v_Vrones = [Vrones; rand(nparams)];
    @test length(v_Vrones) == size(C,2)

    Vrflip = kron(rand(nℓ), [(-1)^n for n = 0:nr-1]);
    v_Vrflip = [Vrflip; rand(nparams)];
    @test length(v_Vrflip) == size(C,2)

    # ∑_n Wkn = 0
    Wrflip = kron(rand(nℓ), [(-1)^n for n = 0:nr-1]);
    v_Wrflip = [rand(nparams); Wrflip];
    @test length(v_Wrflip) == size(C,2)
    @test (C*v_Wrflip)[3] ≈ 0 atol=1e-14

    # ∑_n (-1)^n * Wkn = 0
    Wrones = kron(rand(nℓ), ones(nr));
    v_Wrones = [rand(nparams); Wrones];
    @test length(v_Wrones) == size(C,2)
    @test (C*v_Wrones)[4] ≈ 0 atol=1e-14
end

@testset "invert B" begin
end

Plm(θ, l, m) = SphericalHarmonics.associatedLegendre(θ, l, m, norm = SphericalHarmonics.Orthonormal())
dPlm(θ, l, m) = -ForwardDiff.derivative(θ -> Plm(θ, l, m), θ)/sin(θ)

@testset "legendre integrals" begin
    m = 2
    rtol = 1e-1
    @testset "Pl" begin
        for ℓ1 in m:4, ℓ2 in 0:4, ℓ3 in m:4
            iseven(ℓ1+ℓ2+ℓ3) || continue
            int, err = quadgk(θ -> sin(θ)*Plm(θ, ℓ1, m)*Plm(θ, ℓ2, 0)*Plm(θ, ℓ3, m), 0, pi, rtol=rtol)
            @test intPl1mPl20Pl3m(ℓ1,ℓ2,ℓ3,m) ≈ int atol=max(1e-10, err) rtol=rtol
        end
    end
    @testset "dPl" begin
        for ℓ1 in m:4, ℓ2 in 0:4, ℓ3 in m:4
            iseven(ℓ1+ℓ2+ℓ3) || continue
            int, err = quadgk(θ -> sin(θ)*Plm(θ, ℓ1, m)*dPlm(θ, ℓ2, 0)*Plm(θ, ℓ3, m), 0, pi, rtol=rtol)

            @test begin
                res = isapprox(intPl1mP′l20Pl3m(ℓ1,ℓ2,ℓ3,m), int, atol=max(1e-10, err), rtol=rtol)
                if !res
                    @show ℓ1, ℓ2, ℓ3
                end
                res
            end
        end
    end
end

@testset "trig matrices in Legendre basis" begin
    m = 2;
    fcoeff_1010 = intPl1mPl20Pl3m(1,0,1,0)
    fcoeff_1210 = intPl1mPl20Pl3m(1,2,1,0)
    fcoeff_1230 = intPl1mPl20Pl3m(1,2,3,0)
    C = RossbyWaveSpectrum.costheta_operator(nℓ, m);
    C_el(ℓ1,ℓ2,m) = √(2/3) * intPl1mPl20Pl3m(ℓ1,1,ℓ2,m);
    Sd = RossbyWaveSpectrum.sintheta_dtheta_operator(nℓ, m);
    C2 = C^2;
    @test C*C2 ≈ C2*C ≈ C^3;
    C2_el(ℓ1,ℓ2,m) = 1/3 * (ℓ1 == ℓ2) + 2/3*√(2/5)*intPl1mPl20Pl3m(ℓ1,2,ℓ2,m);
    S2 = I-C2;
    @test C*S2 ≈ S2*C;
    @testset "chain rule" begin
        @test Sd*C ≈ C*Sd - S2
        @test Sd*C2 ≈ -2C*S2 + C2*Sd
    end
    S2_el(ℓ1,ℓ,m) = 2/3 * (ℓ1==ℓ) - 2/3*√(2/5)*intPl1mPl20Pl3m(ℓ1,2,ℓ,m);
    CS2 = C*S2;
    CS2_1 = 4/(3*√3) * fcoeff_1010 - 4/(3*√15) * fcoeff_1210
    CS2_3 = -4/(3*√15) * fcoeff_1230
    CS2_coeffs = [CS2_1, CS2_3]
    CS2_el(ℓ1,ℓ,m) = sum(CS2_coeffs[ℓ2ind]*intPl1mPl20Pl3m(ℓ1,ℓ2,ℓ,m) for (ℓ2ind, ℓ2) in enumerate(1:2:3))
    SdSd = Sd^2;
    SdSdSd = Sd^3;
    CSd = C*Sd;
    S3d = S2*Sd;

    sinθdθPlm(θ, l, m) = sin(θ) * ForwardDiff.derivative(θ -> Plm(θ, l, m), θ)
    sinθdθsinθdθPlm(θ, l, m) = sin(θ) * ForwardDiff.derivative(θ-> sinθdθPlm(θ, l, m), θ)
    sinθdθsinθdθsinθdθPlm(θ, l, m) = sin(θ) * ForwardDiff.derivative(θ-> sinθdθsinθdθPlm(θ, l, m), θ)

    rtol = 1e-1
    @testset "cosθ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*cos(θ)*Plm(θ, l′, m), 0, pi, rtol=rtol)
            @test C[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
            @test C[lind, l′ind] ≈ C_el(l,l′,m) atol=1e-10 rtol=1e-6
        end
    end
    @testset "cos²θ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*cos(θ)^2*Plm(θ, l′, m), 0, pi, rtol=rtol)
            @test C2[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
            @test C2[lind, l′ind] ≈ C2_el(l,l′,m) atol=1e-10  rtol=rtol
        end
    end
    @testset "sin²θ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*sin(θ)^2*Plm(θ, l′, m), 0, pi, rtol=rtol)
            @test S2[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
            @test S2[lind, l′ind] ≈ S2_el(l,l′,m) atol=1e-10  rtol=rtol
        end
    end
    @testset "cosθsin²θ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*cos(θ)*sin(θ)^2*Plm(θ, l′, m), 0, pi, rtol=rtol)
            @test CS2[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
        end
    end
    @testset "sinθdθ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*sinθdθPlm(θ, l′, m), 0, pi, rtol=rtol)
            @test Sd[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
        end
    end
    @testset "sin³θdθ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*sin(θ)^2*sinθdθPlm(θ, l′, m), 0, pi, rtol=rtol)
            @test S3d[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
        end
    end
    @testset "sinθdθ*sinθdθ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*sinθdθsinθdθPlm(θ, l′, m), 0, pi, rtol=rtol)
            @test SdSd[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
        end
    end
    @testset "sinθdθ*sinθdθ*sinθdθ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*sinθdθsinθdθsinθdθPlm(θ, l′, m), 0, pi, rtol=rtol)
            @test SdSdSd[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
        end
    end
end

@testset "angular velocity in legendre and associated legendre basis" begin
    m = 2
    thetaop = RossbyWaveSpectrum.theta_operators(nℓ, m);
    (; thetaGL, PLMfwd, PLMinv) = thetaop;
    # Arbitrary function
    ΔΩ_ℓ_cosθ = OffsetArray([0, √(2/3)], OffsetArrays.Origin(0));
    ΔΩ_ℓ_cos²θ = OffsetArray([1/3*√2, 0, 2/3*√(2/5)], OffsetArrays.Origin(0));
    ΔΩ_ℓ_sin²θ = OffsetArray([2/3*√2, 0, -2/3*√(2/5)], OffsetArrays.Origin(0));
    ΔΩ_ℓ_arr = [(cos, ΔΩ_ℓ_cosθ), (x -> cos(x)^2, ΔΩ_ℓ_cos²θ), (x -> sin(x)^2, ΔΩ_ℓ_sin²θ)];
    for (f, ΔΩ_ℓ) in ΔΩ_ℓ_arr
        ΔΩ = f.(thetaGL);
        ΔΩ_ℓ1ℓ2 = PaddedMatrix(PLMfwd * Diagonal(ΔΩ) * PLMinv, nℓ);
        ℓs = range(m, length = nℓ);
        ΔΩ_ℓ1ℓ2_of = OffsetArray(ΔΩ_ℓ1ℓ2, ℓs, ℓs);
        # The legendre coefficients are 1/3, 0, 2/3
        for ℓ1 in ℓs, ℓ2 in ℓs
            @test ΔΩ_ℓ1ℓ2_of[ℓ1, ℓ2] ≈ sum(ΔΩ_ℓ[ℓ]*intPl1mPl20Pl3m(ℓ1, ℓ, ℓ2, m) for ℓ in axes(ΔΩ_ℓ, 1)) atol=1e-10
        end
    end
end

@testset "vorticity of differential rotation" begin
    m = 2;
    C = RossbyWaveSpectrum.costheta_operator(nℓ, m);
    C2 = C^2;
    C3 = C2*C;
    S2 = I - C2;
    Sd = RossbyWaveSpectrum.sintheta_dtheta_operator(nℓ, m);

    SdI = Sd ⊗ Ir;

    thetaop = RossbyWaveSpectrum.theta_operators(nℓ, m);
    (; thetaGL, PLMfwd, PLMinv) = thetaop;
    chebytheta = reverse(acos.(RossbyWaveSpectrum.chebyshevnodes(20)[1]));

    @testset "sintheta_dtheta_ΔΩ" begin
        function sintheta_dtheta_ΔΩ_operator(ΔΩ_r_Legendre, m, operators)
            (; transforms, params) = operators;
            (; nparams, nℓ) = params;
            (; Tcrfwd, Tcrinv) = transforms;
            ℓs = range(m, length = 2nℓ);
            ΔΩ_r_Legendre_of = OffsetArray(ΔΩ_r_Legendre, :, 0:size(ΔΩ_r_Legendre, 2)-1);
            sinθ_dθ_ΔΩ = zeros(2nparams, 2nparams);
            for ℓ in axes(ΔΩ_r_Legendre_of, 2)
                ℓ == 0 && continue
                ΔΩ_r_ℓ = Diagonal(@view ΔΩ_r_Legendre_of[:, ℓ])
                ΔΩ_cheby_ℓ = Tcrfwd * ΔΩ_r_ℓ * Tcrinv
                nchebyr1, nchebyr2 = size(ΔΩ_cheby_ℓ)
                for (ℓ2ind, ℓ2) in enumerate(ℓs), (ℓ1ind, ℓ1) in enumerate(ℓs)
                    Tℓ1ℓℓ2 = ℓ*(√(2/3)*(intPl1mPl20Pl3m(1,ℓ,ℓ-1,0)*intPl1mPl20Pl3m(ℓ1,ℓ-1,ℓ2,m) +
                        intPl1mPl20Pl3m(1,ℓ,ℓ+1,0)*intPl1mPl20Pl3m(ℓ1,ℓ+1,ℓ2,m)) -
                        √((2ℓ+1)/(2ℓ-1))*intPl1mPl20Pl3m(ℓ1,ℓ-1,ℓ2,m))
                    iszero(Tℓ1ℓℓ2) && continue

                    rowinds = (ℓ1ind-1)*nchebyr1 + 1:ℓ1ind*nchebyr1
                    colinds = (ℓ2ind-1)*nchebyr2 + 1:ℓ2ind*nchebyr2
                    ℓ2ℓ1inds = CartesianIndices((rowinds, colinds))
                    @. sinθ_dθ_ΔΩ[ℓ2ℓ1inds] += ΔΩ_cheby_ℓ * Tℓ1ℓℓ2
                end
            end
            return PaddedMatrix(sinθ_dθ_ΔΩ, nparams)
        end

        f = θ -> cos(θ)
        ΔΩ_r_thetaGL = ones(nr) .* f.(thetaGL)';
        ΔΩ_r_chebytheta = ones(nr) .* f.(chebytheta)';

        ΔΩ_kk′_ℓℓ′ = RossbyWaveSpectrum.real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop);
        @test ΔΩ_kk′_ℓℓ′ ≈ C ⊗ Ir;
        ΔΩ_r_Legendre = RossbyWaveSpectrum.normalizedlegendretransform2(ΔΩ_r_chebytheta);

        @test sintheta_dtheta_ΔΩ_operator(ΔΩ_r_Legendre, m, operators) ≈ -S2 ⊗ Ir;
        @test SdI*ΔΩ_kk′_ℓℓ′ - ΔΩ_kk′_ℓℓ′*SdI ≈ -S2 ⊗ Ir;

        f = θ -> cos(θ)^2
        ΔΩ_r_thetaGL = ones(nr) .* f.(thetaGL)';
        ΔΩ_r_chebytheta = ones(nr) .* f.(chebytheta)';

        ΔΩ_kk′_ℓℓ′ = RossbyWaveSpectrum.real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop);
        @test ΔΩ_kk′_ℓℓ′ ≈ C^2 ⊗ Ir;
        ΔΩ_r_Legendre = RossbyWaveSpectrum.normalizedlegendretransform2(ΔΩ_r_chebytheta);

        @test sintheta_dtheta_ΔΩ_operator(ΔΩ_r_Legendre, m, operators) ≈ (-2C*S2) ⊗ Ir;
        @test SdI*ΔΩ_kk′_ℓℓ′ - ΔΩ_kk′_ℓℓ′*SdI ≈ (-2C*S2) ⊗ Ir;
    end

    @testset "rddr_ΔΩ" begin
        Irddr = Iℓ ⊗ rddr;

        f = θ -> cos(θ)
        ΔΩ_r_thetaGL = ones(nr) .* f.(thetaGL)';
        ΔΩ_r_chebytheta = ones(nr) .* f.(chebytheta)';

        ΔΩ_kk′_ℓℓ′ = RossbyWaveSpectrum.real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop);

        ΔΩ_r_ℓℓ′ = RossbyWaveSpectrum.real_to_r_assocleg(ΔΩ_r_thetaGL, operators, thetaop)
        rddr_ΔΩ = RossbyWaveSpectrum.rddr_operator(ΔΩ_r_ℓℓ′, m, operators)
        @test all(x -> isapprox(0,x,atol=1e-10), rddr_ΔΩ)
        ddr_ΔΩ = RossbyWaveSpectrum.ddr_operator(ΔΩ_r_ℓℓ′, m, operators)
        @test all(x -> isapprox(0,x,atol=1e-10), ddr_ΔΩ)

        fr = r -> r; fθ = θ -> cos(θ)
        ΔΩ_r_thetaGL = fr.(r) .* fθ.(thetaGL)';
        ΔΩ_r_chebytheta = fr.(r) .* fθ.(chebytheta)';

        @test Tcrinv * rddr * Tcrfwd * ΔΩ_r_thetaGL ≈ ΔΩ_r_thetaGL

        ΔΩ_kk′_ℓℓ′ = RossbyWaveSpectrum.real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop);

        ΔΩ_r_ℓℓ′ = RossbyWaveSpectrum.real_to_r_assocleg(ΔΩ_r_thetaGL, operators, thetaop);
        rddr_ΔΩ = RossbyWaveSpectrum.rddr_operator(ΔΩ_r_ℓℓ′, m, operators);
        @test rddr_ΔΩ ≈ ΔΩ_kk′_ℓℓ′ ≈ C ⊗ r_cheby
        ddr_ΔΩ = RossbyWaveSpectrum.ddr_operator(ΔΩ_r_ℓℓ′, m, operators);
        @test ddr_ΔΩ ≈ C ⊗ Ir

        fr = r -> r^2; fθ = θ -> cos(θ)
        ΔΩ_r_thetaGL = fr.(r) .* fθ.(thetaGL)';
        ΔΩ_r_chebytheta = fr.(r) .* fθ.(chebytheta)';

        @test Tcrinv * rddr * Tcrfwd * ΔΩ_r_thetaGL ≈ 2ΔΩ_r_thetaGL

        ΔΩ_kk′_ℓℓ′ = RossbyWaveSpectrum.real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop);

        ΔΩ_r_ℓℓ′ = RossbyWaveSpectrum.real_to_r_assocleg(ΔΩ_r_thetaGL, operators, thetaop);
        rddr_ΔΩ = RossbyWaveSpectrum.rddr_operator(ΔΩ_r_ℓℓ′, m, operators);
        @test rddr_ΔΩ ≈ 2ΔΩ_kk′_ℓℓ′ ≈ C ⊗ 2r_cheby^2
        ddr_ΔΩ = RossbyWaveSpectrum.ddr_operator(ΔΩ_r_ℓℓ′, m, operators);
        @test ddr_ΔΩ ≈ C ⊗ 2r_cheby
    end

    f = θ -> cos(θ)
    ΔΩ_r_thetaGL = ones(nr) .* f.(thetaGL)';
    ΔΩ_r_chebytheta = ones(nr) .* f.(chebytheta)';

    ΔΩ_kk′_ℓℓ′ = RossbyWaveSpectrum.real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop);
    ΔΩ_r_ℓℓ′ = RossbyWaveSpectrum.real_to_r_assocleg(ΔΩ_r_thetaGL, operators, thetaop);
    @test ΔΩ_kk′_ℓℓ′ ≈ C ⊗ Ir;
    ΔΩ_r_Legendre = RossbyWaveSpectrum.normalizedlegendretransform2(ΔΩ_r_chebytheta);

    (; ωΩr, ωΩθ_by_sinθ, sinθ_ωΩθ, invsinθ_dθ_ωΩr, dθ_ωΩθ, drωΩr, cotθ_ωΩθ, ωΩr_plus_cotθ_ωΩθ) =
        RossbyWaveSpectrum.vorticity_terms(ΔΩ_kk′_ℓℓ′, ΔΩ_r_ℓℓ′, ΔΩ_r_Legendre, m, C, Sd, operators);

    @testset "ωr" begin
        # for ΔΩ = cosθ = P1(cosθ), ωr = 3cos²θ - 1
        @test ωΩr ≈ (3C2 - I) ⊗ Ir
        @test all(x -> isapprox(0, x, atol=1e-10), drωΩr)
        # dθωr = -6cosθsinθ, (1/sinθ) * dθωr = -6cosθ
        @test invsinθ_dθ_ωΩr ≈ -6C ⊗ Ir
    end
    @testset "ωθ" begin
        # ωΩθ/sinθ = -(2 + r∂r)ΔΩ = -2ΔΩ if ΔΩ does not vary in r
        # for ΔΩ = cosθ we obtain ωΩθ/sinθ = -2cosθ
        @test ωΩθ_by_sinθ ≈ -2C ⊗ Ir
        # sinθωΩθ = sin²θ * ωΩθ/sinθ = -2sin²θ*cosθ
        @test sinθ_ωΩθ ≈ (-2S2*C) ⊗ Ir
        # ωΩθ = -2cosθsinθ = -sin(2θ)
        # dθ_ωΩθ = -2cos(2θ) = -2(cos²θ - sin²θ)
        @test dθ_ωΩθ ≈ -2(C2 - S2) ⊗ Ir
        @test cotθ_ωΩθ ≈ -2C2 ⊗ Ir
    end

    fr = r -> r^2; fθ = θ -> cos(θ);
    ΔΩ_r_thetaGL = fr.(r) .* fθ.(thetaGL)';
    ΔΩ_r_chebytheta = fr.(r) .* fθ.(chebytheta)';

    ΔΩ_kk′_ℓℓ′ = RossbyWaveSpectrum.real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop);
    ΔΩ_r_ℓℓ′ = RossbyWaveSpectrum.real_to_r_assocleg(ΔΩ_r_thetaGL, operators, thetaop);
    ΔΩ_r_Legendre = RossbyWaveSpectrum.normalizedlegendretransform2(ΔΩ_r_chebytheta);

    (; ωΩr, ωΩθ_by_sinθ, sinθ_ωΩθ, invsinθ_dθ_ωΩr, dθ_ωΩθ, drωΩr, cotθ_ωΩθ, ωΩr_plus_cotθ_ωΩθ) =
        RossbyWaveSpectrum.vorticity_terms(ΔΩ_kk′_ℓℓ′, ΔΩ_r_ℓℓ′, ΔΩ_r_Legendre, m, C, Sd, operators);

    @testset "ωr" begin
        # for ΔΩ = r²cosθ = P1(cosθ), ωr = r²(3cos²θ - 1)
        @test ωΩr ≈ (3C2 - I) ⊗ r_cheby^2
        @test drωΩr ≈ (3C2 - I) ⊗ 2r_cheby
        # dθωr = -6cosθsinθ, (1/sinθ) * dθωr = -6cosθ
        @test invsinθ_dθ_ωΩr ≈ -6C ⊗ r_cheby^2
    end
    @testset "ωθ" begin
        # ωΩθ/sinθ = -(2 + r∂r)ΔΩ
        # for ΔΩ = r²cosθ we obtain ωΩθ/sinθ = -4r²cosθ
        @test ωΩθ_by_sinθ ≈ -4C ⊗ r_cheby^2
        # sinθωΩθ = sin²θ * ωΩθ/sinθ = -4r²sin²θcosθ
        @test sinθ_ωΩθ ≈ (-4S2*C) ⊗ r_cheby^2
        # ωΩθ = -4r²cosθsinθ = -2r²sin(2θ)
        # dθωΩθ = -4r²cos(2θ) = -4r²(cos²θ - sin²θ)
        @test dθ_ωΩθ ≈ -4(C2 - S2) ⊗ r_cheby^2
        @test cotθ_ωΩθ ≈ -4C2 ⊗ r_cheby^2
    end
end

# @testset "eigenvalues" begin
#     (; C) = RossbyWaveSpectrum.constraintmatrix(operators);

#     m = 5
#     M = RossbyWaveSpectrum.twoΩcrossv(nr, nℓ, m; operators);

#     λ, v = RossbyWaveSpectrum.filter_eigenvalues(
#         RossbyWaveSpectrum.uniform_rotation_spectrum, nr, nℓ, m);
#     @test M * v ≈ v * Diagonal(λ) rtol=1e-2
#     @test all(x -> isapprox(x, 0, atol=1e-5), C * v)
# end
