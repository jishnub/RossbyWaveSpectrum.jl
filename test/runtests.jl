using RossbyWaveSpectrum
using RossbyWaveSpectrum: ForwardTransform, InverseTransform, IdentityMatrix, OneHotVector, PaddedMatrix
using RossbyWaveSpectrum: intPl1mPl20Pl3m, intPl1mP′l20Pl3m, PaddedArray
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
(; coordinates, rad_terms, transforms, diff_operators, identities, scratch) = operators;
(; Tcrfwd, Tcrinv) = transforms;
(; r, r_cheby) = coordinates;
(; params) = operators;
(; nchebyr, r_in, r_out, Δr, nparams) = params;
(; rDDr, DDr, rddr, ddr, D2Dr2) = diff_operators;
(; ηρ, ηρ_cheby, onebyr_cheby, onebyr2_cheby) = rad_terms;
r_mid = (r_out + r_in)/2;
rηρ_cheby = r_cheby * ηρ_cheby;
(; Ir, Iℓ) = identities;

m = 5;
Cosθ = RossbyWaveSpectrum.costheta_operator(nℓ, m);
Sinθdθ = RossbyWaveSpectrum.sintheta_dtheta_operator(nℓ, m);
Cos²θ = Cosθ^2;
Cos³θ = Cos²θ*Cosθ;
Sin²θ = I - Cos²θ;
Sin⁻²θ = PaddedArray(inv(Matrix(parent(Sin²θ))), nℓ);
SinθdθI = Sinθdθ ⊗ Ir;

thetaop = RossbyWaveSpectrum.theta_operators(nℓ, m);
(; thetaGL, PLMfwd, PLMinv) = thetaop;

fullfwd = kron(PLMfwd, Tcrfwd);
fullinv = kron(PLMinv, Tcrinv);

(; BC, ZW, MWn) = RossbyWaveSpectrum.constraintmatrix(operators);

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
            @testset "ddr and rddr" begin
                @test Tcrinv * ddr * Tcrfwd * ones(nr) ≈ zeros(nr) atol=1e-10
                @test Tcrinv * ddr * Tcrfwd * r ≈ ones(nr)
                @test Tcrinv * ddr * Tcrfwd * r.^2 ≈ 2r
                @test Tcrinv * ddr * Tcrfwd * r.^3 ≈ 3r.^2
                @test Tcrinv * rddr * Tcrfwd * ones(nr) ≈ zeros(nr) atol=1e-10
                @test Tcrinv * rddr * Tcrfwd * r ≈ r
                @test Tcrinv * rddr * Tcrfwd * r.^2 ≈ 2r.^2
                @test Tcrinv * rddr * Tcrfwd * r.^3 ≈ 3r.^3
            end
        end
    end
end

@testset "boundary conditions constraint" begin
    Vrones = kron(rand(nℓ), ones(nr));
    v_Vrones = [Vrones; rand(nparams)];
    @test length(v_Vrones) == size(BC,2)

    Vrflip = kron(rand(nℓ), [(-1)^n for n = 0:nr-1]);
    v_Vrflip = [Vrflip; rand(nparams)];
    @test length(v_Vrflip) == size(BC,2)

    # ∑_n Wkn = 0
    Wrflip = kron(rand(nℓ), [(-1)^n for n = 0:nr-1]);
    v_Wrflip = [rand(nparams); Wrflip];
    @test length(v_Wrflip) == size(BC,2)
    @test (BC*v_Wrflip)[3] ≈ 0 atol=1e-14

    # ∑_n (-1)^n * Wkn = 0
    Wrones = kron(rand(nℓ), ones(nr));
    v_Wrones = [rand(nparams); Wrones];
    @test length(v_Wrones) == size(BC,2)
    @test (BC*v_Wrones)[4] ≈ 0 atol=1e-14
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
    C_el(ℓ1,ℓ2,m) = √(2/3) * intPl1mPl20Pl3m(ℓ1,1,ℓ2,m);
    @test Cosθ*Cos²θ ≈ Cos²θ*Cosθ ≈ Cos³θ;
    Cos²θ_el(ℓ1,ℓ2,m) = 1/3 * (ℓ1 == ℓ2) + 2/3*√(2/5)*intPl1mPl20Pl3m(ℓ1,2,ℓ2,m);
    Sin²θ = I-Cos²θ;
    @test Cosθ*Sin²θ ≈ Sin²θ*Cosθ;
    @testset "chain rule" begin
        @test Sinθdθ*Cosθ ≈ Cosθ*Sinθdθ - Sin²θ
        @test Sinθdθ*Cos²θ ≈ -2Cosθ*Sin²θ + Cos²θ*Sinθdθ
    end
    Sin²θ_el(ℓ1,ℓ,m) = 2/3 * (ℓ1==ℓ) - 2/3*√(2/5)*intPl1mPl20Pl3m(ℓ1,2,ℓ,m);
    CosθSin²θ = Cosθ*Sin²θ;
    CosθSin²θ_1 = 4/(3*√3) * fcoeff_1010 - 4/(3*√15) * fcoeff_1210
    CosθSin²θ_3 = -4/(3*√15) * fcoeff_1230
    CSin²θ_coeffs = [CosθSin²θ_1, CosθSin²θ_3]
    CSin²θ_el(ℓ1,ℓ,m) = sum(CSin²θ_coeffs[ℓ2ind]*intPl1mPl20Pl3m(ℓ1,ℓ2,ℓ,m) for (ℓ2ind, ℓ2) in enumerate(1:2:3))
    SinθdθSinθdθ = Sinθdθ^2;
    SinθdθSinθdθSinθdθ = Sinθdθ^3;
    CSinθdθ = C*Sinθdθ;
    Sin³θ = Sin²θ*Sinθdθ;

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
            @test Cos²θ[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
            @test Cos²θ[lind, l′ind] ≈ Cos²θ_el(l,l′,m) atol=1e-10  rtol=rtol
        end
    end
    @testset "sin²θ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*sin(θ)^2*Plm(θ, l′, m), 0, pi, rtol=rtol)
            @test Sin²θ[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
            @test Sin²θ[lind, l′ind] ≈ Sin²θ_el(l,l′,m) atol=1e-10  rtol=rtol
        end
    end
    @testset "cosθsin²θ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*cos(θ)*sin(θ)^2*Plm(θ, l′, m), 0, pi, rtol=rtol)
            @test CosθSin²θ[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
        end
    end
    @testset "sinθdθ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*sinθdθPlm(θ, l′, m), 0, pi, rtol=rtol)
            @test Sinθdθ[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
        end
    end
    @testset "sin³θdθ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*sin(θ)^2*sinθdθPlm(θ, l′, m), 0, pi, rtol=rtol)
            @test Sin³θ[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
        end
    end
    @testset "sinθdθ*sinθdθ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*sinθdθsinθdθPlm(θ, l′, m), 0, pi, rtol=rtol)
            @test SinθdθSinθdθ[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
        end
    end
    @testset "sinθdθ*sinθdθ*sinθdθ" begin
        for (lind, l) in enumerate(m .+ (0:nℓ-1)), (l′ind, l′) in enumerate(m .+ (0:nℓ-1))
            int, err = quadgk(θ -> sin(θ)*Plm(θ, l, m)*sinθdθsinθdθsinθdθPlm(θ, l′, m), 0, pi, rtol=rtol)
            @test SinθdθSinθdθSinθdθ[lind, l′ind] ≈ int atol=max(1e-10, err) rtol=rtol
        end
    end
end

@testset "angular velocity in legendre and associated legendre basis" begin
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

        @test sintheta_dtheta_ΔΩ_operator(ΔΩ_r_Legendre, m, operators) ≈ -Sin²θ ⊗ Ir;
        @test SinθdθI*ΔΩ_kk′_ℓℓ′ - ΔΩ_kk′_ℓℓ′*SdI ≈ -Sin²θ ⊗ Ir;

        f = θ -> cos(θ)^2
        ΔΩ_r_thetaGL = ones(nr) .* f.(thetaGL)';
        ΔΩ_r_chebytheta = ones(nr) .* f.(chebytheta)';

        ΔΩ_kk′_ℓℓ′ = RossbyWaveSpectrum.real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop);
        @test ΔΩ_kk′_ℓℓ′ ≈ C^2 ⊗ Ir;
        ΔΩ_r_Legendre = RossbyWaveSpectrum.normalizedlegendretransform2(ΔΩ_r_chebytheta);

        @test sintheta_dtheta_ΔΩ_operator(ΔΩ_r_Legendre, m, operators) ≈ (-2C*Sin²θ) ⊗ Ir;
        @test SdI*ΔΩ_kk′_ℓℓ′ - ΔΩ_kk′_ℓℓ′*SdI ≈ (-2C*Sin²θ) ⊗ Ir;
    end

    @testset "rddr_ΔΩ" begin
        Irddr = Iℓ ⊗ rddr;

        f = θ -> cos(θ)
        ΔΩ_r_thetaGL = ones(nr) .* f.(thetaGL)';
        ΔΩ_r_chebytheta = ones(nr) .* f.(chebytheta)';

        ΔΩ_kk′_ℓℓ′ = RossbyWaveSpectrum.real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop);

        ΔΩ_r_ℓℓ′ = RossbyWaveSpectrum.real_to_r_assocleg(ΔΩ_r_thetaGL, operators, thetaop)
        rddr_ΔΩ = RossbyWaveSpectrum.apply_radial_operator(rddr, ΔΩ_r_ℓℓ′, m, operators)
        @test all(x -> isapprox(0,x,atol=1e-10), rddr_ΔΩ)
        ddr_ΔΩ = RossbyWaveSpectrum.apply_radial_operator(ddr, ΔΩ_r_ℓℓ′, m, operators)
        @test all(x -> isapprox(0,x,atol=1e-10), ddr_ΔΩ)

        fr = r -> r; fθ = θ -> cos(θ)
        ΔΩ_r_thetaGL = fr.(r) .* fθ.(thetaGL)';
        ΔΩ_r_chebytheta = fr.(r) .* fθ.(chebytheta)';

        @test Tcrinv * rddr * Tcrfwd * ΔΩ_r_thetaGL ≈ ΔΩ_r_thetaGL

        ΔΩ_kk′_ℓℓ′ = RossbyWaveSpectrum.real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop);

        ΔΩ_r_ℓℓ′ = RossbyWaveSpectrum.real_to_r_assocleg(ΔΩ_r_thetaGL, operators, thetaop);
        rddr_ΔΩ = RossbyWaveSpectrum.apply_radial_operator(rddr, ΔΩ_r_ℓℓ′, m, operators);
        @test rddr_ΔΩ ≈ ΔΩ_kk′_ℓℓ′ ≈ C ⊗ r_cheby
        ddr_ΔΩ = RossbyWaveSpectrum.apply_radial_operator(ddr, ΔΩ_r_ℓℓ′, m, operators);
        @test ddr_ΔΩ ≈ C ⊗ Ir

        fr = r -> r^2; fθ = θ -> cos(θ)
        ΔΩ_r_thetaGL = fr.(r) .* fθ.(thetaGL)';
        ΔΩ_r_chebytheta = fr.(r) .* fθ.(chebytheta)';

        @test Tcrinv * rddr * Tcrfwd * ΔΩ_r_thetaGL ≈ 2ΔΩ_r_thetaGL

        ΔΩ_kk′_ℓℓ′ = RossbyWaveSpectrum.real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop);

        ΔΩ_r_ℓℓ′ = RossbyWaveSpectrum.real_to_r_assocleg(ΔΩ_r_thetaGL, operators, thetaop);
        rddr_ΔΩ = RossbyWaveSpectrum.apply_radial_operator(rddr, ΔΩ_r_ℓℓ′, m, operators);
        @test rddr_ΔΩ ≈ 2ΔΩ_kk′_ℓℓ′ ≈ C ⊗ 2r_cheby^2
        ddr_ΔΩ = RossbyWaveSpectrum.apply_radial_operator(ddr, ΔΩ_r_ℓℓ′, m, operators);
        @test ddr_ΔΩ ≈ C ⊗ 2r_cheby
    end

    f = θ -> cos(θ)
    ΔΩ_r_thetaGL = ones(nr) .* f.(thetaGL)';
    ΔΩ_r_chebytheta = ones(nr) .* f.(chebytheta)';

    ΔΩ_kk′_ℓℓ′ = RossbyWaveSpectrum.real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop);
    ΔΩ_r_ℓℓ′ = RossbyWaveSpectrum.real_to_r_assocleg(ΔΩ_r_thetaGL, operators, thetaop);
    @test ΔΩ_kk′_ℓℓ′ ≈ C ⊗ Ir;
    ΔΩ_r_Legendre = RossbyWaveSpectrum.normalizedlegendretransform2(ΔΩ_r_chebytheta);

    (; ωΩr, ωΩθ_by_sinθ, sinθ_ωΩθ, invsinθ_dθ_ωΩr, dθ_ωΩθ, drωΩr, cotθ_ωΩθ) =
        RossbyWaveSpectrum.vorticity_terms(ΔΩ_kk′_ℓℓ′, ΔΩ_r_ℓℓ′, ΔΩ_r_Legendre, m, C, Sd, operators);

    @testset "ωr" begin
        # for ΔΩ = cosθ = P1(cosθ), ωr = 3cos²θ - 1
        @test ωΩr ≈ (3Cos²θ - I) ⊗ Ir
        @test all(x -> isapprox(0, x, atol=1e-10), drωΩr)
        # dθωr = -6cosθsinθ, (1/sinθ) * dθωr = -6cosθ
        @test invsinθ_dθ_ωΩr ≈ -6C ⊗ Ir
    end
    @testset "ωθ" begin
        # ωΩθ/sinθ = -(2 + r∂r)ΔΩ = -2ΔΩ if ΔΩ does not vary in r
        # for ΔΩ = cosθ we obtain ωΩθ/sinθ = -2cosθ
        @test ωΩθ_by_sinθ ≈ -2C ⊗ Ir
        # sinθωΩθ = sin²θ * ωΩθ/sinθ = -2sin²θ*cosθ
        @test sinθ_ωΩθ ≈ (-2Sin²θ*C) ⊗ Ir
        # ωΩθ = -2cosθsinθ = -sin(2θ)
        # dθ_ωΩθ = -2cos(2θ) = -2(cos²θ - sin²θ)
        @test dθ_ωΩθ ≈ -2(Cos²θ - Sin²θ) ⊗ Ir
        @test cotθ_ωΩθ ≈ -2Cos²θ ⊗ Ir
    end

    fr = r -> r^2; fθ = θ -> cos(θ);
    ΔΩ_r_thetaGL = fr.(r) .* fθ.(thetaGL)';
    ΔΩ_r_chebytheta = fr.(r) .* fθ.(chebytheta)';

    ΔΩ_kk′_ℓℓ′ = RossbyWaveSpectrum.real_to_chebyassocleg(ΔΩ_r_thetaGL, operators, thetaop);
    ΔΩ_r_ℓℓ′ = RossbyWaveSpectrum.real_to_r_assocleg(ΔΩ_r_thetaGL, operators, thetaop);
    ΔΩ_r_Legendre = RossbyWaveSpectrum.normalizedlegendretransform2(ΔΩ_r_chebytheta);

    (; ωΩr, ωΩθ_by_sinθ, sinθ_ωΩθ, invsinθ_dθ_ωΩr, dθ_ωΩθ, drωΩr, cotθ_ωΩθ) =
        RossbyWaveSpectrum.vorticity_terms(ΔΩ_kk′_ℓℓ′, ΔΩ_r_ℓℓ′, ΔΩ_r_Legendre, m, C, Sd, operators);

    @testset "ωr" begin
        # for ΔΩ = r²cosθ = P1(cosθ), ωr = r²(3cos²θ - 1)
        @test ωΩr ≈ (3Cos²θ - I) ⊗ r_cheby^2
        @test drωΩr ≈ (3Cos²θ - I) ⊗ 2r_cheby
        # dθωr = -6cosθsinθ, (1/sinθ) * dθωr = -6cosθ
        @test invsinθ_dθ_ωΩr ≈ -6C ⊗ r_cheby^2
    end
    @testset "ωθ" begin
        # ωΩθ/sinθ = -(2 + r∂r)ΔΩ
        # for ΔΩ = r²cosθ we obtain ωΩθ/sinθ = -4r²cosθ
        @test ωΩθ_by_sinθ ≈ -4C ⊗ r_cheby^2
        # sinθωΩθ = sin²θ * ωΩθ/sinθ = -4r²sin²θcosθ
        @test sinθ_ωΩθ ≈ (-4Sin²θ*C) ⊗ r_cheby^2
        # ωΩθ = -4r²cosθsinθ = -2r²sin(2θ)
        # dθωΩθ = -4r²cos(2θ) = -4r²(cos²θ - sin²θ)
        @test dθ_ωΩθ ≈ -4(Cos²θ - Sin²θ) ⊗ r_cheby^2
        @test cotθ_ωΩθ ≈ -4Cos²θ ⊗ r_cheby^2
    end
end

@testset "constant differential rotation" begin
    to_blocks(M) = (M[1:nparams, 1:nparams], M[1:nparams, nparams+1:end], M[nparams+1:end, 1:nparams], M[nparams+1:end, nparams+1:end])

    M_uniformrot = RossbyWaveSpectrum.twoΩcrossv(nr, nℓ, m; operators);
    M_uniformrot_11, M_uniformrot_12, M_uniformrot_21, M_uniformrot_22 = to_blocks(M_uniformrot);
    @test [M_uniformrot_11 M_uniformrot_12; M_uniformrot_21 M_uniformrot_22] == M_uniformrot;

    M_diffrot = RossbyWaveSpectrum.diffrotterms(nr, nℓ, m; operators, thetaop, test = true);
    M_diffrot_11, M_diffrot_12, M_diffrot_21, M_diffrot_22 = to_blocks(M_diffrot);
    @test [M_diffrot_11 M_diffrot_12; M_diffrot_21 M_diffrot_22] == M_diffrot;

    @test M_uniformrot_11 ≈ M_diffrot_11 + m * I;
    @test M_uniformrot_22 ≈ M_diffrot_22 + m * I;

    @testset "curl curl terms for uniform differential rotation" begin
        ℓs = range(m, length = 2nℓ);

        ℓℓp1 = ℓs.*(ℓs.+1);
        ℓℓp1_ax2 = Ones(nℓ) .* ℓℓp1[1:nℓ]';
        ℓℓp1_diag = PaddedMatrix(Diagonal(ℓℓp1), nℓ);

        @test Sin²θ .* ℓℓp1_ax2 ≈ Sin²θ * ℓℓp1_diag

        (; ΔΩ_r_thetaGL, ΔΩ_r_Legendre) = RossbyWaveSpectrum.read_angular_velocity(operators, thetaGL, test = true);
        ΔΩ_r_ℓℓ′ = PaddedMatrix((PLMfwd ⊗ Ir) * Diagonal(vec(ΔΩ_r_thetaGL)) * (PLMinv ⊗ Ir), nparams);
        ΔΩ_kk′_ℓℓ′ = PaddedMatrix(fullfwd * Diagonal(vec(ΔΩ_r_thetaGL)) * fullinv, nparams);

        ω_terms = RossbyWaveSpectrum.vorticity_terms(ΔΩ_kk′_ℓℓ′, ΔΩ_r_ℓℓ′, ΔΩ_r_Legendre, m, Cosθ, Sinθdθ, operators);
        (; ωΩr, ωΩθ_by_sinθ, dθ_ωΩθ, cotθ_ωΩθ, sinθ_dθ_ΔΩ_kk′_ℓℓ′) = ω_terms;

        (; T_imW_V, T_imW_imW, T_imW_V1, T_imW_imW1, T_imW_V2, T_imW_imW2, T_imW_V_sum, T_imW_imW_sum,
            im_sinθ_r_by_ρ_curl_ρu_cross_ω_θ_V,
            im_sinθ_r_by_ρ_curl_ρu_cross_ω_θ_imW, sinθ_r_by_ρ_curl_ρu_cross_ω_ϕ_V, sinθ_r_by_ρ_curl_ρu_cross_ω_ϕ_imW,
            Sin²θ_T_imW_V2, Sin²θ_T_imW_imW2) =
        RossbyWaveSpectrum.curl_curl_matrix_terms(m, operators, ω_terms, ΔΩ_kk′_ℓℓ′, ΔΩ_r_ℓℓ′, Cosθ, Sinθdθ, ℓs);

        @testset "W terms" begin
            im_sinθ_r_by_ρ_curl_ρu_cross_ω_θ_imW_exp = -(Sinθdθ^2 ⊗ (2onebyr_cheby * DDr) - 2Cosθ*Sinθdθ ⊗ DDr^2 +
                (2Sin²θ * ℓℓp1_diag) ⊗ onebyr2_cheby - m^2 * ℓℓp1_diag ⊗ onebyr2_cheby + m^2 * Iℓ ⊗ (ddr * DDr));

            @test im_sinθ_r_by_ρ_curl_ρu_cross_ω_θ_imW ≈ im_sinθ_r_by_ρ_curl_ρu_cross_ω_θ_imW_exp

            sinθ_r_by_ρ_curl_ρu_cross_ω_ϕ_imW_exp = m*(-Sinθdθ ⊗ ((ddr + 2onebyr_cheby)*DDr) + (Sinθdθ*ℓℓp1_diag) ⊗ onebyr2_cheby + 2Cosθ ⊗ DDr^2)
            @test sinθ_r_by_ρ_curl_ρu_cross_ω_ϕ_imW ≈ sinθ_r_by_ρ_curl_ρu_cross_ω_ϕ_imW_exp

            Sin²θ_T_imW_imW2_exp = m*(
                (Sin²θ*(ℓℓp1_diag - 2I)) ⊗ DDr^2 - (Sin²θ*(ℓℓp1_diag - 2I)*ℓℓp1_diag) ⊗ onebyr2_cheby
                -(Sin²θ*ℓℓp1_diag) ⊗ (ηρ_cheby * DDr)
            );
            @test Sin²θ_T_imW_imW2 ≈ Sin²θ_T_imW_imW2_exp

            T_imW_imW2_exp = m*(
                (ℓℓp1_diag - 2I) ⊗ DDr^2 - ((ℓℓp1_diag - 2I)*ℓℓp1_diag) ⊗ onebyr2_cheby
                -ℓℓp1_diag ⊗ (ηρ_cheby * DDr)
            );
            @test T_imW_imW2 ≈ (Sin⁻²θ ⊗ Ir) * Sin²θ_T_imW_imW2;
            @test T_imW_imW2 ≈ T_imW_imW2_exp;

            @test T_imW_imW1 ≈ m*ℓℓp1_diag ⊗ (ηρ_cheby * DDr);

            T_imW_imW_exp = m*(
                (ℓℓp1_diag - 2I) ⊗ DDr^2 - ((ℓℓp1_diag - 2I)*ℓℓp1_diag) ⊗ onebyr2_cheby
            );
            @test T_imW_imW_sum ≈ T_imW_imW_exp;
        end

        @testset "V terms" begin
            @test im_sinθ_r_by_ρ_curl_ρu_cross_ω_θ_V ≈ m*(Sinθdθ ⊗ (ddr + 2onebyr_cheby) - 2Cosθ⊗DDr)
            @test sinθ_r_by_ρ_curl_ρu_cross_ω_ϕ_V ≈ (2Sinθdθ^2 ⊗ onebyr_cheby
                -2*(Cosθ*Sinθdθ ⊗ DDr) + Sin²θ*ℓℓp1_diag ⊗ ηρ_cheby +  m^2*(Iℓ ⊗ ddr))

            Sin²θ_T_imW_V2_exp1 = (Sinθdθ^3 ⊗ 2onebyr_cheby -(2Sinθdθ*Cosθ*Sinθdθ) ⊗ DDr
            + (Sinθdθ * Sin²θ * ℓℓp1_diag) ⊗ ηρ_cheby
            - m^2 *(Sinθdθ ⊗ 2onebyr_cheby - 2Cosθ ⊗ DDr)
            );
            @test Sin²θ_T_imW_V2 ≈ Sin²θ_T_imW_V2_exp1;

            Sin²θ_T_imW_V2_exp2 = (-Sin²θ * ℓℓp1_diag * Sinθdθ ⊗ 2onebyr_cheby
            -(2Sinθdθ*Cosθ*Sinθdθ) ⊗ DDr
            + (Sinθdθ * Sin²θ * ℓℓp1_diag) ⊗ ηρ_cheby
            + m^2 *2Cosθ ⊗ DDr
            );
            @test Sin²θ_T_imW_V2 ≈ Sin²θ_T_imW_V2_exp2;

            Sin²θ_T_imW_V2_exp = (Sin²θ ⊗ Ir) * (
                2Sinθdθ ⊗ DDr -(2ℓℓp1_diag*Sinθdθ) ⊗ onebyr_cheby +
                (2Cosθ*ℓℓp1_diag) ⊗ (DDr + ηρ_cheby) +
                (Sinθdθ*ℓℓp1_diag) ⊗ ηρ_cheby
            );

            @test Sin²θ_T_imW_V2 ≈ Sin²θ_T_imW_V2_exp;

            T_imW_V2_exp = (
                2Sinθdθ ⊗ DDr -(2ℓℓp1_diag*Sinθdθ) ⊗ onebyr_cheby +
                (2Cosθ*ℓℓp1_diag) ⊗ (DDr + ηρ_cheby) +
                (Sinθdθ*ℓℓp1_diag) ⊗ ηρ_cheby
            );
            @test T_imW_V2 ≈ T_imW_V2_exp
        end

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
