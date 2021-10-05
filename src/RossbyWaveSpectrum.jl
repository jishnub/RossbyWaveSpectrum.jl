module RossbyWaveSpectrum

using LinearAlgebra
using LinearAlgebra: AbstractTriangular
using MKL
using FillArrays
using Kronecker
using Kronecker: KroneckerProduct
using BlockArrays
using UnPack
using PyPlot
using LaTeXStrings
using OffsetArrays
using DelimitedFiles
using Dierckx
using Distributed
using JLD2
using FastSphericalHarmonics
using FastTransforms
using SphericalHarmonics
using Trapz

export datadir

const SCRATCH = Ref("")
const DATADIR = Ref("")

function __init__()
    SCRATCH[] = get(ENV, "SCRATCH", homedir())
    DATADIR[] = get(ENV, "DATADIR", joinpath(SCRATCH[], "RossbyWaves"))
end

datadir(f) = joinpath(DATADIR[], f)

# Legedre expansion constants
α⁺ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((ℓ-m+1)*(ℓ+m+1)/((2ℓ+1)*(2ℓ+3)))))
α⁻ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((ℓ-m)*(ℓ+m)/((2ℓ-1)*(2ℓ+1)))))

β⁻ℓm(ℓ, m) = (ℓ < abs(m) ? 0.0 : oftype(0.0, √((2ℓ+1)/(2ℓ-1)*(ℓ^2-m^2))))
γ⁺ℓm(ℓ, m) = ℓ * α⁺ℓm(ℓ, m)
γ⁻ℓm(ℓ, m) = ℓ*α⁻ℓm(ℓ, m) - β⁻ℓm(ℓ, m)


function chebyshevnodes(n, a, b)
    nodes = cos.(reverse(pi*((1:n) .- 0.5)./n))
    nodes_scaled = nodes*(b - a)/2 .+ (b + a)/2
    nodes, nodes_scaled
end
chebyshevnodes(n) = chebyshevnodes(n, -1, 1)

"""
    chebyshevpoly(n, x)

Evaluate the Chebyshev polynomials of the first kind ``T_n(x)`` for a degree ``n`` and the
argument ``x``.
"""
function chebyshevpoly(n, x)
    Tc = zeros(n, n);
    Tc[:,1] .= 1
    Tc[:,2] = x
    @views for k = 3:n
        @. Tc[:,k] = 2x*Tc[:,k-1] - Tc[:,k-2]
    end
    Tc
end

"""
    chebyshevderiv(n)

Evaluate the matrix that relates the Chebyshev coefficients of the derivative of a function
``df(x)/dx`` to those of the function ``f(x)``. This assumes that ``-1 <= x <= 1``.
"""
function chebyshevderiv(n)
    M = zeros(n, n)
    for col in 1:n, row in col+1:2:n
        v = (row-1) * (2 - (col == 1))
        M[row, col] = v
    end
    UpperTriangular(transpose(M))
end

"""
    constraintmatrix(params)

Evaluate the matrix ``F`` that captures the constriants on the Chebyshev coefficients
of ``V`` and ``W``, such that
```math
F * [V_{11}, ⋯, V_{1k}, ⋯, V_{n1}, ⋯, V_{nk}, W_{11}, ⋯, W_{1k}, ⋯, W_{n1}, ⋯, W_{nk}] = 0.
```

"""
function constraintmatrix(operators)
    (; params) = operators;
    (; nr, r_in, r_out, nℓ) = params;

    nparams = nr*nℓ;

    Onesr = Ones(1, nr);

    # Radial constraint
    MVn = zeros(2, nr);
    MVno = OffsetArray(MVn, :, 0:nr-1);
    MWn = zeros(2, nr);
    MWno = OffsetArray(MWn, :, 0:nr-1);
    # constraints on W
    for n in 0:nr-1
        MWno[1, n] = 1
        MWno[2, n] = (-1)^n
    end
    # constraints on V
    Δr = r_out - r_in;
    for n in 0:nr-1
        MVno[1, n] = 2n^2/Δr - 1/r_in
        MVno[2, n] = (-1)^n * (2n^2/Δr - 1/r_out)
    end

    C = [
        Ones(1, nℓ) ⊗ MVn Zeros(2, nparams)
        Zeros(2, nparams) Ones(1, nℓ) ⊗ MWn
        ]

    (; C, MVn, MWn)
end

"""
    constraintnullspacematrix(M)

Given a constraint matrix `M` of size `m×n` that satisfies `Mv=0` for the sought eigenvectors `v`, return a matrix `C` of size `n×(n-m)`
whose columns lie in the null space of `M`, and for an arbitrary `y`, the vector `v = C*y` satisfies the constraint `Mv=0`.
"""
function constraintnullspacematrix(M)
    m, n = size(M)
    @assert n >= m
    _, R = qr(M);
    Q, _ = qr(R')
    return collect(Q)[:, end-(n-m)+1:end]
end

function parameters(nr, nℓ)
    nchebyr = nr;
    r_in = 0.71;
    r_out = 0.985;
    Δr = r_out - r_in
    nparams = nchebyr * nℓ;
    return (; nchebyr, r_in, r_out, Δr, nr, nparams, nℓ)
end

include("identitymatrix.jl")
include("forwardinversetransforms.jl")
include("realspectralspace.jl")
include("paddedmatrix.jl")

function chebyshev_forward_inverse(n, boundaries...)
    nodes, r = chebyshevnodes(n, boundaries...);
    Tc = chebyshevpoly(n, nodes);
    Tcfwd = Tc' * 2 /n;
    Tcfwd[1, :] /= 2
    Tcinv = Tc;
    r, ForwardTransform(Tcfwd), InverseTransform(Tcinv)
end

isapproxdiagonal(M) = isapprox(M, Diagonal(M), atol=1e-14, rtol=1e-8)

function spherical_harmonic_transform_plan(ntheta)
    # spherical harmonic transform parameters
    lmax = sph_lmax(ntheta);
    θϕ_sh = FastSphericalHarmonics.sph_points(lmax);
    shtemp = zeros(length.(θϕ_sh));
    PSPH2F = plan_sph2fourier(shtemp);
    PSPHA = plan_sph_analysis(shtemp);
    (; shtemp, PSPHA, PSPH2F, θϕ_sh, lmax);
end

function cosθ_operator(nℓ, m, pad = 4)
    dl = [α⁻ℓm(ℓ,m) for ℓ in m.+(1:nℓ-1+pad)]
    d = zeros(nℓ+pad)
    M = SymTridiagonal(d, dl)'
    PaddedMatrix(M, pad)
end

function sinθdθ_operator(nℓ, m, pad = 4)
    dl = [γ⁻ℓm(ℓ,m) for ℓ in m.+(1:nℓ-1+pad)]
    d = zeros(nℓ+pad)
    du = [γ⁺ℓm(ℓ,m) for ℓ in m.+(0:nℓ-2+pad)]
    M = Tridiagonal(dl, d, du)'
    PaddedMatrix(M, pad)
end

function basic_operators(nr, nℓ)
    params = parameters(nr, nℓ);
    (; nchebyr, r_in, r_out, Δr) = params;
    r, Tcrfwd, Tcrinv = chebyshev_forward_inverse(nr, r_in, r_out);

    dr = chebyshevderiv(nr) * (2/Δr);
    d2r = dr * dr;

    onebyr = 1 ./r;
    onebyr_cheby = Tcrfwd * Diagonal(onebyr) * Tcrinv;
    onebyr2_cheby = Tcrfwd * Diagonal(onebyr)^2 * Tcrinv;

    diml = 696e6;
    dimT = 1/(2 * pi * 453e-9);
    dimrho = 0.2;
    dimp = dimrho * diml^2/dimT^2;
    dimtemp = 6000;
    Nrho = 3;
    npol = 2.6;
    betapol = r_in / r_out; rhoc = 2e-1;
    gravsurf=278; pc = 1e10;
    R = 8.314 * dimrho * dimtemp/dimp;
    G = 6.67e-11; Msol = 2e30;
    zeta0 = (betapol + 1)/(betapol * exp(Nrho/npol) + 1);
    zetai = (1 + betapol - zeta0)/betapol;
    Cp = (npol + 1) * R; gamma_ind = (npol+1)/npol; Cv = npol * R;
    c0 = (2 * zeta0 - betapol - 1)/(1-betapol);
    #^c1 = G * Msol * rhoc/((npol+1)*pc * Tc * (r_out-r_in));
    c1 = (1-zeta0)*(betapol + 1)/(1-betapol)^2;
    ζfn(r) = c0 + c1 * Δr /r
    zeta = ζfn.(r);
    dzeta_dr = @. -c1 * Δr /r^2;
    d2zeta_dr2 = @. 2 * c1 * Δr /r^3;
    grav = @. -pc / rhoc * (npol+1) * dzeta_dr;
    pc = gravsurf .* pc/minimum(grav);
    grav = grav .* gravsurf ./ minimum(grav) * dimT^2/diml;
    ηρ = @. npol/zeta*dzeta_dr;
    ηρ_integral = @. npol * log(zeta)
    ηρ_integral .-= first(ηρ_integral)
    ηρ_definite_integral(r, s) = npol * log(ζfn(s)/ζfn(r))
    # drhrho = @. (- npol/zeta^2 * (dzeta_dr)^2 + npol/zeta * d2zeta_dr2);
    ηρ_cheby = Tcrfwd * Diagonal(ηρ) * Tcrinv;

    DDr = dr + ηρ_cheby;
    D2Dr2 = DDr^2;
    DDr_minus_2byr = DDr - 2*onebyr_cheby;

    # scratch matrices
    # B = D2Dr2 - ℓ(ℓ+1)/r^2
    B = zero(D2Dr2);
    Cℓ′ = zero(DDr);
    BinvCℓ′ = zero(DDr);

    coordinates = (; r);
    transforms = (; Tcrfwd, Tcrinv);
    rad_terms = (; onebyr, onebyr_cheby, ηρ, onebyr2_cheby);
    diff_operators = (; DDr, D2Dr2, DDr_minus_2byr);
    scratch = (; B, Cℓ′, BinvCℓ′);

    (; rad_terms, diff_operators, transforms, coordinates, params, scratch)
end

w1(r, ℓ, r_in, r_out) = r^(ℓ+1)-(r_in^(2ℓ+1))/r^ℓ
w2(r, ℓ, r_in, r_out) = r^(ℓ+1)-(r_out^(2ℓ+1))/r^ℓ
w1w2(r, s, ℓ, r_in, r_out) = w1(r, ℓ, r_in, r_out)*w2(s, ℓ, r_in, r_out)
function Binv_realspace(r, s, ℓ, r_in, r_out, ηρ_definite_integral)
    @assert ℓ >= 0
    @assert r_out > r_in
    @assert r_in <= r <= r_out
    @assert r_in <= s <= r_out
    W = (2ℓ+1)*(r_out^(2ℓ+1) - r_in^(2ℓ+1))
    T = (s <= r) ? w1w2(s, r, ℓ, r_in, r_out) : w1w2(r, s, ℓ, r_in, r_out)
    p = exp(ηρ_definite_integral(r, s))
    p*T/W
end
function Binv_realspace(r::AbstractVector, ℓ, ηρ_definite_integral)
    r_in, r_out = extrema(r)
    Binv_realspace.(r, r', ℓ, r_in, r_out, ηρ_definite_integral)
end

function twoΩcrossv(nr, nℓ, m, operators = basic_operators(nr, nℓ))
    (; params) = operators;
    (; rad_terms, diff_operators, scratch) = operators;

    (; nchebyr) = params;

    (; DDr, D2Dr2, DDr_minus_2byr) = diff_operators;
    (; onebyr_cheby, onebyr2_cheby) = rad_terms;
    (; B, Cℓ′, BinvCℓ′) = scratch;

    nparams = nchebyr * nℓ;
    ℓmin = m;
    ℓs = range(ℓmin, length = nℓ);

    D = 2m * Diagonal(@. 1/(ℓs*(ℓs + 1))) ⊗ I(nchebyr);

    A = zeros(nparams, nparams);
    C = zeros(nparams, nparams);

    (; MWn) = constraintmatrix(operators);
    Q = constraintnullspacematrix(MWn);

    C1 = DDr_minus_2byr

    cosθ = OffsetArray(cosθ_operator(nℓ, m), ℓs, ℓs)
    sinθdθ = OffsetArray(sinθdθ_operator(nℓ, m), ℓs, ℓs)

    for ℓ in ℓs
        @. B = D2Dr2 - ℓ*(ℓ+1) * onebyr2_cheby
        B2 = Q' * B * Q;
        F = lu!(B2)
        ABℓ′top = (ℓ - minimum(m)) * nchebyr + 1;
        ACℓ′vertinds = range(ABℓ′top, length = nr);

        for ℓ′ in ℓ-1:2:ℓ+1
            ℓ′ in ℓs || continue
            @. Cℓ′ = DDr - ℓ′*(ℓ′+1) * onebyr_cheby
            @. Cℓ′ = -2/(ℓ*(ℓ+1))*(ℓ′*(ℓ′+1)*C1*cosθ[ℓ, ℓ′] + Cℓ′*sinθdθ[ℓ, ℓ′])
            BinvCℓ′ .= Q * (F \ (Q' * Cℓ′))
            ABℓ′left = (ℓ′ - minimum(m)) * nchebyr + 1
            ACℓ′horinds = range(ABℓ′left, length = nr);
            ACℓ′ℓ′inds = CartesianIndices((ACℓ′vertinds, ACℓ′horinds));
            A[ACℓ′ℓ′inds] .= Cℓ′
            C[ACℓ′ℓ′inds] .= BinvCℓ′
        end
    end

    [D  A
     C  D];
end

function interp1d(xin, z, xout)
    spline = Spline1D(xin, z)
    spline(xout)
end

function interp2d(xin, yin, z, xout, yout)
    spline = Spline2D(xin, yin, z)
    evalgrid(spline, xout, yout)
end

function read_angular_velocity(r, theta)
    parentdir = joinpath(@__DIR__, "..")
    r_rot_data = vec(readdlm(joinpath(parentdir, "rmesh.orig"))::Matrix{Float64});
    r_rot_data = r_rot_data[1:4:end];

    rot_data = readdlm(joinpath(parentdir, "rot2d.hmiv72d.ave"))::Matrix{Float64};
    rot_data = [rot_data reverse(rot_data[:,2:end], dims = 2)];
    rot_data = (rot_data .- 453.1)./453.1;

    nθ = size(rot_data,2);
    lats_data = LinRange(0, pi, nθ)

    interp2d(lats_data, r_rot_data, rot_data', theta, r)
end

function rotation_velocity_phi(operators)
    (; coordinates, trig_functions) = operators;
    (; r, theta) = coordinates;
    (; sintheta) = trig_functions;

    ΔΩ = read_angular_velocity(r, theta);
    ΔΩ .= 1
    # r along columns, θ along rows
    vϕ = velocity_from_angular_velocity(ΔΩ, r, sintheta);

    Diagonal(vec(ΔΩ)), Diagonal(vec(vϕ))
end

velocity_from_angular_velocity(Ω, r, sintheta) = Ω .* sintheta .* r';

function vorticity_radial_term1(ncheby)
    M = zeros(ncheby, ncheby)
    Mt = OffsetArray(M', 0:ncheby-1, 0:ncheby-1)
    Mt[0, 1] = 2
    for n in 1:ncheby-1, j in abs(n-1):2:min(n+1, ncheby-1)
        Mt[n, j] = 1
    end
    return M
end
function vorticity_radial_term2(ncheby)
    M = zeros(ncheby, ncheby)
    Mt = OffsetArray(M', 0:ncheby-1, 0:ncheby-1)
    Mt[0,0] = 1
    Mt[0,2] = -1
    Mt[1,1] = 0.5
    Mt[1,3] = -0.5
    for n in 2:ncheby-1, j in abs(n-2):2:min(n+2, ncheby-1)
        T = n == j ? 1.0 : sign(n-j)*0.5
        Mt[n, j] = T
    end
    return M
end
function vorticity_radial_from_angular_velocity(Ω::Matrix #= θ along rows, r along columns =#, operators)
    (; params, transforms, identities, diff_operators) = operators;
    (; nchebytheta) = params;
    (; fullfwd) = transforms;
    (; Ir) = identities;
    (; DDtheta) = diff_operators;

    Ωcheby = fullfwd * vec(Ω)

    term1_matrix = kronecker(Ir, vorticity_radial_term1(nchebytheta))
    term2_matrix = kronecker(Ir, vorticity_radial_term2(nchebytheta)) * DDtheta
    (term1_matrix + term2_matrix) * Ωcheby
end

function diffrotterms(nr, ntheta, m, operators = basic_operators(nr, nℓ))
    @unpack trig_functions, rad_terms, diff_operators, transforms = operators;

    @unpack DD_V, DD_W, sintheta_ddtheta_mat, D2Dtheta2, DDtheta, DDr, D2Dr2, DDtheta_realspace, DDr_realspace = diff_operators;
    @unpack costheta_mat, sintheta_mat, cottheta_mat, onebysintheta_mat, sintheta_realspace, costheta_realspace, onebysintheta_realspace = trig_functions;
    @unpack r_mat, onebyr_mat, hrho_mat, onebyr2_mat, drhrho_mat, negr2_mat, r_realspace, onebyr_realspace = rad_terms;
    @unpack fullfwd, fullinv, fullfwdc, fullinvc = transforms;

    laplacianh = horizontal_laplacian(operators, m);

    ΔΩ_realspace, V0_realspace = rotation_velocity_phi(operators);
    V0 = fullfwdc * V0_realspace * fullinvc;
    ΔΩ = fullfwdc * ΔΩ_realspace * fullinvc;

    one_over_rsintheta = onebyr_mat * onebysintheta_mat;
    one_over_rsintheta_realspace = onebyr_realspace * onebysintheta_realspace;

    # Omega is the curl of V = (0,0,V0)
    Omega_r_realspace = one_over_rsintheta_realspace * Diagonal(DDtheta_realspace * diag(sintheta_realspace * V0_realspace));
    Omega_r = fullfwdc * Omega_r_realspace * fullinvc;
    Omega_theta_realspace = Diagonal(-onebyr_realspace * (DDr_realspace * diag(r_realspace * V0_realspace)));
    Omega_theta = fullfwdc * Omega_theta_realspace * fullinvc;

    DDr_Omega_r = fullfwdc * Diagonal(DDr_realspace * diag(Omega_r_realspace)) * fullinvc;
    Dtheta_Omega_r = fullfwdc * Diagonal(DDtheta_realspace * diag(Omega_r_realspace)) * fullinvc;
    DDr_Omega_theta = fullfwdc * Diagonal(DDr_realspace * diag(Omega_theta_realspace)) * fullinvc;
    Dtheta_Omega_theta = fullfwdc * Diagonal(DDtheta_realspace * diag(Omega_theta_realspace)) * fullinvc;

    DDV_V0 = fullfwdc * Diagonal(fullinv * DD_V * fullfwd * diag(V0_realspace)) * fullinvc;

    DDtheta_V0 = fullfwdc * Diagonal(DDtheta_realspace * diag(V0_realspace)) * fullinvc;

    m_over_sintheta = m*onebysintheta_mat;

    # im_vr_V = 0
    im_vr_im_W = collect(-onebyr_mat^2*laplacianh);
    vr_W = im_vr_im_W;
    im_r_vr_im_W = collect(-onebyr_mat*laplacianh);
    im_r2_vr_im_W = collect(-laplacianh);

    im_r_vtheta_V = collect(-m_over_sintheta);
    im_r_vtheta_im_W = collect(DDtheta*DD_W);

    m_over_r_sintheta = collect(m * one_over_rsintheta);

    im_vtheta_V = -m_over_r_sintheta;
    vtheta_V = -im*im_vtheta_V;
    im_vtheta_im_W = collect(DDtheta*(onebyr_mat*DD_W));
    vtheta_W = im_vtheta_im_W;

    vphi_V = collect(-onebyr_mat*DDtheta);
    vphi_im_W = collect(-m_over_r_sintheta*DD_W);

    omega_r_V = collect(-onebyr_mat^2*laplacianh);
    r2_omega_r_V = collect(-laplacianh);

    omega_theta_V = collect(onebyr_mat*DDr*DDtheta);
    omega_theta_im_W = collect(-m_over_r_sintheta*(laplacianh*onebyr2_mat + DDr*DD_W));

    im_omega_phi_V = collect(-m_over_r_sintheta*DD_V);
    im_omega_phi_im_W = collect(onebyr_mat*DDtheta*(DDr*DD_W + onebyr_mat^2 * laplacianh));

    # matrix elements

    # curl vf × ωΩ
    diffrot_V_V_1 = (
        # (Omega_r * (DD_W - 2*onebyr_mat) - DDr_Omega_r + onebyr_mat*Omega_theta*DDtheta) * im_r2_vr_V #= (this is zero) =# -
        -Dtheta_Omega_r * im_r_vtheta_V
        # for a constant rotation rate, this term is -2m
    );

    diffrot_V_W_1 = (
        (Omega_r * (DD_W - 2*onebyr_mat) - DDr_Omega_r + onebyr_mat*Omega_theta*DDtheta) * im_r2_vr_im_W -
        Dtheta_Omega_r * im_r_vtheta_im_W
    );

    # curl uΩ × ωf
    diffrot_V_V_2 = m_over_r_sintheta * V0 * r2_omega_r_V;
    # for a constant rotation rate, this term is mΩ(-∇h^2)

    # diffrot_V_W_2 = m_over_r_sintheta * V0 * r2_omega_r_im_W #= (this is zero) =#;

    diffrot_V_V = diffrot_V_V_1 + diffrot_V_V_2;

    diffrot_V_W = diffrot_V_W_1;

    diffrot_W_V_1 = -(DDtheta + cottheta_mat)*(
        rbyρ_curlρucrossω_ϕ_vϕ_V + rbyρ_curlρucrossω_ϕ_ωr_V +
        rbyρ_curlρucrossω_ϕ_ωθ_V + rbyρ_curlρucrossω_ϕ_ωϕ_V
    ) + m_over_sintheta * (
        rbyρ_curlρucrossω_θ_imvθ_V + rbyρ_curlρucrossω_θ_imωr_V
    );

    # ρ∇u^2 term
    diffrot_W_pre = r_mat * hrho_mat * onebysintheta_mat * (
        DDtheta * sintheta_mat * DDtheta * sintheta_mat * ΔΩ - m^2 * ΔΩ
    );

    diffrot_W_V_2 = diffrot_W_pre * vphi_V;

    diffrot_W_V = diffrot_W_V_1 + diffrot_W_V_2;

    diffrot_W_V_1 = -(DDtheta + cottheta_mat)*(
        rbyρ_curlρucrossω_ϕ_vϕ_imW +
        rbyρ_curlρucrossω_ϕ_ωθ_imW + rbyρ_curlρucrossω_ϕ_ωϕ_imW
    ) + m_over_sintheta * (
        rbyρ_curlρucrossω_θ_imvθ_imW + rbyρ_curlρucrossω_θ_imvr_imW + rbyρ_curlρucrossω_θ_imωr_imW
    );

    diffrot_W_W_2 = diffrot_W_pre * vphi_im_W;

    diffrot_W_W = diffrot_W_W_1 + diffrot_W_W_2;

    OperatorMatrix(diffrot_V_V, diffrot_V_W, diffrot_W_V, diffrot_W_W)
end

function diffrotterms2(nr, ntheta, m, operators = basic_operators(nr, nℓ))
    @unpack trig_functions, rad_terms, diff_operators, transforms, params = operators;

    @unpack nparams = params;
    @unpack DD_V, DD_W, sintheta_ddtheta_mat, D2Dtheta2, DDtheta, DDr, D2Dr2, DDtheta_realspace, DDr_realspace = diff_operators;
    @unpack costheta_mat, sintheta_mat, cottheta_mat, onebysintheta_mat, sintheta_realspace, costheta_realspace, onebysintheta_realspace = trig_functions;
    @unpack r_mat, onebyr_mat, hrho_mat, onebyr2_mat, drhrho_mat, negr2_mat, r_realspace, onebyr_realspace = rad_terms;
    @unpack fullfwd, fullinv, fullfwdc, fullinvc = transforms;

    cos2theta_mat = costheta_mat^2 - sintheta_mat^2;
    sin2theta_mat = 2*sintheta_mat*costheta_mat;

    laplacianh = horizontal_laplacian(operators, m);

    ΔΩ_realspace, V0_realspace = rotation_velocity_phi(operators);
    V0 = fullfwd * V0_realspace * fullinv;
    ΔΩ = fullfwd * ΔΩ_realspace * fullinv;

    one_over_rsintheta = onebyr_mat * onebysintheta_mat;
    one_over_rsintheta_realspace = Diagonal(onebyr_realspace * onebysintheta_realspace);

    # Omega is the curl of V = (0,0,V0)
    Omega_r_realspace = one_over_rsintheta_realspace * Diagonal(DDtheta_realspace * diag(sintheta_realspace * V0_realspace));
    Omega_r = fullfwdc * Omega_r_realspace * fullinvc;
    Omega_theta_realspace = Diagonal(-onebyr_realspace * (DDr_realspace * diag(r_realspace * V0_realspace)));
    Omega_theta = fullfwdc * Omega_theta_realspace * fullinvc;

    DDr_Omega_r = fullfwdc * Diagonal(DDr_realspace * diag(Omega_r_realspace)) * fullinvc;
    Dtheta_Omega_r = fullfwdc * Diagonal(DDtheta_realspace * diag(Omega_r_realspace)) * fullinvc;
    DDr_Omega_theta = fullfwdc * Diagonal(DDr_realspace * diag(Omega_theta_realspace)) * fullinvc;
    Dtheta_Omega_theta = fullfwdc * Diagonal(DDtheta_realspace * diag(Omega_theta_realspace)) * fullinvc;

    DDV_V0 = fullfwdc * Diagonal(fullinv * DD_V * fullfwd * diag(V0_realspace)) * fullinvc;

    DDtheta_V0 = fullfwdc * Diagonal(DDtheta_realspace * diag(V0_realspace)) * fullinvc;

    DDtheta_sintheta_mat = sintheta_mat*DDtheta + costheta_mat;

    m_over_sintheta = m*onebysintheta_mat;

    # im_vr_V = 0
    im_vr_im_W = collect(-onebyr_mat^2*laplacianh);
    vr_W = im_vr_im_W;
    im_r_vr_im_W = collect(-onebyr_mat*laplacianh);
    im_r2_vr_im_W = collect(-laplacianh);
    vr_im_W = -im*im_vr_im_W;

    im_r_vtheta_V = collect(-m_over_sintheta);
    im_r_vtheta_im_W = collect(DDtheta*DD_W);
    im_vtheta_im_W = collect(onebyr_mat * im_r_vtheta_im_W);

    m_over_r_sintheta = m * one_over_rsintheta;

    im_vtheta_V = collect(-m_over_r_sintheta);
    vtheta_V = -im*im_vtheta_V;
    im_vtheta_im_W = collect(DDtheta*(onebyr_mat*DD_W));
    vtheta_W = im_vtheta_im_W;
    vtheta_im_W = -im*im_vtheta_im_W;

    vphi_V = collect(-onebyr_mat*DDtheta);
    im_vphi_V = im*vphi_V;
    r_vphi_V = collect(-DDtheta);
    vphi_im_W = collect(-m_over_r_sintheta*DD_W);
    im_vphi_im_W = im*vphi_im_W;
    r_vphi_im_W = collect(-m_over_sintheta*DD_W);

    omega_r_V = collect(-onebyr_mat^2*laplacianh);
    r2_omega_r_V = collect(-laplacianh);

    omega_theta_V = collect(onebyr_mat*DDr*DDtheta);
    omega_theta_im_W = collect(-m_over_r_sintheta*(laplacianh*onebyr2_mat + DDr*DD_W));

    im_omega_phi_V = collect(-m_over_r_sintheta*DD_V);
    im_omega_phi_im_W = collect(onebyr_mat*DDtheta*(DDr*DD_W + onebyr_mat^2 * laplacianh));

    # T2 terms

    T2r_V = 2sintheta_mat*vphi_V;
    T2r_im_W = 2sintheta_mat*vphi_im_W - m*im_vr_im_W;
    T2θ_V = 2costheta_mat*vphi_V - m*im_vtheta_V;
    T2θ_im_W = 2costheta_mat*vphi_im_W - m*im_vtheta_im_W;
    T2ϕ_V = -(m*im_vphi_V + 2costheta_mat*vtheta_V);
    T2ϕ_im_W = -(m*im_vphi_im_W + 2costheta_mat*vtheta_im_W + 2sintheta_mat*vr_im_W);

    # single curl terms

    curlT2_direct_r_V = r_mat*(m*(DDtheta + 3cottheta_mat)*vphi_V
    -((m^2*I + 2collect(cos2theta_mat))*onebysintheta_mat-2costheta_mat*DDtheta)*im_vtheta_V
    );

    curlT2_direct_r_im_W = r_mat*(m*(DDtheta + 3cottheta_mat)*vphi_im_W
    -((m^2*I + 2collect(cos2theta_mat))*onebysintheta_mat-2costheta_mat*DDtheta)*im_vtheta_im_W
    -(2sintheta_ddtheta_mat + 4costheta_mat)*im_vr_im_W
    );

    curlT2_direct_θ_V = onebyr_mat * (m*onebysintheta_mat*T2r_V + r_mat*(DD_V + onebyr_mat)*real(im*T2ϕ_V));
    curlT2_direct_θ_im_W = onebyr_mat * (m*onebysintheta_mat*T2r_im_W + r_mat*(DD_W + onebyr_mat)*real(im*T2ϕ_im_W));
    curlT2_direct_ϕ_V = onebyr_mat * (r_mat*(DD_V + onebyr_mat)*T2θ_V - DDtheta * T2r_V);
    curlT2_direct_ϕ_im_W = onebyr_mat * (r_mat*(DD_W + onebyr_mat)*T2θ_im_W - DDtheta * T2r_im_W);

    # double curl terms

    curl_curl_T2_direct_r_V = (-r_mat*onebysintheta_mat)*(
        DDtheta_sintheta_mat*real(im*curlT2_direct_ϕ_V) - m*curlT2_direct_θ_V
    );
    curl_curl_T2_direct_r_im_W = (-r_mat*onebysintheta_mat)*(
        DDtheta_sintheta_mat*real(im*curlT2_direct_ϕ_im_W) - m*curlT2_direct_θ_im_W
    );

    OperatorMatrix(curlT2_direct_r_V, curlT2_direct_r_im_W, curl_curl_T2_direct_r_V, curl_curl_T2_direct_r_im_W)
end

function uniform_rotation_spectrum(nr, nℓ, m, operators = basic_operators(nr, nℓ))
    (; C) = RossbyWaveSpectrum.constraintmatrix(operators);

    M = RossbyWaveSpectrum.twoΩcrossv(nr, nℓ, m, operators);

    Z = Zeros(size(C, 1), size(M,2) + size(C, 1) - size(C,2))

    M_constrained = [M C'
                     C Z];

    λ, v = eigen!(M_constrained);
    λ, v[1:end - size(C,1), :]
end

function differential_rotation_spectrum(nr, nℓ, m, operators = basic_operators(nr, nℓ))
    (; C) = RossbyWaveSpectrum.constraintmatrix(operators);

    uniform_rot_operators = RossbyWaveSpectrum.twoΩcrossv(nr, nℓ, m, operators);
    diff_rot_operators = RossbyWaveSpectrum.diffrotterms2(nr, nℓ, m, operators);

    M = uniform_rot_operators + diff_rot_operators;

    Z = Zeros(size(C, 1), size(M,2) + size(C, 1) - size(C,2))

    M_constrained = [M C'
                    C Z];

    λ, v = eigen!(M_constrained);
    λ, v[1:end - size(C,1), :]
end

rossby_ridge(m, ℓ = m) = 2m/(ℓ*(ℓ+1))
eigenvalue_filter(x, m) = isreal(x) && 0 < real(x) < 4rossby_ridge(m)
eigenvector_filter(v, C, atol = 1e-5) = norm(C * v) < atol

function filter_eigenvalues(f, nr, nℓ, m::Integer; operators = basic_operators(nr, nℓ),
        atol_constraint = 1e-5, Δl_cutoff = 5, power_cutoff = 0.9)

    @show m, Libc.gethostname(), BLAS.get_num_threads()
    (; C) = constraintmatrix(operators);
    lam::Vector{ComplexF64}, v::Matrix{ComplexF64} = f(nr, nℓ, m, operators)
    filterfn(λ, v) = begin
        # remove the lagrange multiplier elements
        eigenvalue_filter(λ, m) &&
        eigenvector_filter(v, C, atol_constraint) #&&
        # sphericalharmonic_transform_filter(v, operators, Δl_cutoff, power_cutoff)
    end
    filtinds = axes(lam, 1)[filterfn.(lam, eachcol(v))]
    real(lam[filtinds]), v[:, filtinds]
end

function filter_eigenvalues_mrange(f, nr, nℓ, mr::AbstractVector; operators = basic_operators(nr, nℓ),
    atol_constraint = 1e-5, Δl_cutoff = 5, power_cutoff = 0.9)
    map(m -> filter_eigenvalues(f, nr, nℓ, m; operators, atol_constraint, Δl_cutoff, power_cutoff), mr)
end

function save_eigenvalues(f, nr, nℓ, mr;
    operators = basic_operators(nr, nℓ), atol_constraint = 1e-5,
    Δl_cutoff = 5, power_cutoff = 0.9)

    λv = filter_eigenvalues_mrange(f, nr, nℓ, mr;
        operators, atol_constraint, Δl_cutoff, power_cutoff)
    lam = first.(λv);
    vec = last.(λv);
    fname = joinpath(DATADIR[], "rossby_$(string(nameof(f)))_nr$(nr)_nell$(nℓ).jld2")
    jldsave(fname; lam, vec)
end

plot_eigenvalues_singlem(m, lam) = plot.(m, lam, marker = "o", ms = 4, mfc = "lightcoral", mec= "firebrick")

function plot_rossby_ridges(mr)
    plot(mr, rossby_ridge.(mr), color = "k", label="ℓ = m")
    linecolors = ["r", "b", "g"]
    for (Δℓind, Δℓ) in enumerate(1:3)
        plot(mr, (m -> rossby_ridge(m, m+Δℓ)).(mr), color = linecolors[Δℓind], label="ℓ = m + $Δℓ",
        marker="None")
    end
    # plot(mr, 1.5 .*rossby_ridge.(mr) .- mr, color = "green", label = "-m + 1.5 * 2/(m+1)")
end

function plot_eigenvalues(f, nr, nℓ, mr::AbstractVector; operators = basic_operators(nr, nℓ), atol_constraint = 1e-5, Δl_cutoff = 5)
    λv = filter_eigenvalues_mrange(f, nr, nℓ, mr; operators, atol_constraint, Δl_cutoff)
    lam = map(first, λv)
    plot_eigenvalues(lam, mr)
end

function plot_eigenvalues(fname::String, mr)
    lam = jldopen(fname, "r")["lam"]
    plot_eigenvalues(lam, mr)
end

function plot_eigenvalues(lam::AbstractArray, mr)
    figure()
    plot_eigenvalues_singlem.(mr, lam)
    plot_rossby_ridges(mr)
    xlabel("m", fontsize = 12)
    ylabel(L"\omega/\Omega", fontsize = 12)
    ylim(0.5*2/(maximum(mr)+1), 1.1*2/(minimum(mr)+1))
    legend(loc=1)
end

function plot_eigenvalues_shfilter(fname::String, mr; operators, Δl_cutoff = 5, power_cutoff = 0.9)
    j = jldopen(fname, "r")
    lam = j["lam"]::Vector{Vector{Float64}}
    v = j["vec"]::Vector{Matrix{ComplexF64}}
    for (ind, (m, vm)) in enumerate(zip(mr, v))
        inds_filterred = sphericalharmonic_transform_filter.(eachcol(vm), (operators,), m, Δl_cutoff, power_cutoff)
        inds = axes(vm, 2)[inds_filterred]
        lam[ind] = lam[ind][inds]
        v[ind] = vm[:, inds]
    end
    figure()
    plot_eigenvalues(lam, mr)
end

function remove_spurious_eigenvalues(fname1, fname2, rtol=1e-3)
    lam1 = jldopen(fname1, "r")["lam"]::Vector{Vector{Float64}}
    lam2 = jldopen(fname2, "r")["lam"]::Vector{Vector{Float64}}
    lam = similar(lam1)
    for (ind, (l1, l2)) in enumerate(zip(lam1, lam2))
        lam_i = eltype(first(lam1))[]
        for l1_i in l1, l2_i in l2
            if isapprox(l1_i, l2_i, rtol=rtol)
                push!(lam_i, l1_i)
            end
        end
        lam[ind] = lam_i
    end
    return lam
end

function plot_eigenvalues_remove_spurious(fname1, fname2, mr, rtol=1e-3)
    lam_filt = remove_spurious_eigenvalues(fname1, fname2, rtol)
    for (m, lam) in zip(mr, lam_filt)
        plot(fill(m, length(lam)), lam, "o")
    end
    plot(mr, 1.5 .*rossby_ridge.(mr) .- mr, color = "green", label = "-m + 1.5 * 2/(m+1)")
    xlabel("m", fontsize = 12)
    ylabel(L"\omega/\Omega", fontsize = 12)
    legend()
end

function eigenfunction_rad_sh(v, operators)
    (; params, transforms) = operators;
    (; nparams, nchebyr) = params;
    (; Tcrinv) = transforms
    V_spectrum = v[1:nparams];
    W_spectrum = v[nparams+1:end];
    V_r_ℓm = zero(V_spectrum);
    W_r_ℓm = zero(W_spectrum);
    for ind in 1:nchebyr:nparams-nchebyr+1
        inds = ind .+ (0:nchebyr-1)
        V_r_ℓm[inds] = Tcrinv * V_spectrum[inds]
        W_r_ℓm[inds] = Tcrinv * W_spectrum[inds]
    end
    (; V_r_ℓm, W_r_ℓm)
end

function eigenfunction_realspace(v, operators, m)
    (; params) = operators;
    (; nr, nℓ, nchebyr) = params;
    (; V_r_ℓm, W_r_ℓm) = eigenfunction_rad_sh(v, operators)

    ℓmax = m + nℓ - 1;

    θ, _ = sph_points(ℓmax); nθ = length(θ);
    V_r_θ_m = zeros(eltype(V_r_ℓm), nr, nθ)
    W_r_θ_m = zeros(eltype(W_r_ℓm), nr, nθ)

    for (θind, θi) in enumerate(θ)
        Plcosθ = computePlmcostheta(θi, ℓmax, m, norm = SphericalHarmonics.Orthonormal())
        for (ℓind, ℓ) in enumerate(m:ℓmax)
            Plmcosθ = Plcosθ[(ℓ,m)]
            V_r_θ_m[:, θind] = V_r_ℓm[(ℓind - 1)*nchebyr .+ (1:nchebyr)] * Plmcosθ
            W_r_θ_m[:, θind] = W_r_ℓm[(ℓind - 1)*nchebyr .+ (1:nchebyr)] * Plmcosθ
        end
    end

    (; V_r_θ_m, W_r_θ_m)
end

function eigenfunction_cheby_ℓm_spectrum(v, operators)
    (; params) = operators;
    (; nparams, nr, nℓ) = params;
    V_spectrum = reshape(v[1:nparams], nr, nℓ)
    W_spectrum = reshape(v[nparams+1:end], nr, nℓ)
    (; V_spectrum, W_spectrum)
end

function sphericalharmonic_transform_filter(v, operators, Δl_cutoff = 5, power_cutoff = 0.9)
    (; params) = operators;
    (; nr) = params;
    (; V_r_ℓm, W_r_ℓm) = eigenfunction_rad_sh(v, operators)

    l_cutoff_ind = 1 + Δl_cutoff
    # ensure that most of the power at the surface is below the cutoff
    V_lm_surface = @view V_r_ℓm[nr:nr:end]
    P_frac = sum(abs2, @view V_lm_surface[1:l_cutoff_ind])/ sum(abs2, V_lm_surface)
    P_frac > power_cutoff
end

end # module
