module RossbyWaveSpectrum

using LinearAlgebra
using MKL
using FillArrays
using Kronecker
using BlockArrays
using UnPack
using PyPlot
using LaTeXStrings
using OffsetArrays
using DelimitedFiles
using Dierckx

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
    constraintmatrix(nr, r_in, r_out, Itheta)

Evaluate the matrix ``C`` that captures the constriants on the Chebyshev coefficients
of ``V`` and ``W``, such that
```math
C * [V_{11}, ⋯, V_{1k}, ⋯, V_{n1}, ⋯, V_{nk}, W_{11}, ⋯, W_{1k}, ⋯, W_{n1}, ⋯, W_{nk}] = 0.
```
"""
function constraintmatrix(nr, r_in, r_out, Itheta)
    M = zeros(4, 2nr)
    MV = OffsetArray((@view M[1:2, 1:nr]), :, 0:nr-1)
    MW = OffsetArray((@view M[3:4, nr+1:end]), :, 0:nr-1)
    # constraints on W
    for n in 0:nr-1
        MW[1, n] = 1
        MW[2, n] = (-1)^n
    end
    # constraints on V
    Δr = r_out - r_in
    for n in 0:nr-1
        MV[1, n] = 2n^2/Δr - 1/r_in
        MV[2, n] = (-1)^n * (2n^2/Δr - 1/r_out)
    end
    kronecker(M, Itheta)
end

function boundary_condition_prefactor(nr, ntheta, r_in, r_out)
    F = constraintmatrix(nr, r_in, r_out, IdentityMatrix(ntheta))
    Fadj = collectadj(F);
    collect(I - Fadj * ((F*Fadj) \ F))
end

function parameters(nr, ntheta)
    nchebytheta = ntheta;
    nchebyr = nr;
    r_in = 0.71;
    r_out = 0.985;
    Δr = r_out - r_in
    nparams = nchebytheta * nchebyr
    return (; nchebytheta, nchebyr,nparams, r_in, r_out, Δr)
end

include("identitymatrix.jl")
include("forwardinversetransforms.jl")

function chebyshev_forward_inverse(n, boundaries...)
    nodes, r = chebyshevnodes(n, boundaries...);
    Tc = chebyshevpoly(n, nodes);
    Tcfwd = Tc' * 2 /n;
    Tcfwd[1, :] /= 2
    Tcinv = Tc;
    r, ForwardTransform(Tcfwd), InverseTransform(Tcinv)
end

function reverse_theta(costheta, Tcthetafwd, Tcthetainv)
    costheta = reverse(costheta)
    Tcthetafwd = ForwardTransform(reverse(parent(Tcthetafwd), dims = 2));
    Tcthetainv = InverseTransform(reverse(parent(Tcthetainv), dims = 1));
    return costheta, Tcthetafwd, Tcthetainv
end

function laplacianh_theta(nchebytheta)
    L = zeros(0:nchebytheta-1, 0:nchebytheta-1)
    for n in 0:nchebytheta-1, j in isodd(n):2:n
        if n == j
            T = -n*(n+1)
        else
            T = -n*(2-iszero(j))
        end
        L[j, n] = T
    end
    return parent(L)
end

function basic_operators(nr, ntheta)
    params = parameters(nr, ntheta);
    @unpack nchebytheta, nchebyr, nparams, r_in, r_out, Δr = params;
    r, Tcrfwd, Tcrinv = chebyshev_forward_inverse(nr, r_in, r_out);

    dr = chebyshevderiv(nr) * (2/Δr);
    d2r = dr * dr;

    Ir = IdentityMatrix(nchebyr);
    hrho = zeros(nr);

    diml = 696e6; dimT = 1/(2 * pi * 453e-9); dimrho = 0.2; dimp = dimrho * diml^2/dimT^2; dimtemp = 6000;
    Nrho = 3; npol = 2.6; betapol = r_in / r_out; rhoc = 2e-1; gravsurf=278; pc = 1e10;
    R = 8.314 * dimrho * dimtemp/dimp; G = 6.67e-11; Msol = 2e30;
    zeta0 = (betapol + 1)/(betapol * exp(Nrho/npol) + 1);
    zetai = (1 + betapol - zeta0)/betapol;
    Cp = (npol + 1) * R; gamma_ind = (npol+1)/npol; Cv = npol * R;
    c0 = (2 * zeta0 - betapol - 1)/(1-betapol);
    #^c1 = G * Msol * rhoc/((npol+1)*pc * Tc * (r_out-r_in));
    c1 = (1-zeta0)*(betapol + 1)/(1-betapol)^2;
    zeta = @. c0 + c1 * Δr /r;
    dzeta_dr = @. -c1 * Δr /r^2;
    d2zeta_dr2 = @. 2 * c1 * Δr /r^3;
    grav = @. -pc / rhoc * (npol+1) * dzeta_dr;
    pc = gravsurf .* pc/minimum(grav);
    grav = grav .* gravsurf ./ minimum(grav) * dimT^2/diml;
    hrho = @. npol/zeta*dzeta_dr;
    drhrho = @. (- npol/zeta^2 * (dzeta_dr)^2 + npol/zeta * d2zeta_dr2);

    # theta terms

    costheta, Tcthetafwd, Tcthetainv = reverse_theta(chebyshev_forward_inverse(ntheta)...);

    theta = acos.(costheta);
    sintheta, cottheta = sin.(theta), cot.(theta);
    sintheta_spectral = Tcthetafwd * Diagonal(sintheta) * Tcthetainv
    sintheta_mat = kronecker(Ir, sintheta_spectral);
    onebysintheta_mat = kronecker(Ir, Tcthetafwd * Diagonal(1 ./ sintheta) * Tcthetainv);
    costheta_mat = kronecker(Ir, Tcthetafwd * Diagonal(costheta) * Tcthetainv);
    cottheta_mat = kronecker(Ir, Tcthetafwd * Diagonal(cottheta) * Tcthetainv);
    cosec2theta_mat = onebysintheta_mat^2;

    dtheta = -sintheta_spectral * chebyshevderiv(ntheta);
    d2theta = dtheta * dtheta;

    DDtheta = kronecker(Ir, dtheta);
    D2Dtheta2 = kronecker(Ir, d2theta);
    sintheta_ddtheta_mat = sintheta_mat * DDtheta;
    laplacianhtheta = kronecker(Ir, laplacianh_theta(nchebytheta));

    fullinv = kronecker(Tcrinv, Tcthetainv)
    fullfwd = kronecker(Tcrfwd, Tcthetafwd)

    DDtheta_realspace = fullinv * DDtheta * fullfwd;
    D2Dtheta2_realspace = fullinv * D2Dtheta2 * fullfwd;

    Itheta = IdentityMatrix(nchebytheta);

    # Derivative matrix wrt r
    DDr = kronecker(dr, Itheta);
    D2Dr2 = kronecker(d2r, Itheta);
    r_mat = kronecker(Tcrfwd * Diagonal(r) * Tcrinv, Itheta);
    negr2_mat = kronecker(Tcrfwd * Diagonal(-r.^2) * Tcrinv, Itheta);
    onebyr_mat = kronecker(Tcrfwd * Diagonal(1 ./ r) * Tcrinv, Itheta);
    onebyr2_mat = kronecker(Tcrfwd * Diagonal(1 ./ r.^2) * Tcrinv, Itheta);
    hrho_mat = kronecker(Tcrfwd * Diagonal(hrho) * Tcrinv, Itheta);
    drhrho_mat = kronecker(Tcrfwd * Diagonal(drhrho) * Tcrinv, Itheta);

    DDr_realspace = fullinv * DDr * fullfwd;
    D2Dr2_realspace = fullinv * D2Dr2 * fullfwd;

    DD_V = DDr + hrho_mat;
    DD_W = DD_V;

    fullfwd = kronecker(Tcrfwd, Tcthetafwd);
    fullinv = kronecker(Tcrinv, Tcthetainv);

    coordinates = (; r, theta)
    transforms = (; Tcrfwd, Tcrinv, Tcthetafwd, Tcthetainv, fullfwd, fullinv)
    trig_functions = (; costheta_mat, sintheta_mat, cottheta_mat, onebysintheta_mat, cosec2theta_mat,
                        sintheta, cottheta, costheta);
    rad_terms = (; onebyr_mat, hrho_mat, onebyr2_mat, drhrho_mat, r_mat, negr2_mat);
    identities = (; Itheta, Ir);
    diff_operators = (; DD_V, DD_W, sintheta_ddtheta_mat, D2Dtheta2, DDtheta, DDr, D2Dr2,
        laplacianhtheta, dtheta, d2theta, dr, d2r, DDr_realspace, D2Dr2_realspace, DDtheta_realspace, D2Dtheta2_realspace);

    (; trig_functions, identities, rad_terms, diff_operators, transforms, coordinates, params)
end

function horizontal_laplacian(operators, m)
    @unpack trig_functions, diff_operators = operators;
    @unpack laplacianhtheta = diff_operators;
    @unpack cosec2theta_mat = trig_functions;
    laplacianh = laplacianhtheta - m^2 * cosec2theta_mat;
end

struct OperatorMatrix{T}
    VV :: Matrix{T}
    VW :: Matrix{T}
    WV :: Matrix{T}
    WW :: Matrix{T}
end

function OperatorMatrix(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix)
    T = mapreduce(eltype, promote_type, (A, B, C, D))
    OperatorMatrix(map(x -> convert(Matrix{T}, x), (A, B, C, D))...)
end

struct RHSMatrix{T}
    R11 :: Matrix{T}
    R22 :: Matrix{T}
end

function RHSMatrix(A::AbstractMatrix, B::AbstractMatrix)
    T = mapreduce(eltype, promote_type, (A, B))
    RHSMatrix(map(x -> convert(Matrix{T}, x), (A, B))...)
end

function Base.:+(A::OperatorMatrix, B::OperatorMatrix)
    OperatorMatrix(A.VV + B.VV, A.VW + B.VW, A.WV + B.WV, A.WW + B.WW)
end

function Base.:\(R::RHSMatrix, M::OperatorMatrix)
    [R.R11\M.VV   R.R11\M.VW
     R.R22\M.WV   R.R22\M.WW]
end

function twoΩcrossv(nr, ntheta, m, operators = basic_operators(nr, ntheta))
    @unpack params = operators;
    @unpack trig_functions, rad_terms, diff_operators = operators;

    @unpack nparams = params;

    @unpack DD_V, DD_W, sintheta_ddtheta_mat, D2Dtheta2, DDtheta, DDr, D2Dr2 = diff_operators;
    @unpack costheta_mat, sintheta_mat, cottheta_mat, onebysintheta_mat = trig_functions;
    @unpack onebyr_mat, hrho_mat, onebyr2_mat, drhrho_mat = rad_terms;

    laplacianh = horizontal_laplacian(operators, m);

    W_Z1 = -(D2Dr2 + 2hrho_mat * DDr + hrho_mat^2 + drhrho_mat);
    W_Z2 = -onebyr2_mat * laplacianh;
    W_Z = collect(W_Z1) + collect(W_Z2);

    RHS_22 = laplacianh * W_Z1  + laplacianh * W_Z2;
    RHS_11 = collect(laplacianh);

    twocostheta_laplacian = 2costheta_mat * laplacianh;
    twosintheta_ddtheta = 2sintheta_ddtheta_mat;
    twosintheta_ddtheta_laplacian = twosintheta_ddtheta * laplacianh;

    EqV_V = -2m * Eye(nparams);
    EqV_W = -twocostheta_laplacian * (DD_W - 2onebyr_mat) + twosintheta_ddtheta * DD_W + twosintheta_ddtheta_laplacian * onebyr_mat;

    EqW_V = twocostheta_laplacian * (DD_V - 2onebyr_mat) + (-twosintheta_ddtheta * DD_V) + (-twosintheta_ddtheta_laplacian * onebyr_mat);
    EqW_W = -2m * W_Z;

    uniform_rot_operators = OperatorMatrix(EqV_V, EqV_W, EqW_V, EqW_W);
    RHS = RHSMatrix(RHS_11, RHS_22);

    (; uniform_rot_operators, RHS)
end

function interp2d(xin, yin, z, xout, yout)
    spline = Spline2D(xin, yin, z)
    evalgrid(spline, xout, yout)
end

function read_angular_velocity(r, theta)
    r_rot_data = vec(readdlm("../rmesh.orig")::Matrix{Float64});
    r_rot_data = r_rot_data[1:4:end];

    rot_data = readdlm("../rot2d.hmiv72d.ave")::Matrix{Float64};
    rot_data = [rot_data reverse(rot_data[:,2:end], dims = 2)];
    rot_data = (rot_data .- 453.1)./453.1;

    rot_data .= 1

    nθ = size(rot_data,2);
    lats_data = LinRange(0, pi, nθ)

    interp2d(lats_data, r_rot_data, rot_data', theta, r)
end

function rotation_velocity_phi(operators)
    @unpack coordinates, trig_functions, transforms = operators;
    @unpack r, theta = coordinates;
    @unpack sintheta = trig_functions;

    ΔΩ = read_angular_velocity(r, theta);
    ΔΩ .= 1
    # r along columns, θ along rows
    vϕ = velocity_from_angular_velocity(ΔΩ, r, sintheta);

    Diagonal(vec(vϕ))
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
    @unpack params, transforms, identities, diff_operators = operators;
    @unpack nchebytheta = params;
    @unpack fullfwd = transforms;
    @unpack Ir = identities;
    @unpack DDtheta = diff_operators;

    Ωcheby = fullfwd * vec(Ω)

    term1_matrix = kronecker(Ir, vorticity_radial_term1(nchebytheta))
    term2_matrix = kronecker(Ir, vorticity_radial_term2(nchebytheta)) * DDtheta
    (term1_matrix + term2_matrix) * Ωcheby
end

function diffrotterms(nr, ntheta, m, operators = basic_operators(nr, ntheta))
    @unpack trig_functions, rad_terms, diff_operators, transforms = operators;

    @unpack DD_V, DD_W, sintheta_ddtheta_mat, D2Dtheta2, DDtheta, DDr, D2Dr2 = diff_operators;
    @unpack costheta_mat, sintheta_mat, cottheta_mat, onebysintheta_mat = trig_functions;
    @unpack r_mat, onebyr_mat, hrho_mat, onebyr2_mat, drhrho_mat, negr2_mat = rad_terms;
    @unpack fullfwd, fullinv = transforms;

    laplacianh = horizontal_laplacian(operators, m);

    V0_realspace = rotation_velocity_phi(operators);
    V0 = fullfwd * V0_realspace * fullinv;

    one_over_rsintheta = onebyr_mat * onebysintheta_mat;

    # Omega is the curl of V = (0,0,V0)
    Omega_r = one_over_rsintheta * fullfwd * Diagonal(fullinv * DDtheta * fullfwd * diag(fullinv * sintheta_mat * V0 * fullfwd)) * fullinv;
    Omega_theta = fullfwd * Diagonal(fullinv * (-onebyr_mat * (DDr * fullfwd * diag(fullinv * r_mat * V0 * fullfwd)))) * fullinv;

    DDr_Omega_r = fullfwd * Diagonal(fullinv * DDr * fullfwd * diag(fullinv * Omega_r * fullfwd)) * fullinv;
    Dtheta_Omega_r = fullfwd * Diagonal(fullinv * DDtheta * fullfwd * diag(fullinv * Omega_r * fullfwd)) * fullinv;
    Dtheta_Omega_theta = fullfwd * Diagonal(fullinv * DDtheta * fullfwd * diag(fullinv * Omega_theta * fullfwd)) * fullinv;

    DDV_V0 = fullfwd * Diagonal(fullinv * DD_V * fullfwd * diag(V0_realspace)) * fullinv;

    DDtheta_V0 = fullfwd * Diagonal(fullinv * DDtheta * fullfwd * diag(V0_realspace)) * fullinv;

    DDtheta_V0_Sym = DDtheta_V0 + V0 * DDtheta;

    DDtheta_DDV_V0_sym = DDtheta * DDV_V0 + DDV_V0 * DDtheta;



    m_over_sintheta = m*onebysintheta_mat;

    im_r_vr_im_W = -onebyr_mat*laplacianh;
    im_r2_vr_im_W = -laplacianh;

    im_r_vtheta_V = -m_over_sintheta;
    im_r_vtheta_im_W = DDtheta*DD_W;

    m_over_r_sintheta = m * one_over_rsintheta;

    im_vtheta_V = -m_over_r_sintheta;
    im_vtheta_im_W = DDtheta*(onebyr_mat*DD_W);

    vphi_V = -onebyr_mat*DDtheta;
    vphi_im_W = -m_over_r_sintheta*DD_W;

    omega_r_V = -onebyr_mat^2*laplacianh;
    r2_omega_r_V = -laplacianh;

    omega_theta_V = onebyr_mat*DDr*DDtheta;
    omega_theta_im_W = -m_over_r_sintheta*(laplacianh*onebyr2_mat + DDr*DD_W);

    im_omega_phi_V = -m_over_r_sintheta*DD_V;
    im_omega_phi_im_W = onebyr_mat*DDtheta*(DDr*DD_W + onebyr_mat^2 * laplacianh);

    # matrix elements

    # curl vf × ωΩ
    diffrot_V_V_1 = -Dtheta_Omega_r*im_r_vtheta_V;
    diffrot_V_W_1 = -(
        (Omega_r * (DD_W - 2*onebyr_mat) - DDr_Omega_r + onebyr_mat*Omega_theta*DDtheta + Dtheta_Omega_r*DDtheta*DD_W) * im_r2_vr_im_W -
        Dtheta_Omega_r * im_r_vtheta_im_W
    );

    # curl uΩ × ωf
    diffrot_V_V_2 = m_over_r_sintheta * V0 * r2_omega_r_V;
    # diffrot_V_W_2 = 0;

    diffrot_V_V = diffrot_V_V_1 + diffrot_V_V_2;

    diffrot_V_W = diffrot_V_W_1;

    diffrot_W_V = -(
        (DDtheta_DDV_V0_sym - DDtheta_V0_Sym)

    )

    OperatorMatrix(diffrot_V_V, diffrot_V_W, diffrot_W_V, diffrot_W_W)
end

function uniform_rotation_spectrum(nr, ntheta, m, operators = basic_operators(nr, ntheta))
    @unpack r_in, r_out = RossbyWaveSpectrum.parameters(nr, ntheta);

    F = RossbyWaveSpectrum.boundary_condition_prefactor(nr, ntheta, r_in, r_out);

    uniform_rot_operators, RHS = RossbyWaveSpectrum.twoΩcrossv(nr, ntheta, m, operators);

    M = RHS \ uniform_rot_operators;

    eigen!(F * M);
end

function differential_rotation_spectrum(nr, ntheta, m, operators = basic_operators(nr, ntheta))
    @unpack r_in, r_out = RossbyWaveSpectrum.parameters(nr, ntheta);

    F = RossbyWaveSpectrum.boundary_condition_prefactor(nr, ntheta, r_in, r_out);

    uniform_rot_operators, RHS = RossbyWaveSpectrum.twoΩcrossv(nr, ntheta, m, operators);
    diff_rot_operators = RossbyWaveSpectrum.diffrotterms(nr, ntheta, m, operators);

    M = RHS \ (uniform_rot_operators + diff_rot_operators)

    eigen!(F * M);
end

rossby_ridge(m) = 2/(m+1)
eigenvalue_filter(x, m) = abs(imag(x)) < abs(real(x))*1e-6 && 0 < real(x) < 4rossby_ridge(m)

function filter_eigenvalues(f, nr, ntheta, m, operators = basic_operators(nr, ntheta))
    lam::Vector{ComplexF64}, _ = f(nr, ntheta, m, operators)
    filterfn(x) = eigenvalue_filter(x, m)
    real.(filter(filterfn, lam))
end

function plot_eigenvalues_singlem(f, nr, ntheta, m, operators = basic_operators(nr, ntheta))
    lam = filter_eigenvalues(f, nr, ntheta, m, operators)
    plot.(m, lam, marker = "o", ms = 3)
end

function plot_eigenvalues(f, nr, ntheta, mr::AbstractVector)
    operators = basic_operators(nr, ntheta)
    plot_eigenvalues_singlem.((f,), nr, ntheta, mr, (operators,))
    plot(mr, rossby_ridge.(mr), color = "k", label="2/(m+1)")
    multiplier = 2.5
    plot(mr, multiplier .*rossby_ridge.(mr), color = "brown", label = "$multiplier * 2/(m+1)")
    xlabel("m", fontsize = 12)
    ylabel(L"\omega/\Omega", fontsize = 12)
    legend()
end

end # module
