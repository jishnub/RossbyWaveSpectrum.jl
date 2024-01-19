function eigensystem_satisfy_filter(Feig::FilteredEigen, m::Integer, ind::Integer, args...; kwargs...)
    λ, v = Feig[m][ind]
    eigensystem_satisfy_filter(λ, v, args...; kwargs...)
end
function eigensystem_satisfy_frac(Feig::FilteredEigen, m::Integer, ind::Integer, args...; kwargs...)
    λ, v = Feig[m][ind]
    eigensystem_satisfy_frac(λ, v, args...; kwargs...)
end

function eigensystem_satisfy_frac(λ::Number, v::StructVector{<:Complex},
        AB::Tuple{StructMatrix{<:Complex}, AbstractMatrix{<:Real}}, args...; kw...)

    eigensystem_satisfy_frac(λ, v, map(computesparse, AB), args...; kw...)
end

function eigensystem_satisfy_filter(λ::Number, v::StructVector{<:Complex},
        AB::Tuple{StructMatrix{<:Complex}, AbstractMatrix{<:Real}}, args...; kw...)

    eigensystem_satisfy_filter(λ, v, map(computesparse, AB), args...; kw...)
end

function eigensystem_satisfy_frac(λ::Number, v::StructVector{<:Complex},
        AB::Tuple{StructMatrix{<:Complex}, AbstractMatrix{<:Real}},
        MVcache = allocate_MVcache(size(AB[1], 1)))

    A, B = AB;
    Av, λBv = MVcache;

    mul!(Av.re, A.re, v.re)
    mul!(Av.re, A.im, v.im, -1.0, 1.0)
    mul!(Av.im, A.re, v.im)
    mul!(Av.im, A.im, v.re,  1.0, 1.0)

    mul!(λBv.re, B, v.re)
    mul!(λBv.im, B, v.im)
    λBv .*= λ

    normAv = norm(Av)
    normλBv = norm(λBv)
    Av .-= λBv
    normmax = max(normAv, normλBv)
    norm(Av)/normmax
end

"""
    eigensystem_satisfy_filter(λ, v, (A,B), MVcache = allocate_MVcache(size(A, 1));
        rtol = RossbyWaveSpectrum.DefaultFilterParams[:eigen_rtol])

Return whether the eigenvalue ``λ`` and mode ``v`` satisfy
``A\\mathbf{v}=λ B\\mathbf{v}`` to within the relative tolerance `rtol`.
"""
function eigensystem_satisfy_filter(ω_over_Ω0::Number, v::StructVector{<:Complex},
        AB::Tuple{StructMatrix{<:Complex}, AbstractMatrix{<:Real}},
        MVcache = allocate_MVcache(size(AB[1], 1));
        rtol = DefaultFilterParams[:eigen_rtol])

    eigensystem_satisfy_frac(ω_over_Ω0, v, AB, MVcache) <= rtol
end
