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

const TStructSparseComplexMat{T} = @NamedTuple{re::SparseMatrixCSC{T, Int64}, im::SparseMatrixCSC{T, Int64}}
function eigensystem_satisfy_frac(λ::Number, v::StructVector{<:Complex},
        AB::Tuple{StructArray{Complex{T},2,TStructSparseComplexMat{T}}, SparseMatrixCSC{T, Int64}},
        MVcache::NTuple{2, StructArray{<:Complex,1}} = allocate_MVcache(size(AB[1], 1))) where {T<:Real}

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

function eigensystem_satisfy_filter(λ::Number, v::StructVector{<:Complex},
        AB::Tuple{StructArray{Complex{T},2,TStructSparseComplexMat{T}}, SparseMatrixCSC{T, Int64}},
        MVcache::NTuple{2, StructArray{<:Complex,1}} = allocate_MVcache(size(AB[1], 1));
        rtol = DefaultFilterParams[:eigen_rtol]) where {T<:Real}

    eigensystem_satisfy_frac(λ, v, AB, MVcache) <= rtol
end
