struct PaddedMatrix{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    p :: M
    pad :: Int
end

Base.parent(A::PaddedMatrix) = A.p
Base.size(A::PaddedMatrix) = size(A.p) .- A.pad
function Base.getindex(A::PaddedMatrix, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    @inbounds parent(A)[i,j]
end

function Base.:*(A::PaddedMatrix, B::PaddedMatrix)
    pad_new = min(A.pad, B.pad)
    PaddedMatrix(parent(A) * parent(B), pad_new)
end

Base.:*(A::PaddedMatrix, B::AbstractVector) = Matrix(A) * B
Base.:*(A::PaddedMatrix, B::AbstractMatrix) = Matrix(A) * B
Base.:*(B::AbstractMatrix, A::PaddedMatrix) = B * Matrix(A)
for T in [:Diagonal, :AbstractTriangular]
    @eval Base.:*(A::PaddedMatrix, B::$T) = Matrix(A) * B
    @eval Base.:*(A::$T, B::PaddedMatrix) = A * Matrix(B)
end

Base.:^(A::PaddedMatrix, n::Int) = PaddedMatrix(parent(A)^n, A.pad)
