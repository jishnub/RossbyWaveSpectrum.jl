struct PaddedArray{T,N,M<:AbstractArray{T,N}} <: AbstractArray{T,N}
    p :: M
    pad :: Int
end

const PaddedMatrix{T,M<:AbstractMatrix{T}} = PaddedArray{T,2,M}
const PaddedVector{T,M<:AbstractVector{T}} = PaddedArray{T,1,M}
PaddedMatrix(A::AbstractMatrix{T}, pad::Integer) where {T} = PaddedArray{T,2,typeof(A)}(A, pad)
PaddedVector(A::AbstractVector{T}, pad::Integer) where {T} = PaddedArray{T,1,typeof(A)}(A, pad)

Base.parent(A::PaddedArray) = A.p
Base.size(A::PaddedArray) = size(A.p) .- A.pad
Base.@propagate_inbounds function Base.getindex(A::PaddedArray{<:Any,N}, i::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(A, i...)
    @inbounds parent(A)[i...]
end
pad(P::PaddedArray) = P.pad
pad(A::AbstractArray) = 0

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

Base.:*(a::Number, B::PaddedMatrix) = PaddedMatrix(a*parent(B), B.pad)
Base.:*(B::PaddedMatrix, a::Number) = PaddedMatrix(parent(B)*a, B.pad)

Base.:^(A::PaddedMatrix, n::Int) = PaddedMatrix(parent(A)^n, A.pad)

Base.:+(I::UniformScaling, B::PaddedMatrix) = PaddedMatrix(I + parent(B), B.pad)
Base.:+(B::PaddedMatrix, I::UniformScaling) = PaddedMatrix(parent(B) + I, B.pad)
Base.:-(I::UniformScaling, B::PaddedMatrix) = PaddedMatrix(I - parent(B), B.pad)
Base.:-(B::PaddedMatrix, I::UniformScaling) = PaddedMatrix(parent(B) - I, B.pad)
Base.:-(A::PaddedMatrix) = PaddedMatrix(-parent(A), A.pad)

function Base.:+(A::PaddedMatrix, B::PaddedMatrix)
    A.pad == B.pad || throw(ArgumentError("pads of the matrices don't match"))
    PaddedMatrix(parent(A) + parent(B), A.pad)
end
function Base.:-(A::PaddedMatrix, B::PaddedMatrix)
    A.pad == B.pad || throw(ArgumentError("pads of the matrices don't match"))
    PaddedMatrix(parent(A) - parent(B), A.pad)
end

_matrix(M::Union{Matrix, AbstractTriangular, Diagonal}) = M
_matrix(M::AbstractMatrix) = Matrix(M)
Base.:\(P::PaddedMatrix, ::UniformScaling) = PaddedMatrix(_matrix(parent(P)) \ I, P.pad)

function Base.replace_in_print_matrix(A::PaddedMatrix, i::Integer, j::Integer, s::AbstractString)
    Base.replace_in_print_matrix(parent(A), i, j, s)
end

function kronbig(K::KroneckerProduct{<:Any, <:PaddedMatrix})
    A, B = getmatrices(K);
    kron(parent(A), B)
end
function kronbig(K::KroneckerProduct{<:Any, <:Any, <:PaddedMatrix})
    A, B = getmatrices(K);
    kron(A, parent(B))
end
function kronbig(K::KroneckerProduct{<:Any, <:PaddedMatrix, <:PaddedMatrix})
    A, B = getmatrices(K);
    kron(parent(A), parent(B))
end

const KronProdAnyPadded = Union{KroneckerProduct{<:Any, <:PaddedMatrix}, KroneckerProduct{<:Any, <:Any, <:PaddedMatrix}}

function pad(K::KroneckerProduct{<:Any, <:PaddedMatrix})
    A, B = getmatrices(K)
    pad(A) * size(B, 1)
end
function pad(K::KroneckerProduct{<:Any, <:Any, <:PaddedMatrix})
    A, B = getmatrices(K)
    size(A, 1) * pad(B)
end
function pad(K::KroneckerProduct{<:Any, <:PaddedMatrix, <:PaddedMatrix})
    A, B = getmatrices(K)
    pad(A) * pad(B)
end

function Base.:*(K::KronProdAnyPadded, P::PaddedMatrix)
    M = kronbig(K) * parent(P)
    PaddedMatrix(M, P.pad)
end
function Base.:*(P::PaddedMatrix, K::KronProdAnyPadded)
    M = parent(P) * kronbig(K)
    PaddedMatrix(M, P.pad)
end

Base.:*(K::KroneckerProduct, P::PaddedMatrix) = Matrix(K) * P
Base.:*(P::PaddedMatrix, K::KroneckerProduct) = P * Matrix(K)

function Base.:+(P::PaddedArray, K::KronProdAnyPadded)
    M = parent(P) + kronbig(K)
    PaddedMatrix(M, P.pad)
end
Base.:+(K::KronProdAnyPadded, P::PaddedArray) = P + K
function Base.:+(K1::KronProdAnyPadded, K2::KronProdAnyPadded)
    M = kronbig(K1) + kronbig(K2)
    PaddedMatrix(M, pad(K1))
end

function Base.:-(P::PaddedArray, K::KronProdAnyPadded)
    M = parent(P) - kronbig(K)
    PaddedMatrix(M, P.pad)
end
function Base.:-(K::KronProdAnyPadded, P::PaddedArray)
    M = kronbig(K) - parent(P)
    PaddedMatrix(M, P.pad)
end
function Base.:-(K1::KronProdAnyPadded, K2::KronProdAnyPadded)
    M = kronbig(K1) - kronbig(K2)
    PaddedMatrix(M, pad(K1))
end

const KroneckerProductPaddedIdentity1{R<:AbstractMatrix} = Kronecker.KroneckerProduct{<:Any, <:PaddedMatrix{<:Any, <:IdentityMatrix}, R}
const KroneckerProductPaddedIdentityDiagonal1{R<:AbstractMatrix} = Kronecker.KroneckerProduct{<:Any, <:PaddedMatrix{<:Any, <:Union{IdentityMatrix, Diagonal}}, R}
const KroneckerProductPaddedIdentity2{R<:AbstractMatrix} = Kronecker.KroneckerProduct{<:Any, R, <:PaddedMatrix{<:Any, <:IdentityMatrix}}
const KroneckerProductPaddedIdentityDiagonal2{R<:AbstractMatrix} = Kronecker.KroneckerProduct{<:Any, R, <:PaddedMatrix{<:Any, <:Union{IdentityMatrix, Diagonal}}}
const KroneckerProductPaddedIdentity12 = Kronecker.KroneckerProduct{<:Any, <:PaddedMatrix{<:Any, <:IdentityMatrix}, <:PaddedMatrix{<:Any, <:IdentityMatrix}}
const KroneckerProductPaddedIdentityDiagonal12 = Kronecker.KroneckerProduct{<:Any, <:PaddedMatrix{<:Any, <:Union{IdentityMatrix, Diagonal}}, <:PaddedMatrix{<:Any, <:Union{IdentityMatrix, Diagonal}}}
const KroneckerProductPaddedIdentity{R<:AbstractMatrix} = Union{KroneckerProductIdentity1{R}, KroneckerProductIdentity2{R}}

function _replace_in_print_matrix_1(K, i, j, s)
    _, B = getmatrices(K)
    k, l = size(B)
    m, n = (i - 1) % k + 1, (j - 1) % l + 1
    if cld(i, k) != cld(j, l) || !isnonzero(B, m, n)
        return Base.replace_with_centered_mark(s)
    end
    return s
end
function Base.replace_in_print_matrix(K::KroneckerProductPaddedIdentityDiagonal1, i::Integer, j::Integer, s::AbstractString)
    _replace_in_print_matrix_1(K, i, j, s)
end
function Base.replace_in_print_matrix(K::KroneckerProductPaddedIdentityDiagonal1{<:IdentityMatrixOrDiagonal}, i::Integer, j::Integer, s::AbstractString)
    _replace_in_print_matrix_1(K, i, j, s)
end

function _replace_in_print_matrix_2(K, i, j, s)
    A, B = getmatrices(K)
    k, l = size(B)
    m, n = cld(i, k), cld(j, l)
    if ((i - 1) % k + 1) != ((j - 1) % l + 1) || !isnonzero(A, m, n)
        return Base.replace_with_centered_mark(s)
    end
    return s
end
function Base.replace_in_print_matrix(K::KroneckerProductPaddedIdentityDiagonal2, i::Integer, j::Integer, s::AbstractString)
    _replace_in_print_matrix_2(K, i, j, s)
end
function Base.replace_in_print_matrix(K::KroneckerProductPaddedIdentityDiagonal2{<:IdentityMatrixOrDiagonal}, i::Integer, j::Integer, s::AbstractString)
    _replace_in_print_matrix_2(K, i, j, s)
end

function Base.replace_in_print_matrix(K::KroneckerProductPaddedIdentityDiagonal12, i::Integer, j::Integer, s::AbstractString)
    _, B = getmatrices(K)
    k, l = size(B)
    if cld(i, k) != cld(j, l) || ((i - 1) % k + 1) != ((j - 1) % l + 1)
        return Base.replace_with_centered_mark(s)
    end
    return s
end

function Base.:+(K::KroneckerProductPaddedIdentity1, I::UniformScaling)
    A, B = getmatrices(K)
    A ⊗ (B + I)
end
function Base.:+(I::UniformScaling, K::KroneckerProductPaddedIdentity1)
    A, B = getmatrices(K)
    A ⊗ (I + B)
end
function Base.:+(I::UniformScaling, K::KroneckerProductPaddedIdentity2)
    A, B = getmatrices(K)
    (I + A) ⊗ B
end
function Base.:+(K::KroneckerProductPaddedIdentity2, I::UniformScaling)
    A, B = getmatrices(K)
    (A + I) ⊗ B
end
function Base.:-(K::KroneckerProductPaddedIdentity1)
    A, B = getmatrices(K)
    A ⊗ (-B)
end
function Base.:-(K::KroneckerProductPaddedIdentity2)
    A, B = getmatrices(K)
    (-A) ⊗ B
end
