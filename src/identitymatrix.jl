struct IdentityMatrix <: AbstractMatrix{Bool} n::Int end
Base.size(r::IdentityMatrix) = (r.n, r.n)
Base.axes(r::IdentityMatrix) = (Base.OneTo(r.n),Base.OneTo(r.n))
Base.length(r::IdentityMatrix) = r.n^2
Base.getindex(r::IdentityMatrix, i::Int, j::Int) = i == j
Base.:(==)(A::IdentityMatrix, I::UniformScaling) = true
Base.:(==)(I::UniformScaling, A::IdentityMatrix) = true
Base.inv(I::IdentityMatrix) = I
Base.one(I::IdentityMatrix) = I
Base.zero(I::IdentityMatrix) = Diagonal(Zeros(I.n))
Base.copy(I::IdentityMatrix) = I
LinearAlgebra.adjoint(I::IdentityMatrix) = I
LinearAlgebra.transpose(I::IdentityMatrix) = I
LinearAlgebra.factorize(I::IdentityMatrix) = I
LinearAlgebra.det(::IdentityMatrix) = 1
LinearAlgebra.tr(I::IdentityMatrix) = I.n
LinearAlgebra.diag(I::IdentityMatrix) = Ones(I.n)

function Base.replace_in_print_matrix(A::IdentityMatrix, i::Integer, j::Integer, s::AbstractString)
    i==j ? s : Base.replace_with_centered_mark(s)
end

Base.show(io::IO, I::IdentityMatrix) = print(io, typeof(I), "(", I.n, ")")

function Base.:(*)(I::IdentityMatrix, J::IdentityMatrix)
    if I.n != J.n
        throw(DimensionMismatch("second dimension of I, $(I.n), does not match first dimension of J, $(J.n)"))
    end
    I
end

Base.:(*)(x::Number, I::IdentityMatrix) = x * Eye(I.n)
Base.:(*)(I::IdentityMatrix, x::Number) = Eye(I.n) * x
Base.:(/)(I::IdentityMatrix, x::Number) = Eye(I.n) / x

function size_check(A, B)
    @noinline throw_dimerr(sA1, sB1) = throw(DimensionMismatch("second dimension of A, $sA1, does not match first dimension of B, $sB1"))
    size(A, 2) == size(B, 1) || throw_dimerr(size(A, 2), size(B, 1))
    return nothing
end

Base.:*(I::IdentityMatrix, V::AbstractVector) = (size_check(I, V); V)
Base.:*(I::IdentityMatrix, M::AbstractMatrix) = (size_check(I, M); M)
Base.:*(M::AbstractMatrix, I::IdentityMatrix) = (size_check(M, I); M)
Base.:*(I::IdentityMatrix, M::Diagonal) = (size_check(I, M); M)
Base.:*(M::Diagonal, I::IdentityMatrix) = (size_check(M, I); M)

function LinearAlgebra.kron(I::IdentityMatrix, D::Diagonal)
    d = diag(D)
    Diagonal(repeat(d, I.n))
end

function LinearAlgebra.kron(D::Diagonal, I::IdentityMatrix)
    d = diag(D)
    Diagonal(kron(d, Ones(I.n)))
end

function Base.broadcasted(::Broadcast.DefaultArrayStyle{2}, ::typeof(*), I::IdentityMatrix, J::IdentityMatrix)
    I.n == J.n || throw(DimensionMismatch("dimensions of I, $(size(I)), are incompatible with that of J, $(size(J))"))
    I
end
Base.broadcasted(::Broadcast.DefaultArrayStyle{2}, ::typeof(*), x::Number, I::IdentityMatrix) = x * Eye(I.n)
Base.broadcasted(::Broadcast.DefaultArrayStyle{2}, ::typeof(*), I::IdentityMatrix, x::Number) = x * Eye(I.n)
Base.broadcasted(::Broadcast.DefaultArrayStyle{2}, ::typeof(^), I::IdentityMatrix, x::Number) = I
Base.broadcasted(::Broadcast.DefaultArrayStyle{2}, ::typeof(Base.literal_pow), ::Base.RefValue, I::IdentityMatrix, x::Base.RefValue{<:Val}) = I

const IdentityMatrixOrDiagonal = Union{IdentityMatrix, Diagonal}
const KroneckerProductIdentity1{R<:AbstractMatrix} = Kronecker.KroneckerProduct{<:Any, IdentityMatrix, R}
const KroneckerProductIdentityDiagonal1{R<:AbstractMatrix} = Kronecker.KroneckerProduct{<:Any, IdentityMatrixOrDiagonal, R}
const KroneckerProductIdentity2{R<:AbstractMatrix} = Kronecker.KroneckerProduct{<:Any, R, IdentityMatrix}
const KroneckerProductIdentityDiagonal2{R<:AbstractMatrix} = Kronecker.KroneckerProduct{<:Any, R, IdentityMatrixOrDiagonal}
const KroneckerProductIdentity12 = Kronecker.KroneckerProduct{<:Any, IdentityMatrix, IdentityMatrix}
const KroneckerProductIdentityDiagonal12 = Kronecker.KroneckerProduct{<:Any, IdentityMatrixOrDiagonal, IdentityMatrixOrDiagonal}
const KroneckerProductIdentity{R<:AbstractMatrix} = Union{KroneckerProductIdentity1{R}, KroneckerProductIdentity2{R}}

getnonidentity(K::KroneckerProductIdentity1) = first(getmatrices(K))
getidentity(K::KroneckerProductIdentity1) = last(getmatrices(K))
getnonidentity(K::KroneckerProductIdentity2) = first(getmatrices(K))
getidentity(K::KroneckerProductIdentity2) = last(getmatrices(K))

function Base.:+(K1::KroneckerProductIdentity1, K2::KroneckerProductIdentity1, Krest::KroneckerProductIdentity1...)
    Ks = (K2, Krest...)
    A1, B1 = getmatrices(K1)
    A2s = map(first∘getmatrices, Ks)
    B2s = map(last∘getmatrices, Ks)
    all(x -> x === A1, A2s) || throw(ArgumentError("identity matrices (first terms in the kronecker product) differ"))
    kronecker(A1, +(B1, B2s...))
end

function Base.:+(K1::KroneckerProductIdentity2, K2::KroneckerProductIdentity2, Krest::KroneckerProductIdentity2...)
    Ks = (K2, Krest...)
    A1, B1 = getmatrices(K1)
    A2s = map(first∘getmatrices, Ks)
    B2s = map(last∘getmatrices, Ks)
    all(x -> x === B1, B2s) || throw(ArgumentError("identity matrices (second terms in the kronecker product) differ"))
    kronecker(+(A1, A2s...), B1)
end

function Base.:-(K1::KroneckerProductIdentity1, K2::KroneckerProductIdentity1)
    A1, B1 = getmatrices(K1)
    A2, B2 = getmatrices(K2)
    A1 === A2 || throw(ArgumentError("identity matrices (first terms in the kronecker product) differ"))
    kronecker(A1, B1 - B2)
end

function Base.:-(K1::KroneckerProductIdentity2, K2::KroneckerProductIdentity2)
    A1, B1 = getmatrices(K1)
    A2, B2 = getmatrices(K2)
    B1 === B2 || throw(ArgumentError("identity matrices (second terms in the kronecker product) differ"))
    kronecker(A1 - A2, B1)
end

function Base.:-(K::KroneckerProductIdentity1)
    A, B = getmatrices(K)
    kronecker(A, -B)
end

function Base.:-(K::KroneckerProductIdentity2)
    A, B = getmatrices(K)
    kronecker(-A, B)
end

function Base.:(*)(x::Number, K::KroneckerProductIdentity2)
    A, B = getmatrices(K)
    kronecker(x*A, B)
end

function Base.:(*)(x::Number, K::KroneckerProductIdentity1)
    A, B = getmatrices(K)
    kronecker(A, x*B)
end

Base.:*(D::Diagonal, K::KroneckerProductIdentity{<:Diagonal}) = D * Diagonal(K)
Base.:*(K::KroneckerProductIdentity{<:Diagonal}, D::Diagonal) = Diagonal(K) * D

Base.:*(M::Matrix, K::KroneckerProductIdentity) = M * collect(K)
Base.:*(K::KroneckerProductIdentity, M::Matrix) = collect(K) * M

function Base.:(/)(K::KroneckerProductIdentity1, x::Number)
    A, B = getmatrices(K)
    kronecker(A, B/x)
end

function Base.:(/)(K::KroneckerProductIdentity2, x::Number)
    A, B = getmatrices(K)
    kronecker(A/x, B)
end

function Base.:(*)(K1::KroneckerProductIdentity1, K2::KroneckerProductIdentity1)
    A1, B1 = getmatrices(K1)
    A2, B2 = getmatrices(K2)
    A1 === A2 || throw(ArgumentError("identity matrices (first terms in the kronecker product) differ"))
    kronecker(A1, B1 * B2)
end

function Base.:(*)(K1::KroneckerProductIdentity2, K2::KroneckerProductIdentity2)
    A1, B1 = getmatrices(K1)
    A2, B2 = getmatrices(K2)
    B1 === B2 || throw(ArgumentError("identity matrices (second terms in the kronecker product) differ"))
    kronecker(A1 * A2, B1)
end

function Base.broadcasted(::Broadcast.AbstractArrayStyle{2}, ::typeof(Base.literal_pow), r::Base.RefValue, K::KroneckerProductIdentity, x::Base.RefValue{<:Val})
    A, B = getmatrices(K)
    kronecker(Broadcast.broadcast(Base.literal_pow, r[], A, x[]), Broadcast.broadcast(Base.literal_pow, r[], B, x[]))
end

isnonzero(::AbstractMatrix, i, j) = true
isnonzero(::Diagonal, i, j) = i == j
isnonzero(::UpperTriangular, i, j) = i <= j

function Base.replace_in_print_matrix(K::KroneckerProductIdentityDiagonal1, i::Integer, j::Integer, s::AbstractString)
    _, B = getmatrices(K)
    k, l = size(B)
    m, n = (i - 1) % k + 1, (j - 1) % l + 1
    if cld(i, k) != cld(j, l) || !isnonzero(B, m, n)
        return Base.replace_with_centered_mark(s)
    end
    return s
end

function Base.replace_in_print_matrix(K::KroneckerProductIdentityDiagonal2, i::Integer, j::Integer, s::AbstractString)
    A, B = getmatrices(K)
    k, l = size(B)
    m, n = cld(i, k), cld(j, l)
    if ((i - 1) % k + 1) != ((j - 1) % l + 1) || !isnonzero(A, m, n)
        return Base.replace_with_centered_mark(s)
    end
    return s
end

function Base.replace_in_print_matrix(K::KroneckerProductIdentityDiagonal12, i::Integer, j::Integer, s::AbstractString)
    _, B = getmatrices(K)
    k, l = size(B)
    if cld(i, k) != cld(j, l) || ((i - 1) % k + 1) != ((j - 1) % l + 1)
        return Base.replace_with_centered_mark(s)
    end
    return s
end

function collectadj(K::KroneckerProductIdentity1)
    A, B = getmatrices(K)
    kronecker(A, collect(B'))
end

function collectadj(K::KroneckerProductIdentity2)
    A, B = getmatrices(K)
    kronecker(collect(A'), B)
end

Base.copy(K::KroneckerProductIdentity) = kronecker(getmatrices(K)...)

_collect(A) = collect(A)
_collect(A::IdentityMatrix) = A
collectadj(K::KroneckerProductIdentity) = kronecker(map(_collect, getmatrices(K))...)

function LinearAlgebra.:(\)(K1::KroneckerProductIdentity2, K2::KroneckerProductIdentity2)
    A1, B1 = getmatrices(K1)
    A2, B2 = getmatrices(K2)
    B1 === B2 || throw(ArgumentError("identity matrices (second terms in the kronecker product) differ"))
    kronecker(A1 \ A2, B1)
end

function LinearAlgebra.:(\)(K1::KroneckerProductIdentity1, K2::KroneckerProductIdentity1)
    A1, B1 = getmatrices(K1)
    A2, B2 = getmatrices(K2)
    A1 === A2 || throw(ArgumentError("identity matrices (first terms in the kronecker product) differ"))
    kronecker(A1, B1 \ B2)
end

function Base.:-(I::UniformScaling, K::KroneckerProductIdentity1)
    A, B = getmatrices(K)
    kronecker(A, I - B)
end
function Base.:-(K::KroneckerProductIdentity1, I::UniformScaling)
    A, B = getmatrices(K)
    kronecker(A, B - I)
end

function Base.:-(I::UniformScaling, K::KroneckerProductIdentity2)
    A, B = getmatrices(K)
    kronecker(I - A, B)
end
function Base.:-(K::KroneckerProductIdentity2, I::UniformScaling)
    A, B = getmatrices(K)
    kronecker(A - I, B)
end

function Base.:+(I::UniformScaling, K::KroneckerProductIdentity1)
    A, B = getmatrices(K)
    kronecker(A, I + B)
end
function Base.:+(K::KroneckerProductIdentity1, I::UniformScaling)
    A, B = getmatrices(K)
    kronecker(A, B + I)
end

function Base.:+(I::UniformScaling, K::KroneckerProductIdentity2)
    A, B = getmatrices(K)
    kronecker(I + A, B)
end
function Base.:+(K::KroneckerProductIdentity2, I::UniformScaling)
    A, B = getmatrices(K)
    kronecker(A + I, B)
end

function LinearAlgebra.diag(K::KroneckerProductIdentity)
    A, B = getmatrices(K)
    kron(diag(A), diag(B))
end

function Base.showarg(io::IO, K::KroneckerProductIdentity, toplevel)
    A, B = getmatrices(K)
    if !toplevel
        print(io, "::")
    end
    print(io, nameof(typeof(K)), "(")
    Base.showarg(io, A, false)
    print(io, ", ")
    Base.showarg(io, B, false)
    print(io, ")")
end

struct OneHotVector{T} <: AbstractVector{T}
    n :: Int
    i :: Int
end
OneHotVector(n, i) = OneHotVector{Int}(n, i)
OneHotVector(i) = OneHotVector{Int}(i, i)
Base.size(v::OneHotVector) = (v.n,)
Base.length(v::OneHotVector) = v.n
Base.getindex(v::OneHotVector{T}, i::Int) where {T} = T(v.i == i)

Base.getindex(I::IdentityMatrix, ::Colon, j::Int) = OneHotVector(I.n, j)
Base.getindex(I::IdentityMatrix, j::Int, ::Colon) = OneHotVector(I.n, j)
Base.getindex(I::IdentityMatrix, ::Colon, ::Colon) = I

function Base.replace_in_print_matrix(O::OneHotVector, i::Integer, j::Integer, s::AbstractString)
    i == O.i ? s : Base.replace_with_centered_mark(s)
end
