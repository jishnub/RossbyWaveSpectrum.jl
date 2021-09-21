struct RealSpace{T, V <: AbstractVector{T}, F<:AbstractMatrix, I<:AbstractMatrix} <: AbstractMatrix{T}
    f :: V
    fwd :: F
    inv :: I
end
RealSpace(v::AbstractVector, fwd, inv) = RealSpace{eltype(v), typeof(v), typeof(fwd), typeof(inv)}(v, fwd, inv)
RealSpace(D::AbstractMatrix, fwd, inv) = RealSpace(diag(D), fwd, inv)

Base.parent(R::RealSpace) = R.f
Base.size(S::RealSpace) = (s = size(parent(S), 1); (s, s))
Base.getindex(S::RealSpace, i::Int...) = Diagonal(S)[i...]
Base.setindex!(S::RealSpace, v, i::Int...) = (Diagonal(S)[i...] = v; S)
LinearAlgebra.Diagonal(S::RealSpace) = Diagonal(diag(S))
LinearAlgebra.diag(S::RealSpace) = parent(S)
Base.:*(R::RealSpace, ::IdentityMatrix) = R
Base.:*(::IdentityMatrix, R::RealSpace) = R
Base.:*(R::RealSpace, M::AbstractMatrix) = Diagonal(R) * M
Base.:*(M::AbstractMatrix, ::RealSpace) = error("Right multiplication by RealSpace is not defined for $(typeof(M))")
Base.:*(R1::RealSpace, R2::RealSpace) = RealSpace(diag(R1) .* diag(R2), R1.fwd, R1.inv)
Base.:-(R::RealSpace) = RealSpace(-parent(R), R.fwd, R.inv)
Base.:+(R1::RealSpace, R2::RealSpace) = RealSpace(parent(R1) + parent(R2), R.fwd, R.inv)
Base.:-(R1::RealSpace, R2::RealSpace) = RealSpace(parent(R1) - parent(R2), R.fwd, R.inv)

function Base.:*(K1::KroneckerProductIdentity1{<:RealSpace}, K2::KroneckerProductIdentity2{<:RealSpace})
    A2 = last(getmatrices(K1))
    B1 = first(getmatrices(K2))
    RealSpace(kron(A2, B1), A2.fwd, A2.inv)
end

function Base.:*(K2::KroneckerProductIdentity2{<:RealSpace}, K1::KroneckerProductIdentity1{<:RealSpace})
    A1 = first(getmatrices(K2))
    B2 = last(getmatrices(K1))
    RealSpace(kron(A1, B2), A1.fwd, A1.inv)
end

Base.:*(K1::KroneckerProductIdentity{<:RealSpace}, R::RealSpace) = RealSpace(K1, R.fwd, R.inv) * R
Base.:*(R::RealSpace, K1::KroneckerProductIdentity{<:RealSpace}) = R * RealSpace(K1, R.fwd, R.inv)

isnonzero(R::RealSpace, i, j) = i == j
function Base.replace_in_print_matrix(R::RealSpace, i::Integer, j::Integer, s::AbstractString)
    Base.replace_in_print_matrix(Diagonal(R), i, j, s)
end

function Base.replace_in_print_matrix(::Kronecker.KroneckerProduct{<:Any, <:RealSpace, <:RealSpace}, i::Integer, j::Integer, s::AbstractString)
    i == j ? s : Base.replace_with_centered_mark(s)
end

struct SpectralSpaceOperator{T, M<:AbstractArray{T}} <: AbstractMatrix{T}
    m :: M
end

Base.parent(S::SpectralSpaceOperator) = S.m
Base.size(S::SpectralSpaceOperator) = size(parent(S))
Base.getindex(S::SpectralSpaceOperator, i::Int...) = parent(S)[i...]
Base.setindex!(S::SpectralSpaceOperator, v, i::Int...) = (parent(S)[i...] = v; S)

SpectralSpaceOperator(R::RealSpace) = SpectralSpaceOperator(R.fwd * Diagonal(R) * R.inv)
isnonzero(S::SpectralSpaceOperator, i, j) = isnonzero(parent(S), i, j)

Base.:*(S::SpectralSpaceOperator, R::RealSpace) = RealSpace(R.inv * parent(S) * R.fwd * parent(R), R.fwd, R.inv)
Base.:*(R::RealSpace, S::SpectralSpaceOperator) = SpectralSpaceOperator(R) * S
Base.:*(S1::SpectralSpaceOperator, S2::SpectralSpaceOperator) = SpectralSpaceOperator(parent(S1) * parent(S2))
Base.:*(S::SpectralSpaceOperator, n::Number) = SpectralSpaceOperator(parent(S)*n)
Base.:*(n::Number, S::SpectralSpaceOperator) = SpectralSpaceOperator(parent(S)*n)
Base.:+(S1::SpectralSpaceOperator, S2::SpectralSpaceOperator) = SpectralSpaceOperator(parent(S1) + parent(S2))
Base.:-(S1::SpectralSpaceOperator, S2::SpectralSpaceOperator) = SpectralSpaceOperator(parent(S1) - parent(S2))
Base.:-(S::SpectralSpaceOperator) = SpectralSpaceOperator(-parent(S))
Base.:+(S::SpectralSpaceOperator, R::RealSpace) = S + SpectralSpaceOperator(R)
Base.:+(R::RealSpace, S::SpectralSpaceOperator) = SpectralSpaceOperator(R) + S
Base.:-(S::SpectralSpaceOperator, R::RealSpace) = S - SpectralSpaceOperator(R)
Base.:-(R::RealSpace, S::SpectralSpaceOperator) = SpectralSpaceOperator(R) - S
Base.inv(S::SpectralSpaceOperator) = SpectralSpaceOperator(inv(parent(S)))

function Base.replace_in_print_matrix(R::SpectralSpaceOperator, i::Integer, j::Integer, s::AbstractString)
    Base.replace_in_print_matrix(parent(R), i, j, s)
end

struct OperatorStyle <: Broadcast.AbstractArrayStyle{2} end
OperatorStyle(::Val{2}) = OperatorStyle()
OperatorStyle(::Val{N}) where {N} = Broadcast.AbstractArrayStyle{N}()

Broadcast.BroadcastStyle(::Type{<:SpectralSpaceOperator}) = OperatorStyle()
Broadcast.BroadcastStyle(::Type{<:RealSpace}) = OperatorStyle()
Base.similar(bc::Broadcast.Broadcasted{OperatorStyle}, ::Type{ElType}) where {ElType} =
    SpectralSpaceOperator(similar(Array{ElType}, axes(bc)))
