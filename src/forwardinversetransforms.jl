abstract type Transform{T} <: AbstractMatrix{T} end
struct ForwardTransform{T, A <: AbstractMatrix{T}} <: Transform{T}
    a :: A
end
struct InverseTransform{T, A <: AbstractMatrix{T}} <: Transform{T}
    a :: A
end
Base.parent(F::Transform) = F.a

function Base.showarg(io::IO, T::Transform, toplevel)
    if !toplevel
        print(io, "::")
    end
    print(io, nameof(typeof(T)))
    if toplevel
        print(io, "(")
        Base.showarg(io, parent(T), false)
        print(io, ")")
    end
end

for f in [:size, :axes, :length]
    @eval Base.$f(F::Transform) = $f(parent(F))
end
@eval Base.getindex(A::Transform, i::Int, j::Int) = parent(A)[i, j]
@eval Base.getindex(A::Transform, i::Int) = parent(A)[i]
@eval Base.:(*)(A::Transform, B::AbstractMatrix) = parent(A) * B
@eval Base.:(*)(A::Transform, B::Diagonal) = parent(A) * B
@eval function Base.:(*)(M::Transform, I::IdentityMatrix)
    sizeM2 = size(M, 2)
    if I.n != sizeM2
        throw(DimensionMismatch("second dimension of M, $sizeM2, does not match first dimension of I, $(I.n)"))
    end
    M
end
@eval Base.:(*)(A::AbstractMatrix, B::Transform) = A * parent(B)
@eval function Base.:(*)(I::IdentityMatrix, M::Transform)
    sizeB1 = size(M, 1)
    if I.n != sizeB1
        throw(DimensionMismatch("second dimension of I, $(I.n), does not match first dimension of M, $sizeB1"))
    end
    M
end
@eval Base.:(*)(A::Diagonal, B::Transform) = A * parent(B)
@eval Base.:(*)(A::Transform, V::AbstractVector) = parent(A) * V
@eval Base.:(*)(A::AbstractVector, B::Transform) = A * parent(B)
@eval Base.:(*)(A::Transform, B::Transform) = parent(A) * parent(A)
@eval Base.:(*)(A::Transform, B::AbstractTriangular) = parent(A) * B
@eval Base.:(*)(A::AbstractTriangular, B::Transform) = A * parent(B)

function Base.:(*)(F::ForwardTransform, I::InverseTransform)
    if size(F, 2) != size(I, 1)
        throw(DimensionMismatch("Second axis of the forward transform, $(size(F, 2)), doesn't match the first axis of the inverse transform $(size(I, 1))"))
    end
    size(F, 1) == size(F, 2) || throw(DimensionMismatch("forward transform matrix is not square"))
    size(I, 1) == size(I, 2) || throw(DimensionMismatch("inverse transform matrix is not square"))
    IdentityMatrix(size(F, 1))
end

function Base.:(*)(I::InverseTransform, F::ForwardTransform)
    if size(I, 2) != size(F, 1)
        throw(DimensionMismatch("Second axis of the inverse transform, $(size(I, 2)), doesn't match the first axis of the forward transform $(size(F, 1))"))
    end
    size(F, 1) == size(F, 2) || throw(DimensionMismatch("forward transform matrix is not square"))
    size(I, 1) == size(I, 2) || throw(DimensionMismatch("inverse transform matrix is not square"))
    IdentityMatrix(size(F, 1))
end

LinearAlgebra.factorize(T::Transform) = factorize(parent(T))

Base.:*(K::Kronecker.GeneralizedKroneckerProduct, M::ForwardTransform) = K * collect(M)
Base.:*(K::Kronecker.GeneralizedKroneckerProduct, M::InverseTransform) = K * collect(M)

Base.:*(M::Matrix, K::Kronecker.KroneckerProduct{<:Any, <:InverseTransform, <:InverseTransform}) = M * collect(K)
Base.:*(K::Kronecker.KroneckerProduct{<:Any, <:ForwardTransform, <:ForwardTransform}, M::Matrix) = collect(K) * M
