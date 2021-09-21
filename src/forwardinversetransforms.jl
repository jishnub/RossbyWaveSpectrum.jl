struct ForwardTransform{T, A <: AbstractMatrix{T}} <: AbstractMatrix{T}
    a :: A
end
struct InverseTransform{T, A <: AbstractMatrix{T}} <: AbstractMatrix{T}
    a :: A
end
Base.parent(F::ForwardTransform) = F.a
Base.parent(F::InverseTransform) = F.a
for T in [:ForwardTransform, :InverseTransform]
    for f in [:size, :axes, :length]
        @eval Base.$f(F::$T) = $f(parent(F))
    end
    @eval Base.getindex(A::$T, i::Int, j::Int) = parent(A)[i, j]
    @eval Base.getindex(A::$T, i::Int) = parent(A)[i]
    @eval Base.:(*)(A::$T, B::AbstractMatrix) = parent(A) * B
    @eval Base.:(*)(A::$T, B::Diagonal) = parent(A) * B
    @eval function Base.:(*)(M::$T, I::IdentityMatrix)
        sizeM2 = size(M, 2)
        if I.n != sizeM2
            throw(DimensionMismatch("second dimension of M, $sizeM2, does not match first dimension of I, $(I.n)"))
        end
        M
    end
    @eval Base.:(*)(A::AbstractMatrix, B::$T) = A * parent(B)
    @eval function Base.:(*)(I::IdentityMatrix, M::$T)
        sizeB1 = size(M, 1)
        if I.n != sizeB1
            throw(DimensionMismatch("second dimension of I, $(I.n), does not match first dimension of M, $sizeB1"))
        end
        M
    end
    @eval Base.:(*)(A::Diagonal, B::$T) = A * parent(B)
    @eval Base.:(*)(A::$T, V::AbstractVector) = parent(A) * V
    @eval Base.:(*)(A::AbstractVector, B::$T) = A * parent(B)
    @eval Base.:(*)(A::$T, B::$T) = parent(A) * parent(A)
    @eval Base.:(*)(A::$T, B::LinearAlgebra.AbstractTriangular) = parent(A) * B
    @eval Base.:(*)(A::LinearAlgebra.AbstractTriangular, B::$T) = A * parent(B)
end

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

Base.:*(K::Kronecker.GeneralizedKroneckerProduct, M::ForwardTransform) = K * collect(M)
Base.:*(K::Kronecker.GeneralizedKroneckerProduct, M::InverseTransform) = K * collect(M)
