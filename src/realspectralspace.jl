struct RealSpace{T, V <: AbstractVector{T}} <: AbstractVector{T}
    f :: V
end

Base.IndexStyle(::RealSpace{<:Any, V}) where {V} = IndexStyle(V)

RealSpace(R::RealSpace) = R

Base.parent(R::RealSpace) = R.f
Base.size(R::RealSpace) = size(parent(R))
Base.getindex(S::RealSpace, i::Int...) = parent(S)[i...]
Base.setindex!(S::RealSpace, v, i::Int...) = (parent(S)[i...] = v; S)
Base.:*(::IdentityMatrix, R::RealSpace) = R
for T in [:Diagonal, :AbstractMatrix, :AbstractTriangular]
    @eval Base.:*(M::$T, R::RealSpace) = M * parent(R)
end
Base.:*(D::Diagonal{<:Any, <:RealSpace}, R::RealSpace) = diag(D) .* R
Base.:*(K::KroneckerProduct, R::RealSpace) = RealSpace(K * parent(R))

struct RealSpaceStyle <: Broadcast.AbstractArrayStyle{1} end
Base.BroadcastStyle(::Type{<:RealSpace}) = RealSpaceStyle()
Base.similar(bc::Broadcast.Broadcasted{RealSpaceStyle}, ::Type{ElType}) where {ElType} =
    RealSpace(similar(Array{ElType}, axes(bc)))
RealSpaceStyle(::Val{N}) where {N} = Broadcast.DefaultArrayStyle{N}()
RealSpaceStyle(::Val{1}) = RealSpaceStyle()

abstract type Operator{T} <: AbstractMatrix{T} end
Base.size(S::Operator) = size(parent(S))
Base.getindex(S::Operator, i::Int...) = parent(S)[i...]
Base.setindex!(S::Operator, v, i::Int...) = (parent(S)[i...] = v; S)

struct LazyMul{T, A<:AbstractMatrix, B<:AbstractMatrix} <: AbstractMatrix{T}
    a :: A
    b :: B

    function LazyMul(a::A, b::B) where {A<:AbstractMatrix, B<:AbstractMatrix}
        size(a, 2) == size(b, 1) ||
        throw(DimensionMismatch(
            "second dimension of A, $(size(a, 2)) is incompatible with first dimension of B $(size(b, 1))"))
        new{promote_type(eltype(A), eltype(B)), A, B}(a, b)
    end
end

LazyMul(l::LazyMul) = l

Base.Matrix(l::LazyMul) = Matrix(l.a) * Matrix(l.b)
Base.collect(l::LazyMul) = Matrix(l)

Base.size(l::LazyMul) = (size(l.a, 1), size(l.b, 2))
Base.getindex(l::LazyMul, i::Int, j::Int) = dot((@view l.a[i, :]), (@view l.b[:, j]))

LazyMul(a, b, c, d...) = LazyMul(lazymul(lazymul(a, b), c), d...)

const LazyMul1{T<:AbstractMatrix} = LazyMul{<:Any, T}
const LazyMul2{T<:AbstractMatrix} = LazyMul{<:Any, <:AbstractMatrix, T}
const LazyMulTransform1 = LazyMul1{<:Transform}
const LazyMulTransform2 = LazyMul2{<:Transform}
const LazyMulTransform = Union{LazyMul1{<:Transform}, LazyMul2{<:Transform}}
const TransformOrLazyTransform = Union{LazyMulTransform, Transform}
const TransformOrLazyTransform1 = Union{LazyMulTransform1, Transform}
const TransformOrLazyTransform2 = Union{LazyMulTransform2, Transform}

for op in [:+, :-]
    f = Symbol("Base.:", op)
    @eval $f(l1::LazyMul, l2::LazyMul) = $op(Matrix(l1), Matrix(l2))
    @eval $f(l1::LazyMul, l2::AbstractMatrix) = $op(Matrix(l1), l2)
    @eval $f(l1::AbstractMatrix, l2::LazyMul) = $op(Matrix(l1), Matrix(l2))
end
Base.:-(l::LazyMul) = LazyMul(-l.a, l.b)
Base.:-(l::LazyMul{Transform, Transform}) = LazyMul(-l.a, l.b)
Base.:-(l::LazyMulTransform2) = LazyMul(-l.a, l.b)
Base.:-(l::LazyMulTransform1) = LazyMul(l.a, -l.b)
Base.:^(S::LazyMul, n::Integer) = reduce(*, fill(S, n))


lazymul(a, b) = a * b
lazymul(a::Transform, b) = LazyMul(a, b)
lazymul(a, b::Transform) = LazyMul(a, b)
lazymul(a::Transform, b::Transform) = LazyMul(a, b)
lazymul(a, b::TransformOrLazyTransform1) = LazyMul(a, b)
lazymul(a, b::TransformOrLazyTransform2) = LazyMul(a * b.a, b.b)
lazymul(a::TransformOrLazyTransform2, b) = LazyMul(a, b)
lazymul(a::TransformOrLazyTransform2, b::TransformOrLazyTransform) = LazyMul(a, b)
lazymul(a::TransformOrLazyTransform2, b::Transform) = LazyMul(a, b)
lazymul(a::TransformOrLazyTransform1, b) = LazyMul(a.a, a.b * b)
lazymul(a::TransformOrLazyTransform1, b::TransformOrLazyTransform) = LazyMul(a, b)
lazymul(a::TransformOrLazyTransform1, b::Transform) = LazyMul(a, b)
lazymul(a::TransformOrLazyTransform2, b::TransformOrLazyTransform1) = LazyMul(a.a, a.b * b.a, b.b)
lazymul(a::TransformOrLazyTransform1, b::TransformOrLazyTransform2) = LazyMul(a, b)
lazymul(F::ForwardTransform, I::InverseTransform) = F * I
lazymul(I::InverseTransform, F::ForwardTransform) = I * F

lazymul(T::Transform, M::LazyMulTransform2) = lazymul(T * M.a, M.b)

lazymul(a::ForwardTransform, l::LazyMul1{<:InverseTransform}) = lazymul(a * l.a, l.b)
lazymul(a::InverseTransform, l::LazyMul1{<:ForwardTransform}) = lazymul(a * l.a, l.b)
lazymul(l::LazyMul2{<:InverseTransform}, a::ForwardTransform) = lazymul(l.a, l.b * a)
lazymul(l::LazyMul2{<:ForwardTransform}, a::InverseTransform) = lazymul(l.a, l.b * a)

lazymul(a, b, c...) = lazymul(lazymul(a, b), c...)

lazymul(a::AbstractMatrix, ::IdentityMatrix) = a
lazymul(::IdentityMatrix, a::AbstractMatrix) = a
lazymul(I::IdentityMatrix, ::IdentityMatrix) = I

Base.:*(m::AbstractMatrix, l::LazyMul) = lazymul(lazymul(m, l.a), l.b)
Base.:*(l::LazyMul, m::AbstractMatrix) = lazymul(l.a, lazymul(l.b, m))

Base.:*(l::LazyMul2{<:ForwardTransform}, ::InverseTransform) = l.a
Base.:*(l::LazyMul2{<:InverseTransform}, ::ForwardTransform) = l.a
Base.:*(::InverseTransform, l::LazyMul1{<:ForwardTransform}) = l.b
Base.:*(::ForwardTransform, l::LazyMul1{<:InverseTransform}) = l.b
Base.:*(l::LazyMul, I::Transform) = Matrix(l) * I
Base.:*(I::Transform, l::LazyMul) = I * Matrix(l)

for T in [:AbstractTriangular, :Diagonal]
    @eval Base.:*(l::LazyMul, A::$T) = Matrix(l) * A
    @eval Base.:*(A::$T, l::LazyMul) = A * Matrix(l)
    @eval Base.:*(l::LazyMul2{<:Transform}, A::$T) = lazymul(l.a, l.b * A)
    @eval Base.:*(A::$T, l::LazyMul1{<:Transform}) = lazymul(A * l.a, l.b)
end

Base.:*(n::Number, l::LazyMulTransform1) = LazyMul(l.a, n*l.b)
Base.:*(l::LazyMulTransform1, n::Number) = LazyMul(l.a, l.b*n)
Base.:*(n::Number, l::LazyMulTransform2) = LazyMul(n*l.a, l.b)
Base.:*(l::LazyMulTransform2, n::Number) = LazyMul(l.a*n, l.b)

Base.:*(l1::LazyMul, l2::LazyMul) = lazymul(l1.a, l1.b * l2.a, l2.b)

function Base.showarg(io::IO, l::LazyMul, toplevel)
    if !toplevel
        print(io, "::")
    end
    print(io, "LazyMul(")
    Base.showarg(io, l.a, false)
    print(io, ", ")
    Base.showarg(io, l.b, false)
    print(io, ")")
end

struct RealSpaceOperator{T, M<:AbstractMatrix{T}, F<:ForwardTransform, I<:InverseTransform} <: Operator{T}
    m :: M
    fwd :: F
    inv :: I

    function RealSpaceOperator(m::M, fwd::F, inv::I) where {M,F,I}
        size(m) == size(fwd) == size(inv) ||
        throw(DimensionMismatch("Sizes of matrices are inconsistent, received $(size(m)), $(size(fwd)), $(size(inv))"))
        T = reduce(promote_type, (eltype(M), eltype(F), eltype(I)))
        new{T, M, F, I}(m, fwd, inv)
    end
end
Base.parent(R::RealSpaceOperator) = R.m

isnonzero(R::Operator, m, n) = isnonzero(parent(R), m, n)

function Base.showarg(io::IO, R::RealSpaceOperator, toplevel)
    if !toplevel
        print(io, "::")
    end
    print(io, "RealSpaceOperator(")
    Base.showarg(io, parent(R), false)
    print(io, ")")
end

struct SpectralSpaceOperator{T, M<:AbstractMatrix{T}, F<:ForwardTransform, I<:InverseTransform} <: Operator{T}
    m :: M
    fwd :: F
    inv :: I

    function SpectralSpaceOperator(m::M, fwd::F, inv::I) where {M,F,I}
        size(m) == size(fwd) == size(inv) ||
        throw(DimensionMismatch("Sizes of matrices are inconsistent, received $(size(m)), $(size(fwd)), $(size(inv))"))
        T = reduce(promote_type, (eltype(M), eltype(F), eltype(I)))
        new{T, M, F, I}(m, fwd, inv)
    end
end
Base.parent(R::SpectralSpaceOperator) = R.m
function Base.showarg(io::IO, R::SpectralSpaceOperator, toplevel)
    if !toplevel
        print(io, "::")
    end
    print(io, "SpectralSpaceOperator(")
    Base.showarg(io, parent(R), false)
    print(io, ")")
end

RealSpaceOperator(R::RealSpace, fwd, inv) = RealSpaceOperator(Diagonal(R), fwd, inv)
Base.:*(S::RealSpaceOperator, R::RealSpace) = RealSpace(parent(S) * R)
Base.:*(R::RealSpace, S::RealSpaceOperator) = RealSpaceOperator(R, S.fwd, S.inv) * S

SpectralSpaceOperator(R::RealSpace, fwd, inv) = SpectralSpaceOperator(RealSpaceOperator(R, fwd, inv))

SpectralSpaceOperator(R::RealSpaceOperator) =
    SpectralSpaceOperator(lazymul(R.fwd, parent(R), R.inv), R.fwd, R.inv)

RealSpaceOperator(R::SpectralSpaceOperator) =
    RealSpaceOperator(lazymul(R.inv, parent(R), R.fwd), R.fwd, R.inv)

realspaceoperator(R, args...) = RealSpaceOperator(R, args...)
realspaceoperator(R::RealSpaceOperator, args...) = R
realspaceoperator(R::IdentityMatrix, args...) = R
const IdentityOrOperator = Union{IdentityMatrix, Operator}
realspaceoperator(K::KroneckerProduct{<:Any, <:IdentityOrOperator, <:IdentityOrOperator}) =
    kronecker(map(realspaceoperator, getmatrices(K))...)
spectralspaceoperator(R, args...) = SpectralSpaceOperator(R, args...)
spectralspaceoperator(R::SpectralSpaceOperator, args...) = R
spectralspaceoperator(R::IdentityMatrix, args...) = R
spectralspaceoperator(K::KroneckerProduct{<:Any, <:IdentityOrOperator, <:IdentityOrOperator}) =
    kronecker(map(spectralspaceoperator, getmatrices(K))...)

LinearAlgebra.diag(R::RealSpaceOperator) = diag(parent(R))

const RealSpaceOpReal = RealSpaceOperator{<:Any, <:Diagonal{<:Any, <:RealSpace}}
const KronProdRealReal = Kronecker.KroneckerProduct{<:Any, <:RealSpaceOpReal, <:RealSpaceOpReal}
function Base.replace_in_print_matrix(::KronProdRealReal, i::Integer, j::Integer, s::AbstractString)
    i == j ? s : Base.replace_with_centered_mark(s)
end

Base.:+(R::RealSpaceOpReal, D::Union{Diagonal, IdentityMatrix, UniformScaling}) = Diagonal(R) + D
Base.:+(D::Union{Diagonal, IdentityMatrix, UniformScaling}, R::RealSpaceOpReal) = D + Diagonal(R)

for T in [:RealSpaceOperator, :SpectralSpaceOperator]
    @eval $T(R::$T) = R
    @eval Base.:*(S1::$T, S2::$T) = $T(parent(S1) * parent(S2), S1.fwd, S1.inv)
    @eval Base.:*(S::$T, n::Number) = $T(parent(S)*n, S.fwd, S.inv)
    @eval Base.:*(n::Number, S::$T) = $T(parent(S)*n, S.fwd, S.inv)
    @eval Base.:+(S1::$T, S2::$T, S3s::$T...) = $T(+(parent(S1), parent(S2), map(parent, S3s)...), S1.fwd, S1.inv)
    @eval Base.:-(S1::$T, S2::$T) = $T(parent(S1) - parent(S2), S1.fwd, S1.inv)
    @eval Base.:-(S::$T) = $T(-parent(S), S.fwd, S.inv)
    @eval Base.inv(S::$T) = $T(inv(parent(S)), S.fwd, S.inv)
    @eval Base.:^(S::$T, n::Integer) = $T(parent(S)^n, S.fwd, S.inv)
    @eval $T(K::KroneckerProductIdentity{<:$T}) = $T(K, getnonidentity(K).fwd, getnonidentity(K).inv)
end

Base.:+(S1::Operator, S2::Operator) = realspaceoperator(S1) + realspaceoperator(S2)
Base.:-(S1::Operator, S2::Operator) = realspaceoperator(S1) - realspaceoperator(S2)

Base.:*(S::SpectralSpaceOperator, R::RealSpace) = realspaceoperator(S) * R

Base.:*(R::RealSpaceOperator, S::SpectralSpaceOperator) = R * realspaceoperator(S)
Base.:*(S::SpectralSpaceOperator, R::RealSpaceOperator) = realspaceoperator(S) * R

Base.:*(R::RealSpace, S::Operator) = RealSpaceOperator(R, S.fwd, S.inv) * S

Base.:+(S::SpectralSpaceOperator, R::RealSpaceOperator) = realspaceoperator(S) + R
Base.:+(R::RealSpaceOperator, S::SpectralSpaceOperator) = R + realspaceoperator(S)

Base.:-(S::Operator, R::RealSpace) = realspaceoperator(S) - realspaceoperator(R, S.fwd, S.inv)
Base.:-(R::RealSpace, S::Operator) = realspaceoperator(R, S.fwd, S.inv) - S

function Base.replace_in_print_matrix(R::Operator, i::Integer, j::Integer, s::AbstractString)
    Base.replace_in_print_matrix(parent(R), i, j, s)
end

find_operator(bc::Base.Broadcast.Broadcasted) = find_operator(bc.args)
find_operator(args::Tuple) = find_operator(find_operator(args[1]), Base.tail(args))
find_operator(x) = x
find_operator(::Tuple{}) = nothing
find_operator(a::Operator, rest) = a
find_operator(::Any, rest) = find_aac(rest)

struct RealSpaceOperatorStyle <: Broadcast.AbstractArrayStyle{2} end
Base.BroadcastStyle(::Type{<:RealSpaceOperator}) = RealSpaceOperatorStyle()
function Base.similar(bc::Broadcast.Broadcasted{RealSpaceOperatorStyle}, ::Type{ElType}) where {ElType}
    A = find_operator(bc)
    RealSpaceOperator(similar(Array{ElType}, axes(bc)), A.fwd, A.inv)
end
RealSpaceOperatorStyle(::Val{N}) where {N} = Broadcast.DefaultArrayStyle{N}()
RealSpaceOperatorStyle(::Val{2}) = RealSpaceOperatorStyle()

struct SpectralSpaceOperatorStyle <: Broadcast.AbstractArrayStyle{2} end
Base.BroadcastStyle(::Type{<:SpectralSpaceOperator}) = SpectralSpaceOperatorStyle()
function Base.similar(bc::Broadcast.Broadcasted{SpectralSpaceOperatorStyle}, ::Type{ElType}) where {ElType}
    A = find_operator(bc)
    SpectralSpaceOperator(similar(Array{ElType}, axes(bc)), A.fwd, A.inv)
end
SpectralSpaceOperatorStyle(::Val{N}) where {N} = Broadcast.DefaultArrayStyle{N}()
SpectralSpaceOperatorStyle(::Val{2}) = SpectralSpaceOperatorStyle()
