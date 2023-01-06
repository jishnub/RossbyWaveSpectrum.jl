module ApproxFunAssociatedLegendre

using DomainSets
using ApproxFunBase
using ApproxFunBase: SubOperator, UnsetSpace, ConcreteMultiplication, MultiplicationWrapper,
	ConstantTimesOperator, ConversionWrapper
import ApproxFunBase: domainspace, rangespace, Multiplication, setspace, Conversion,
	spacescompatible
using BandedMatrices
import BandedMatrices: BandedMatrix
using ApproxFunOrthogonalPolynomials
using ApproxFunSingularities
using LazyArrays
using BlockBandedMatrices

export NormalizedPlm
export sinθdθ_Operator
export cosθ_Operator
export sinθdθ_plus_2cosθ_Operator
export HorizontalLaplacian
export expand
export kronmatrix

struct NormalizedPlm{T, NJS <: Space{ChebyshevInterval{T},T}} <: Space{ChebyshevInterval{T},T}
	m :: Int
	jacobispace :: NJS
end
azimuthalorder(p::NormalizedPlm) = p.m

const TJ = JacobiWeight{<:NormalizedPolynomialSpace{<:Jacobi{<:ChebyshevInterval}}}
const JacobiMaybeNormalized = Union{Jacobi{<:ChebyshevInterval},
	NormalizedPolynomialSpace{<:Jacobi{<:ChebyshevInterval}}}

ApproxFunBase.domain(::NormalizedPlm{T}) where {T} = ChebyshevInterval{T}()
NormalizedPlm(m::Int) = NormalizedPlm(m, JacobiWeight(m/2, m/2, NormalizedJacobi(m,m)))
NormalizedPlm(; m::Int) = NormalizedPlm(m)
Base.show(io::IO, sp::NormalizedPlm) = print(io, "NormalizedPlm(m=", azimuthalorder(sp), ")")

spacescompatible(b::typeof(NormalizedLegendre()), a::NormalizedPlm) =
	spacescompatible(a, b)
function spacescompatible(a::NormalizedPlm, b::typeof(NormalizedLegendre()))
	iseven(azimuthalorder(a)) && spacescompatible(a.jacobispace, b)
end

Conversion(J::JacobiMaybeNormalized, sp::NormalizedPlm) = Conversion(JacobiWeight(0,0,J), sp)
function Conversion(J::TJ, sp::NormalizedPlm)
	m = azimuthalorder(sp)
	C = (-1)^m * Conversion(J, sp.jacobispace)
	S = ApproxFunBase.SpaceOperator(C, J, sp)
	ApproxFunBase.ConversionWrapper(S)
end
Conversion(sp::NormalizedPlm, J::JacobiMaybeNormalized) = Conversion(sp, JacobiWeight(0,0,J))
function Conversion(sp::NormalizedPlm, J::TJ)
	m = azimuthalorder(sp)
	C = (-1)^m * Conversion(sp.jacobispace, J)
	S = ApproxFunBase.SpaceOperator(C, sp, J)
	ApproxFunBase.ConversionWrapper(S)
end

# Assume that only one m is being used
function Base.union(A::NormalizedPlm, B::NormalizedPlm)
	@assert azimuthalorder(A) == azimuthalorder(B)
	A
end

function ApproxFunBase.maxspace(A::NormalizedPlm, B::NormalizedPlm)
	@assert azimuthalorder(A) == azimuthalorder(B)
	A
end

function Base.:(==)(A::NormalizedPlm, B::NormalizedPlm)
	@assert azimuthalorder(A) == azimuthalorder(B)
	true
end

function assertLegendre(sp::JacobiMaybeNormalized)
	csp = ApproxFunBase.canonicalspace(sp)
	@assert csp == Legendre() "multiplication is only defined in Legendre space"
end
function assertLegendre(sp::NormalizedPlm)
	@assert azimuthalorder(sp) == 0 "multiplication is only defined in Legendre space"
end

function ApproxFunBase.union_rule(a::NormalizedPlm, b::Union{ConstantSpace, PolynomialSpace})
	azimuthalorder(a) == 0 || return ApproxFunBase.NoSpace()
	union(Legendre(), b)
end

# Legendre polynomial norm
plnorm2(ℓ) = (2ℓ+1)/2
plnorm(ℓ) = sqrt(plnorm2(ℓ))

# Associated Legendre polynomial norm
plmnorm(l, m) = exp(LegendrePolynomials.logplm_norm(l, m))

function ApproxFunBase.spacescompatible(a::NormalizedPlm, b::NormalizedPlm)
	azimuthalorder(a) == 0 || azimuthalorder(b) == 0 ||
		azimuthalorder(a) == azimuthalorder(b) || return false
	ApproxFunBase.spacescompatible(a.jacobispace, b.jacobispace)
end

function ApproxFunBase.evaluate(v::Vector, s::NormalizedPlm, x)
	m = azimuthalorder(s)
	evaluate(v, s.jacobispace, x) * (-1)^m
end

for f in [:plan_transform, :plan_transform!, :plan_itransform, :plan_itransform!]
	@eval function ApproxFunBase.$f(sp::NormalizedPlm, v::Vector)
		m = azimuthalorder(sp)
		ApproxFunBase.$f(sp.jacobispace, v .* (-1)^m)
	end
end

function _Fun(f, sp::NormalizedPlm)
	F = Fun(f, sp.jacobispace)
	m = azimuthalorder(sp)
	c = coefficients(F) .* (-1)^m
	Fun(sp, c)
end
ApproxFunBase.Fun(f, sp::NormalizedPlm) = _Fun(f, sp)
ApproxFunBase.Fun(f::Fun, sp::NormalizedPlm) = _Fun(f, sp)
ApproxFunBase.Fun(f::typeof(identity), sp::NormalizedPlm) = _Fun(f, sp)

function ApproxFunBase.Derivative(sp::NormalizedPlm, k::Int)
	m = azimuthalorder(sp)
	ApproxFunBase.DerivativeWrapper((-1)^m * Derivative(sp.jacobispace, k), k)
end

abstract type PlmSpaceOperator{T, DS<:NormalizedPlm} <: Operator{T} end

domainspace(p::PlmSpaceOperator) = p.ds
rangespace(p::PlmSpaceOperator) = p.ds
function ApproxFunBase.promotedomainspace(P::PlmSpaceOperator, sp::NormalizedPlm)
	ApproxFunBase.setspace(P, sp)
end

index_to_ℓ(i, m) = m + i - 1

for P in [:sinθdθ_Operator, :cosθ_Operator, :sinθdθ_plus_2cosθ_Operator]
	@eval begin
		struct $P{T,DS} <: PlmSpaceOperator{T,DS}
			ds :: DS
		end
		$P{T}(ds) where {T} = $P{T, typeof(ds)}(ds)
		$P(ds) = $P{Float64}(ds)

		Base.convert(::Type{Operator{T}}, h::$P) where {T} =
			$P{T}(h.ds)::Operator{T}

		ApproxFunBase.setspace(::$P{T}, sp::Space) where {T} =
			$P{T}(sp)
	end
end

BandedMatrices.bandwidths(::Union{sinθdθ_Operator, cosθ_Operator, sinθdθ_plus_2cosθ_Operator}) = (1,1)

C⁻ℓm(ℓ, m, T = Float64) = (ℓ < abs(m) ? T(0) : convert(T, √(T(ℓ - m) * T(ℓ + m) / (T(2ℓ - 1) * T(2ℓ + 1)))))
C⁺ℓm(ℓ, m, T) = C⁻ℓm(ℓ+1, m, T)
β⁻ℓm(ℓ, m, T = Float64) = (ℓ < abs(m) ? T(0) : convert(T, √(T(2ℓ + 1) / T(2ℓ - 1) * T(ℓ^2 - m^2))))
S⁺ℓm(ℓ, m, T = Float64) = convert(T, ℓ) * C⁺ℓm(ℓ, m, T)
S⁻ℓm(ℓ, m, T = Float64) = convert(T, ℓ) * C⁻ℓm(ℓ, m, T) - β⁻ℓm(ℓ, m, T)

function tridiag_getindex(::Type{T}, m, i, j, topdiagfn, botdiagfn) where {T}
	if j == i+1
		ℓ = index_to_ℓ(i, m)
		topdiagfn(ℓ+1, m, T)
	elseif j == i-1
		ℓ = index_to_ℓ(i, m)
		botdiagfn(ℓ-1, m, T)
	else
		zero(T)
	end
end

function Base.getindex(P::sinθdθ_Operator{T}, i::Int, j::Int) where {T}
	m = azimuthalorder(domainspace(P))
	tridiag_getindex(T, m, i, j, S⁻ℓm, S⁺ℓm)
end

function Base.getindex(P::cosθ_Operator{T}, i::Int, j::Int) where {T}
	m = azimuthalorder(domainspace(P))
	tridiag_getindex(T, m, i, j, C⁻ℓm, C⁺ℓm)
end

function Base.getindex(P::sinθdθ_plus_2cosθ_Operator{T}, i::Int, j::Int) where {T}
	m = azimuthalorder(domainspace(P))
	C = tridiag_getindex(T, m, i, j, C⁻ℓm, C⁺ℓm)
	S = tridiag_getindex(T, m, i, j, S⁻ℓm, S⁺ℓm)
	S + 2C
end

function ApproxFunBase.Multiplication(f::Fun, sp::NormalizedPlm)
	Multiplication(Fun(f, NormalizedLegendre()), sp)
end
function ApproxFunBase.Multiplication(f::Fun{<:JacobiMaybeNormalized}, sp::NormalizedPlm)
	assertLegendre(space(f))
	g = Fun(f, NormalizedPlm(0))
	Multiplication(g, sp)
end
function ApproxFunBase.Multiplication(f::Fun{<:NormalizedPlm}, sp::NormalizedPlm)
	assertLegendre(space(f))
	ConcreteMultiplication(f, sp)
end
function BandedMatrices.bandwidths(M::ConcreteMultiplication{<:NormalizedPlm, <:NormalizedPlm})
	f = M.f
	nc = max(ncoefficients(f), 1) # treat empty vectors as 0
	(nc-1, nc-1)
end
function ApproxFunBase.rangespace(M::ConcreteMultiplication{<:NormalizedPlm, <:NormalizedPlm})
	ApproxFunBase.domainspace(M)
end
function _getindex(M::ConcreteMultiplication{<:NormalizedPlm, <:NormalizedPlm}, i, j)
	sp = domainspace(M)
	f = M.f
	g = Fun(space(f).jacobispace, coefficients(f))
	sp = domainspace(M)
	spj = sp.jacobispace
	x = Multiplication(g, spj)[i, j]
end
Base.getindex(M::ConcreteMultiplication{<:NormalizedPlm, <:NormalizedPlm}, i::Int, j::Int) =
	_getindex(M, i, j)
Base.getindex(M::ConcreteMultiplication{<:NormalizedPlm, <:NormalizedPlm}, i::UnitRange{Int}, j::UnitRange{Int}) =
	_getindex(M, i, j)
Base.getindex(M::ConcreteMultiplication{<:NormalizedPlm, <:NormalizedPlm}, i::AbstractRange{Int}, j::AbstractRange{Int}) =
	_getindex(M, first(i):last(i), first(j):last(j))[1:step(i):end, 1:step(j):end]

struct HorizontalLaplacian{T,DS<:Space} <: PlmSpaceOperator{T,DS}
	ds :: DS
	diagshift::Int
end
HorizontalLaplacian{T}(ds, diagshift = 0) where {T} = HorizontalLaplacian{T, typeof(ds)}(ds, diagshift)
HorizontalLaplacian(args...) = HorizontalLaplacian{Float64}(args...)
ApproxFunBase.setspace(P::HorizontalLaplacian{T}, sp::Space) where {T} =
	HorizontalLaplacian{T, typeof(sp)}(sp, P.diagshift)

Base.convert(::Type{Operator{T}}, h::HorizontalLaplacian) where {T} =
	HorizontalLaplacian{T}(h.ds, h.diagshift)::Operator{T}

BandedMatrices.bandwidths(::HorizontalLaplacian) = (0,0)

Base.:(+)(H::HorizontalLaplacian{T}, ds::Int) where {T} = HorizontalLaplacian{T}(H.ds, H.diagshift + ds)
Base.:(+)(ds::Int, H::HorizontalLaplacian) = H + ds
Base.:(-)(H::HorizontalLaplacian{T}, ds::Int) where {T} = HorizontalLaplacian{T}(H.ds, H.diagshift - ds)

function Base.getindex(P::HorizontalLaplacian{T,<:NormalizedPlm}, i::Int, j::Int) where {T}
	m = azimuthalorder(domainspace(P))
	ds = P.diagshift
	if j == i
		ℓ = index_to_ℓ(i, m)
		convert(T, -ℓ*(ℓ+1) + ds)
	else
		zero(T)
	end
end

function ApproxFunBase.Multiplication(f::Fun{<:NormalizedPlm}, sp::NormalizedPolynomialSpace{<:Jacobi})
	assertLegendre(space(f))
	Multiplication(Fun(f, NormalizedLegendre()), sp)
end

###################################################################################

expand(A::PlusOperator, B::PlusOperator) = mapreduce(x -> expand(x,B), +, A.ops)
function expand(A::PlusOperator, B)
	Be = expand(B)
	mapreduce(x -> expand(expand(x), Be), +, A.ops)
end
function expand(A, B::PlusOperator)
	Ae = expand(A)
	mapreduce(x -> expand(Ae, expand(x)), +, B.ops)
end
expand(A, B) = expand(A) * expand(B)
expand(A) = A
expand(P::PlusOperator) = mapreduce(expand, +, P.ops)
expand(T::TimesOperator) = mapfoldr(expand, expand, T.ops)
expand(C::ConversionWrapper) = expand(C.op)
expand(C::MultiplicationWrapper) = expand(C.op)
expand(C::ConstantOperator) = convert(Number, C)

function expand(C::ConstantTimesOperator)
	(; λ, op) = C
	expand(λ, expand(op))
end
function expand(λ::Number, K::KroneckerOperator)
	A, B = K.ops
	(λ * A) ⊗ B
end
function expand(K::KroneckerOperator, λ::Number)
	A, B = K.ops
	A ⊗ (B * λ)
end
expand(λ::Number, P::PlusOperator) = mapreduce(op -> expand(λ, op), +, P.ops)
function expand(λ::Number, T::TimesOperator)
	(; ops) = T
	ops[1] *= λ
	foldr(expand, ops)
end
function expand(n::Number, C::ConstantTimesOperator)
	(; λ) = C
	n2 = n * λ
	expand(n2, C.op)
end
expand(λ::Number, O::Operator) = λ * O
expand(O::Operator, λ::Number) = O * λ

# function expand(A::MultiplicationWrapper, B::MultiplicationWrapper)
# 	expand(A.op, B.op)
# end
# function expand(M::MultiplicationWrapper, K::Operator)
# 	expand(M.op, K)
# end
# function expand(K::Operator, M::MultiplicationWrapper)
# 	expand(K, M.op)
# end

# function expand(C::ConversionWrapper{<:KroneckerOperator}, D::ConversionWrapper{<:KroneckerOperator})
# 	expand(C.op, D.op)
# end
# function expand(C::ConversionWrapper{<:KroneckerOperator}, O::Operator)
# 	expand(C.op, O)
# end
# function expand(O::Operator, C::ConversionWrapper{<:KroneckerOperator})
# 	expand(O, C.op)
# end

# function expand(K::KroneckerOperator, C::ConstantOperator)
# 	(; λ) = C
# 	expand(K, λ)
# end

function expand(C::ConstantTimesOperator, K::KroneckerOperator)
	(; λ, op) = C
	expand(op, expand(λ, K))
end
function expand(K::KroneckerOperator, C::ConstantTimesOperator)
	(; λ, op) = C
	expand(expand(K, λ), op)
end

##################################################################################

function kronmatrix(M::MultiplicationWrapper, args...)
	kronmatrix(M.op, args...)
end
function kronmatrix(P::PlusOperator, args...)
	mapfoldl(op -> kronmatrix(op, args...), (x,y) -> x .+= y, P.ops)
end
function kronmatrix(K::KroneckerOperator, nr, nℓ::Integer)
	kronmatrix(K::KroneckerOperator, nr, 1:nℓ, 1:nℓ)
end
function kronmatrix(K::KroneckerOperator, nr,
		ℓindrange_row::AbstractRange, ℓindrange_col::AbstractRange)
	Or, Oθ = K.ops
	ℓindrange_row_ur = first(ℓindrange_row):last(ℓindrange_row)
	ℓindrange_col_ur = first(ℓindrange_col):last(ℓindrange_col)
	O1full = Oθ[ℓindrange_row_ur, ℓindrange_col_ur]
	O1 = O1full[1:step(ℓindrange_row):end, 1:step(ℓindrange_col):end]
	kronmatrix(Matrix(O1), Matrix(Or[1:nr, 1:nr]))
end

function kronmatrix(A::BandedMatrix, B::AbstractMatrix)
	rows = Fill(size(B,1), size(A,1))
	cols = Fill(size(B,2), size(A,2))
	BlockBandedMatrix(Kron(A, B), rows, cols, bandwidths(A))
end
function kronmatrix(A::BandedMatrix, B::BandedMatrix)
	rows = Fill(size(B,1), size(A,1))
	cols = Fill(size(B,2), size(A,2))
	BandedBlockBandedMatrix(Kron(A, B), rows, cols, bandwidths(A), bandwidths(B))
end
kronmatrix(A::AbstractMatrix, B::AbstractMatrix) = kron(A, B)

end # module ApproxFunAssociatedLegendre
