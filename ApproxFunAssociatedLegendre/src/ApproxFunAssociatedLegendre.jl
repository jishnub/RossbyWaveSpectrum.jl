module ApproxFunAssociatedLegendre

using DomainSets
using ApproxFunBase
using ApproxFunBase: SubOperator, UnsetSpace, ConcreteMultiplication, MultiplicationWrapper,
	ConstantTimesOperator, ConversionWrapper, SpaceOperator
import ApproxFunBase: domainspace, rangespace, Multiplication, setspace, Conversion,
	spacescompatible, maxspace_rule, Fun, coefficients, DefiniteIntegral, DefiniteIntegralWrapper,
	canonicalspace, Derivative, DerivativeWrapper, hasconversion
using BandedMatrices
import BandedMatrices: BandedMatrix, bandwidths
using ApproxFunOrthogonalPolynomials
using ApproxFunSingularities
using BlockBandedMatrices
using HalfIntegers

export NormalizedPlm
export sinθdθ_Operator
export cosθ_Operator
export sinθdθ_plus_2cosθ_Operator
export HorizontalLaplacian
export expand
export kronmatrix
export kronmatrix!

const JacobiWeightNormalized{T,P<:PolynomialSpace{ChebyshevInterval{T}}} =
	JacobiWeight{<:NormalizedPolynomialSpace{P,ChebyshevInterval{T}}, ChebyshevInterval{T}}

const JacobiWeightMaybeNormalized{T,P<:PolynomialSpace{ChebyshevInterval{T}}} =
	Union{JacobiWeightNormalized{T,P}, JacobiWeight{P,ChebyshevInterval{T}}}

const PolynomialMaybeNormalized{T,P<:PolynomialSpace{ChebyshevInterval{T}}} =
	Union{P,NormalizedPolynomialSpace{P,ChebyshevInterval{T}}}

struct NormalizedPlm{T<:Real, NJS<:Space{ChebyshevInterval{T},T}} <: Space{ChebyshevInterval{T},T}
	m :: Int
	jacobispace :: NJS
end
azimuthalorder(p::NormalizedPlm) = p.m

canonicalspace(A::NormalizedPlm) = A.jacobispace
function canonicalspace_zerostrip(A::NormalizedPlm)
	azimuthalorder(A) == 0 ? NormalizedLegendre(domain(A)) : canonicalspace(A)
end

ApproxFunBase.domain(::NormalizedPlm{T}) where {T} = ChebyshevInterval{T}()
function NormalizedPlm(m::Int)
	sp = NormalizedUltraspherical(NormalizedJacobi(m,m))
	NormalizedPlm(m, JacobiWeight(half(m), half(m), sp))
end
NormalizedPlm(; m::Int) = NormalizedPlm(m)
Base.show(io::IO, sp::NormalizedPlm) = print(io, "NormalizedPlm(m=", azimuthalorder(sp), ")")

spacescompatible(b::PolynomialMaybeNormalized, a::NormalizedPlm) =
	spacescompatible(a, b)
function spacescompatible(a::NormalizedPlm, b::PolynomialMaybeNormalized)
	iseven(azimuthalorder(a)) && spacescompatible(canonicalspace(a), b)
end

hasconversion(J::PolynomialMaybeNormalized, sp::NormalizedPlm) = hasconversion(J, canonicalspace(sp))
hasconversion(sp::NormalizedPlm, J::PolynomialMaybeNormalized) = hasconversion(canonicalspace(sp), J)
function hasconversion(J::JacobiWeightMaybeNormalized{T}, sp::NormalizedPlm{T}) where {T}
	hasconversion(J, canonicalspace(sp))
end
function hasconversion(sp::NormalizedPlm{T}, J::JacobiWeightMaybeNormalized{T}) where {T}
	hasconversion(canonicalspace(sp), J)
end

function Conversion(J::PolynomialMaybeNormalized, sp::NormalizedPlm)
	C = Conversion(J, canonicalspace_zerostrip(sp))
	ConversionWrapper(SpaceOperator(C, J, sp))
end
function Conversion(J::JacobiWeightMaybeNormalized, sp::NormalizedPlm)
	C = Conversion(J, canonicalspace(sp))
	ConversionWrapper(SpaceOperator(C, J, sp))
end
function Conversion(sp::NormalizedPlm, J::PolynomialMaybeNormalized)
	C = Conversion(canonicalspace_zerostrip(sp), J)
	ConversionWrapper(SpaceOperator(C, sp, J))
end
function Conversion(sp::NormalizedPlm, J::JacobiWeightMaybeNormalized)
	C = Conversion(canonicalspace(sp), J)
	ConversionWrapper(SpaceOperator(C, sp, J))
end

# Assume that only one m is being used
function Base.union(A::NormalizedPlm, B::NormalizedPlm)
	@assert azimuthalorder(A) == azimuthalorder(B)
	A
end

function maxspace_rule(A::Union{PolynomialMaybeNormalized, JacobiWeightMaybeNormalized}, B::NormalizedPlm)
	maxspace(A, canonicalspace(B))
end
function maxspace_rule(A::NormalizedPlm, B::Union{PolynomialMaybeNormalized, JacobiWeightMaybeNormalized})
	maxspace(canonicalspace(A), B)
end
function maxspace_rule(A::NormalizedPlm, B::NormalizedPlm)
	@assert azimuthalorder(A) == azimuthalorder(B)
	A
end

function Base.:(==)(A::NormalizedPlm, B::NormalizedPlm)
	@assert azimuthalorder(A) == azimuthalorder(B)
	true
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

function spacescompatible(a::NormalizedPlm, b::NormalizedPlm)
	azimuthalorder(a) == 0 || azimuthalorder(b) == 0 ||
		azimuthalorder(a) == azimuthalorder(b) || return false
	spacescompatible(canonicalspace(a), canonicalspace(b))
end

function ApproxFunBase.evaluate(v::Vector, s::NormalizedPlm, x)
	evaluate(v, canonicalspace(s), x)
end

for f in [:plan_transform, :plan_transform!, :plan_itransform, :plan_itransform!]
	@eval function ApproxFunBase.$f(sp::NormalizedPlm, v::Vector)
		ApproxFunBase.$f(canonicalspace(sp), v)
	end
end

function _Fun(f, sp::NormalizedPlm)
	F = Fun(f, canonicalspace(sp))::Fun{<:JacobiWeight{<:NormalizedUltraspherical}}
	c = coefficients(F)
	Fun(sp, c)
end
Fun(f::typeof(identity), sp::NormalizedPlm) = _Fun(f, sp)

canonicalfun(f::Fun{<:NormalizedPlm}) = Fun(canonicalspace(f), coefficients(f))
maybecanonicalfun(f::Fun{<:NormalizedPlm}) = canonicalfun(f)
maybecanonicalfun(f::Fun) = f

# abs2 necessarily produces a polynomial, so we may decompose this in a Chebyshev space
Base.abs2(f::Fun{<:NormalizedPlm,<:Real}) = Fun(abs2(canonicalfun(f)), Chebyshev(domain(f)))
Base.abs2(f::Fun{<:NormalizedPlm,<:Complex}) = abs2(real(f)) + abs2(imag(f))

function coefficients(f::AbstractVector,
		sp::JacobiWeight{<:Any,ChebyshevInterval{T}}, snp::NormalizedPlm{T}) where {T<:Real}

	coefficients(f, sp, canonicalspace(snp))
end

function coefficients(f::AbstractVector,
		snp::NormalizedPlm{T}, sp::JacobiWeight{<:Any,ChebyshevInterval{T}}) where {T<:Real}

	coefficients(f, canonicalspace(snp), sp)
end

function coefficients(f::AbstractVector, snp1::NormalizedPlm, snp2::NormalizedPlm)
	coefficients(f, snp1.jacobispace, snp2.jacobispace)
end

Base.@constprop :aggressive function Derivative(sp::NormalizedPlm, k::Int)
	DerivativeWrapper(Derivative(canonicalspace(sp), k), k)
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

bandwidths(::Union{sinθdθ_Operator, cosθ_Operator, sinθdθ_plus_2cosθ_Operator}) = (1,1)

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

function Multiplication(f::Fun, sp::NormalizedPlm)
	g = maybecanonicalfun(f)
	jsp = canonicalspace(sp)
	M = Multiplication(g, jsp) → jsp
	S = SpaceOperator(M, sp, sp)
	MultiplicationWrapper(f, S)
end

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

bandwidths(::HorizontalLaplacian) = (0,0)

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

function Multiplication(f::Fun{<:NormalizedPlm}, sp::PolynomialMaybeNormalized)
	Multiplication(canonicalfun(f), sp)
end

# Definite integrals
function DefiniteIntegral(S::NormalizedPlm)
	D = DefiniteIntegral(canonicalspace(S))
	DefiniteIntegralWrapper(D, S)
end

Base.sum(f::Fun{<:NormalizedPlm}) = sum(Fun(space(f).jacobispace, coefficients(f)))

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

function expand(C::ConstantTimesOperator, K::KroneckerOperator)
	(; λ, op) = C
	expand(op, expand(λ, K))
end
function expand(K::KroneckerOperator, C::ConstantTimesOperator)
	(; λ, op) = C
	expand(expand(K, λ), op)
end

##################################################################################

function addto!(x, y)
	for i in eachindex(x,y)
		x[i] += y[i]
	end
	return x
end

function kronmatrix!(C, P::PlusOperator, nr,
		ℓindrange_row::AbstractRange, ℓindrange_col::AbstractRange)

	C .= 0
	X = zeros(eltype(P), nr*length(ℓindrange_row), nr*length(ℓindrange_col))
	mapfoldl(addto!, P.ops, init=C) do op
		let Y = X
			kronmatrix!(Y, op, nr, ℓindrange_row, ℓindrange_col)
		end
	end
	return C
end
function kronmatrix(P::PlusOperator, nr,
		ℓindrange_row::AbstractRange, ℓindrange_col::AbstractRange)

	C = zeros(eltype(P), nr*length(ℓindrange_row), nr*length(ℓindrange_col))
	kronmatrix!(C, P, nr, ℓindrange_row, ℓindrange_col)
	return C
end
function _kronmatrix(Kops::NTuple{2,Operator{T}}, nr,
		ℓindrange_row::AbstractRange, ℓindrange_col::AbstractRange) where {T}

	Or::Operator{T}, Oθ::Operator{T} = Kops
	ℓindrange_row_ur = first(ℓindrange_row):last(ℓindrange_row)
	ℓindrange_col_ur = first(ℓindrange_col):last(ℓindrange_col)
	O1full = Oθ[ℓindrange_row_ur, ℓindrange_col_ur]::AbstractMatrix{T}
	O1 = O1full[1:step(ℓindrange_row):end, 1:step(ℓindrange_col):end]::AbstractMatrix{T}
	Matrix(O1)::Matrix{T}, Matrix(Or[1:nr, 1:nr])::Matrix{T}
end
function kronmatrix(K::KroneckerOperator, nr,
		ℓindrange_row::AbstractRange, ℓindrange_col::AbstractRange)

	C = zeros(eltype(K), nr*length(ℓindrange_row), nr*length(ℓindrange_col))
	kronmatrix!(C, K, nr, ℓindrange_row, ℓindrange_col)
end
function kronmatrix!(C, K::KroneckerOperator, nr,
		ℓindrange_row::AbstractRange, ℓindrange_col::AbstractRange)

	A, B = _kronmatrix(K.ops, nr, ℓindrange_row, ℓindrange_col)
	kron!(C, A, B)
	return C
end

end # module ApproxFunAssociatedLegendre
