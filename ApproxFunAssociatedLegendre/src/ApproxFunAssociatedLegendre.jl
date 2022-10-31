module ApproxFunAssociatedLegendre

using DomainSets
using ApproxFunBase
using ApproxFunBase: SubOperator, UnsetSpace, ConcreteMultiplication
import ApproxFunBase: domainspace, rangespace, Multiplication, setspace
using BandedMatrices
import BandedMatrices: BandedMatrix
using WignerSymbols
using LegendrePolynomials
using ApproxFunOrthogonalPolynomials
using ApproxFunSingularities
using SpecialFunctions
using LazyArrays
using BlockBandedMatrices

export NormalizedPlm

struct NormalizedPlm{T, NJS <: Space{ChebyshevInterval{T},T}} <: Space{ChebyshevInterval{T},T}
	m :: Int
	jacobispace :: NJS
end
ApproxFunBase.domain(n::NormalizedPlm{T}) where {T} = ChebyshevInterval{T}()
NormalizedPlm(m::Int) = NormalizedPlm(m, JacobiWeight(m/2, m/2, NormalizedJacobi(m,m)))
Base.show(io::IO, sp::NormalizedPlm) = print(io, "NormalizedPlm(", sp.m, ")")

const JacobiMaybeNormalized = Union{Jacobi, NormalizedPolynomialSpace{<:Jacobi}}
function assertLegendre(sp::JacobiMaybeNormalized)
	csp = ApproxFunBase.canonicalspace(sp)
	@assert csp == Legendre() "multiplication is only defined in Legendre space"
end
function assertLegendre(sp::NormalizedPlm)
	@assert sp.m == 0 "multiplication is only defined in Legendre space"
end

function ApproxFunBase.union_rule(a::NormalizedPlm, b::Union{ConstantSpace, PolynomialSpace})
	a.m == 0 || return ApproxFunBase.NoSpace()
	union(Legendre(), b)
end

# Legendre polynomial norm
plnorm2(ℓ) = (2ℓ+1)/2
plnorm(ℓ) = sqrt(plnorm2(ℓ))

# Associated Legendre polynomial norm
plmnorm(l, m) = exp(LegendrePolynomials.logplm_norm(l, m))

function ApproxFunBase.spacescompatible(a::NormalizedPlm, b::NormalizedPlm)
	a.m == 0 || b.m == 0 || a.m == b.m || return false
	ApproxFunBase.spacescompatible(a.jacobispace, b.jacobispace)
end

function ApproxFunBase.evaluate(v::Vector, s::NormalizedPlm, x)
	m = s.m
	evaluate(v, s.jacobispace, x) * (-1)^m
end

for f in [:plan_transform, :plan_transform!]
	@eval function ApproxFunBase.$f(sp::NormalizedPlm, v::AbstractVector)
		m = sp.m
		ApproxFunBase.$f(sp.jacobispace, v .* (-1)^m)
	end
end

function _Fun(f, sp::NormalizedPlm)
	F = Fun(f, sp.jacobispace)
	m = sp.m
	c = coefficients(F) .* (-1)^m
	Fun(sp, c)
end
ApproxFunBase.Fun(f, sp::NormalizedPlm) = _Fun(f, sp)
ApproxFunBase.Fun(f::Fun, sp::NormalizedPlm) = _Fun(f, sp)
ApproxFunBase.Fun(f::typeof(identity), sp::NormalizedPlm) = _Fun(f, sp)

function ApproxFunBase.Derivative(sp::NormalizedPlm, k::Int)
	m = sp.m
	ApproxFunBase.DerivativeWrapper((-1)^m * Derivative(sp.jacobispace, k), k)
end

abstract type PlmSpaceOperator{DS<:Space} <: Operator{Float64} end
(::Type{O})() where {O<:PlmSpaceOperator} = O(UnsetSpace())

domainspace(p::PlmSpaceOperator) = p.ds
rangespace(p::PlmSpaceOperator) = p.ds
function ApproxFunBase.promotedomainspace(P::PlmSpaceOperator, sp::NormalizedPlm)
	ApproxFunBase.setspace(P, sp)
end

index_to_ℓ(i, m) = m + i - 1

struct sinθ∂θ_Operator{DS<:Space} <: PlmSpaceOperator{DS}
	ds :: DS
end

ApproxFunBase.setspace(P::sinθ∂θ_Operator, sp::Space) =
	sinθ∂θ_Operator{typeof(sp)}(sp)

BandedMatrices.bandwidths(C::sinθ∂θ_Operator) = (1,1)

C⁻ℓm(ℓ, m, T = Float64) = (ℓ < abs(m) ? T(0) : convert(T, √(T(ℓ - m) * T(ℓ + m) / (T(2ℓ - 1) * T(2ℓ + 1)))))
C⁺ℓm(ℓ, m, T) = C⁻ℓm(ℓ+1, m, T)
β⁻ℓm(ℓ, m, T = Float64) = (ℓ < abs(m) ? T(0) : convert(T, √(T(2ℓ + 1) / T(2ℓ - 1) * T(ℓ^2 - m^2))))
S⁺ℓm(ℓ, m, T = Float64) = convert(T, ℓ) * C⁺ℓm(ℓ, m, T)
S⁻ℓm(ℓ, m, T = Float64) = convert(T, ℓ) * C⁻ℓm(ℓ, m, T) - β⁻ℓm(ℓ, m, T)

function Base.getindex(P::sinθ∂θ_Operator{<:NormalizedPlm}, i::Int, j::Int)
	m = domainspace(P).m
	if j == i+1
		ℓ = index_to_ℓ(i, m)
		S⁻ℓm(ℓ+1, m, Float64)
	elseif j == i-1
		ℓ = index_to_ℓ(i, m)
		S⁺ℓm(ℓ-1, m, Float64)
	else
		zero(Float64)
	end
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
function Base.getindex(M::ConcreteMultiplication{<:NormalizedPlm, <:NormalizedPlm}, i::Int, j::Int)
	j > i && return getindex(M, j, i) # preserve symmetry
	sp = domainspace(M)
	m = sp.m
	ℓ = index_to_ℓ(i, m)
	ℓ′ = index_to_ℓ(j, m)
	bw = bandwidth(M, 1)
	abs(ℓ - ℓ′) <= bw || return zero(eltype(M))
	fc = coefficients(M.f)
	T = promote_type(eltype(fc), ApproxFunBase.prectype(sp))
	s = zero(T)

	for (ℓ′′, fℓ′′) in zip(range(0, length=length(fc)), fc)
		iseven(ℓ + ℓ′′ + ℓ′) || continue
		WignerSymbols.δ(ℓ, ℓ′′, ℓ′) || continue
		pre = √(2ℓ′′+1)
		w1 = wigner3j(T, ℓ, ℓ′′, ℓ′, 0, 0)
		w2 = wigner3j(T, ℓ, ℓ′′, ℓ′, -m, 0)
		s += fℓ′′ * pre * w1 * w2
	end
	x = (-1)^m * √((2ℓ+1)*(2ℓ′+1)/2) * s
	eltype(M)(x)
end

struct HorizontalLaplacian{DS<:Space} <: PlmSpaceOperator{DS}
	ds :: DS
end
ApproxFunBase.setspace(P::HorizontalLaplacian, sp::Space) =
	HorizontalLaplacian{typeof(sp)}(sp)

BandedMatrices.bandwidths(M::HorizontalLaplacian) = (0,0)
function Base.getindex(P::HorizontalLaplacian{<:NormalizedPlm}, i::Int, j::Int)
	m = domainspace(P).m
	if j == i
		ℓ = index_to_ℓ(i, m)
		convert(Float64, -ℓ*(ℓ+1))
	else
		zero(Float64)
	end
end

function matrix(P::PlusOperator, nr, nθ)
	mapfoldl(op -> matrix(op, nr, nθ), +, P.ops)
end
matrix(T::TimesOperator, nr, nθ) = matrix(KroneckerOperator(T), nr, nθ)
function matrix(K::KroneckerOperator, nr, nθ)
	Oθ, Or = K.ops
	matrix(Oθ[1:nθ, 1:nθ], Or[1:nr, 1:nr])
end

function matrix(A::BandedMatrix, B::AbstractMatrix)
	rows = Fill(size(B,1), size(A,1))
	cols = Fill(size(B,2), size(A,2))
	BlockBandedMatrix(Kron(A, B), rows, cols, bandwidths(A))
end
function matrix(A::BandedMatrix, B::BandedMatrix)
	rows = Fill(size(B,1), size(A,1))
	cols = Fill(size(B,2), size(A,2))
	BandedBlockBandedMatrix(Kron(A, B), rows, cols, bandwidths(A), bandwidths(B))
end
matrix(A::AbstractMatrix, B::AbstractMatrix) = kron(A, B)

end # module ApproxFunAssociatedLegendre
