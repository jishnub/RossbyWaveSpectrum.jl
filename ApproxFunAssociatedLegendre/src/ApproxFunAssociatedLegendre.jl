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

export ColatitudeDomain
export PlmSpace, NormalizedPlmSpace

struct ColatitudeDomain{T} <: AbstractInterval{T} end
ColatitudeDomain() = ColatitudeDomain{Float64}()

Base.in(x::Real, ::ColatitudeDomain) = 0 <= x <= pi
Base.isempty(::ColatitudeDomain) = false

for f in [:endpoints, :closedendpoints]
	@eval DomainSets.$f(c::ColatitudeDomain{T}) where {T} = (T(0), T(pi))
end

abstract type AbstractPlmSpace{T} <: Space{ColatitudeDomain{T},T} end
struct PlmSpace{T} <: AbstractPlmSpace{T}
	m :: Int
end
PlmSpace(m::Int) = PlmSpace{Float64}(m)
Base.show(io::IO, sp::PlmSpace) = print(io, "PlmSpace(", azimuthalorder(sp), ")")

struct NormalizedPlmSpace{T} <: AbstractPlmSpace{T}
	parent :: PlmSpace{T}
end

NormalizedPlmSpace(m::Int) = NormalizedPlmSpace(PlmSpace(m))

Base.show(io::IO, sp::NormalizedPlmSpace) = print(io, "NormalizedPlmSpace(", azimuthalorder(sp), ")")

ApproxFunBase.domain(::AbstractPlmSpace{T}) where {T} = ColatitudeDomain{T}()

azimuthalorder(x::PlmSpace) = x.m
azimuthalorder(x::NormalizedPlmSpace) = x.parent.m

function ApproxFunBase.spacescompatible(a::AbstractPlmSpace, b::AbstractPlmSpace)
	azimuthalorder(a) == azimuthalorder(b)
end

function ApproxFunBase.evaluate(v::Vector, s::NormalizedPlmSpace, x)
	m = azimuthalorder(s)
	lr = range(m, length=length(v))
	itr = LegendrePolynomials.LegendrePolynomialIterator(x, m)
	sum(((Pl, vl),) -> Pl * vl, zip(itr, v))
end

Plmnorm(l, m) = exp(LegendrePolynomials.logplm_norm(l, m))
function ApproxFunBase.evaluate(v::Vector, s::PlmSpace, x)
	m = azimuthalorder(s)
	lr = range(m, length=length(v))
	itr = LegendrePolynomials.LegendrePolynomialIterator(x, m)
	sum(((l, Pl, vl),) -> Plmnorm(l, m) * Pl * vl, zip(lr, itr, v))
end

_equivalentlegendre(sp::PlmSpace) = Legendre()
_equivalentlegendre(sp::NormalizedPlmSpace) = NormalizedLegendre()
function equivalentlegendre(sp::AbstractPlmSpace)
	assert_m_zero(azimuthalorder(sp))
	_equivalentlegendre(sp)
end

function throw_m_error(m)
	throw(ArgumentError("only Legendre transforms (m=0) are supported, received m = $m"))
end
assert_m_zero(m) = m == 0 || throw_m_error(m)
for f in [:plan_transform, :plan_transform!]
	@eval function ApproxFunBase.$f(sp::AbstractPlmSpace, v::AbstractVector)
		m = azimuthalorder(sp)
		assert_m_zero(m)
		spL = equivalentlegendre(sp)
		ApproxFunBase.$f(spL, v)
	end
end

function _Fun(f, sp)
	m = azimuthalorder(sp)
	assert_m_zero(m)
	spL = equivalentlegendre(sp)
	F = Fun(f, spL)
	c = coefficients(F)
	Fun(sp, c)
end
ApproxFunBase.Fun(f, sp::AbstractPlmSpace) = _Fun(f, sp)
ApproxFunBase.Fun(f::Fun, sp::AbstractPlmSpace) = _Fun(f, sp)
ApproxFunBase.Fun(f::typeof(identity), sp::AbstractPlmSpace) = _Fun(f, sp)

abstract type PlmSpaceOperator{T} <: Operator{T} end
(::Type{T})() where {T<:PlmSpaceOperator} = T(UnsetSpace())

domainspace(p::PlmSpaceOperator) = p.ds
rangespace(p::PlmSpaceOperator) = p.ds

struct sinθ∂θ_Operator{T,DS<:Space{<:Any,T}} <: PlmSpaceOperator{T}
	ds :: DS
end

ApproxFunBase.setspace(P::sinθ∂θ_Operator, sp::Space) =
	sinθ∂θ_Operator{ApproxFunBase.prectype(sp),typeof(sp)}(sp)

function ApproxFunBase.promotedomainspace(P::PlmSpaceOperator, sp::AbstractPlmSpace)
	ApproxFunBase.setspace(P, sp)
end

BandedMatrices.bandwidths(C::sinθ∂θ_Operator) = (1,1)

C⁻ℓm(ℓ, m, T = Float64) = (ℓ < abs(m) ? T(0) : convert(T, √(T(ℓ - m) * T(ℓ + m) / (T(2ℓ - 1) * T(2ℓ + 1)))))
C⁺ℓm(ℓ, m, T) = C⁻ℓm(ℓ+1, m, T)
β⁻ℓm(ℓ, m, T = Float64) = (ℓ < abs(m) ? T(0) : convert(T, √(T(2ℓ + 1) / T(2ℓ - 1) * T(ℓ^2 - m^2))))
S⁺ℓm(ℓ, m, T = Float64) = convert(T, ℓ) * C⁺ℓm(ℓ, m, T)
S⁻ℓm(ℓ, m, T = Float64) = convert(T, ℓ) * C⁻ℓm(ℓ, m, T) - β⁻ℓm(ℓ, m, T)

function Base.getindex(P::sinθ∂θ_Operator{T,<:NormalizedPlmSpace}, i::Int, j::Int) where {T}
	m = azimuthalorder(domainspace(P))
	if j == i+1
		ℓ = m + i - 1
		S⁻ℓm(ℓ+1, m, T)
	elseif j == i-1
		ℓ = m + i - 1
		S⁺ℓm(ℓ-1, m, T)
	else
		zero(T)
	end
end

function ApproxFunBase.Multiplication(f::Fun{<:AbstractPlmSpace}, sp::AbstractPlmSpace)
	@assert azimuthalorder(space(f)) == 0
	ConcreteMultiplication(f, sp)
end
function BandedMatrices.bandwidths(M::ConcreteMultiplication{<:AbstractPlmSpace, <:AbstractPlmSpace})
	f = M.f
	nc = max(ncoefficients(f), 1) # treat empty vectors as 0
	(nc-1, nc-1)
end
function ApproxFunBase.rangespace(M::ConcreteMultiplication{<:AbstractPlmSpace, <:AbstractPlmSpace})
	ApproxFunBase.domainspace(M)
end
function Base.getindex(M::ConcreteMultiplication{<:NormalizedPlmSpace, <:NormalizedPlmSpace}, i::Int, j::Int)
	j > i && return getindex(M, j, i) # preserve symmetry
	sp = domainspace(M)
	m = azimuthalorder(sp)
	ℓ = i - 1 + m
	ℓ′ = j - 1 + m
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

end # module ApproxFunAssociatedLegendre
