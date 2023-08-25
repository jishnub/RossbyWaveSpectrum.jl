using IntervalSets: AbstractInterval
import IntervalSets: endpoints, closedendpoints
import ApproxFun: Space
using ApproxFunBase

struct UniqueInterval{T, I<:AbstractInterval{T}} <: AbstractInterval{T}
	parentinterval :: I
end

Base.parent(U::UniqueInterval) = U.parentinterval

for f in [:endpoints, :closedendpoints]
	@eval $f(m::UniqueInterval) = $f(m.parentinterval)
end

Base.in(x, m::UniqueInterval) = in(x, m.parentinterval)
Base.isempty(m::UniqueInterval) = isempty(m.parentinterval)

ApproxFunBase.domainscompatible(a::UniqueInterval, b::UniqueInterval) = a == b
ApproxFunBase.isambiguous(a::UniqueInterval) = false

Space(a::UniqueInterval) = Chebyshev(a)

Base.:(==)(a::UniqueInterval, b::UniqueInterval) = (@assert a.parentinterval == b.parentinterval; true)

function Base.split(a::UniqueInterval, pts)
	@assert all(((x,y),) -> x==y, zip(endpoints(a), pts))
	a
end

function ApproxFunBase.Fun(g::Fun{<:Chebyshev{<:UniqueInterval}}, d::UniqueInterval)
	@assert domain(g) == d
	g
end

function Base.show(io::IO, m::UniqueInterval)
	print(io, "UniqueInterval(")
	show(io, m.parentinterval)
	print(io, ")")
end
