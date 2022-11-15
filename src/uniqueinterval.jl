using IntervalSets: AbstractInterval
import IntervalSets: endpoints, closedendpoints

struct UniqueInterval{T, I<:AbstractInterval{T}} <: AbstractInterval{T}
	parentinterval :: I
end

for f in [:endpoints, :closedendpoints]
	@eval $f(m::UniqueInterval) = $f(m.parentinterval)
end

Base.in(x, m::UniqueInterval) = in(x, m.parentinterval)
Base.isempty(m::UniqueInterval) = isempty(m.parentinterval)

ApproxFunBase.domainscompatible(a::UniqueInterval, b::UniqueInterval) = a == b
ApproxFunBase.isambiguous(a::UniqueInterval) = false

Base.:(==)(a::UniqueInterval, b::UniqueInterval) = (@assert a.parentinterval == b.parentinterval; true)

function Base.show(io::IO, m::UniqueInterval)
	print(io, "UniqueInterval(")
	show(io, m.parentinterval)
	print(io, ")")
end
