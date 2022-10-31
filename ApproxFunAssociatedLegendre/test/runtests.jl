using Test
using ApproxFunAssociatedLegendre

using Aqua
@testset "project quality" begin
    Aqua.test_all(ApproxFunAssociatedLegendre, ambiguities = false)
end
