juliaup add 1.10.1
julia +1.10.1 -e \
'import Pkg;
Pkg.activate("ApproxFunAssociatedLegendre"); Pkg.instantiate();
Pkg.activate("."); Pkg.instantiate();
Pkg.activate("RossbyPlots"); Pkg.instantiate();'
