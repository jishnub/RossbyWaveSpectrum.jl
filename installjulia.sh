source ./juliaversion
juliaversion=${1:-$RossbyWaveJuliaVersion}
juliaup add $juliaversion
julia +$juliaversion -e \
'import Pkg;
Pkg.activate("ApproxFunAssociatedLegendre"); Pkg.instantiate();
Pkg.activate("."); Pkg.instantiate();
Pkg.activate("RossbyPlots"); Pkg.instantiate();'
