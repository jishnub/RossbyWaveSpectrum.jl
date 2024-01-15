#! /bin/bash
set -euo pipefail

curl -fsSL https://install.julialang.org | sh -s -- --yes
TMPFILE=$(mktemp)
cat > $TMPFILE <<- EOM
source ~/.bashrc
juliaup add 1.10.0
julia +1.10.0 -e \
'import Pkg;
Pkg.activate("ApproxFunAssociatedLegendre"); Pkg.instantiate();
Pkg.activate("."); Pkg.instantiate();
Pkg.activate("RossbyPlots"); Pkg.instantiate();'
rm $TMPFILE
exit 0
EOM
bash --rcfile $TMPFILE -i
echo "Installation successful. You may want to either source your shell's rc script, or launch a new interactive shell."
