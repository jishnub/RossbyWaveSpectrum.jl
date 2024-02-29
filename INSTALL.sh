#! /bin/bash
set -euo pipefail

curl -fsSL https://install.julialang.org | sh -s -- --yes
source ./juliaversion
juliaversion=${1:-$RossbyWaveJuliaVersion}
bash --rcfile ~/.bashrc -i ./installjulia.sh $juliaversion
if [ $? == 0 ]; then
	echo "Installation successful."
	echo "You may want to either source your shell's rc script, or launch a new interactive shell."
	echo "RossbyWaveJuliaVersion=$juliaversion" > juliaversioninstalled
fi
