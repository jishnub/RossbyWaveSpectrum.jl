#! /bin/bash
set -euo pipefail

RESOLVE=false
source ./juliaversion
juliaversion=$RossbyWaveJuliaVersion
while getopts ":rv:" flag; do
    case ${flag} in
        v)
            juliaversion=$OPTARG
            ;;
    esac
done
curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel=$juliaversion
bash --rcfile ~/.bashrc -i ./installjulia.sh $@
if [ $? == 0 ]; then
	echo "Installation successful."
	echo "You may want to either source your shell's rc script, or launch a new interactive shell."
	echo "RossbyWaveJuliaVersion=$juliaversion" > juliaversioninstalled
fi
