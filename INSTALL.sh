#! /bin/bash
set -euo pipefail

args=$@
source ./juliaversion
juliaversion=$RossbyWaveJuliaVersion
VERSIONFLAG=false
while getopts ":rv:" flag; do
    case ${flag} in
        v)
            VERSIONFLAG=true
            juliaversion=$OPTARG
            ;;
    esac
done
# backward compatibility with version as a positional argument
if [ $VERSIONFLAG == false ]; then
    shift $((OPTIND - 1))
    juliaversion=${1:-$juliaversion}
fi
echo "using Julia version $juliaversion"
curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel=$juliaversion
bash --rcfile ~/.bashrc -i ./installjulia.sh $args
if [ $? == 0 ]; then
	echo "Installation successful."
	echo "You may want to either source your shell's rc script, or launch a new interactive shell."
	echo "RossbyWaveJuliaVersion=$juliaversion" > juliaversioninstalled
fi
