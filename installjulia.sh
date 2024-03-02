source ./juliaversion
juliaversion=$RossbyWaveJuliaVersion
RESOLVE=false
VERSIONFLAG=false
while getopts ":rv:" flag; do
    case ${flag} in
        r)
        	RESOLVE=true
            ;;
        v)
			VERSIONFLAG=true
            juliaversion=$OPTARG
            ;;
        ?)
            echo "Invalid option: -${OPTARG}"
            exit 1
            ;;
    esac
done
# backward compatibility with version as a positional argument
if [ $VERSIONFLAG == false ]; then
	shift $((OPTIND - 1))
	juliaversion=${1:-$juliaversion}
fi

juliaup add $juliaversion
if [ $RESOLVE == true ]; then
	echo "Resolving environments on Julia version $juliaversion"
	julia +$juliaversion -e \
	'import Pkg;
	Pkg.activate("ApproxFunAssociatedLegendre"); Pkg.resolve(); Pkg.instantiate();
	Pkg.activate("."); Pkg.resolve(); Pkg.instantiate();
	Pkg.activate("RossbyPlots"); Pkg.resolve(); Pkg.instantiate();
	'
else
	julia +$juliaversion -e \
	'import Pkg;
	Pkg.activate("ApproxFunAssociatedLegendre"); Pkg.instantiate();
	Pkg.activate("."); Pkg.instantiate();
	Pkg.activate("RossbyPlots"); Pkg.instantiate();
	'
fi
