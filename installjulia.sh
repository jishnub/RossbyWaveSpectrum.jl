source ./juliaversion
juliaversion=$RossbyWaveJuliaVersion
RESOLVE=false
while getopts ":rv:" flag; do
    case ${flag} in
        r)
        	RESOLVE=true
            ;;
        v)
            juliaversion=$OPTARG
            ;;
        ?)
            echo "Invalid option: -${OPTARG}"
            exit 1
            ;;
    esac
done
juliaup add $juliaversion
if [ $RESOLVE == true ]; then
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
