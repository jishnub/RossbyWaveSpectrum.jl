#! /bin/bash
set -euo pipefail

curl -fsSL https://install.julialang.org | sh -s -- --yes
bash --rcfile ~/.bashrc -i ./installjulia.sh
if [ $? == 0 ]; then
	echo "Installation successful."
	echo "You may want to either source your shell's rc script, or launch a new interactive shell."
fi
