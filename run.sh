#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/braids-matplotlib-cache}"
mkdir -p "$MPLCONFIGDIR"

#methods=(hdiv mixed lm)
methods=(lm)
initial_conditions=(hopf E3 E3-positive)
#initial_conditions=(hopf E3 E3-positive)
# Complete input combinations (3 methods x 3 initial conditions).
# Uncomment any command below to run that experiment independently.
# IC=hopf       python hdiv.py
# IC=hopf       python mixed.py
# IC=hopf       python lm.py
# IC=E3         python hdiv.py
# IC=E3         python mixed.py
# IC=E3         python lm.py
# IC=E3-positive python hdiv.py
# IC=E3-positive python mixed.py
# IC=E3-positive python lm.py

for ic in "${initial_conditions[@]}"; do
    if [[ "$ic" == "hopf" ]]; then
        bc="closed"
    else
        bc="periodic"
    fi
    for method in "${methods[@]}"; do
        echo "============================================================"
        echo "Running ${method}.py with IC=${ic}, BC=${bc}"
        echo "============================================================"
        IC="$ic" python "${method}.py"
    done
done

echo "============================================================"
echo "Generating Matplotlib, PGFPlots, and error outputs"
echo "============================================================"
python plt.py
./plot-all.sh

echo "All nine experiments and plots are complete."
