#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/braids-matplotlib-cache}"
mkdir -p "$MPLCONFIGDIR"

#methods=(hdiv mixed lm)
methods=(mixed lm hdiv)
initial_conditions=(hopf)

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
echo "Generating all plots"
echo "============================================================"
python plt.py

echo "All nine experiments and plots are complete."
