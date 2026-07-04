#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/braids-matplotlib-cache}"
mkdir -p "$MPLCONFIGDIR"

methods=(hdiv mixed lm)
initial_conditions=(E3-positive)

for ic in "${initial_conditions[@]}"; do
    for method in "${methods[@]}"; do
        echo "============================================================"
        echo "Running ${method}.py with IC=${ic}"
        echo "============================================================"
        IC="$ic" python "${method}.py"
    done
done

echo "============================================================"
echo "Generating E3 plots"
echo "============================================================"
bash plot-e3/run.sh

echo "============================================================"
echo "Generating E3-positive plots"
echo "============================================================"
bash plot-e3-positive/run.sh

echo "============================================================"
echo "Generating Hopf plots"
echo "============================================================"
bash plot-hopf/run.sh

echo "All nine experiments and plots are complete."
