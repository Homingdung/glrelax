#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PV_PYTHON:-}" ]]; then
    pvpython="$PV_PYTHON"
elif command -v pvpython >/dev/null 2>&1; then
    pvpython="$(command -v pvpython)"
elif [[ -x /Applications/ParaView-5.13.2.app/Contents/bin/pvpython ]]; then
    pvpython=/Applications/ParaView-5.13.2.app/Contents/bin/pvpython
else
    echo "error: pvpython not found; set PV_PYTHON to the ParaView Python executable." >&2
    exit 1
fi

methods=(mixed lm hdiv)
cases=(e3 e3-positive hopf)
missing=0

for case_name in "${cases[@]}"; do
    for method in "${methods[@]}"; do
        pvd_file="${method}-${case_name}/parker.pvd"
        if [[ ! -f "$pvd_file" ]]; then
            echo "missing: $pvd_file" >&2
            missing=1
        fi
    done
done

if (( missing )); then
    echo "No trace plots were generated. Add all parker.pvd datasets first." >&2
    exit 1
fi

echo "Rendering E3 trace plots"
"$pvpython" trace-plot-e3.py

echo "Rendering E3-positive trace plots"
"$pvpython" trace-plot-e3-positive.py

echo "Rendering Hopf trace plots"
"$pvpython" trace-plot-hopf.py

echo "Generated 27 trace PNGs under plot-e3/trace/, plot-e3-positive/trace/, and plot-hopf/trace/."
