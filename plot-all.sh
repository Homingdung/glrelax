#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

command -v pdflatex >/dev/null 2>&1 || {
    echo "error: pdflatex is required (install a TeX distribution with pgfplots)." >&2
    exit 1
}

PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || {
    echo "error: Python interpreter not found: $PYTHON" >&2
    exit 1
}

methods=(mixed lm hdiv)
cases=(e3 e3-positive hopf)
missing=0

for case_name in "${cases[@]}"; do
    plot_dir="plot-${case_name}"
    if [[ ! -d "$plot_dir" ]]; then
        echo "missing: $plot_dir/ (LaTeX plot templates)" >&2
        missing=1
    fi

    for method in "${methods[@]}"; do
        data_file="${method}-${case_name}/data.csv"

        if [[ ! -f "$data_file" ]]; then
            echo "missing: $data_file" >&2
            missing=1
        elif ! head -n 1 "$data_file" | tr -d '\r' | grep -Eq '(^|,)t(,|$).*[, ]energy(,|$)' || \
             ! head -n 1 "$data_file" | tr -d '\r' | grep -Eq '(^|,)helicity(,|$)'; then
            echo "invalid columns: $data_file (required: t, energy, helicity)" >&2
            missing=1
        fi

        for kind in plot error; do
            tex_file="${plot_dir}/${kind}-${method}-${case_name}.tex"
            if [[ ! -f "$tex_file" ]]; then
                echo "missing: $tex_file" >&2
                missing=1
            fi
        done

    done
done

initial_relaxation_tex="plot-e3/plot-e3-comparison.tex"
if [[ ! -f "$initial_relaxation_tex" ]]; then
    echo "missing: $initial_relaxation_tex" >&2
    missing=1
fi

if (( missing )); then
    echo "No PDFs were generated. Add/fix all CSV files, then rerun ./plot-all.sh." >&2
    exit 1
fi

echo "Computing conservation errors"
"$PYTHON" filter.py

for case_name in "${cases[@]}"; do
    for method in "${methods[@]}"; do
        error_file="errors/${method}-${case_name}-error.csv"
        if ! head -n 1 "$error_file" | tr -d '\r' | grep -Eq '^t,helicity_error,divB_error$'; then
            echo "invalid columns: $error_file" >&2
            exit 1
        fi
    done
done

for case_name in "${cases[@]}"; do
    plot_dir="plot-${case_name}"
    for method in "${methods[@]}"; do
        for kind in plot error; do
            tex_file="${kind}-${method}-${case_name}.tex"
            echo "Compiling ${plot_dir}/${tex_file}"
            (
                cd "$plot_dir"
                pdflatex -interaction=nonstopmode -halt-on-error "$tex_file" >/dev/null
            )
        done
    done
done

echo "Compiling ${initial_relaxation_tex}"
(
    cd plot-e3
    pdflatex -interaction=nonstopmode -halt-on-error \
        "$(basename "$initial_relaxation_tex")" >/dev/null
)

echo "Removing generated intermediate files"
for case_name in "${cases[@]}"; do
    plot_dir="plot-${case_name}"
    for method in "${methods[@]}"; do
        rm -f "errors/${method}-${case_name}-error.csv"
        for kind in plot error; do
            stem="${plot_dir}/${kind}-${method}-${case_name}"
            rm -f "${stem}.aux" "${stem}.log" "${stem}.out" \
                "${stem}.toc" "${stem}.fls" "${stem}.fdb_latexmk" \
                "${stem}.synctex.gz"
        done
    done
done
initial_relaxation_stem="${initial_relaxation_tex%.tex}"
rm -f "${initial_relaxation_stem}.aux" "${initial_relaxation_stem}.log" \
    "${initial_relaxation_stem}.out" "${initial_relaxation_stem}.toc" \
    "${initial_relaxation_stem}.fls" "${initial_relaxation_stem}.fdb_latexmk" \
    "${initial_relaxation_stem}.synctex.gz"
rmdir errors 2>/dev/null || true

echo "Generated 19 PDFs under the three plot directories; intermediate files removed."
