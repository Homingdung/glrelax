# LaTeX / PGFPlots figures

The three plot directories contain 18 standalone PGFPlots documents:

- `plot-e3/`: three energy plots and three error plots.
- `plot-e3-positive/`: wrappers that reuse the `plot-e3/` templates.
- `plot-hopf/`: three energy/helicity plots and three error plots.

Run error calculation and all PGFPlots figures from the repository root:

```bash
./plot-all.sh
```

The command performs these stages in order:

1. Validate all nine `data.csv` inputs.
2. Run `filter.py` to generate nine temporary error CSV files.
3. Compile the 18 standalone PGFPlots PDFs.
4. Delete the temporary error CSV and LaTeX auxiliary files, leaving the PDFs
   alongside the source `.tex` files.

Set `PYTHON` if the required interpreter is not on `PATH`:

```bash
PYTHON=/path/to/python ./plot-all.sh
```

Trace rendering is intentionally separate because it is slow. Render all three
cases with:

```bash
./trace-plot-all.sh
```

This validates all nine `parker.pvd` inputs, runs the E3, E3-positive, and Hopf
trace scripts, and writes 27 PNG files to the corresponding `plot-*/trace/`
directories. Override the ParaView executable when needed:

```bash
PV_PYTHON=/path/to/pvpython ./trace-plot-all.sh
```

Each main data file must be named `<method>-<case>/data.csv`, where `method` is
`mixed`, `lm`, or `hdiv`, and `case` is `e3`, `e3-positive`, or `hopf`. Its
header must include `t,helicity,energy` (additional columns are allowed).

Each error file must be named `errors/<method>-<case>-error.csv` and have this
header:

```text
t,helicity_error,divB_error
```

The E3 error figures plot `divB_error`; the Hopf error figures plot both
`helicity_error` and `divB_error`. Error axes use logarithmic scale, so error
values must be strictly positive.
