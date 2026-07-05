from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
METHODS = ("mixed", "lm", "hdiv")
CASES = ("e3", "e3-positive", "hopf")
REQUIRED_COLUMNS = {"t", "helicity", "divB"}


def process_file(input_path: Path, output_path: Path) -> None:
    data = pd.read_csv(input_path)
    if data.empty:
        raise ValueError(f"{input_path}: data file is empty")

    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{input_path}: missing required column(s): {names}")

    errors = pd.DataFrame(
        {
            "t": data["t"],
            "helicity_error": (data["helicity"] - data["helicity"].iloc[0]).abs(),
            "divB_error": (data["divB"] - data["divB"].iloc[0]).abs(),
        }
    )

    # At t=0 both differences are exactly zero and cannot be shown on a log axis.
    errors = errors.iloc[1:]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(output_path, index=False)
    print(f"Wrote {output_path.relative_to(ROOT)}")


def main() -> None:
    for case_name in CASES:
        for method in METHODS:
            stem = f"{method}-{case_name}"
            process_file(
                ROOT / stem / "data.csv",
                ROOT / "errors" / f"{stem}-error.csv",
            )


if __name__ == "__main__":
    main()
