import glob
import os
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_params(directory):
    param_path = os.path.join(directory, "param.txt")
    params = {}
    if os.path.exists(param_path):
        with open(param_path) as stream:
            for line in stream:
                fields = line.split()
                if len(fields) >= 2 and not line.lstrip().startswith("#"):
                    params[fields[0]] = fields[1]
    return params


def data_paths_from_args():
    if len(sys.argv) > 1:
        paths = []
        for path in sys.argv[1:]:
            paths.append(path if path.endswith(".csv") else os.path.join(path, "data.csv"))
    else:
        paths = glob.glob("*/data.csv")

    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(
            "No data.csv found. Run a scheme first, or pass data directories/CSV files."
        )

    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Missing data file(s): " + ", ".join(missing))
    return paths


def legacy_background_energy(params):
    """Recover the E3 background energy for CSV files without that column."""
    if params.get("ic") not in ("E3", "E3-positive"):
        return None
    lengths = params.get("Lx,Ly,Lz")
    if lengths is None:
        return None

    volume = 1.0
    for length in lengths.split(","):
        volume *= float(length)
    return volume


def plot_data(data_path):
    output_dir = os.path.dirname(data_path) or "."
    data = pd.read_csv(data_path)
    params = read_params(output_dir)
    periodic = params.get("bc") == "periodic"
    helicity_label = "generalized helicity" if periodic else "helicity"
    helicity_abs_label = r"$|H_G|$" if periodic else r"$|H|$"

    required = {"t", "helicity", "energy"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{data_path}: missing column(s): {', '.join(sorted(missing))}")

    if (data["helicity"] < 0).any():
        n_negative = int((data["helicity"] < 0).sum())
        minimum = float(data["helicity"].min())
        print(f"[warning] {data_path}: {helicity_label} is negative at "
              f"{n_negative}/{len(data)} time steps (min = {minimum:.6e}).")

    figure, axes = plt.subplots(figsize=(10, 6))
    axes.plot(data["t"], data["energy"], label="total energy", color="tab:red",
              linewidth=2.0)
    if (params.get("ic") in ("E3", "E3-positive")
            and "background_energy" in data.columns):
        background = float(data["background_energy"].iloc[0])
    else:
        background = legacy_background_energy(params)
    if background is not None:
        axes.axhline(background, label=f"background energy ({background:g})",
                     color="black", linestyle="--", linewidth=2.0, zorder=3)
    axes.plot(data["t"], data["helicity"], label=helicity_label, color="tab:blue")
    axes.plot(data["t"], data["helicity"].abs(), label=helicity_abs_label,
              linestyle="--", color="tab:green")
    axes.set_xlabel("Time")
    axes.set_ylabel("Values")
    axes.legend(loc="upper right")
    figure.tight_layout()

    output_path = os.path.join(output_dir, "plot.png")
    figure.savefig(output_path, dpi=750, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")


def main():
    paths = data_paths_from_args()
    for path in paths:
        plot_data(path)
    print(f"Generated {len(paths)} plot(s).")


if __name__ == "__main__":
    main()
