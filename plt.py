import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def data_path_from_args():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        return path if path.endswith(".csv") else os.path.join(path, "data.csv")

    candidates = glob.glob("hdiv-*/data.csv") + glob.glob("mixed-*/data.csv") + glob.glob("lm-*/data.csv")
    if not candidates:
        raise FileNotFoundError("No data.csv found. Run a scheme first, or pass a data directory.")
    return max(candidates, key=os.path.getmtime)


data_path = data_path_from_args()
output_dir = os.path.dirname(data_path) or "."
data_new = pd.read_csv(data_path)


def legacy_energy_series(data, directory):
    """Recover E3 free energy from CSV files written before that column existed."""
    param_path = os.path.join(directory, "param.txt")
    params = {}
    if os.path.exists(param_path):
        with open(param_path) as f:
            for line in f:
                fields = line.split()
                if len(fields) >= 2 and not line.lstrip().startswith("#"):
                    params[fields[0]] = fields[1]

    if params.get("ic") in ("E3", "E3-positive") and "Lx,Ly,Lz" in params:
        volume = 1.0
        for length in params["Lx,Ly,Lz"].split(","):
            volume *= float(length)
        print("\033[93m[warning]\033[0m free_energy is absent; "
              f"recovering legacy E3 free energy as energy - {volume:g}.")
        return data["energy"] - volume, "free energy"

    print("\033[93m[warning]\033[0m free_energy is absent; plotting legacy energy data.")
    return data["energy"], "energy"

if (data_new["helicity"] < 0).any():
    n_neg = int((data_new["helicity"] < 0).sum())
    h_min = float(data_new["helicity"].min())
    print(f"\033[93m[warning]\033[0m helicity is negative at {n_neg}/{len(data_new)} time steps "
          f"(min = {h_min:.6e}); plotting |H| as well.")

plt.figure(figsize=(10, 6))

if "free_energy" in data_new.columns:
    energy_values, energy_label = data_new["free_energy"], "free energy"
else:
    energy_values, energy_label = legacy_energy_series(data_new, output_dir)
plt.plot(data_new["t"], energy_values, label=energy_label, color="tab:red")
plt.plot(data_new["t"], data_new["helicity"], label="helicity", color="tab:blue")
plt.plot(data_new["t"], data_new["helicity"].abs(), label=r"$|H|$", linestyle="--", color="tab:green")
# plt.plot(data_new["t"], data_new["divB"], label=r"$\nabla\cdot \mathbf{B}$", linestyle=":", color="tab:purple")

plt.xlabel("Time")
plt.ylabel("Values")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot.png"), dpi=750, bbox_inches="tight")
plt.show()
