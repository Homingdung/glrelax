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

if (data_new["helicity"] < 0).any():
    n_neg = int((data_new["helicity"] < 0).sum())
    h_min = float(data_new["helicity"].min())
    print(f"\033[93m[warning]\033[0m helicity is negative at {n_neg}/{len(data_new)} time steps "
          f"(min = {h_min:.6e}); plotting |H| as well.")

plt.figure(figsize=(10, 6))

plt.plot(data_new["t"], data_new["energy"], label=r"energy", color="tab:red")
plt.plot(data_new["t"], data_new["helicity"], label="helicity", color="tab:blue")
plt.plot(data_new["t"], data_new["helicity"].abs(), label=r"$|H|$", linestyle="--", color="tab:green")
# plt.plot(data_new["t"], data_new["divB"], label=r"$\nabla\cdot \mathbf{B}$", linestyle=":", color="tab:purple")

plt.xlabel("Time")
plt.ylabel("Values")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot.png"), dpi=750, bbox_inches="tight")
plt.show()
