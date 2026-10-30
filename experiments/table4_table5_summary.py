"""Print Tables 4 and 5 (regression part) from saved results.

Reads results/regression_runs.csv (made by table4_table5_regression.py) and the
GP predictions in T-KRR-SIAM/<dataset>_all_runs_gp.csv. All values are
mean ± std over the 10 splits (population std).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
runs = pd.read_csv(ROOT / "results/regression_runs.csv")
runtimes = pd.read_csv(ROOT / "results/regression_runtimes.csv").set_index("dataset")["runtime_seconds"]


def ms(values, dec=3):
    return f"{np.mean(values):.{dec}f} ± {np.std(values):.{dec}f}"


def gp_metrics(name):
    """Per-run RMSE, Gaussian NLL and 95% coverage of the GP baseline (None if no file)."""
    path = ROOT / "T-KRR-SIAM" / f"{name}_all_runs_gp.csv"
    if not path.exists():
        return None
    out = {"rmse": [], "nll": [], "cov": []}
    for _, g in pd.read_csv(path).groupby("run"):
        y, m, s = g["y_test"].values, g["prediction_mean"].values, g["prediction_std"].values
        out["rmse"].append(np.sqrt(np.mean((y - m) ** 2)))
        out["nll"].append(np.mean(0.5 * np.log(2 * np.pi * s**2) + 0.5 * (y - m) ** 2 / s**2))
        out["cov"].append(np.mean((y >= m - 1.959963985 * s) & (y <= m + 1.959963985 * s)) * 100)
    return out


general, uq = [], []
for name, g in runs.groupby("dataset"):
    gp = gp_metrics(name)
    general.append({
        "Dataset": name.capitalize(),
        "RMSE": ms(g.rmse),
        "GP RMSE": ms(gp["rmse"]) if gp else "N/A",
        "Effective R": ms(g.R_effective, 1),
        "Runtime (s)": f"{runtimes[name]:.2f}",
        "Time/run (s)": ms(g.run_time, 2) if "run_time" in g else "N/A",
    })
    uq.append({
        "Dataset": name.capitalize(),
        "Analytic NLL": ms(g.analytic_nll),
        "MC NLL": ms(g.mc_nll),
        "GP NLL": ms(gp["nll"]) if gp else "N/A",
        "Analytic Cov. (%)": ms(g.analytic_coverage, 1),
        "MC Cov. (%)": ms(g.mc_coverage, 1),
        "GP Cov. (%)": ms(gp["cov"], 1) if gp else "N/A",
    })

pd.set_option("display.width", 200)
print("\nTABLE 4 (regression): accuracy, effective rank and run time")
print(pd.DataFrame(general).to_string(index=False))
print("\nTABLE 5: NLL and 95% coverage (analytic vs. Monte Carlo vs. GP)")
print(pd.DataFrame(uq).to_string(index=False))
print(f"\nTotal runtime: {runtimes.sum():.2f} s")


# TABLE 4 (regression): accuracy, effective rank and run time
#  Dataset          RMSE       GP RMSE Effective R Runtime (s) Time/run (s)
#  Airfoil 1.738 ± 0.149 2.293 ± 0.189   9.5 ± 0.5       41.94  4.19 ± 0.97
# Concrete 5.391 ± 1.275 5.565 ± 0.494   5.3 ± 0.5       47.36  4.74 ± 1.55
#   Energy 0.496 ± 0.143 1.296 ± 0.275  10.1 ± 1.3       69.67  6.97 ± 2.04
#    Yacht 0.368 ± 0.126 0.402 ± 0.124   5.2 ± 0.6       16.11  1.61 ± 0.19

# TABLE 5: NLL and 95% coverage (analytic vs. Monte Carlo vs. GP)
#  Dataset  Analytic NLL        MC NLL        GP NLL Analytic Cov. (%) MC Cov. (%) GP Cov. (%)
#  Airfoil 1.976 ± 0.096 1.977 ± 0.119 2.281 ± 0.080        94.7 ± 1.8  91.9 ± 1.0  95.2 ± 1.0
# Concrete 3.344 ± 0.409 3.336 ± 0.423 3.078 ± 0.076        88.1 ± 3.4  87.2 ± 4.5  93.3 ± 2.1
#   Energy 1.528 ± 0.279 0.856 ± 0.654 1.679 ± 0.214        99.4 ± 0.9  87.1 ± 4.1  93.0 ± 3.3
#    Yacht 1.010 ± 0.415 0.821 ± 1.164 0.105 ± 0.319       100.0 ± 0.0  87.7 ± 6.4  92.6 ± 6.6
