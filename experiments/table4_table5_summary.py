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



# ANNEALING SWEEPS = 20 

# TABLE 4 (regression): accuracy, effective rank and run time
#  Dataset          RMSE       GP RMSE Effective R Runtime (s) Time/run (s)
#  Airfoil 1.723 ± 0.180 2.293 ± 0.189   9.4 ± 0.8       75.05  7.51 ± 1.38
# Concrete 4.969 ± 0.487 5.565 ± 0.494   8.3 ± 0.5      125.74 12.57 ± 2.83
#   Energy 0.412 ± 0.060 0.448 ± 0.037   6.5 ± 0.7       62.95  6.30 ± 0.73
#    Yacht 0.401 ± 0.154 0.402 ± 0.124   4.6 ± 0.7       34.21  3.42 ± 0.64

# TABLE 5: NLL and 95% coverage (analytic vs. Monte Carlo vs. GP)
#  Dataset  Analytic NLL        MC NLL        GP NLL Analytic Cov. (%) MC Cov. (%) GP Cov. (%)
#  Airfoil 1.940 ± 0.101 1.942 ± 0.103 2.281 ± 0.080        95.0 ± 1.2  95.0 ± 1.1  95.2 ± 1.0
# Concrete 3.030 ± 0.140 3.020 ± 0.126 3.078 ± 0.076        91.3 ± 2.9  91.4 ± 2.9  93.3 ± 2.1
#   Energy 0.537 ± 0.137 0.542 ± 0.137 0.608 ± 0.091        96.4 ± 2.0  96.4 ± 2.2  93.9 ± 2.0
#    Yacht 0.347 ± 0.350 0.378 ± 0.388 0.105 ± 0.319        94.8 ± 4.1  94.8 ± 4.1  92.6 ± 6.6



# NO ANNEALING

# TABLE 4 (regression): accuracy, effective rank and run time
#  Dataset          RMSE       GP RMSE Effective R Runtime (s) Time/run (s)
#  Airfoil 1.700 ± 0.171 2.293 ± 0.189   7.8 ± 0.4       55.67  5.57 ± 0.65
# Concrete 5.121 ± 0.524 5.565 ± 0.494   6.5 ± 0.5       80.73  8.07 ± 2.44
#   Energy 0.433 ± 0.044 0.448 ± 0.037   5.0 ± 0.4       54.60  5.46 ± 0.73
#    Yacht 0.584 ± 0.140 0.402 ± 0.124   2.9 ± 0.3       24.57  2.46 ± 0.19

# TABLE 5: NLL and 95% coverage (analytic vs. Monte Carlo vs. GP)
#  Dataset  Analytic NLL        MC NLL        GP NLL Analytic Cov. (%) MC Cov. (%) GP Cov. (%)
#  Airfoil 1.940 ± 0.102 1.941 ± 0.101 2.281 ± 0.080        94.2 ± 1.2  94.1 ± 1.1  95.2 ± 1.0
# Concrete 3.062 ± 0.129 3.071 ± 0.135 3.078 ± 0.076        92.6 ± 2.6  92.8 ± 2.5  93.3 ± 2.1
#   Energy 0.600 ± 0.093 0.605 ± 0.097 0.608 ± 0.091        95.2 ± 1.9  95.5 ± 2.1  93.9 ± 2.0
#    Yacht 0.812 ± 0.356 0.842 ± 0.422 0.105 ± 0.319        93.2 ± 2.7  92.3 ± 3.3  92.6 ± 6.6