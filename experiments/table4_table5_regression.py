"""Tables 4 and 5 (regression): BTN-Kernel machines on Airfoil, Concrete, Energy and Yacht.

For each data set and each of 10 random 90/10 splits:
  - train the model and compute RMSE,
  - analytic NLL and 95% coverage from the Student's-t predictive,
  - the same NLL and coverage from 3000 Monte Carlo samples of the posterior.

Per-run results are saved to results/regression_runs.csv and run times to
results/regression_runtimes.csv. Print the tables with table4_table5_summary.py.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from btnkm import BTNKM
from btnkm.data import load_csv, split, student_t_nll, student_t_coverage

# n_features: number of input columns; the next column is the target.
# a0: shape of the Gamma prior on the noise precision (b0 = 1e-3 for all).
DATASETS = {
    "airfoil":  dict(n_features=5, a0=1e-3),
    "concrete": dict(n_features=8, a0=1e-3),
    "energy":   dict(n_features=9, a0=1e-2),
    "yacht":    dict(n_features=6, a0=1e-2),
}
M, R_MAX = 20, 25
N_RUNS, MC_SAMPLES = 10, 3000

rows, runtimes = [], []
for name, cfg in DATASETS.items():
    print(f"\n=== {name} ===")
    data = load_csv(name)
    X, y = data[:, : cfg["n_features"]], data[:, cfg["n_features"]]

    start = time.time()
    for run in range(N_RUNS):
        X_train, X_test, y_train, y_test = split(X, y, seed=run)
        y_mean, y_std = y_train.mean(), y_train.std()
        y_train = (y_train - y_mean) / y_std

        model = BTNKM().train(
            X_train, y_train, M=M, R=R_MAX,
            a0=cfg["a0"], b0=1e-3, c0=1e-5, d0=1e-6, g0=1e-6, h0=1e-6, max_iter=50,
        )

        # Analytic Student's-t predictive, back on the original scale
        nu = 2.0 * model.a
        mean, std = model.predict(X_test)
        mean, std = mean * y_std + y_mean, std * y_std

        # Monte Carlo check (on the standardised scale, then converted back)
        mc = model.mc_predict(X_test, S=MC_SAMPLES, y_true=(y_test - y_mean) / y_std, seed=run)
        lower, upper = mc["lower_95"] * y_std + y_mean, mc["upper_95"] * y_std + y_mean

        row = dict(
            dataset=name,
            run=run,
            rmse=np.sqrt(np.mean((mean - y_test) ** 2)),
            analytic_nll=student_t_nll(y_test, mean, std, nu),
            mc_nll=mc["nll"] + np.log(y_std),  # change of variables back to the original scale
            analytic_coverage=student_t_coverage(y_test, mean, std, nu),
            mc_coverage=np.mean((y_test >= lower) & (y_test <= upper)) * 100,
            std_ratio=np.mean(std / (mc["std"] * y_std)),
            R_effective=model.rank,
        )
        rows.append(row)
        print(
            f"  run {run}: RMSE={row['rmse']:.4f}  NLL={row['analytic_nll']:.4f} (MC {row['mc_nll']:.4f})  "
            f"cov={row['analytic_coverage']:.1f}% (MC {row['mc_coverage']:.1f}%)  R={row['R_effective']}"
        )
    runtimes.append(dict(dataset=name, runtime_seconds=time.time() - start))

(ROOT / "results").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(ROOT / "results/regression_runs.csv", index=False)
pd.DataFrame(runtimes).to_csv(ROOT / "results/regression_runtimes.csv", index=False)
print("\nSaved results/regression_runs.csv and results/regression_runtimes.csv")
