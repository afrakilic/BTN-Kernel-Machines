"""Table 4 (classification): BTN-Kernel machines on Spambase and Adult.

Labels are mapped to {-1, +1}, the model is fitted as a regression and the
predicted class is the sign of the predictive mean. Reports the
misclassification rate, NLL and effective rank over 10 random 90/10 splits.

Usage:  python experiments/table4_classification.py spambase     (~8 min)
        python experiments/table4_classification.py adult        (slow: ~5.5 h)
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from btnkm import BTNKM
from btnkm.data import load_csv, split, classification_nll

# n_features: number of input columns (the next column is the label)
DATASETS = {
    "spambase": dict(n_features=57, M=30, R=25, c0=1e-5),
    "adult":    dict(n_features=96, M=40, R=10, c0=1e-6)
}

names = sys.argv[1:] or list(DATASETS)
for name in names:
    cfg = DATASETS[name]
    data = load_csv(name)
    X = data[:, : cfg["n_features"]]
    y = np.where(data[:, cfg["n_features"]] > 0, 1, -1)

    errors, nlls, ranks = [], [], []
    run_times = []  # time of each run (s)
    start = time.time()
    for run in range(10):
        t0 = time.time()
        X_train, X_test, y_train, y_test = split(X, y, seed=run)
        model = BTNKM().train(
            X_train, y_train, M=cfg["M"], R=cfg["R"],
            a0=1e-2, b0=1e-3, c0=cfg["c0"], d0=1e-6, g0=1e-6, h0=1e-6, max_iter=50
        )
        mean, std = model.predict(X_test)
        errors.append(1 - accuracy_score(y_test, np.sign(mean)))
        nlls.append(classification_nll(y_test, mean, std))
        ranks.append(model.rank)
        run_times.append(time.time() - t0)
        print(f"  run {run}: error={errors[-1]:.4f}  NLL={nlls[-1]:.4f}  R={model.rank}"
              f"  time={run_times[-1]:.2f} s")

    print(f"\n{name}: total runtime for 10 runs: {time.time() - start:.2f} s")
    print(f"Misclassification: {np.mean(errors):.4f} ± {np.std(errors):.4f}")
    print(f"NLL:               {np.mean(nlls):.4f} ± {np.std(nlls):.4f}")
    print(f"Effective R:       {np.mean(ranks):.1f} ± {np.std(ranks):.1f}")
    print(f"Time per run:      {np.mean(run_times):.2f} ± {np.std(run_times):.2f} s") 
    
# ANNEALING SWEEPS = 20

# spambase: total runtime for 10 runs: 1182.54 s
# Misclassification: 0.0544 ± 0.0084
# NLL:               0.1637 ± 0.0231
# Effective R:       3.3 ± 0.6
# Time per run:      118.25 ± 22.40 s

# adult: total runtime for 10 runs: 19549.88 s
# Misclassification: 0.1449 ± 0.0052
# NLL:               0.3324 ± 0.0067
# Effective R:       4.1 ± 0.8
# Time per run:      1954.99 ± 386.01 s

# NO ANNEALING 

# spambase: total runtime for 10 runs: 428.13 s
# Misclassification: 0.7759 ± 0.3425
# NLL:               0.6154 ± 0.1199
# Effective R:       3.5 ± 2.1
# Time per run:      42.81 ± 10.94 s

# adult: total runtime for 10 runs: 10835.89 s
# Misclassification: 0.1474 ± 0.0073
# NLL:               0.3349 ± 0.0076
# Effective R:       3.4 ± 0.5
# Time per run:      1083.59 ± 231.68 s

# NO ANNEALING, WITH THE BTT-KM SETTINGS (control experiment for the BTT-KM paper, Sec. 4.2)
# Purpose: check whether BTN-KM also trains without annealing when it uses the same
# settings as BTT-KM. This isolates the effect of the TN format (CPD vs TT).
# Settings: anneal_sweeps=0 and a0=1e-1 (as for BTT-KM on Spambase). The other settings
# are unchanged and already match BTT-KM: b0=1e-3, c0=1e-5, d0=g0=h0=1e-6, M=30,
# random N(0, I) initialization, and a warm-up of 5 iterations in which tau, lambda
# and delta are kept fixed and no ranks are truncated. R=25 matches the size of a
# BTT-KM core with R_max=5.
#
# Result: BTN-KM fails in 7 of the 10 splits.
#   - 6 runs converge to rank 1 with an error of 0.23-0.28.
#   - 1 run (run 7) truncates all rank components ("Rank becomes 0").
#     Its predictions are NaN, so it reports error=1.0 and NLL=ln(2).
#   - Only 3 runs reach a good solution (rank 2, error 0.04-0.06).
# With the same settings, BTT-KM trains stably on all splits (error 0.094 ± 0.036).
# With annealing, BTN-KM reaches 0.054 ± 0.008 (see above).
#
# spambase: total runtime for 10 runs: 1787.51 s
# Misclassification: 0.2464 ± 0.2687
# NLL:               0.3495 ± 0.1788
# Effective R:       1.5 ± 0.5
# Time per run:      178.75 ± 20.10 s