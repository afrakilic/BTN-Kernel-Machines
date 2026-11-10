"""Table 4 (classification): BTN-Kernel machines on Spambase and Adult.

Labels are mapped to {-1, +1}, the model is fitted as a regression and the
predicted class is the sign of the predictive mean. Reports the
misclassification rate, NLL and effective rank over 10 random 90/10 splits.

Usage:  python experiments/table4_classification.py spambase
        python experiments/table4_classification.py adult     (slow: ~18 h)
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
    # "spambase": dict(n_features=57, M=30, R=10, c0=1e-5),
    "adult":    dict(n_features=96, M=40, R=10, c0=1e-6),
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
            a0=1e-2, b0=1e-3, c0=cfg["c0"], d0=1e-6, g0=1e-6, h0=1e-6, max_iter=50,
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
    

# spambase: total runtime for 10 runs: 833.61 s
# Misclassification: 0.0599 ± 0.0068
# NLL:               0.1922 ± 0.0335
# Effective R:       4.2 ± 1.1
# Time per run:      83.36 ± 8.36 s

# adult: total runtime for 10 runs: 20312.93 s
# Misclassification: 0.1434 ± 0.0044
# NLL:               0.3296 ± 0.0083
# Effective R:       6.3 ± 0.5
# Time per run:      2031.29 ± 142.21 s