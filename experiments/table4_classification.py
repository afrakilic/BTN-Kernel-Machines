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
    "spambase": dict(n_features=57, M=30, R=10, c0=1e-5),
    "adult":    dict(n_features=96, M=40, R=10, c0=1e-6),
}

name = sys.argv[1] if len(sys.argv) > 1 else "spambase"
cfg = DATASETS[name]
data = load_csv(name)
X = data[:, : cfg["n_features"]]
y = np.where(data[:, cfg["n_features"]] > 0, 1, -1)

errors, nlls, ranks = [], [], []
start = time.time()
for run in range(10):
    X_train, X_test, y_train, y_test = split(X, y, seed=run)
    model = BTNKM().train(
        X_train, y_train, M=cfg["M"], R=cfg["R"],
        a0=1e-2, b0=1e-3, c0=cfg["c0"], d0=1e-6, g0=1e-6, h0=1e-6, max_iter=50,
    )
    mean, std = model.predict(X_test)
    errors.append(1 - accuracy_score(y_test, np.sign(mean)))
    nlls.append(classification_nll(y_test, mean, std))
    ranks.append(model.rank)
    print(f"  run {run}: error={errors[-1]:.4f}  NLL={nlls[-1]:.4f}  R={model.rank}")

print(f"\n{name}: total runtime for 10 runs: {time.time() - start:.2f} s")
print(f"Misclassification: {np.mean(errors):.4f} ± {np.std(errors):.4f}")
print(f"NLL:               {np.mean(nlls):.4f} ± {np.std(nlls):.4f}")
print(f"Effective R:       {np.mean(ranks):.1f} ± {np.std(ranks):.1f}")
