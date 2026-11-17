"""Tables 1 and 2: effect of the initial rank R_max (10, 25, 50).

Concrete, Energy and Airfoil, M = 20, 10 random 90/10 splits. Prints RMSE,
NLL (Table 1) and the effective rank R_eff (Table 2), as mean ± std.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from btnkm import BTNKM
from btnkm.data import load_csv, split, student_t_nll

# number of input columns; the next column is the target
# (Energy: 8 building inputs, heating load as target; cooling load is not used)
N_INPUTS = {"concrete": 8, "energy": 8, "airfoil": 5}
DATASETS = ["concrete", "energy", "airfoil"]
M = 20
R_MAX = [10, 25, 50]

# results[name][R_max] = {"rmse": [...], "nll": [...], "r_eff": [...]}, one entry per split
results = {name: {r: {"rmse": [], "nll": [], "r_eff": []} for r in R_MAX} for name in DATASETS}

for name in DATASETS:
    data = load_csv(name)
    n = N_INPUTS[name]
    X, y = data[:, :n], data[:, n]
    for trial in range(10):
        X_train, X_test, y_train, y_test = split(X, y, seed=trial)
        y_mean, y_std = y_train.mean(), y_train.std()
        y_train = (y_train - y_mean) / y_std

        for r in R_MAX:
            model = BTNKM().train(
                X_train, y_train, M=M, R=r,
                a0=1e-3, b0=1e-3, c0=1e-5, d0=1e-6, g0=1e-6, h0=1e-6, max_iter=50,
            )
            mean, std = model.predict(X_test)
            mean, std = mean * y_std + y_mean, std * y_std
            results[name][r]["rmse"].append(np.sqrt(np.mean((mean - y_test) ** 2)))
            results[name][r]["nll"].append(student_t_nll(y_test, mean, std, 2.0 * model.a))
            results[name][r]["r_eff"].append(model.rank_history[-1])

for name in DATASETS:
    print(f"\n{name.capitalize()}")
    for r in R_MAX:
        v = results[name][r]
        print(
            f"  R_max={r:2d}:  RMSE = {np.mean(v['rmse']):.3f} ± {np.std(v['rmse']):.3f}   "
            f"NLL = {np.mean(v['nll']):.3f} ± {np.std(v['nll']):.3f}   "
            f"R_eff = {np.mean(v['r_eff']):.1f} ± {np.std(v['r_eff']):.1f}"
        )



# Concrete
#   R_max=10:  RMSE = 5.537 ± 0.576   NLL = 3.170 ± 0.175   R_eff = 4.3 ± 0.6
#   R_max=25:  RMSE = 5.153 ± 0.602   NLL = 3.094 ± 0.171   R_eff = 5.3 ± 0.5
#   R_max=50:  RMSE = 5.012 ± 0.637   NLL = 3.052 ± 0.163   R_eff = 5.2 ± 0.7

# Energy
#   R_max=10:  RMSE = 0.462 ± 0.066   NLL = 0.648 ± 0.134   R_eff = 4.3 ± 0.8
#   R_max=25:  RMSE = 0.467 ± 0.060   NLL = 0.667 ± 0.122   R_eff = 3.8 ± 0.7
#   R_max=50:  RMSE = 0.463 ± 0.058   NLL = 0.655 ± 0.123   R_eff = 3.3 ± 0.5

# Airfoil
#   R_max=10:  RMSE = 2.092 ± 0.214   NLL = 2.163 ± 0.097   R_eff = 4.1 ± 0.3
#   R_max=25:  RMSE = 1.723 ± 0.180   NLL = 1.940 ± 0.101   R_eff = 9.4 ± 0.8
#   R_max=50:  RMSE = 1.689 ± 0.199   NLL = 1.923 ± 0.120   R_eff = 8.4 ± 0.9