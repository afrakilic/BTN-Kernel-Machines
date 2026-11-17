"""Figure 4 and Table 3: effect of the feature precisions delta_d.

Concrete, Energy and Airfoil, R_max = 25, M = 20 and 50, 10 random 90/10
splits. Each model is trained with delta_d on and off. Prints RMSE, NLL and
R_eff (Table 3) and plots the average effective feature dimension per input
(Figure 4, saved to results/figure4.pdf).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from btnkm import BTNKM
from btnkm.data import load_csv, split, student_t_nll

# number of input columns; the next column is the target
# (Energy: 8 building inputs, heating load as target; cooling load is not used)
N_INPUTS = {"concrete": 8, "energy": 8, "airfoil": 5}
DATASETS = ["concrete", "energy", "airfoil"]
M_MAX = [20, 50]
R_MAX = 25
COLORS = ["#A40000", "#00008B"]

# results[name][mode][M] = {"metrics": [(rmse, nll, R_eff), ...], "M_eff": [...]}
results = {n: {mode: {m: {"metrics": [], "M_eff": []} for m in M_MAX} for mode in ["on", "off"]} for n in DATASETS}

for name in DATASETS:
    data = load_csv(name)
    n = N_INPUTS[name]
    X, y = data[:, :n], data[:, n]
    for trial in range(10):
        X_train, X_test, y_train, y_test = split(X, y, seed=trial)
        y_mean, y_std = y_train.mean(), y_train.std()
        y_train = (y_train - y_mean) / y_std

        for M in M_MAX:
            for mode in ["off", "on"]:
                model = BTNKM().train(
                    X_train, y_train, M=M, R=R_MAX,
                    a0=1e-3, b0=1e-3, c0=1e-5, d0=1e-6, g0=1e-6, h0=1e-6, max_iter=50,
                    delta_update=(mode == "on"),
                )
                mean, std = model.predict(X_test)
                mean, std = mean * y_std + y_mean, std * y_std
                rmse = np.sqrt(np.mean((mean - y_test) ** 2))
                nll = student_t_nll(y_test, mean, std, 2.0 * model.a)
                results[name][mode][M]["metrics"].append((rmse, nll, model.rank))
                results[name][mode][M]["M_eff"].append(model.M_eff)

# Table 3
print("\nAveraged results over 10 trials:")
for name in DATASETS:
    print(f"\n{name.capitalize()}:")
    for mode in ["on", "off"]:
        for M in M_MAX:
            rmse, nll, r = np.array(results[name][mode][M]["metrics"]).T
            print(
                f"  M_max = {M:3d}, delta {mode:3s} -> RMSE = {np.mean(rmse):.4f} ± {np.std(rmse):.4f}, "
                f"NLL = {np.mean(nll):.4f} ± {np.std(nll):.4f}, R = {np.mean(r):.4f} ± {np.std(r):.4f}"
            )

# Figure 4: average effective feature dimension per input, delta on (solid) and off (faded)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
width = 0.35
for col, name in enumerate(DATASETS):
    ax = axs[col]
    n_inputs = len(results[name]["off"][M_MAX[0]]["M_eff"][0])
    x = np.arange(n_inputs)
    for i, (M, color) in enumerate(zip(M_MAX, COLORS)):
        for mode, alpha in [("off", 0.3), ("on", 1)]:
            avg = np.mean(np.vstack(results[name][mode][M]["M_eff"]), axis=0)
            ax.bar(x + i * width, avg, width=width, color=color, alpha=alpha,
                   label=rf"$\delta_d$: {mode}, $M_{{\max}} = {M}$")
    ax.axhline(y=50, color="#00008B", linestyle="--", linewidth=2)
    ax.axhline(y=20, color="#A40000", linestyle="--", linewidth=2)
    ax.set_title(name.capitalize(), fontsize=24)
    ax.set_xlabel("Features", fontsize=22, labelpad=15)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([str(i) for i in range(1, n_inputs + 1)], fontsize=22)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines[["top", "right"]].set_visible(False)
    if col != 0:
        ax.spines["left"].set_visible(False)

axs[0].set_ylabel(r"$\bar{M}_{\text{eff}}$", fontsize=22)
handles, labels = axs[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(0.9, 0.5), fontsize=22, frameon=False)
plt.tight_layout(rect=[0, 0, 0.88, 1])
(ROOT / "results").mkdir(exist_ok=True)
plt.savefig(ROOT / "results/figure4.pdf", bbox_inches="tight")
plt.show()


# Averaged results over 10 trials:
# Concrete:
#   M_max =  20, delta on  -> RMSE = 5.1525 ± 0.6017, NLL = 3.0940 ± 0.1714, R = 5.3000 ± 0.4583
#   M_max =  50, delta on  -> RMSE = 5.3468 ± 0.6633, NLL = 3.0939 ± 0.1690, R = 5.8000 ± 0.6000
#   M_max =  20, delta off -> RMSE = 5.0276 ± 0.5265, NLL = 3.0355 ± 0.1268, R = 5.2000 ± 0.4000
#   M_max =  50, delta off -> RMSE = 5.0164 ± 0.6248, NLL = 3.0340 ± 0.1543, R = 6.0000 ± 0.6325

# Energy:
#   M_max =  20, delta on  -> RMSE = 0.4673 ± 0.0604, NLL = 0.6672 ± 0.1216, R = 3.8000 ± 0.7483
#   M_max =  50, delta on  -> RMSE = 0.4564 ± 0.0582, NLL = 0.6448 ± 0.1289, R = 3.1000 ± 0.3000
#   M_max =  20, delta off -> RMSE = 0.4579 ± 0.0629, NLL = 0.6517 ± 0.1198, R = 4.0000 ± 0.6325
#   M_max =  50, delta off -> RMSE = 0.4459 ± 0.0604, NLL = 0.6249 ± 0.1317, R = 3.1000 ± 0.3000

# Airfoil:
#   M_max =  20, delta on  -> RMSE = 1.7233 ± 0.1797, NLL = 1.9396 ± 0.1013, R = 9.4000 ± 0.8000
#   M_max =  50, delta on  -> RMSE = 1.7135 ± 0.1579, NLL = 1.9423 ± 0.0916, R = 7.4000 ± 1.1136
#   M_max =  20, delta off -> RMSE = 1.7117 ± 0.1524, NLL = 1.9345 ± 0.0926, R = 9.6000 ± 0.6633
#   M_max =  50, delta off -> RMSE = 1.7067 ± 0.1618, NLL = 1.9420 ± 0.0923, R = 7.7000 ± 0.9000
