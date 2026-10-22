"""Figure 3: effective rank R_eff and variational lower bound during training.

Concrete, Energy and Airfoil (one split, seed 1), M = 20, R_max = 10, 25, 50,
150 iterations. Saves results/figure3.pdf.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from btnkm import BTNKM
from btnkm.data import load_csv, split

DATASETS = ["concrete", "energy", "airfoil"]
M = 20
R_MAX = [10, 25, 50]
STYLES = [("-", "o", "#A40000"), ("--", "s", "#00008B"), ("-.", "d", "#006B3C")]

fig, axs = plt.subplots(2, 3, figsize=(15.5, 7))
for col, name in enumerate(DATASETS):
    data = load_csv(name)
    X_train, _, y_train, _ = split(data[:, :-1], data[:, -1], seed=1)
    y_train = (y_train - y_train.mean()) / y_train.std()

    max_len = 0
    for r, (ls, marker, color) in zip(R_MAX, STYLES):
        model = BTNKM().train(
            X_train, y_train, M=M, R=r,
            a0=1e-3, b0=1e-3, c0=1e-5, d0=1e-6, g0=1e-6, h0=1e-6,
            max_iter=150, lower_bound_tol=1e-10,
        )
        style = dict(linestyle=ls, marker=marker, color=color, markersize=3)
        axs[0, col].plot(model.rank_history, label=rf"$R_{{\max}} = {r}$", **style)
        axs[1, col].plot(np.arange(1, len(model.lower_bound) + 1), model.lower_bound, alpha=0.8, **style)
        max_len = max(max_len, len(model.rank_history))

    axs[0, col].set_title(name.capitalize(), fontsize=24)
    axs[1, col].set_xlabel("Iteration", fontsize=22)
    for ax in axs[:, col]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_xticks(np.arange(0, max_len + 1, 50))

axs[0, 0].set_ylabel(r"$R_{\text{eff}}$", fontsize=22)
axs[1, 0].set_ylabel("LB", fontsize=22)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.93, 0.5), fontsize=22, frameon=False)
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.subplots_adjust(wspace=0.5, hspace=0.3)
(ROOT / "results").mkdir(exist_ok=True)
# plt.savefig(ROOT / "results/figure3.pdf", bbox_inches="tight")
plt.show()
