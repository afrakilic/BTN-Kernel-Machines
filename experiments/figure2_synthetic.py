"""Figure 2: recovering a known low-rank structure from synthetic data.

The data come from a rank-3 CP model with D = 3 inputs, in which each factor
uses a random subset of at least 3 of its 5 feature rows (so that the rank-3 CP
model is unique), plus a little noise. The data are
generated with the model's own feature map (btnkm.utils.features), so that the true
row and rank structure exists in the basis the model uses. BTNKM.train is run for
100 iterations from R_max = 5 and M = 5, without rank pruning so that all 5 columns
are shown. The figure shows |W^(d)| for each factor, with the learned rank
precisions lambda (columns) and feature precisions delta_d (rows).
Saves results/figure2.pdf.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from btnkm import BTNKM
from btnkm.utils import features

np.random.seed(1)

#  True model 
D, R_TRUE, R, I, N = 3, 3, 5, 5, 100
e = 1e-3 * np.random.randn(N)
rows = [np.sort(np.random.choice(I, np.random.randint(3, I), replace=False)) for _ in range(D)]  # >= 3 rows
W_true = [np.random.randn(len(r), R_TRUE) for r in rows]
X = np.random.randn(N, D)
Phi = features(X, I)
y = np.sum(
    (Phi[0][:, rows[0]] @ W_true[0]) * (Phi[1][:, rows[1]] @ W_true[1]) * (Phi[2][:, rows[2]] @ W_true[2]),
    axis=1,
) + e

# Training
model = BTNKM().train(
    X, y, M=I, R=R,
    a0=1e-3, b0=1e-3, c0=1e-6, d0=1e-6, g0=1e-6, h0=1e-6,
    max_iter=100, lower_bound_tol=0,  # run all 100 iterations
    prune_rank=False,                 # keep all R columns in the figure
    verbose=False
)
W_D, delta, lambda_R, tau = model.W_D, model.delta, model.lambda_R, model.tau


print("true rows per factor:", [r.tolist() for r in rows], "\n")
with np.printoptions(formatter={"float_kind": lambda x: f"{x:.2e}"}):
    for d in range(D):
        print(f"W_D[{d + 1}]:\n{W_D[d]}\n")
    for d in range(D):
        print(f"delta[{d + 1}]:\n{delta[d]}\n")
    print(f"lambda_R:\n{lambda_R}\n\ntau:\n{tau}\n")
# share of f carried by each rank-1 component (components above 1% count as active)
comp = np.prod([Phi[d] @ W_D[d] for d in range(D)], axis=0)
share = np.sum(comp**2, axis=0) / np.sum(comp**2)
print(f"share of f per component: {np.round(share, 4)}  -> rank {int(np.sum(share > 0.01))}")
for d in range(D):
    print(f"Singular values of W_D[{d}]: {np.linalg.svd(W_D[d], compute_uv=False)}")

#Figure 2
def fmt(val):
    if abs(val) > 1e2:
        base, exp = f"{val:.2e}".split("e")
        return f"{round(float(base))} \\times 10^{{{int(exp)}}}"
    return f"{val:.2f}"


W_abs = [np.abs(W) for W in W_D]
vmin = max(min(W.min() for W in W_abs), 1e-5)
vmax = max(W.max() for W in W_abs)

fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), constrained_layout=True)
for d, ax in enumerate(axes):
    heatmap = sns.heatmap(
        W_abs[d], cmap="gray_r", norm=LogNorm(vmin=vmin, vmax=vmax), ax=ax, cbar=False, square=True,
        xticklabels=[f"${fmt(v)}$" for v in lambda_R], yticklabels=[f"${fmt(v)}$" for v in delta[d]],
        linewidths=0.01, linecolor="lightgray",
    )
    ax.set_title(f"$\\tilde{{W}}^{{({d + 1})}}$", fontsize=18)
    ax.tick_params(axis="x", rotation=45, labelsize=12.1)
    ax.tick_params(axis="y", rotation=0, labelsize=12.1)

cbar = fig.colorbar(heatmap.get_children()[0], ax=axes.ravel().tolist(), orientation="vertical", fraction=0.02, pad=0.05)
cbar.ax.tick_params(labelsize=10)
t_left = fig.text(-0.00, 0.5, r"$\tilde{\boldsymbol{\delta}}_d$", va="center", ha="center", rotation="vertical", fontsize=15)
t_bottom = fig.text(0.5, -0.05, r"$\tilde{\boldsymbol{\lambda}}$", va="center", ha="center", fontsize=15)
(ROOT / "results").mkdir(exist_ok=True)
plt.savefig(ROOT / "results/figure2.pdf", bbox_inches="tight", bbox_extra_artists=[t_left, t_bottom])
plt.show()