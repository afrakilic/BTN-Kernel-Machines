"""Figure 2: recovering a known low-rank structure from synthetic data.

The data come from a rank-3 CP model with D = 3 inputs, in which each factor
uses only a random subset of its 5 feature rows, plus a little noise.
The variational updates are run for 100 iterations starting from
R_max = 5 and M = 5. The figure shows |W^(d)| for each factor, with the learned rank
precisions lambda (columns) and feature precisions delta_d (rows).
Saves results/figure2.pdf.

Unlike btnkm.BTNKM, this script uses the features without the +0.2 offset,
pinv, no precision floor and no rank pruning, so it keeps its own update loop.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from btnkm.utils import pure_power_features, dotkron, phi_cov, gram_and_rhs, safe_division

np.random.seed(1)

# ---- True model ----
D, R_TRUE, R, I, N = 3, 3, 5, 5, 500
e = 1e-3 * np.random.randn(N)
rows = [np.sort(np.random.choice(I, np.random.randint(1, I), replace=False)) for _ in range(D)]
W_true = [1e1 * np.random.randn(len(r), R_TRUE) for r in rows]
X = np.random.rand(N, D)
Phi = pure_power_features(X, I)
y = np.sum(
    (Phi[0][:, rows[0]] @ W_true[0]) * (Phi[1][:, rows[1]] @ W_true[1]) * (Phi[2][:, rows[2]] @ W_true[2]),
    axis=1,
) + e

# ---- Priors and initialisation ----
a0, b0 = 1e-3, 1e-3
c0, d0 = 1e-6 * np.ones(R), 1e-6 * np.ones(R)
g0, h0 = 1e-6 * np.ones(I), 1e-6 * np.ones(I)

W_D = [np.random.randn(I, R) for _ in range(D)]
WSigma_D = [0.1 * np.kron(np.eye(R), np.eye(I)) for _ in range(D)]
lambda_R = c0 / d0
delta = [[g0 / h0] for _ in range(D)]
tau = a0 / b0

hadamard_V = np.ones((N, R**2))
hadamard_mean = np.ones((N, R))
for d in range(D):
    hadamard_V = hadamard_V * phi_cov(Phi[d], WSigma_D[d], R)
    hadamard_mean = hadamard_mean * (Phi[d] @ W_D[d])

# ---- Variational updates ----
for it in range(100):
    for d in range(D):
        hadamard_V = safe_division(hadamard_V, phi_cov(Phi[d], WSigma_D[d], R))
        hadamard_mean = safe_division(hadamard_mean, Phi[d] @ W_D[d])

        PhiPhi_V = dotkron(Phi[d], Phi[d]).T @ hadamard_V
        cc, cy = gram_and_rhs(Phi[d], hadamard_mean, y)
        V_temp = np.reshape(
            np.transpose(np.reshape(PhiPhi_V, (I, I, R, R), order="F"), axes=(0, 2, 1, 3)),
            (I * R, I * R),
            order="F",
        )
        WSigma_D[d] = np.linalg.pinv(tau * (cc + V_temp) + np.kron(lambda_R * np.eye(R), delta[d] * np.eye(I)))
        W_D[d] = np.reshape(tau * WSigma_D[d] @ cy, (I, R), order="F")

        hadamard_V = hadamard_V * phi_cov(Phi[d], WSigma_D[d], R)
        hadamard_mean = hadamard_mean * (Phi[d] @ W_D[d])

    # feature precisions delta_d
    for d in range(D):
        mtemp = np.diag(W_D[d] @ (lambda_R * np.eye(R)) @ W_D[d].T)
        vtemp = np.diag(
            np.reshape(
                WSigma_D[d].reshape(I, R, I, R).transpose(0, 2, 1, 3).reshape(I**2, R**2)
                @ (lambda_R * np.eye(R)).ravel(order="F"),
                (I, I),
            )
        )
        delta[d] = (g0 + R / 2) / (h0 * np.ones(I) + (mtemp + vtemp) / 2)

    # rank precisions lambda
    c_N = (0.5 * D * I) + c0
    d_N = 0
    for d in range(D):
        mtemp = np.diag(W_D[d].T @ (delta[d] * np.eye(I)) @ W_D[d])
        vtemp = np.diag(
            np.reshape(
                (delta[d] * np.eye(I)).ravel(order="F").T
                @ WSigma_D[d].reshape(I, R, I, R).transpose(0, 2, 1, 3).reshape(I**2, R**2),
                (R, R),
            )
        )
        d_N += mtemp + vtemp
    lambda_R = c_N / (d0 + (0.5 * d_N))

    # noise precision tau
    residual = y - np.sum(hadamard_mean, axis=1)
    tau = (a0 + (N / 2)) / (b0 + (0.5 * (np.dot(residual, residual) + np.sum(np.sum(hadamard_V, axis=1)))))

# ---- Print the learned quantities ----
with np.printoptions(formatter={"float_kind": lambda x: f"{x:.2e}"}):
    for d in range(D):
        print(f"W_D[{d + 1}]:\n{W_D[d]}\n")
    for d in range(D):
        print(f"delta[{d + 1}]:\n{delta[d]}\n")
    print(f"lambda_R:\n{lambda_R}\n\ntau:\n{tau}\n")
for d in range(D):
    print(f"Singular values of W_D[{d}]: {np.linalg.svd(W_D[d], compute_uv=False)}")


# ---- Figure ----
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
    ax.tick_params(axis="y", labelsize=12.1)

cbar = fig.colorbar(heatmap.get_children()[0], ax=axes.ravel().tolist(), orientation="vertical", fraction=0.02, pad=0.05)
cbar.ax.tick_params(labelsize=10)
t_left = fig.text(-0.00, 0.5, r"$\boldsymbol{\delta}_d$", va="center", ha="center", rotation="vertical", fontsize=15)
t_bottom = fig.text(0.5, -0.05, r"$\boldsymbol{\lambda}$", va="center", ha="center", fontsize=15)
(ROOT / "results").mkdir(exist_ok=True)
# plt.savefig(ROOT / "results/figure2.pdf", bbox_inches="tight", bbox_extra_artists=[t_left, t_bottom])
plt.show()
