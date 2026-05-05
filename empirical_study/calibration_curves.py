import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

# ── Load data ────────────────────────────────────────────────────────────────
btn_concrete = pd.read_csv("data/concrete_results_all_runs.csv")
btn_energy   = pd.read_csv("data/energy_results_all_runs.csv")
btn_airfoil  = pd.read_csv("data/airfoil_results_all_runs.csv")
gp_concrete  = pd.read_csv("data/concrete_all_runs_gp.csv")
gp_energy    = pd.read_csv("data/energy_all_runs_gp.csv")
gp_airfoil   = pd.read_csv("data/airfoil_all_runs_gp.csv")

datasets = ["Concrete", "Energy", "Airfoil"]
btn_data = [btn_concrete, btn_energy, btn_airfoil]
gp_data  = [gp_concrete,  gp_energy,  gp_airfoil]

# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_calibration_one_run(df, levels):
    """Empirical coverage at each nominal level for a single run."""
    z     = stats.norm.ppf((1 + levels) / 2)
    mu    = df["prediction_mean"].values
    sigma = df["prediction_std"].values
    y     = df["y_test"].values
    return np.array([
        np.mean((y >= mu - zi * sigma) & (y <= mu + zi * sigma))
        for zi in z
    ])

def compute_calibration_all_runs(df, levels):
    """Mean and std of calibration curves across all runs."""
    run_col = "run" if "run" in df.columns else df.columns[0]
    curves  = []
    for run_id in sorted(df[run_col].unique()):
        run_df = df[df[run_col] == run_id]
        curves.append(compute_calibration_one_run(run_df, levels))
    curves = np.array(curves)          # (n_runs, n_levels)
    return curves.mean(axis=0), curves.std(axis=0)

# ── Plot settings ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

levels     = np.linspace(0.01, 0.99, 99)
COLOR_BTN  = "#d6604d"   # red   — BTN-KM
COLOR_GP   = "#2166ac"   # blue  — GP
COLOR_DIAG = "#aaaaaa"   # gray diagonal

fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True, sharex=True)
fig.subplots_adjust(wspace=0.12, left=0.1, right=0.97, bottom=0.22, top=0.88)

for ax, name, btn_df, gp_df in zip(axes, datasets, btn_data, gp_data):

    btn_mean, btn_std = compute_calibration_all_runs(btn_df, levels)
    gp_mean,  gp_std  = compute_calibration_all_runs(gp_df,  levels)

    # perfect calibration diagonal
    ax.plot([0, 1], [0, 1], color=COLOR_DIAG, lw=0.8,
            linestyle="--", zorder=1, label="Ideal")

    # GP — dashed line + shaded band
    ax.fill_between(levels,
                    gp_mean - gp_std, gp_mean + gp_std,
                    color=COLOR_GP, alpha=0.12, zorder=2)
    ax.plot(levels, gp_mean, color=COLOR_GP, lw=1.2,
            linestyle=(0, (4, 2)), zorder=3, label="GP")

    # BTN-KM — solid line + shaded band
    ax.fill_between(levels,
                    btn_mean - btn_std, btn_mean + btn_std,
                    color=COLOR_BTN, alpha=0.12, zorder=4)
    ax.plot(levels, btn_mean, color=COLOR_BTN, lw=1.4,
            linestyle="solid", zorder=5, label="BTN-KM")

    ax.set_title(name, fontsize=9, fontweight="bold", pad=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.tick_params(labelsize=8, length=3, pad=2)

# shared axis labels
fig.text(0.535, 0.04, "Nominal coverage", ha="center", fontsize=9)
axes[0].set_ylabel("Empirical coverage", fontsize=9)

# legend on first panel
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels,
               loc="upper left",
               fontsize=7.5,
               frameon=True,
               framealpha=0.9,
               edgecolor="#cccccc",
               borderpad=0.5,
               handlelength=1.6,
               labelspacing=0.3)

plt.savefig("calibration_curves.pdf", dpi=300, bbox_inches="tight")
plt.savefig("calibration_curves.png", dpi=300, bbox_inches="tight")
print("Saved.")

plt.show()