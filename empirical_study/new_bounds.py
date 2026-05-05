import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Load data ────────────────────────────────────────────────────────────────
btn_concrete = pd.read_csv("data/concrete_results_all_runs.csv")
btn_energy   = pd.read_csv("data/energy_results_all_runs.csv")
btn_airfoil  = pd.read_csv("data/airfoil_results_all_runs.csv")
gp_concrete  = pd.read_csv("data/concrete_all_runs_gp.csv")
gp_energy    = pd.read_csv("data/energy_all_runs_gp.csv")
gp_airfoil   = pd.read_csv("data/airfoil_all_runs_gp.csv")

datasets = {
    "Concrete": (btn_concrete, gp_concrete),
    "Energy":   (btn_energy,   gp_energy),
    "Airfoil":  (btn_airfoil,  gp_airfoil),
}

RUN_ID = 7

def get_run(df, run_id=RUN_ID):
    """Extract one run, reset index."""
    return df[df["run"] == df["run"].unique()[run_id]].copy().reset_index(drop=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "serif",
    "font.size":           13,
    "axes.linewidth":      0.8,
    "xtick.major.width":   0.8,
    "ytick.major.width":   0.8,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
})

COLOR_BTN  = "#d6604d"   # red   — BTN-KM
COLOR_GP   = "#2166ac"   # blue  — GP
COLOR_TRUE = "#333333"   # dark  — true values

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=False)
fig.subplots_adjust(wspace=0.32,
                    left=0.08, right=0.97,
                    bottom=0.18, top=0.90)

for ax, (name, (btn_df, gp_df)) in zip(axes, datasets.items()):

    # Sort both runs by BTN's y_test order so x-axis is consistent
    btn_run = get_run(btn_df).sort_values("y_test").reset_index(drop=True)
    gp_run  = get_run(gp_df).sort_values("y_test").reset_index(drop=True)

    y         = btn_run["y_test"].values
    mu_btn    = btn_run["prediction_mean"].values
    sigma_btn = btn_run["prediction_std"].values
    mu_gp     = gp_run["prediction_mean"].values
    sigma_gp  = gp_run["prediction_std"].values
    x         = np.arange(len(y))

    # GP CI band (behind)
    ax.fill_between(x,
                    mu_gp - 1.96 * sigma_gp,
                    mu_gp + 1.96 * sigma_gp,
                    color=COLOR_GP, alpha=0.20, linewidth=0, zorder=1)

    # BTN-KM CI band
    ax.fill_between(x,
                    mu_btn - 1.96 * sigma_btn,
                    mu_btn + 1.96 * sigma_btn,
                    color=COLOR_BTN, alpha=0.20, linewidth=0, zorder=2)

    # GP mean scatter
    ax.scatter(x, mu_gp, color=COLOR_GP, s=6, zorder=3,
               linewidths=0, label="GP mean")

    # BTN-KM mean scatter
    ax.scatter(x, mu_btn, color=COLOR_BTN, s=6, zorder=4,
               linewidths=0, label="BTN-KM mean")

    # true values line on top
    ax.plot(x, y, color=COLOR_TRUE, lw=1.0, zorder=5, label="True values")

    ax.set_title(name, fontsize=14, fontweight="bold", pad=6)
    ax.set_xlabel("Test sample (sorted by true value)", fontsize=11)
    ax.tick_params(labelsize=11, length=4, pad=3)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))

axes[0].set_ylabel("Target value", fontsize=12)

# ── Shared legend at bottom ───────────────────────────────────────────────────
legend_elements = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_BTN,
               markersize=7, label="BTN-KM mean", lw=0),
    plt.matplotlib.patches.Patch(facecolor=COLOR_BTN, alpha=0.25,
                                 label="BTN-KM 95% CI"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_GP,
               markersize=7, label="GP mean", lw=0),
    plt.matplotlib.patches.Patch(facecolor=COLOR_GP, alpha=0.25,
                                 label="GP 95% CI"),
    plt.Line2D([0], [0], color=COLOR_TRUE, lw=1.0, label="True values"),
]
fig.legend(handles=legend_elements,
           loc="lower center", ncol=5,
           fontsize=11, frameon=True,
           framealpha=0.9, edgecolor="#cccccc",
           borderpad=0.6, handlelength=1.4,
           columnspacing=1.5,
           bbox_to_anchor=(0.535, 0.01))

plt.savefig("prediction_intervals.pdf", dpi=300, bbox_inches="tight")
plt.savefig("prediction_intervals.png", dpi=300, bbox_inches="tight")
print("Saved.")
plt.show()