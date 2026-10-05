"""
Loads raw per-fold results produced by run_experiments.py (BTN-Kernel
machines) and the MATLAB GP baseline CSVs, and prints two summary tables:
one for general performance (now including GP RMSE for comparison), and
one for NLL/UQ validation that includes GP NLL and GP coverage alongside
the analytic and MC BTN-Kernel results.
"""
import os, sys

sys.path.append(os.getcwd())
from config import *  # Import everything from config.py

RESULTS_PATH = "data/uq_validation_all_runs.csv"
RUNTIMES_PATH = "data/uq_validation_runtimes.csv"

GP_BASE_PATH = "/Users/hakilic/Desktop/submissions/SIAM/T-KRR-SIAM"
GP_FILES = {
    "airfoil": "airfoil_all_runs_gp.csv",
    "concrete": "concrete_all_runs_gp.csv",
    "energy": "energy_all_runs_gp.csv",
    "yacht": "yacht_all_runs_gp.csv",
}

N_STD = 1.959963985  # for coverage interval width, matching the BTN-Kernel analysis

df = pd.read_csv(RESULTS_PATH)
runtimes_df = pd.read_csv(RUNTIMES_PATH)

# ----------------------------------------------------------------------
# Compute GP NLL, coverage, and RMSE per dataset from MATLAB exports
# ----------------------------------------------------------------------
gp_stats_by_dataset = {}

for dataset_name, filename in GP_FILES.items():
    filepath = os.path.join(GP_BASE_PATH, filename)
    if not os.path.exists(filepath):
        print(f"WARNING: GP results file not found for {dataset_name}: {filepath}")
        continue

    gp_df = pd.read_csv(filepath)

    per_run_nll = []
    per_run_coverage = []
    per_run_rmse = []
    for run_id, run_group in gp_df.groupby("run"):
        y_true = run_group["y_test"].values
        pred_mean = run_group["prediction_mean"].values
        pred_std = run_group["prediction_std"].values

        nll = np.mean(
            0.5 * np.log(2 * np.pi * pred_std**2)
            + 0.5 * ((y_true - pred_mean) ** 2) / (pred_std**2)
        )
        per_run_nll.append(nll)

        lower = pred_mean - N_STD * pred_std
        upper = pred_mean + N_STD * pred_std
        coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100
        per_run_coverage.append(coverage)

        rmse = np.sqrt(np.mean((y_true - pred_mean) ** 2))
        per_run_rmse.append(rmse)

    gp_stats_by_dataset[dataset_name] = {
        "nll_mean": np.mean(per_run_nll),
        "nll_std": np.std(per_run_nll, ddof=0),
        "cov_mean": np.mean(per_run_coverage),
        "cov_std": np.std(per_run_coverage, ddof=0),
        "rmse_mean": np.mean(per_run_rmse),
        "rmse_std": np.std(per_run_rmse, ddof=0),
    }

# ----------------------------------------------------------------------
# Table 1: General performance stats (RMSE, Effective R, runtime) — BTN vs. GP
# ----------------------------------------------------------------------
general_metrics = {
    "rmse": "RMSE",
}

general_rows = []
for dataset_name, group in df.groupby("dataset"):
    row = {"Dataset": dataset_name.capitalize(), "N runs": len(group)}
    for col, label in general_metrics.items():
        mean_val = group[col].mean()
        std_val = group[col].std(ddof=0)
        row[label] = f"{mean_val:.3f} ± {std_val:.3f}"

    r_mean = group["R_effective"].mean()
    r_std = group["R_effective"].std(ddof=0)
    row["Effective R"] = f"{r_mean:.1f} ± {r_std:.1f}"

    if dataset_name in gp_stats_by_dataset:
        gp_stats = gp_stats_by_dataset[dataset_name]
        row["GP RMSE"] = f"{gp_stats['rmse_mean']:.3f} ± {gp_stats['rmse_std']:.3f}"
    else:
        row["GP RMSE"] = "N/A"

    runtime_row = runtimes_df[runtimes_df["dataset"] == dataset_name]
    if not runtime_row.empty:
        row["Total runtime (s)"] = f"{runtime_row['runtime_seconds'].values[0]:.2f}"
    else:
        row["Total runtime (s)"] = "N/A"

    general_rows.append(row)

general_df = pd.DataFrame(general_rows)
general_column_order = ["Dataset", "N runs", "RMSE", "GP RMSE", "Effective R", "Total runtime (s)"]
general_df = general_df[general_column_order]

# ----------------------------------------------------------------------
# Table 2: NLL / UQ validation stats (analytic vs. MC vs. GP) — BTN
# ----------------------------------------------------------------------
nll_metrics = {
    "analytic_nll": "Analytic NLL",
    "mc_nll": "MC NLL",
    "analytic_coverage": "Analytic Cov. (%)",
    "mc_coverage": "MC Cov. (%)",
}

nll_rows = []
for dataset_name, group in df.groupby("dataset"):
    row = {"Dataset": dataset_name.capitalize(), "N runs": len(group)}
    for col, label in nll_metrics.items():
        mean_val = group[col].mean()
        std_val = group[col].std(ddof=0)
        row[label] = f"{mean_val:.3f} ± {std_val:.3f}"

    if dataset_name in gp_stats_by_dataset:
        gp_stats = gp_stats_by_dataset[dataset_name]
        row["GP NLL"] = f"{gp_stats['nll_mean']:.3f} ± {gp_stats['nll_std']:.3f}"
        row["GP Cov. (%)"] = f"{gp_stats['cov_mean']:.1f} ± {gp_stats['cov_std']:.1f}"
    else:
        row["GP NLL"] = "N/A"
        row["GP Cov. (%)"] = "N/A"

    nll_rows.append(row)

nll_df = pd.DataFrame(nll_rows)
nll_column_order = [
    "Dataset", "N runs",
    "Analytic NLL", "MC NLL", "GP NLL",
    "Analytic Cov. (%)", "MC Cov. (%)", "GP Cov. (%)",
]
nll_df = nll_df[nll_column_order]

# ----------------------------------------------------------------------
# Print both tables
# ----------------------------------------------------------------------
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

print("\n" + "=" * 100)
print("GENERAL PERFORMANCE — SUMMARY ACROSS DATASETS (BTN-Kernel vs. GP)")
print("=" * 100)
print(general_df.to_string(index=False))

print("\n" + "=" * 100)
print("UNCERTAINTY QUANTIFICATION VALIDATION (NLL, COVERAGE, ANALYTIC vs. MC vs. GP) — SUMMARY ACROSS DATASETS")
print("=" * 100)
print(nll_df.to_string(index=False))
print("=" * 100)

print(f"\nTotal runtime across all datasets: {runtimes_df['runtime_seconds'].sum():.2f} seconds")



# ====================================================================================================
# GENERAL PERFORMANCE — SUMMARY ACROSS DATASETS (BTN-Kernel vs. GP)
# ====================================================================================================
#  Dataset  N runs          RMSE GP RMSE Effective R Total runtime (s)
#  Airfoil      10 1.738 ± 0.149     N/A   9.5 ± 0.5             49.14
# Concrete      10 5.391 ± 1.275     N/A   5.3 ± 0.5             50.50
#   Energy      10 0.496 ± 0.143     N/A  10.1 ± 1.3             66.20
#    Yacht      10 0.368 ± 0.126     N/A   5.2 ± 0.6             26.06

# ====================================================================================================
# UNCERTAINTY QUANTIFICATION VALIDATION (NLL, COVERAGE, ANALYTIC vs. MC vs. GP) — SUMMARY ACROSS DATASETS
# ====================================================================================================
#  Dataset  N runs  Analytic NLL        MC NLL GP NLL Analytic Cov. (%)    MC Cov. (%) GP Cov. (%)
#  Airfoil      10 1.976 ± 0.096 1.977 ± 0.119    N/A    94.702 ± 1.777 91.921 ± 0.973         N/A
# Concrete      10 3.344 ± 0.409 3.336 ± 0.423    N/A    88.058 ± 3.420 87.184 ± 4.529         N/A
#   Energy      10 1.528 ± 0.279 0.856 ± 0.654    N/A    99.351 ± 0.871 87.143 ± 4.125         N/A
#    Yacht      10 1.010 ± 0.415 0.821 ± 1.164    N/A   100.000 ± 0.000 87.742 ± 6.419         N/A
# ====================================================================================================