"""
Runs BTN-Kernel Machines training + analytic prediction + Monte Carlo
predictive validation across multiple UCI datasets, for 10 random splits
each. Saves per-fold results to a single CSV for later analysis.

Analytic NLL/coverage use the exact Student's-t predictive (3.40) --
no Gaussian moment-matching. MC NLL/coverage use the mixture-of-Gaussians
estimate and empirical quantiles returned by mc_predictive_check, with no
distributional assumption imposed.

Dependencies and configurations are centralized in `config.py`.
"""

import os, sys

sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
from functions.BTN_KM import btnkm
from functions.utils import mc_predictive_check


# ----------------------------------------------------------------------
# Dataset configurations: path, target column, feature slice, hyperparams
# ----------------------------------------------------------------------
DATASETS = {
    
    # "naval": dict(
    #     path="data/naval.csv",
    #     n_features=16,
    #     input_dimension=6,      # lower than the shared default -- degree-20 monomials
    #                              # cause severe ill-conditioning on this dataset (see diagnosis)
    #     max_rank=25,
    #     tau_shape=1e-4, tau_scale=1e-3,
    #     has_header=False,       # this file has no header row, unlike the others
    # ),
    "airfoil": dict(
        path="data/airfoil.csv",
        n_features=5,
        input_dimension=20,
        max_rank=25,
        tau_shape=1e-3, tau_scale=1e-3,
        has_header=True,
    ),
    "concrete": dict(
        path="data/concrete.csv",
        n_features=8,
        input_dimension=20,
        max_rank=25,
        tau_shape=1e-3, tau_scale=1e-3,
        has_header=True,
    ),
    "energy": dict(
        path="data/energy.csv",
        n_features=9,
        input_dimension=20,
        max_rank=25,
        tau_shape=1e-2, tau_scale=1e-3,
        has_header=True,
    ),
    "yacht": dict(
        path="data/yacht.csv",
        n_features=6,
        input_dimension=20,
        max_rank=25,
        tau_shape=1e-2, tau_scale=1e-3,
        has_header=True,
    ),
}

N_RUNS = 10
S_MC = 3000
COVERAGE_LEVEL = 0.95  # nominal coverage for both analytic and MC intervals

all_results = []
dataset_runtimes = []

overall_start = time.time()

for dataset_name, cfg in DATASETS.items():
    print("\n" + "=" * 70)
    print(f"Running experiments for: {dataset_name}")
    print("=" * 70)

    input_dimension = cfg["input_dimension"]
    max_rank = cfg["max_rank"]
    n_features = cfg["n_features"]

    a, b = cfg["tau_shape"], cfg["tau_scale"]
    c, d = 1e-5 * np.ones(max_rank), 1e-6 * np.ones(max_rank)
    g, h = 1e-6 * np.ones(input_dimension), 1e-6 * np.ones(input_dimension)

    # Load the dataset -- naval has no header row and is whitespace-delimited,
    # unlike the other UCI files which have a header row and are comma-delimited.
    if cfg.get("has_header", True):
        df = pd.read_csv(cfg["path"], header=None)
        df.columns = df.iloc[0]
        df = df[1:]
        df.reset_index(drop=True, inplace=True)
        df = df.values.astype(float)
    else:
        df = pd.read_csv(cfg["path"], header=None, sep=r"\s+")
        df = df.values.astype(float)

    X = df[:, :n_features]
    y = df[:, n_features]

    dataset_start = time.time()
    for i in range(N_RUNS):
        np.random.seed(i)
        indices = np.random.permutation(len(X))
        split_index = int(0.90 * len(X))
        X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
        y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_std[X_std == 0] = 1
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std

        y_mean = y_train.mean()
        y_std = y_train.std()
        y_train = (y_train - y_mean) / y_std
        y_test_standardized = (y_test - y_mean) / y_std

        # ---------------- Train ----------------
        model = btnkm(X_train.shape[1])
        R, _, _, _, _, _, _ = model.train(
            features=X_train,
            target=y_train,
            input_dimension=input_dimension,
            max_rank=max_rank,
            shape_parameter_tau=a,
            scale_parameter_tau=b,
            shape_parameter_lambda=c,
            scale_parameter_lambda=d,
            shape_parameter_delta=g,
            scale_parameter_delta=h,
            max_iter=50,
            precision_update=True,
            lambda_update=True,
            delta_update=True,
            plot_results=False,
            prune_rank=True,
        )

        # model.a is the posterior shape parameter a_N (eq. 3.32); nu_y = 2*a_N
        # is a scalar, shared across all test points (eq. 3.40).
        nu_y = 2.0 * model.a

        # ---------------- Analytic prediction ----------------
        prediction_mean, prediction_std, _ = model.predict(
            features=X_test, input_dimension=input_dimension
        )
        prediction_mean_unscaled = prediction_mean * y_std + y_mean
        prediction_std_unscaled = prediction_std * y_std  # sqrt(Var(y_i)) from eq. (3.40)

        # Recover the Student's-t scale parameter from the reported predictive
        # std: Var = scale^2 * nu/(nu-2)  =>  scale = std / sqrt(nu/(nu-2))
        scale_i = prediction_std_unscaled / np.sqrt(nu_y / (nu_y - 2))

        # Exact Student's-t NLL: the density BTN-Kernel machines actually
        # output (3.40), with no further Gaussian approximation layered on.
        analytic_nll = -np.mean(
            student_t.logpdf(y_test, df=nu_y, loc=prediction_mean_unscaled, scale=scale_i)
        )

        rmse = np.sqrt(np.mean((prediction_mean_unscaled - y_test) ** 2))

        # Exact t-quantile interval (heavier tails than a Gaussian +-z*sigma)
        t_crit = student_t.ppf(0.5 + COVERAGE_LEVEL / 2, df=nu_y)
        lower_a = prediction_mean_unscaled - t_crit * scale_i
        upper_a = prediction_mean_unscaled + t_crit * scale_i
        coverage_analytic = np.mean((y_test >= lower_a) & (y_test <= upper_a)) * 100

        # ---------------- Monte Carlo predictive check ----------------
        mc_result = mc_predictive_check(
            model, X_test, input_dimension, S=S_MC, y_true=y_test_standardized, seed=i
        )
        mc_mean_unscaled = mc_result["mc_mean"] * y_std + y_mean
        mc_std_unscaled = mc_result["mc_std_total"] * y_std
        mc_nll = mc_result["nll_mc"] + np.log(y_std)  # change-of-variables term for unscaling

        # Use the empirical MC quantiles directly (already computed with no
        # distributional assumption inside mc_predictive_check) rather than
        # rebuilding a Gaussian +-z*sigma interval from the MC mean/std --
        # that would silently reimpose a Gaussian shape on an otherwise
        # assumption-free check.
        lower_mc = mc_result["lower_95"] * y_std + y_mean
        upper_mc = mc_result["upper_95"] * y_std + y_mean
        coverage_mc = np.mean((y_test >= lower_mc) & (y_test <= upper_mc)) * 100

        std_ratio = np.mean(prediction_std_unscaled / mc_std_unscaled)

        all_results.append(dict(
            dataset=dataset_name,
            run=i,
            rmse=rmse,
            analytic_nll=analytic_nll,
            mc_nll=mc_nll,
            analytic_coverage=coverage_analytic,
            mc_coverage=coverage_mc,
            std_ratio=std_ratio,
            R_effective=R,
        ))

        print(f"  [run {i}] RMSE={rmse:.4f}  analytic_NLL={analytic_nll:.4f}  "
              f"MC_NLL={mc_nll:.4f}  analytic_cov={coverage_analytic:.1f}%  "
              f"MC_cov={coverage_mc:.1f}%  std_ratio={std_ratio:.2f}  R={R}")

    dataset_elapsed = time.time() - dataset_start
    dataset_runtimes.append(dict(dataset=dataset_name, runtime_seconds=dataset_elapsed))
    print(f"  Completed {dataset_name} in {dataset_elapsed:.2f} seconds")

overall_elapsed = time.time() - overall_start

# ----------------------------------------------------------------------
# Save all raw per-fold results and runtimes
# ----------------------------------------------------------------------
results_df = pd.DataFrame(all_results)
results_df.to_csv("data/uq_validation_all_runs.csv", index=False)

runtimes_df = pd.DataFrame(dataset_runtimes)
runtimes_df.to_csv("data/uq_validation_runtimes.csv", index=False)

print("\n" + "=" * 70)
print(f"Total runtime across all datasets: {overall_elapsed:.2f} seconds")
print("=" * 70)
print("Saved raw per-fold results to data/uq_validation_all_runs.csv")
print("Saved per-dataset runtimes to data/uq_validation_runtimes.csv")