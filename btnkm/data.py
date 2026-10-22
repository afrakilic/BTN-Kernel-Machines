"""Data loading, train/test splits and evaluation metrics shared by all experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_csv(name):
    """Load data/<name>.csv as a float array. The first row is always skipped (header)."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv", header=None, low_memory=False)
    return df.iloc[1:].values.astype(float)


def split(X, y, seed, train_frac=0.90):
    """Random train/test split (seeded) and input standardisation with training statistics."""
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    n_train = int(train_frac * len(X))
    X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
    y_train, y_test = y[idx[:n_train]], y[idx[n_train:]]

    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_std[X_std == 0] = 1
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    return X_train, X_test, y_train, y_test


def student_t_scale(std, nu):
    """Scale parameter of a Student's-t with standard deviation `std` and nu degrees of freedom."""
    return std / np.sqrt(nu / (nu - 2))


def student_t_nll(y, mean, std, nu):
    """Average NLL of y under the Student's-t predictive (mean, std, nu)."""
    return -np.mean(student_t.logpdf(y, df=nu, loc=mean, scale=student_t_scale(std, nu)))


def student_t_coverage(y, mean, std, nu, level=0.95):
    """Percentage of y inside the central `level` interval of the Student's-t predictive."""
    scale = student_t_scale(std, nu)
    t_crit = student_t.ppf(0.5 + level / 2, df=nu)
    return np.mean((y >= mean - t_crit * scale) & (y <= mean + t_crit * scale)) * 100


def classification_nll(y, mean, std, eps=1e-15):
    """Binary NLL for labels y in {-1, +1}, using P(y = +1) = P(f > 0) under N(mean, std^2)."""
    p = np.clip(norm.sf(0, loc=mean, scale=std), eps, 1 - eps)
    y01 = (y + 1) // 2
    return -np.mean(y01 * np.log(p) + (1 - y01) * np.log(1 - p))
