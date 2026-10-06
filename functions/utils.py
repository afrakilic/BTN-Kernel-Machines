"""
HELPER FUNCTIONS
Dependencies and configurations are centralized in `config.py`.
"""

import os, sys

sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
import numpy as np


def pure_power_features_full(X, input_dimension):
    """
    Pure-Power Polynomial Features

    Parameters:
    X : numpy.ndarray
        Input array of shape (n_samples, n_features).
    M : int
        Maximum power (degree) of the polynomial features.

    Returns:
    Mati : numpy.ndarray
        Array of shape (n_features, n_samples, M) containing unit-norm pure-power features.
    """
    # Compute the pure-power features
    Mati = np.power(X.T[:, :, np.newaxis], np.arange(input_dimension))

    # Normalize each sample's features along the last axis to have a unit norm
    norms = np.linalg.norm(
        Mati, axis=2, keepdims=True
    )  # Compute norms along the power axis
    Mati = Mati / norms  # Normalize features to unit norm

    return Mati


def columnwise_kronecker(A, B):
    """Compute the columnwise Kronecker product of two matrices A and B."""
    # Check dimensions
    if A.shape[1] != B.shape[1]:
        raise ValueError("Number of columns in A and B must be the same.")

    # Dimensions of input matrices
    m, n = A.shape
    p, _ = B.shape

    # Vectorized: column i is kron(A[:, i], B[:, i]), i.e. entry (a*p + b, i)
    # = A[a, i] * B[b, i]. One broadcasted multiply instead of a Python loop
    # with n separate np.kron calls.
    return (A[:, None, :] * B[None, :, :]).reshape(m * p, n)


def dotkron(*matrices):
    """
    Computes the row-wise right-Kronecker product of two or three matrices.

    Parameters:
        matrices: Two or three matrices with the same number of rows.

    Returns:
        y: The resulting row-wise right-Kronecker product.

    Raises:
        ValueError: If the matrices do not have the same number of rows
                    or if the number of matrices is not 2 or 3.
    """
    if len(matrices) == 2:
        L, R = matrices
        r1, c1 = L.shape
        r2, c2 = R.shape

        if r1 != r2:
            raise ValueError("Matrices should have equal rows!")

        # Row-wise right-Kronecker product for two matrices.
        # Column j*c1 + i holds L[:, i] * R[:, j] (L index varies fastest),
        # identical to the old tile/kron version but with a single
        # broadcasted multiply and one allocation.
        y = (R[:, :, None] * L[:, None, :]).reshape(r1, c1 * c2)

    elif len(matrices) == 3:
        L, M, R = matrices
        r1, _ = L.shape
        r2, _ = M.shape
        r3, _ = R.shape

        if r1 != r2 or r2 != r3:
            raise ValueError("Matrices should have equal rows!")

        # Recursive call for three matrices
        y = dotkron(L, dotkron(M, R))

    else:
        raise ValueError("Please input 2 or 3 matrices!")

    return y


def temp(
    Phi, V, R, Phi2=None
):  # her bir RXR block'u vectorize edip rowlarina koyuyor yeni matrix'in
    """
    Phi2 : optional precomputed dotkron(Phi, Phi), shape (N, I*I).
           Phi never changes during training, so pass this in to avoid
           rebuilding the (N, I^2) array on every call.
    """
    I = Phi.shape[1]
    V = np.reshape(V, (I, R, I, R), order="F")
    V_permuted = np.transpose(V, axes=(0, 2, 1, 3))
    result = np.reshape(V_permuted, (I**2, R**2))
    if Phi2 is None:
        Phi2 = dotkron(Phi, Phi)
    return Phi2 @ result


def dotkronX(A, B, y):
    """
    Computes the Kronecker dot product for large matrices using batch processing.

    Parameters:
    A : np.ndarray (N, DA)
        First input matrix
    B : np.ndarray (N, DB)
        Second input matrix
    y : np.ndarray (N, 1)
        Target vector
    batch_size : int, optional
        Number of samples to process in each batch (default is 10000)

    Returns:
    CC : np.ndarray (DA*DB, DA*DB)
        Computed Kronecker product matrix
    Cy : np.ndarray (DA*DB, 1)
        Computed target vector
    """
    N, DA = A.shape
    _, DB = B.shape

    CC = np.zeros((DA * DB, DA * DB))
    Cy = np.zeros((DA * DB, 1))
    y = y.reshape(-1, 1)  # Ensure y has shape (N, 1)

    batch_size = 10000
    for n in range(0, N, batch_size):
        # Slice end is exclusive, so n + batch_size covers every row.
        # (The old n + batch_size - 1 silently dropped one row per batch.)
        idx = min(n + batch_size, N)

        # Same column ordering as before (A index fastest), built with one
        # broadcasted multiply instead of np.tile + np.kron.
        Kb = dotkron(A[n:idx, :], B[n:idx, :])

        CC += Kb.T @ Kb  # Accumulate Kronecker product
        Cy += Kb.T @ y[n:idx, :]

    return CC, Cy


def safe_division(X, A, epsilon=1):
    A_safe = np.where(A == 0, epsilon, A)
    Y = X / A_safe
    return Y


def safelog(x):
    return np.log(np.clip(x, 1e-10, 1e10))  # x to range [1e-10, 1e10]


def mc_predictive_check(model, features, input_dimension, S=3000, y_true=None, seed=None):
    """
    Monte Carlo estimate of the predictive distribution from the fitted
    posteriors in `model`, used to check the analytic Student's-t
    predictive (eq. 3.40) against direct sampling of the same integral
    (eq. 3.39), with no distributional form imposed.

    NLL is Rao-Blackwellized: each sample is scored under its own exact
    Gaussian likelihood (eq. 3.2), averaged in probability space via
    logsumexp, rather than moment-matched to a single fitted distribution.
    Coverage uses empirical quantiles of the posterior predictive draws.

    Parameters
    ----------
    model : btnkm
        Trained model (needs self.W_D, self.V, self.a, self.b).
    features : np.ndarray (N, D_in)
        Test inputs, same format as model.predict().
    input_dimension : int
        Same value used at training time.
    S : int
        Number of MC samples.
    y_true : np.ndarray (N,), optional
        True targets (same scale as model predictions). If given, MC NLL
        and coverage are computed.
    seed : int, optional

    Returns
    -------
    dict
        mc_mean, mc_std_epistemic, mc_std_total, lower_95, upper_95,
        nll_mc, coverage_95, and the raw samples.
    """
    if seed is not None:
        np.random.seed(seed)

    D = len(model.W_D)
    I, R = model.W_D[0].shape
    N = features.shape[0]

    Phi = pure_power_features_full(features, input_dimension) + 0.2  # D arrays, each (N, I)

    # sample tau ~ Gamma(a_N, rate=b_N)
    tau_samples = np.random.gamma(shape=model.a, scale=1.0 / model.b, size=S)  # (S,)

    # sample each factor matrix W^(d) ~ N(vec(W_tilde^(d)), Sigma^(d));
    # mean is flattened order="F" to match training, and samples are
    # reshaped back the same way via (S, R, I) -> transpose
    W_samples = []
    for d in range(D):
        mean_d = model.W_D[d].flatten(order="F")
        cov_d = model.V[d]
        flat_samples = np.random.multivariate_normal(mean_d, cov_d, size=S)
        W_samples.append(flat_samples.reshape(S, R, I).transpose(0, 2, 1))  # (S, I, R)

    # propagate every sampled model through the CPD prediction formula
    hadamard = np.ones((S, N, R))
    for d in range(D):
        hadamard *= np.einsum("ni,sir->snr", Phi[d], W_samples[d])
    preds_samples = hadamard.sum(axis=2)  # (S, N)

    # full posterior predictive draws (add sampled observation noise)
    noise = np.random.randn(S, N) / np.sqrt(tau_samples)[:, None]
    y_samples = preds_samples + noise

    # empirical summaries
    mc_mean = preds_samples.mean(axis=0)
    mc_std_epistemic = preds_samples.std(axis=0)
    mc_std_total = y_samples.std(axis=0)
    lower_95 = np.percentile(y_samples, 2.5, axis=0)
    upper_95 = np.percentile(y_samples, 97.5, axis=0)

    nll_mc, coverage = None, None
    if y_true is not None:
        log_dens = (
            -0.5 * np.log(2 * np.pi / tau_samples)[:, None]
            - 0.5 * tau_samples[:, None] * (y_true[None, :] - preds_samples) ** 2
        )
        log_p = logsumexp(log_dens, axis=0) - np.log(S)
        nll_mc = -np.mean(log_p)
        coverage = np.mean((y_true >= lower_95) & (y_true <= upper_95)) * 100

    return {
        "mc_mean": mc_mean,
        "mc_std_epistemic": mc_std_epistemic,
        "mc_std_total": mc_std_total,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "nll_mc": nll_mc,
        "coverage_95": coverage,
        "preds_samples": preds_samples,
        "tau_samples": tau_samples,
    }