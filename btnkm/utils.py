"""Feature map and small linear-algebra helpers used by the model."""

import numpy as np


def pure_power_features(X, M):
    """Unit-norm pure-power features.

    Returns an array of shape (D, N, M): for every input dimension d, the
    powers x_d^0 ... x_d^(M-1), scaled so that each row has unit norm.
    """
    Phi = np.power(X.T[:, :, np.newaxis], np.arange(M))
    return Phi / np.linalg.norm(Phi, axis=2, keepdims=True)


def features(X, M):
    """Feature map used by the model: unit-norm pure powers plus a 0.2 offset."""
    return pure_power_features(X, M) + 0.2


def dotkron(L, R):
    """Row-wise Kronecker product: row n is kron(R[n], L[n])."""
    N, c1 = L.shape
    c2 = R.shape[1]
    if R.shape[0] != N:
        raise ValueError("Matrices should have equal rows!")
    return (R[:, :, None] * L[:, None, :]).reshape(N, c1 * c2)


def columnwise_kronecker(A, B):
    """Column-wise Kronecker product: column i is kron(A[:, i], B[:, i])."""
    if A.shape[1] != B.shape[1]:
        raise ValueError("Number of columns in A and B must be the same.")
    m, n = A.shape
    p = B.shape[0]
    return (A[:, None, :] * B[None, :, :]).reshape(m * p, n)


def phi_cov(Phi, V, R):
    """Per-sample second-moment term of one factor's covariance.

    Row n holds the R x R block phi_n^T V_(r, r') phi_n, flattened to R^2 entries.
    """
    I = Phi.shape[1]
    V = np.reshape(V, (I, R, I, R), order="F").transpose(0, 2, 1, 3)
    return dotkron(Phi, Phi) @ np.reshape(V, (I**2, R**2))


def gram_and_rhs(A, B, y, batch_size=10000):
    """Return K^T K and K^T y for K = dotkron(A, B), built in batches."""
    N, DA = A.shape
    DB = B.shape[1]
    KK = np.zeros((DA * DB, DA * DB))
    Ky = np.zeros((DA * DB, 1))
    y = y.reshape(-1, 1)
    for n in range(0, N, batch_size):
        end = min(n + batch_size, N)
        K = dotkron(A[n:end], B[n:end])
        KK += K.T @ K
        Ky += K.T @ y[n:end]
    return KK, Ky


def safe_division(X, A, epsilon=1):
    """X / A, with zeros in A replaced by epsilon."""
    return X / np.where(A == 0, epsilon, A)


def safelog(x):
    """log(x) with x clipped to [1e-10, 1e10]."""
    return np.log(np.clip(x, 1e-10, 1e10))
