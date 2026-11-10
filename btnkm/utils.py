"""Feature map and small linear-algebra helpers used by the model.

The factor updates work on the products phi_i * phi_j of each feature vector with itself.
These are symmetric in (i, j), and the R x R covariance blocks are symmetric in (r, r'), so
the helpers below only use the pairs i <= j and r <= r' and process the samples in row chunks
of about `target_elems` entries.
"""

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


def columnwise_kronecker(A, B):
    """Column-wise Kronecker product: column i is kron(A[:, i], B[:, i])."""
    if A.shape[1] != B.shape[1]:
        raise ValueError("Number of columns in A and B must be the same.")
    m, n = A.shape
    p = B.shape[0]
    return (A[:, None, :] * B[None, :, :]).reshape(m * p, n)


def dotkron(L, R):
    """Row-wise Kronecker product: row n is kron(R[n], L[n])."""
    N, c1 = L.shape
    c2 = R.shape[1]
    if R.shape[0] != N:
        raise ValueError("Matrices should have equal rows!")
    return (R[:, :, None] * L[:, None, :]).reshape(N, c1 * c2)


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


def _row_chunk(N, row_elems, target_elems):
    """Number of rows per chunk so that a chunk holds at most ~target_elems entries."""
    return min(N, max(1, target_elems // max(1, row_elems)))


def sym_pairs(I):
    """Pairs (i, j) with i <= j, and for every (a, b) the index of its pair (row a * I + b)."""
    iu, ju = np.triu_indices(I)
    pos = np.empty((I, I), dtype=int)
    pos[iu, ju] = np.arange(iu.size)
    pos[ju, iu] = np.arange(iu.size)
    return iu, ju, pos.ravel()


def moments_sym(Phi, hadamard_V, hadamard_mean, iu, ju, ru, rv, target_elems=5_000_000):
    """sum_n kron(phi_n, phi_n) (hadamard_V_n + kron(m_n, m_n))^T on the symmetric pairs.

    Rows: feature pairs i <= j; columns: rank pairs r <= r' (hadamard_V is stored on the
    rank pairs only, since its R x R blocks are symmetric). Built in row chunks.
    """
    N = Phi.shape[0]
    chunk = _row_chunk(N, iu.size, target_elems)
    acc = np.zeros((iu.size, ru.size))
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        P2 = Phi[s:e, iu] * Phi[s:e, ju]
        acc += P2.T @ (hadamard_V[s:e] + hadamard_mean[s:e, ru] * hadamard_mean[s:e, rv])
    return acc


def cov_blocks(Sigma, I, R, ru, rv):
    """The I x I blocks Sigma[(., r), (., r')] for the rank pairs r <= r', as an I x (I * K) matrix."""
    V2 = np.reshape(Sigma, (I, R, I, R), order="F").transpose(0, 2, 1, 3)  # (i, j, r, r')
    return V2[:, :, ru, rv].reshape(I, I * ru.size)


def phi_cov_pairs(Phi, VM, K, target_elems=5_000_000):
    """phi_n^T Sigma(r, r') phi_n for all samples n and rank pairs r <= r', built in row chunks.

    VM = cov_blocks(...). Computed without forming kron(phi_n, phi_n).
    """
    N, I = Phi.shape
    chunk = _row_chunk(N, I * K, target_elems)
    out = np.empty((N, K))
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        T = (Phi[s:e] @ VM).reshape(e - s, I, K)
        out[s:e] = np.einsum("nj,njk->nk", Phi[s:e], T)
    return out