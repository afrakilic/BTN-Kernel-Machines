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

def dotkron(L, R):
    """Row-wise Kronecker product: row n is kron(R[n], L[n])."""
    N, c1 = L.shape
    c2 = R.shape[1]
    if R.shape[0] != N:
        raise ValueError("Matrices should have equal rows!")
    return (R[:, :, None] * L[:, None, :]).reshape(N, c1 * c2)


def safelog(x):
    """log(x) with x clipped to [1e-10, 1e10]."""
    return np.log(np.clip(x, 1e-10, 1e10))





