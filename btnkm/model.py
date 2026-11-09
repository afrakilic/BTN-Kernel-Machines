"""BTN-Kernel machine.

Model: y = <Phi(x), W> + e, with the weight tensor W in CP (CPD) format,
W = [[W^(1), ..., W^(D)]], each factor W^(d) of size M x R.

Priors (all with Gamma hyperpriors):
  - noise precision tau,
  - one precision per rank component lambda_r (shared by all factors) -> prunes the rank,
  - one precision per feature row delta_d,m (per factor)              -> prunes feature dims.

`train` runs mean-field variational inference (a Bayesian ALS sweep over the
factors, then closed-form updates of delta, lambda and tau). `predict` returns the
mean and std of the Student's-t predictive distribution, and `mc_predict`
samples the same predictive distribution by Monte Carlo as a check.
"""

import numpy as np
from scipy.special import gammaln, logsumexp
from tqdm import trange

from .utils import (
    features,
    columnwise_kronecker,
    safe_division,
    safelog,
    sym_pairs,
    moments_sym,
    cov_blocks,
    phi_cov_pairs,
)


class BTNKM:
    def train(
        self,
        X,
        y,
        M,
        R,
        a0,
        b0,
        c0,
        d0,
        g0,
        h0,
        max_iter=100,
        seed=0,
        lambda_update=True,
        delta_update=True,
        precision_update=True,
        prune_rank=True,
        rank_tol=1e-5,
        lower_bound_tol=1e-4,
        verbose=True,
        cache_cov=True,
    ):
        """Fit the model.

        X, y      : training inputs (N, D) and targets (N,)
        M, R      : feature dimension per input and initial (maximum) rank
        a0, b0    : Gamma prior of the noise precision tau
        c0, d0    : Gamma prior of the rank precisions lambda
        g0, h0    : Gamma prior of the feature precisions delta
        rank_tol  : rank components explaining less than this % of the variance are pruned
        cache_cov : keep each factor's covariance term in memory (D x N x R(R+1)/2 floats)
                    instead of recomputing it; same results, faster

        After training, the model stores:
          rank          final rank R_eff
          rank_history  R_eff after every iteration
          lower_bound   variational lower bound per iteration
          M_eff         effective feature dimension of each input (rows with > 0.25% of the variance)
        """
        np.random.seed(seed)
        N, D = X.shape
        I = M

        c0 = c0 * np.ones(R)
        d0 = d0 * np.ones(R)
        g0 = g0 * np.ones(I)
        h0 = h0 * np.ones(I)

        tau = a0 / b0
        c_N, d_N = c0, d0
        lambda_R = c0 / d0
        g_N = [g0 for _ in range(D)]
        h_N = [h0 for _ in range(D)]
        delta = [[g0 / h0] for _ in range(D)]

        # Factor means and covariances
        W_D = [np.random.randn(I, R) for _ in range(D)]
        WSigma_D = [0.1 * np.kron(np.eye(R), np.eye(I)) for _ in range(D)]

        Phi = features(X, M)
        iu, ju, pos = sym_pairs(I)  # symmetric pairs of kron(phi_n, phi_n)
        # hadamard_V holds symmetric R x R blocks per sample: keep only the pairs r <= r'
        ru, rv, posR = sym_pairs(R)

        def cov_term(d):
            # phi_n^T Sigma_d(r, r') phi_n for all samples n and rank pairs r <= r'
            return phi_cov_pairs(Phi[d], cov_blocks(WSigma_D[d], I, R, ru, rv), ru.size)

        LB = np.zeros(max_iter)
        rank_history = []

        # Hadamard products over all factors of the second moments and of the means
        hadamard_V = np.ones((N, ru.size))
        hadamard_mean = np.ones((N, R))
        cache = [None] * D  # covariance term of every factor (if cache_cov), so it is not recomputed
        for d in range(D):
            term = cov_term(d)
            if cache_cov:
                cache[d] = term
            hadamard_V = hadamard_V * term
            hadamard_mean = hadamard_mean * (Phi[d] @ W_D[d])

        pbar = trange(max_iter, desc="Training", leave=True, disable=not verbose)
        for it in pbar:

            # ---- Factor updates q(W^(d)) ----
            for d in range(D):
                hadamard_mean = safe_division(hadamard_mean, Phi[d] @ W_D[d])
                hadamard_V = safe_division(hadamard_V, cache[d] if cache_cov else cov_term(d))
                PhiPhi_V = moments_sym(Phi[d], hadamard_V, hadamard_mean, iu, ju, ru, rv)[pos][:, posR]
                EGG = np.reshape(
                    np.transpose(np.reshape(PhiPhi_V, (I, I, R, R), order="F"), axes=(0, 2, 1, 3)),
                    (I * R, I * R),
                    order="F",
                )
                cy = (Phi[d].T @ (hadamard_mean * np.reshape(y, (-1, 1)))).reshape(-1, 1, order="F")

                A = tau * EGG + np.kron(lambda_R * np.eye(R), delta[d] * np.eye(I))
                try:
                    WSigma_D[d] = np.linalg.inv(A)
                except np.linalg.LinAlgError:
                    WSigma_D[d] = np.linalg.pinv(A)
                W_D[d] = np.reshape(tau * WSigma_D[d] @ cy, (I, R), order="F")

                # add the updated factor back
                term = cov_term(d)
                if cache_cov:
                    cache[d] = term
                hadamard_V = hadamard_V * term
                hadamard_mean = hadamard_mean * (Phi[d] @ W_D[d])

            # ---- Feature precisions q(delta_d) ----
            if delta_update:
                for d in range(D):
                    mtemp = np.diag(W_D[d] @ (lambda_R * np.eye(R)) @ W_D[d].T)
                    vtemp = np.diag(
                        np.reshape(
                            WSigma_D[d].reshape(I, R, I, R).transpose(0, 2, 1, 3).reshape(I**2, R**2)
                            @ (lambda_R * np.eye(R)).ravel(order="F"),
                            (I, I),
                        )
                    )
                    g_N[d] = g0 + R / 2
                    h_N[d] = h0 * np.ones(I) + (mtemp + vtemp) / 2
                    delta[d] = g_N[d] / h_N[d]
                    delta[d][delta[d] < 1e-5] = 1e-5

            # ---- Rank precisions q(lambda) ----
            if lambda_update:
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
                d_N = d0 + (0.5 * d_N)
                lambda_R = c_N / d_N
                lambda_R[lambda_R < 1e-5] = 1e-5

            # ---- Noise precision q(tau) ----
            residual = y - np.sum(hadamard_mean, axis=1)
            ss_error = np.dot(residual, residual)
            covariance = np.sum(hadamard_V @ np.where(ru == rv, 1.0, 2.0))  # off-diagonal pairs count twice
            if precision_update:
                a_N = a0 + (N / 2)
                b_N = b0 + (0.5 * (ss_error + covariance))
            else:
                a_N, b_N = a0, b0
            tau = a_N / b_N

            # ---- Variational lower bound ----
            LB[it] = self._lower_bound(
                W_D, WSigma_D, lambda_R, delta, ss_error + covariance , a_N, b_N, b0, c_N, d_N, d0, g_N, h_N, h0, I, R
            )

            # ---- Rank pruning ----
            rankest = R
            if prune_rank:
                Wall = np.vstack(W_D)
                power = np.diag(Wall.T @ Wall)
                var_explained = power / np.sum(power) * 100
                rankest = np.sum(var_explained > rank_tol)
                if rankest == 0:
                    print("Rank becomes 0 !!!")
                    break
                if R != rankest:
                    removed = np.where(~(var_explained > rank_tol))[0]
                    lambda_R = np.delete(lambda_R, removed)
                    rows = np.r_[[np.arange(I * r, I * r + I) for r in removed]]
                    for d in range(D):
                        W_D[d] = np.delete(W_D[d], removed, axis=1)
                        WSigma_D[d] = np.delete(WSigma_D[d], rows, axis=1)
                        WSigma_D[d] = np.delete(WSigma_D[d], rows, axis=0)
                    hadamard_mean = np.delete(hadamard_mean, removed, axis=1)
                    # drop every rank pair that involves a removed component
                    keep = ~(np.isin(ru, removed) | np.isin(rv, removed))
                    hadamard_V = hadamard_V[:, keep]
                    if cache_cov:
                        cache = [C[:, keep] for C in cache]
                    c0 = c0[:rankest]
                    d0 = d0[:rankest]
                    R = rankest
                    ru, rv, posR = sym_pairs(R)
            rank_history.append(R)

            # ---- Convergence check ----
            rel_change = (LB[it] - LB[it - 1]) / LB[2] if it > 2 else np.nan
            pbar.set_postfix({"RelChan": f"{rel_change:.2e}", "R": rankest})
            if it > 5 and abs(rel_change) < lower_bound_tol:
                if verbose:
                    print("======= Converged =======")
                break

        # Store the posterior
        self.W_D, self.V, self.a, self.b = W_D, WSigma_D, a_N, b_N
        self.M = M
        self.rank = R
        self.rank_history = rank_history
        self.lower_bound = LB[:it]
        self.M_eff = np.array(
            [np.sum((np.diag(W @ W.T) / np.sum(np.diag(W @ W.T))) * 100 >= 0.25) for W in W_D]
        )
        return self

    @staticmethod
    def _lower_bound(W_D, WSigma_D, lambda_R, delta, ss_error, a_N, b_N, b0, c_N, d_N, d0, g_N, h_N, h0, I, R):
        """Variational lower bound (up to constants); only used to check convergence."""
        t1 = -0.5 * (a_N / b_N) * ss_error
        # trace(kron(diag(lambda), diag(delta)) (w w^T + Sigma)) only needs the diagonal of (w w^T + Sigma)
        t2 = 0.0
        for d in range(len(W_D)):
            w = W_D[d].ravel(order="F")
            t2 += np.kron(lambda_R, np.ravel(delta[d])) @ (w**2 + np.diag(WSigma_D[d]))
        t2 = -0.5 * t2
        t3 = 0.5 * np.sum([safelog(np.linalg.slogdet(S)) for S in WSigma_D])
        t4 = np.sum(np.multiply(c_N, (1 - safelog(d_N) - (d0 / d_N))))
        t5 = sum(
            np.sum(
                gammaln(np.array(g_N[d]))
                + np.array(g_N[d]) * (1 - safelog(np.array(h_N[d])) - (h0 / np.array(h_N[d])))
            )
            for d in range(len(g_N))
        )
        t6 = a_N * (1 - safelog(b_N) - (b0 / b_N))
        return t1 + t2 + t3 + t4 + t5 + t6

    def predict(self, X):
        """Mean and standard deviation of the Student's-t predictive distribution.

        The predictive has nu = 2 * a_N degrees of freedom; `std` is its standard
        deviation (not its scale parameter).
        """
        Phi = features(X, self.M)
        N, D = X.shape[0], len(self.W_D)
        R = self.W_D[0].shape[1]

        PW = [Phi[k] @ self.W_D[k] for k in range(D)]
        prod = np.ones((N, R))
        for d in range(D):
            prod = np.multiply(prod, PW[d])
        mean = np.sum(prod, axis=1)

        # sum over factors of g_d(x)^T Sigma_d g_d(x), one value per test point
        quad = np.zeros(N)
        for d in range(D):
            others = np.ones((N, R))
            for k in range(D):
                if k != d:
                    others *= PW[k]
            g = columnwise_kronecker(others.T, Phi[d].T)
            quad += np.sum((g.T @ self.V[d]) * g.T, axis=1)

        S = (2 * self.a / (2 * self.a - 2)) * ((self.b / self.a) + (quad))
        std = np.sqrt(S)
        return mean, std

    def mc_predict(self, X, S=3000, y_true=None, seed=None):
        """Monte Carlo check of the predictive distribution.

        Samples every factor from q(W^(d)) and tau from q(tau), pushes them through
        the model and adds noise. If y_true is given, returns the MC NLL (each
        sample scored with its own Gaussian likelihood, averaged with logsumexp)
        and the coverage (%) of the empirical 2.5-97.5% interval.
        """
        if seed is not None:
            np.random.seed(seed)
        D = len(self.W_D)
        I, R = self.W_D[0].shape
        N = X.shape[0]
        Phi = features(X, self.M)

        tau_samples = np.random.gamma(shape=self.a, scale=1.0 / self.b, size=S)
        W_samples = []
        for d in range(D):
            flat = np.random.multivariate_normal(self.W_D[d].flatten(order="F"), self.V[d], size=S)
            W_samples.append(flat.reshape(S, R, I).transpose(0, 2, 1))  # (S, I, R)

        prod = np.ones((S, N, R))
        for d in range(D):
            prod *= np.einsum("ni,sir->snr", Phi[d], W_samples[d])
        f_samples = prod.sum(axis=2)
        y_samples = f_samples + np.random.randn(S, N) / np.sqrt(tau_samples)[:, None]

        out = {
            "mean": f_samples.mean(axis=0),
            "std": y_samples.std(axis=0),
            "lower_95": np.percentile(y_samples, 2.5, axis=0),
            "upper_95": np.percentile(y_samples, 97.5, axis=0),
            "nll": None,
            "coverage_95": None,
        }
        if y_true is not None:
            log_dens = (
                -0.5 * np.log(2 * np.pi / tau_samples)[:, None]
                - 0.5 * tau_samples[:, None] * (y_true[None, :] - f_samples) ** 2
            )
            out["nll"] = -np.mean(logsumexp(log_dens, axis=0) - np.log(S))
            out["coverage_95"] = np.mean((y_true >= out["lower_95"]) & (y_true <= out["upper_95"])) * 100
        return out