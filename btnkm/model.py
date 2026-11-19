"""BTN-Kernel machine.

Model: y = <Phi(x), W> + e, with the weight tensor W in CP (CPD) format,
W = [[W^(1), ..., W^(D)]], each factor W^(d) of size M x R.

Priors (all with Gamma hyperpriors):
  - noise precision tau,
  - one precision per rank component lambda_r (shared by all factors) -> prunes the rank,
  - one precision per feature row delta_d,m (per factor)              -> prunes feature dims.

`train` runs mean-field variational inference (a Bayesian ALS sweep over the
factors, then closed-form updates of delta, lambda and tau). `predict` returns the
mean and std of the predictive distribution under q, and `mc_predict`
samples the same predictive distribution by Monte Carlo as a check.

Factor update (as in the paper), with ⊙ the row-wise Khatri-Rao product (row n of
Phi ⊙ Phi is kron(phi_n, phi_n)) and ⊛ the Hadamard product:

    E[G^T G] = R{ (Phi_d ⊙ Phi_d)^T  ⊛_{k != d} (Phi_k ⊙ Phi_k) (W_k ⊗ W_k + R{Sigma_k}) }
    E[G]^T y = vec( Phi_d^T ( ⊛_{k != d} (Phi_k W_k)  *  y ) )
    V_d      = ( E[tau] E[G^T G] + kron(diag(lambda), diag(delta_d)) )^-1
    vec(W_d) = E[tau] V_d E[G]^T y

W_k ⊗ W_k is the I^2 x R^2 Kronecker product of the mean factor with itself, and R{Sigma_k}
is the IR x IR covariance rearranged to the same I^2 x R^2 layout. With use_sigma=False the
R{Sigma_k} terms are left out of the sum, E[G^T G] becomes G^T G of the mean factors, and
the factor update is the deterministic (ridge-regularised) ALS update.
"""

import numpy as np
from scipy.special import gammaln, logsumexp
from tqdm import trange

from .utils import features, dotkron, safelog


# ---- Rearrangements and row-wise Khatri-Rao products used in the formulas above ----

def rearrange_sigma(Sigma, I, R):
    """R{Sigma}: the IR x IR covariance of vec(W) (column-major, index i + I*r) rearranged
    into the I^2 x R^2 layout of kron(W, W): entry [i*I + j, r*R + r'] = Sigma[(i, r), (j, r')]."""
    S4 = Sigma.reshape(R, I, R, I)  # S4[r, i, r', j] = Sigma[i + I r, j + I r']
    return S4.transpose(1, 3, 0, 2).reshape(I * I, R * R)


def rearrange_gram(X, I, R):
    """R{X}: the inverse rearrangement, from the I^2 x R^2 layout [i*I + j, r*R + r'] back to
    the IR x IR layout of vec(W): entry [(i, r), (j, r')]."""
    return X.reshape(I, I, R, R).transpose(2, 0, 3, 1).reshape(I * R, I * R)


def rowkron_times(Phi, K, chunk=2000):
    """(Phi ⊙ Phi) K, built in row chunks so that Phi ⊙ Phi (N x I^2) is never stored whole."""
    out = np.empty((Phi.shape[0], K.shape[1]))
    for s in range(0, Phi.shape[0], chunk):
        out[s : s + chunk] = dotkron(Phi[s : s + chunk], Phi[s : s + chunk]) @ K
    return out


def rowkron_T_times(Phi, H, chunk=2000):
    """(Phi ⊙ Phi)^T H, built in row chunks."""
    out = np.zeros((Phi.shape[1] ** 2, H.shape[1]))
    for s in range(0, Phi.shape[0], chunk):
        out += dotkron(Phi[s : s + chunk], Phi[s : s + chunk]).T @ H[s : s + chunk]
    return out


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
        use_sigma=True,
        anneal_sweeps=20,
    ):
        """Fit the model.

        X, y      : training inputs (N, D) and targets (N,)
        M, R      : feature dimension per input and initial (maximum) rank
        a0, b0    : Gamma prior of the noise precision tau
        c0, d0    : Gamma prior of the rank precisions lambda
        g0, h0    : Gamma prior of the feature precisions delta
        rank_tol  : rank components explaining less than this % of the variance are pruned
        use_sigma : include R{Sigma_k} in the second moments W_k ⊗ W_k + R{Sigma_k}.
                    False gives the deterministic ALS updates of the factor means.
        anneal_sweeps : the second moments use W_k ⊗ W_k + beta R{Sigma_k}, with beta growing
                    linearly from 0 (sweep 0: deterministic ALS) to 1 at sweep `anneal_sweeps`;
                    after that the updates are the full mean-field updates. This keeps the
                    variance of the product of the other factors from dominating the first
                    updates, while the factor means are still random. 0 = full updates from the start.

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
        WSigma_D = [1e-5 * np.kron(np.eye(R), np.eye(I)) for _ in range(D)]

        Phi = features(X, M)

        def sigma_weight(it):
            # beta: weight of R{Sigma_k} in the second moments (0 -> 1 over anneal_sweeps)
            if not use_sigma:
                return 0.0
            return 1.0 if anneal_sweeps <= 0 else min(1.0, it / anneal_sweeps)

        def second_moment(k, beta):
            # (Phi_k ⊙ Phi_k) (W_k ⊗ W_k + beta R{Sigma_k}): N x R^2, row n is the R x R matrix
            # E[(phi_n^T W_k)^T (phi_n^T W_k)] (row-major, index r*R + r')
            K = np.kron(W_D[k], W_D[k])
            if beta > 0:
                K = K + beta * rearrange_sigma(WSigma_D[k], I, R)
            return rowkron_times(Phi[k], K)

        LB = np.zeros(max_iter)
        rank_history = []

        # Per-factor moments: first moments Phi_k W_k (N x R), second moments T_k (N x R^2)
        beta = sigma_weight(0)
        PW = [Phi[k] @ W_D[k] for k in range(D)]
        T = [second_moment(k, beta) for k in range(D)]

        pbar = trange(max_iter, desc="Training", leave=True, disable=not verbose)
        for it in pbar:
            if sigma_weight(it) != beta:  # annealing: recompute the second moments with the new beta
                beta = sigma_weight(it)
                T = [second_moment(k, beta) for k in range(D)]

            # ---- Factor updates q(W^(d)) ----
            for d in range(D):
                # Hadamard products over the other factors k != d
                E_g = np.ones((N, R))        # ⊛_{k != d} Phi_k W_k
                E_ggT = np.ones((N, R * R))  # ⊛_{k != d} (Phi_k ⊙ Phi_k)(W_k ⊗ W_k + R{Sigma_k})
                for k in range(D):
                    if k != d:
                        E_g = E_g * PW[k]
                        E_ggT = E_ggT * T[k]

                E_GtG = rearrange_gram(rowkron_T_times(Phi[d], E_ggT), I, R)              # E[G^T G]
                E_Gty = (Phi[d].T @ (E_g * np.reshape(y, (-1, 1)))).reshape(-1, 1, order="F")  # E[G]^T y

                A = tau * E_GtG + np.kron(lambda_R * np.eye(R), delta[d] * np.eye(I))
                
                try:  # inverse via eigendecomposition, with tiny eigenvalues clipped
                    eigvals, eigvecs = np.linalg.eigh(A)
                    eigvals = np.maximum(eigvals, 1e-10 * np.maximum(eigvals.max(), 1e-10))
                    WSigma_D[d] = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                except np.linalg.LinAlgError:
                    try:
                        WSigma_D[d] = np.linalg.pinv(A)
                    except np.linalg.LinAlgError:
                        raise RuntimeError(f"INVERSION_FAILED at core d={d}, iter it={it}")
                W_D[d] = np.reshape(tau * WSigma_D[d] @ E_Gty, (I, R), order="F")

                # moments of the updated factor
                PW[d] = Phi[d] @ W_D[d]
                T[d] = second_moment(d, beta)

            # ---- Feature precisions q(delta_d) ----
            if delta_update:
                for d in range(D):
                    mtemp = np.diag(W_D[d] @ (lambda_R * np.eye(R)) @ W_D[d].T)
                    vtemp = np.diag(
                        np.reshape(
                            rearrange_sigma(WSigma_D[d], I, R)
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
                            @ rearrange_sigma(WSigma_D[d], I, R),
                            (R, R),
                        )
                    )
                    d_N += mtemp + vtemp
                d_N = d0 + (0.5 * d_N)
                lambda_R = c_N / d_N
                lambda_R[lambda_R < 1e-5] = 1e-5

            # ---- Noise precision q(tau) ----
            # Hadamard products over all factors: E[f] per rank component, E[f^2] per rank pair
            E_f = np.ones((N, R))         # ⊛_k Phi_k W_k
            E_ff = np.ones((N, R * R))    # ⊛_k (Phi_k ⊙ Phi_k)(W_k ⊗ W_k + R{Sigma_k})
            for k in range(D):
                E_f = E_f * PW[k]
                E_ff = E_ff * T[k]
            # E||y - f||^2 = y^T y - 2 y^T E[f] + sum_n E[f_n^2]
            err = y @ y - 2 * y @ np.sum(E_f, axis=1) + np.sum(E_ff)
            if precision_update:
                a_N = a0 + (N / 2)
                b_N = b0 + (0.5 * err)
            else:
                a_N, b_N = a0, b0
            tau = a_N / b_N

            # ---- Variational lower bound ----
            LB[it] = self._lower_bound(
                W_D, WSigma_D, lambda_R, delta, err, a_N, b_N, b0, c_N, d_N, d0, g_N, h_N, h0, I, R
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
                    # drop the removed components from the factor moments: their columns of
                    # Phi_k W_k, and every rank pair (r, r') that involves one of them
                    kept = ~np.isin(np.arange(R), removed)
                    PW = [P[:, kept] for P in PW]
                    T = [T_k[:, np.outer(kept, kept).ravel()] for T_k in T]
                    c0 = c0[:rankest]
                    d0 = d0[:rankest]
                    R = rankest
            rank_history.append(R)

            # ---- Convergence check (from 5 iterations after the annealing on) ----
            rel_change = (LB[it] - LB[it - 1]) / abs(LB[it - 1]) if it > 0 else np.nan
            pbar.set_postfix({"RelChan": f"{rel_change:.2e}", "R": rankest})
            if it > anneal_sweeps + 5 and abs(rel_change) < lower_bound_tol:
                if verbose:
                    print("======= Converged =======")
                break

        # Store the posterior
        self.W_D, self.V, self.a, self.b = W_D, WSigma_D, a_N, b_N
        self.lambda_R, self.delta, self.tau = lambda_R, delta, tau  # posterior means of the precisions
        self.M = M
        self.use_sigma = use_sigma
        self.rank = R
        self.rank_history = rank_history
        self.lower_bound = LB[:it+1]
        self.M_eff = np.array(
            [np.sum((np.diag(W @ W.T) / np.sum(np.diag(W @ W.T))) * 100 >= 0.25) for W in W_D]
        )
        return self

    @staticmethod
    def _lower_bound(W_D, WSigma_D, lambda_R, delta, expected_sq_err, a_N, b_N, b0, c_N, d_N, d0, g_N, h_N, h0, I, R):
        """Variational lower bound (up to constants); only used to check convergence."""
        t1 = -0.5 * (a_N / b_N) * expected_sq_err  # expected_sq_err = E||y - f||^2
        t2 = 0.0
        for d in range(len(W_D)):
            w = W_D[d].ravel(order="F")
            t2 += np.kron(lambda_R, np.ravel(delta[d])) @ (w**2 + np.diag(WSigma_D[d]))
        t2 = -0.5 * t2
        t3 = 0.5 * np.sum([np.linalg.slogdet(S)[1] for S in WSigma_D])
        t4 = np.sum(np.multiply(c_N, (1 - safelog(d_N) - (d0 / d_N))))
        t5 = sum(
            np.sum(
                np.array(g_N[d]) * (1 - safelog(np.array(h_N[d])) - (h0 / np.array(h_N[d])))
            )
            for d in range(len(g_N))
        )
        t6 = a_N * (1 - safelog(b_N) - (b0 / b_N))
        return t1 + t2 + t3 + t4 + t5 + t6

    def predict(self, X):
        """Mean and std of the Student's-t predictive of Eq. (3.41):
        """
        Phi = features(X, self.M)
        N, D = X.shape[0], len(self.W_D)
        PW = [Phi[d] @ self.W_D[d] for d in range(D)]  # W_d^T phi_d(x_i) for every i, (N, R)
        mean = np.sum(np.prod(PW, axis=0), axis=1)

        quad = np.zeros(N)
        for d in range(D):
            h = np.prod([PW[k] for k in range(D) if k != d], axis=0)  # ⊛_{k≠d}, (N, R)
            g = (h[:, :, None] * Phi[d][:, None, :]).reshape(N, -1)   # h ⊗ phi_d, (N, I R)
            quad += np.sum((g @ self.V[d]) * g, axis=1)

        S_inv = self.b / self.a + quad
        nu = 2 * self.a
        return mean, np.sqrt(nu / (nu - 2) * S_inv)

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