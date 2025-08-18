
import numpy as np

def armse(trajs_true: np.ndarray, trajs_est: np.ndarray):
    err = trajs_est - trajs_true
    mse = np.mean(np.sum(err**2, axis=2))
    return float(np.sqrt(mse))

def nees(errors: np.ndarray, Ps: np.ndarray):
    N, T, n = errors.shape
    out = np.zeros((N, T))
    for i in range(N):
        for k in range(T):
            e = errors[i, k]
            P = Ps[i, k]
            try:
                out[i, k] = float(e.T @ np.linalg.solve(P, e))
            except np.linalg.LinAlgError:
                out[i, k] = np.nan
    return out

def nis(innovs: np.ndarray, Ss: np.ndarray):
    N, T, m = innovs.shape
    out = np.zeros((N, T))
    for i in range(N):
        for k in range(T):
            v = innovs[i, k]
            S = Ss[i, k]
            try:
                out[i, k] = float(v.T @ np.linalg.solve(S, v))
            except np.linalg.LinAlgError:
                out[i, k] = np.nan
    return out

def divergence_flags(Ps: np.ndarray, thresh: float = 1e12):
    N, T, n, _ = Ps.shape
    flags = np.zeros(N, dtype=bool)
    for i in range(N):
        bad = False
        for k in range(T):
            P = 0.5*(Ps[i,k] + Ps[i,k].T)
            try:
                np.linalg.cholesky(P + 1e-12*np.eye(n))
            except np.linalg.LinAlgError:
                bad = True; break
            if np.any(np.diag(P) > thresh):
                bad = True; break
        flags[i] = bad
    return flags
