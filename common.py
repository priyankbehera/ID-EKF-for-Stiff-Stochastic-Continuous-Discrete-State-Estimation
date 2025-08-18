
import numpy as np
import time

def cholesky_psd(A):
    try:
        L = np.linalg.cholesky(0.5*(A + A.T))
        return L, True
    except np.linalg.LinAlgError:
        return None, False

def is_psd(A, tol=1e-10):
    A = 0.5*(A + A.T)
    try:
        np.linalg.cholesky(A + tol*np.eye(A.shape[0]))
        return True
    except np.linalg.LinAlgError:
        return False

def timer():
    class T:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *exc):
            self.dt = time.perf_counter() - self.t0
    return T()

def set_seed(seed: int):
    rng = np.random.default_rng(seed)
    return rng

def chi2_bounds_mean_nees(alpha: float, state_dim: int, m_runs: int):
    import scipy.stats as st
    dof = m_runs * state_dim
    lower = st.chi2.ppf(alpha/2, dof) / m_runs
    upper = st.chi2.ppf(1 - alpha/2, dof) / m_runs
    return lower, upper
