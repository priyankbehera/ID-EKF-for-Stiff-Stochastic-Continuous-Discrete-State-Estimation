# filters.py
# Conventional (non–square-root) continuous–discrete filters:
#   - CDEKF : matrix MDE time update  P' = J P + P J^T + GQcG^T,  x' = f(x)
#   - CDUKF : same P' as EKF, but x' = E[f(X)] via UKF sigma points (paper weights)
#   - CDCKF : same P' as EKF, but x' = E[f(X)] via cubature points

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple
from scipy.integrate import solve_ivp

# ----------------------------- Utilities -----------------------------------

def symmetrize(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + P.T)

def ensure_psd(P: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    P = symmetrize(P.astype(float, copy=False))
    try:
        np.linalg.cholesky(P + eps * np.eye(P.shape[0]))
        return P
    except np.linalg.LinAlgError:
        return P + (10 * eps) * np.eye(P.shape[0])

def safe_cholesky(A: np.ndarray, eps: float = 1e-12, max_tries: int = 6) -> np.ndarray:
    A = ensure_psd(A)
    I = np.eye(A.shape[0])
    for i in range(max_tries):
        try:
            return np.linalg.cholesky(A + (10.0**i) * eps * I)
        except np.linalg.LinAlgError:
            pass
    # Fallback: eigen clip
    w, V = np.linalg.eigh(symmetrize(A))
    w = np.clip(w, eps, None)
    return V @ np.diag(np.sqrt(w))

# ----------------------------- Model carrier --------------------------------

@dataclass
class ContinuousModel:
    f:   Callable[[float, np.ndarray], np.ndarray]
    J:   Callable[[float, np.ndarray], np.ndarray]
    G:   Callable[[float], np.ndarray]
    Qc:  Callable[[float], np.ndarray]

# --------------------------- Sigma / Cubature sets --------------------------

def ukf_sigma_points_paper(x: np.ndarray, P: np.ndarray):
    """
    Paper-style UKF sigma points (Automatica'18 style):
      Xi_0 = x
      Xi_i = x + sqrt(3)*S e_i
      Xi_{i+n} = x - sqrt(3)*S e_i
    Weights: wm = [(3-n)/3, 1/6, ..., 1/6]
    """
    n = x.size
    S = safe_cholesky(P)
    Xi = np.zeros((2*n + 1, n), dtype=float)
    Xi[0] = x
    for i in range(n):
        col = S[:, i]
        Xi[1 + i]     = x + np.sqrt(3.0) * col
        Xi[1 + n + i] = x - np.sqrt(3.0) * col
    wm = np.full(2*n + 1, 1.0/6.0, dtype=float)
    wm[0] = (3.0 - n) / 3.0
    return Xi, wm

def ckf_cubature_points(x: np.ndarray, P: np.ndarray):
    """
    3rd-degree spherical-radial cubature points:
      Xi_i = x + sqrt(n)*S e_i,  Xi_{i+n} = x - sqrt(n)*S e_i,  weights = 1/(2n)
    """
    n = x.size
    S = safe_cholesky(P)
    Xi = np.zeros((2*n, n), dtype=float)
    for i in range(n):
        col = S[:, i]
        Xi[i]     = x + np.sqrt(n) * col
        Xi[n + i] = x - np.sqrt(n) * col
    W = np.full(2*n, 1.0/(2*n), dtype=float)
    return Xi, W

# --------------------------- CD-EKF (matrix MDE) ----------------------------

class CDEKF:
    def __init__(self, cm: ContinuousModel, h_fun, H_fun, R,
                 rtol: float = 1e-6, atol: float = 1e-9, max_step: float = 0.1, method: str = "BDF"):
        self.cm = cm
        self.h  = h_fun
        self.H  = H_fun
        self.R  = np.asarray(R, dtype=float)
        self.rtol, self.atol, self.max_step, self.method = rtol, atol, max_step, method

    def _integrate_mde(self, t0: float, t1: float, x0: np.ndarray, P0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = x0.size
        def rhs(t, y):
            x = y[:n]
            P = y[n:].reshape(n, n)
            Fx = self.cm.f(t, x)
            Jx = self.cm.J(t, x)
            dP = Jx @ P + P @ Jx.T + self.cm.G(t) @ self.cm.Qc(t) @ self.cm.G(t).T
            return np.hstack([Fx, dP.ravel()])
        y0 = np.hstack([x0, P0.ravel()])
        sol = solve_ivp(rhs, (t0, t1), y0, method=self.method,
                        rtol=self.rtol, atol=self.atol, max_step=self.max_step)
        x1 = sol.y[:n, -1]
        P1 = sol.y[n:, -1].reshape(n, n)
        return x1, ensure_psd(P1)

    def predict(self, t_prev: float, t_curr: float, x_prev: np.ndarray, P_prev: np.ndarray):
        return self._integrate_mde(t_prev, t_curr, x_prev, P_prev)

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray):
        H = self.H(x_pred)
        y = z - self.h(x_pred)
        S = ensure_psd(H @ P_pred @ H.T + self.R)
        K = np.linalg.solve(S, (H @ P_pred.T)).T  # P H^T S^{-1}
        x_upd = x_pred + K @ y
        P_upd = ensure_psd(P_pred - K @ S @ K.T)
        return x_upd, P_upd, y, S

# ----------------------- CD-UKF (matrix MDE + UKF mean) ---------------------

class CDUKF:
    def __init__(self, cm: ContinuousModel, h_fun, R,
                 rtol: float = 1e-6, atol: float = 1e-9, max_step: float = 0.1, method: str = "BDF"):
        self.cm = cm
        self.h  = h_fun
        self.R  = np.asarray(R, dtype=float)
        self.rtol, self.atol, self.max_step, self.method = rtol, atol, max_step, method

    def _integrate_mde_ukfmean(self, t0: float, t1: float, x0: np.ndarray, P0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = x0.size
        def rhs(t, y):
            x = y[:n]
            P = y[n:].reshape(n, n)
            Xi, wm = ukf_sigma_points_paper(x, P)
            Fvals  = np.array([self.cm.f(t, xi) for xi in Xi])
            xdot   = np.sum(wm[:, None] * Fvals, axis=0)
            Jx = self.cm.J(t, x)
            dP = Jx @ P + P @ Jx.T + self.cm.G(t) @ self.cm.Qc(t) @ self.cm.G(t).T
            return np.hstack([xdot, dP.ravel()])
        y0 = np.hstack([x0, P0.ravel()])
        sol = solve_ivp(rhs, (t0, t1), y0, method=self.method,
                        rtol=self.rtol, atol=self.atol, max_step=self.max_step)
        x1 = sol.y[:n, -1]
        P1 = sol.y[n:, -1].reshape(n, n)
        return x1, ensure_psd(P1)

    def predict(self, t_prev: float, t_curr: float, x_prev: np.ndarray, P_prev: np.ndarray):
        return self._integrate_mde_ukfmean(t_prev, t_curr, x_prev, P_prev)

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray):
        Xi, wm = ukf_sigma_points_paper(x_pred, P_pred)
        Zsig = np.array([self.h(xi) for xi in Xi])
        z_pred = np.sum(wm[:, None] * Zsig, axis=0)
        S = self.R.copy()
        Pxz = np.zeros((x_pred.size, z_pred.size))
        for i in range(Xi.shape[0]):
            dz = (Zsig[i] - z_pred).reshape(-1, 1)
            dx = (Xi[i]   - x_pred).reshape(-1, 1)
            wi = wm[i]
            S  += wi * (dz @ dz.T)
            Pxz += wi * (dx @ dz.T)
        S = ensure_psd(S)
        K = np.linalg.solve(S, Pxz.T).T
        y = z - z_pred
        x_upd = x_pred + K @ y
        P_upd = ensure_psd(P_pred - K @ S @ K.T)
        return x_upd, P_upd, y, S

# ----------------------- CD-CKF (matrix MDE + CKF mean) ---------------------

class CDCKF:
    def __init__(self, cm: ContinuousModel, h_fun, R,
                 rtol: float = 1e-6, atol: float = 1e-9, max_step: float = 0.1, method: str = "BDF"):
        self.cm = cm
        self.h  = h_fun
        self.R  = np.asarray(R, dtype=float)
        self.rtol, self.atol, self.max_step, self.method = rtol, atol, max_step, method

    def _integrate_mde_ckfmean(self, t0: float, t1: float, x0: np.ndarray, P0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = x0.size
        def rhs(t, y):
            x = y[:n]
            P = y[n:].reshape(n, n)
            Xi, W = ckf_cubature_points(x, P)
            Fvals = np.array([self.cm.f(t, xi) for xi in Xi])
            xdot  = np.sum(W[:, None] * Fvals, axis=0)
            Jx = self.cm.J(t, x)
            dP = Jx @ P + P @ Jx.T + self.cm.G(t) @ self.cm.Qc(t) @ self.cm.G(t).T
            return np.hstack([xdot, dP.ravel()])
        y0 = np.hstack([x0, P0.ravel()])
        sol = solve_ivp(rhs, (t0, t1), y0, method=self.method,
                        rtol=self.rtol, atol=self.atol, max_step=self.max_step)
        x1 = sol.y[:n, -1]
        P1 = sol.y[n:, -1].reshape(n, n)
        return x1, ensure_psd(P1)

    def predict(self, t_prev: float, t_curr: float, x_prev: np.ndarray, P_prev: np.ndarray):
        return self._integrate_mde_ckfmean(t_prev, t_curr, x_prev, P_prev)

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray):
        Xi, W = ckf_cubature_points(x_pred, P_pred)
        Zsig = np.array([self.h(xi) for xi in Xi])
        z_pred = np.sum(W[:, None] * Zsig, axis=0)
        S = self.R.copy()
        Pxz = np.zeros((x_pred.size, z_pred.size))
        for i in range(Xi.shape[0]):
            dz = (Zsig[i] - z_pred).reshape(-1, 1)
            dx = (Xi[i]   - x_pred).reshape(-1, 1)
            wi = W[i]
            S  += wi * (dz @ dz.T)
            Pxz += wi * (dx @ dz.T)
        S = ensure_psd(S)
        K = np.linalg.solve(S, Pxz.T).T
        y = z - z_pred
        x_upd = x_pred + K @ y
        P_upd = ensure_psd(P_pred - K @ S @ K.T)
        return x_upd, P_upd, y, S
    
# ---- CDIDEKF: continuous–discrete Information-form EKF (uses IDKalman ops) ----
# --- add these imports near the top of filters.py, with your other imports ---
from IDKalman.COVtoINF import cov_to_inf
from IDKalman.INFtoCOV import inf_to_cov
from IDKalman.Mupdate import mupdate

class CDIDEKF:
    """
    Continuous–discrete IDEKF:
      - Time update: matrix MDE in covariance form (same as EKF) using solve_ivp
      - Measurement update: information-form update via IDKalman.mupdate
    """
    def __init__(self, cm: ContinuousModel, h_fun, H_fun, R,
                 rtol: float = 1e-12, atol: float = 1e-12,
                 max_step: float = 1e-1, method: str = "Radau"):
        self.cm = cm
        self.h  = h_fun
        self.H  = H_fun
        self.R  = np.asarray(R, dtype=float)
        self.rtol, self.atol = rtol, atol
        self.max_step, self.method = max_step, method

    def _integrate_mde(self, t0: float, t1: float, x0: np.ndarray, P0: np.ndarray):
        n = x0.size
        def rhs(t, y):
            x = y[:n]
            P = y[n:].reshape(n, n)
            Fx = self.cm.f(t, x)
            Jx = self.cm.J(t, x)
            Gt = self.cm.G(t)
            Qc = self.cm.Qc(t)
            dP = Jx @ P + P @ Jx.T + Gt @ Qc @ Gt.T
            return np.hstack([Fx, dP.ravel()])
        y0 = np.hstack([x0, P0.ravel()])
        sol = solve_ivp(rhs, (t0, t1), y0, method=self.method,
                        rtol=self.rtol, atol=self.atol, max_step=self.max_step)
        x1 = sol.y[:n, -1]
        P1 = sol.y[n:, -1].reshape(n, n)
        return x1, ensure_psd(P1)

    def predict(self, t_prev: float, t_curr: float, x_prev: np.ndarray, P_prev: np.ndarray):
        return self._integrate_mde(t_prev, t_curr, x_prev, P_prev)

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray):
        # Build linearization at x_pred
        Hk = self.H(x_pred)
        # Convert prior covariance to information parameters
        B, V, _ = cov_to_inf(P_pred, P_pred.shape[0])
        u = x_pred.reshape(-1, 1)

        # Wrap nonlinear measurement if needed
        h_wrapped = None
        if callable(self.h):
            def h_wrapped(u_vec):
                return np.asarray(self.h(np.asarray(u_vec).reshape(-1))).reshape(-1, 1)

        # Information-form measurement update (IDKalman)
        z_col = np.asarray(z).reshape(-1, 1)
        u_post, V_post, B_post, _, _ = mupdate(1, z_col, u, B, V, self.R, Hk, h_wrapped)

        # Convert back to mean/covariance
        x_upd = np.asarray(u_post).reshape(-1)
        P_upd = ensure_psd(inf_to_cov(np.asarray(V_post).reshape(-1),
                                      np.asarray(B_post), x_upd.size))

        # Provide innovation y and S for logging (computed in covariance form)
        zhat = (self.h(x_pred) if callable(self.h) else (Hk @ x_pred.reshape(-1,1)).ravel())
        y = (np.asarray(z).ravel() - np.asarray(zhat).ravel())
        S = ensure_psd(Hk @ P_pred @ Hk.T + self.R)

        return x_upd, P_upd, y, S
