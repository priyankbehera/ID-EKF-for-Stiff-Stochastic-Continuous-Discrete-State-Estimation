# filters.py
# Conventional (non–square-root) continuous–discrete filters:
#   - CDEKF : matrix MDE time update (x' = f, P' = J P + P J^T + GQcG^T)
#   - CDUKF : matrix MDE for P, BUT x' = E[f(X)] via UKF sigma points (paper weights)
#   - CDCKF : matrix MDE for P, x' = E[f(X)] via cubature points

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
    """
    Continuous–discrete model for CD filters.

    f(t,x): drift, returns (n,)
    J(t,x): Jacobian of f wrt x, returns (n,n)
    G(t): diffusion gain, returns (n,q)
    Qc(t): continuous-time noise covariance, returns (q,q)
    """
    f: Callable[[float, np.ndarray], np.ndarray]
    J: Callable[[float, np.ndarray], np.ndarray]
    G: Callable[[float], np.ndarray]
    Qc: Callable[[float], np.ndarray]

# --------------------------- Sigma / Cubature sets --------------------------
# (Both use the Cholesky of P as the scale matrix.)

def ukf_sigma_points_paper(x: np.ndarray, P: np.ndarray):
    """
    Paper-style UKF sigma points (no alpha/kappa):
      Xi_0 = x
      Xi_i = x + sqrt(3) * S e_i
      Xi_{i+n} = x - sqrt(3) * S e_i
    Weights:
      wm = [(3-n)/3, 1/6, ..., 1/6]  (length 2n+1)
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
    Third-degree spherical-radial cubature points:
      Xi_i     = x + sqrt(n) * S e_i
      Xi_{i+n} = x - sqrt(n) * S e_i
    Equal weights 1/(2n).
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
                 rtol: float = 1e-12, atol: float = 1e-12, max_step: float = 1e-1):
        self.cm = cm
        self.h  = h_fun
        self.H  = H_fun
        self.R  = np.asarray(R, dtype=float)
        self.rtol, self.atol, self.max_step = rtol, atol, max_step

    def _integrate_mde(self, t0: float, t1: float, x0: np.ndarray, P0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        sol = solve_ivp(rhs, (t0, t1), y0, method="Radau",
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
        # K = P H^T S^{-1} (use solve for stability)
        K = np.linalg.solve(S, (H @ P_pred.T)).T
        x_upd = x_pred + K @ y
        P_upd = ensure_psd(P_pred - K @ S @ K.T)
        return x_upd, P_upd, y, S

# ----------------------- CD-UKF (matrix MDE + UKF mean) ---------------------

class CDUKF:
    """
    Time update: matrix MDE for P; mean derivative x' = E[f(X)] via UKF sigma points.
    Measurement: unscented transform with the same sigma points/weights.
    """
    def __init__(self, cm: ContinuousModel, h_fun, R,
                 rtol: float = 1e-12, atol: float = 1e-12, max_step: float = 1e-1):
        self.cm = cm
        self.h  = h_fun
        self.R  = np.asarray(R, dtype=float)
        self.rtol, self.atol, self.max_step = rtol, atol, max_step

    def _integrate_mde_ukfmean(self, t0: float, t1: float, x0: np.ndarray, P0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = x0.size

        def rhs(t, y):
            x = y[:n]
            P = y[n:].reshape(n, n)

            # UKF mean derivative (sigma-point average)
            Xi, wm = ukf_sigma_points_paper(x, P)
            Fvals = np.array([self.cm.f(t, xi) for xi in Xi])    # (2n+1, n)
            xdot  = np.sum(wm[:, None] * Fvals, axis=0)

            # Matrix MDE for P
            Jx = self.cm.J(t, x)
            Gt = self.cm.G(t)
            Qc = self.cm.Qc(t)
            dP = Jx @ P + P @ Jx.T + Gt @ Qc @ Gt.T

            return np.hstack([xdot, dP.ravel()])

        y0 = np.hstack([x0, P0.ravel()])
        sol = solve_ivp(rhs, (t0, t1), y0, method="Radau",
                        rtol=self.rtol, atol=self.atol, max_step=self.max_step)
        x1 = sol.y[:n, -1]
        P1 = sol.y[n:, -1].reshape(n, n)
        return x1, ensure_psd(P1)

    def predict(self, t_prev: float, t_curr: float, x_prev: np.ndarray, P_prev: np.ndarray):
        return self._integrate_mde_ukfmean(t_prev, t_curr, x_prev, P_prev)

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray):
        # Unscented measurement transform
        Xi, wm = ukf_sigma_points_paper(x_pred, P_pred)
        Zsig = np.array([self.h(xi) for xi in Xi])                 # (2n+1, m)
        z_pred = np.sum(wm[:, None] * Zsig, axis=0)                # (m,)

        # Innovation covariance and cross-covariance
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
    """
    Time update: matrix MDE for P; mean derivative x' = E[f(X)] via cubature points.
    Measurement: CKF (cubature) transform.
    """
    def __init__(self, cm: ContinuousModel, h_fun, R,
                 rtol: float = 1e-12, atol: float = 1e-12, max_step: float = 1e-1):
        self.cm = cm
        self.h  = h_fun
        self.R  = np.asarray(R, dtype=float)
        self.rtol, self.atol, self.max_step = rtol, atol, max_step

    def _integrate_mde_ckfmean(self, t0: float, t1: float, x0: np.ndarray, P0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = x0.size

        def rhs(t, y):
            x = y[:n]
            P = y[n:].reshape(n, n)

            # CKF mean derivative (cubature average)
            Xi, W = ckf_cubature_points(x, P)
            Fvals = np.array([self.cm.f(t, xi) for xi in Xi])     # (2n, n)
            xdot  = np.sum(W[:, None] * Fvals, axis=0)

            # Matrix MDE for P (same as EKF/UKF)
            Jx = self.cm.J(t, x)
            Gt = self.cm.G(t)
            Qc = self.cm.Qc(t)
            dP = Jx @ P + P @ Jx.T + Gt @ Qc @ Gt.T

            return np.hstack([xdot, dP.ravel()])

        y0 = np.hstack([x0, P0.ravel()])
        sol = solve_ivp(rhs, (t0, t1), y0, method="Radau",
                        rtol=self.rtol, atol=self.atol, max_step=self.max_step)
        x1 = sol.y[:n, -1]
        P1 = sol.y[n:, -1].reshape(n, n)
        return x1, ensure_psd(P1)

    def predict(self, t_prev: float, t_curr: float, x_prev: np.ndarray, P_prev: np.ndarray):
        return self._integrate_mde_ckfmean(t_prev, t_curr, x_prev, P_prev)

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray):
        # Cubature measurement transform
        Xi, W = ckf_cubature_points(x_pred, P_pred)
        Zsig = np.array([self.h(xi) for xi in Xi])                 # (2n, m)
        z_pred = np.sum(W[:, None] * Zsig, axis=0)                 # (m,)

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
