# filters.py
# Continuous–discrete filters as in the paper:
#   - CDEKF: matrix MDE time update (x' = f, P' = J P + P J^T + GQcG^T)
#   - CDUKF: matrix MDE time update, BUT mean derivative E[f(X)] via UKF sigma points (paper weights)
#   - CDCKF: square-root MDE (S' form) time update, mean derivative via cubature points

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple
from scipy.integrate import solve_ivp

# ----------------------------- Utilities -----------------------------------

def symmetrize(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + P.T)

def ensure_psd(P: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    P = symmetrize(P)
    try:
        np.linalg.cholesky(P + eps * np.eye(P.shape[0]))
        return P
    except np.linalg.LinAlgError:
        return P + (10 * eps) * np.eye(P.shape[0])

def safe_cholesky(A: np.ndarray, eps: float = 1e-12, max_tries: int = 6) -> np.ndarray:
    A = symmetrize(A)
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

def phi_operator(M: np.ndarray) -> np.ndarray:
    """Φ(M) operator used in SR covariance ODEs: lower-triangular part + 1/2 diag."""
    L = np.tril(M, -1)
    D = np.diag(np.diag(M))
    return L + 0.5 * D

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

def ukf_sigma_points_paper(x: np.ndarray, S: np.ndarray):
    """
    Paper-style UKF sigma points (no alpha/kappa):
      Xi_0 = x
      Xi_i = x + sqrt(3) S e_i
      Xi_{i+n} = x - sqrt(3) S e_i
    Weights:
      wm = [(3-n)/3, 1/6, ..., 1/6]  (length 2n+1)
    """
    n = x.size
    Xi = np.zeros((2*n + 1, n))
    Xi[0] = x
    for i in range(n):
        col = S[:, i]
        Xi[1 + i]     = x + np.sqrt(3.0) * col
        Xi[1 + n + i] = x - np.sqrt(3.0) * col
    wm = np.full(2*n + 1, 1.0/6.0)
    wm[0] = (3.0 - n) / 3.0
    return Xi, wm

def ckf_cubature_points(x: np.ndarray, S: np.ndarray):
    """
    Third-degree spherical-radial cubature points:
      Xi_i = x + sqrt(n) S e_i
      Xi_{i+n} = x - sqrt(n) S e_i
    Equal weights 1/(2n).
    """
    n = x.size
    Xi = np.zeros((2*n, n))
    for i in range(n):
        col = S[:, i]
        Xi[i]     = x + np.sqrt(n) * col
        Xi[n + i] = x - np.sqrt(n) * col
    W = np.full(2*n, 1.0/(2*n))
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
        K = np.linalg.solve(S, (H @ P_pred.T)).T  # P H^T S^{-1}
        x_upd = x_pred + K @ y
        P_upd = ensure_psd(P_pred - K @ S @ K.T)
        return x_upd, P_upd, y, S

# ----------------------- CD-UKF (matrix MDE + UKF mean) ---------------------

class CDUKF:
    """
    Time update uses matrix MDE for P, but x' = E[f(X)] via UKF sigma points
    with the paper's weights/scaling. Measurement uses the same UKF transform.
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
            S = safe_cholesky(ensure_psd(P))
            Xi, wm = ukf_sigma_points_paper(x, S)
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
        Sx = safe_cholesky(ensure_psd(P_pred))
        Xi, wm = ukf_sigma_points_paper(x_pred, Sx)
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

# ----------------------- CD-CKF (square-root MDE + CKF mean) ----------------

class CDCKF:
    """
    Time update uses SR-MDE for P (evolve S with S' = S Φ(M)),
    where M = S^{-1}(J+J^T+GQG^T)S^{-T}. Mean derivative via cubature points.
    Measurement uses the CKF transform.
    """
    def __init__(self, cm: ContinuousModel, h_fun, R,
                 rtol: float = 1e-12, atol: float = 1e-12, max_step: float = 1e-1):
        self.cm = cm
        self.h  = h_fun
        self.R  = np.asarray(R, dtype=float)
        self.rtol, self.atol, self.max_step = rtol, atol, max_step

    def _integrate_sr_mde_ckfmean(self, t0: float, t1: float, x0: np.ndarray, P0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = x0.size
        S0 = safe_cholesky(ensure_psd(P0))

        def rhs(t, y):
            x = y[:n]
            S = y[n:].reshape(n, n)

            # CKF mean derivative (cubature average)
            Xi, W = ckf_cubature_points(x, S)
            Fvals = np.array([self.cm.f(t, xi) for xi in Xi])     # (2n, n)
            xdot  = np.sum(W[:, None] * Fvals, axis=0)

            # SR-MDE for S
            Jx  = self.cm.J(t, x)
            GQG = self.cm.G(t) @ self.cm.Qc(t) @ self.cm.G(t).T
            Sinv = np.linalg.inv(S)
            M   = Sinv @ (Jx + Jx.T + GQG) @ Sinv.T
            Sdot = S @ phi_operator(M)

            return np.hstack([xdot, Sdot.ravel()])

        y0 = np.hstack([x0, S0.ravel()])
        sol = solve_ivp(rhs, (t0, t1), y0, method="Radau",
                        rtol=self.rtol, atol=self.atol, max_step=self.max_step)
        x1 = sol.y[:n, -1]
        S1 = sol.y[n:, -1].reshape(n, n)
        P1 = ensure_psd(S1 @ S1.T)
        return x1, P1

    def predict(self, t_prev: float, t_curr: float, x_prev: np.ndarray, P_prev: np.ndarray):
        return self._integrate_sr_mde_ckfmean(t_prev, t_curr, x_prev, P_prev)

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray):
        Sx = safe_cholesky(ensure_psd(P_pred))
        Xi, W = ckf_cubature_points(x_pred, Sx)
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
