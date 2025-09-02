
import numpy as np

def symmetrize(P):
    return 0.5*(P + P.T)

def ensure_psd(P, eps=1e-10):
    P = symmetrize(P)
    try:
        np.linalg.cholesky(P + eps*np.eye(P.shape[0]))
        return P
    except np.linalg.LinAlgError:
        return P + (10*eps)*np.eye(P.shape[0])

def safe_cholesky(A, eps=1e-12, max_tries=6):
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

class DiscreteModel:
    def __init__(self, g_fun, F_fun, Qk):
        self.g = g_fun
        self.F = F_fun
        self.Q = Qk

class EKF:
    def __init__(self, dm, h_fun, H_fun, R):
        self.dm = dm
        self.h = h_fun
        self.H = H_fun
        self.R = R

    def predict(self, x, P):
        F = self.dm.F(x)
        x_pred = self.dm.g(x)
        P_pred = ensure_psd(F @ P @ F.T + self.dm.Q)
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z):
        H = self.H(x_pred)
        y = z - self.h(x_pred)
        S = ensure_psd(H @ P_pred @ H.T + self.R)
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_upd = x_pred + K @ y
        P_upd = ensure_psd(P_pred - K @ S @ K.T)
        return x_upd, P_upd, y, S


def unscented_points(x, P, alpha=1e-3, beta=2.0, kappa=0.0):
    n = x.shape[0]
    lam = alpha**2 * (n + kappa) - n
    c = n + lam
    P = ensure_psd(P)
    if not np.all(np.isfinite(P)):
        P = np.eye(n) * 1e-6
    U = safe_cholesky((c) * P)
    Wm = np.full(2*n + 1, 1.0 / (2*c))
    Wc = np.full(2*n + 1, 1.0 / (2*c))
    Wm[0] = lam / c
    Wc[0] = lam / c + (1 - alpha**2 + beta)
    Xi = np.zeros((2*n + 1, n))
    Xi[0] = x
    for i in range(n):
        Xi[1 + i]       = x + U[:, i]
        Xi[1 + n + i]   = x - U[:, i]
    return Xi, Wm, Wc

class UKF:
    def __init__(self, dm, h_fun, R, alpha=1e-3, beta=2.0, kappa=0.0):
        self.dm = dm
        self.h = h_fun
        self.R = R
        self.alpha, self.beta, self.kappa = alpha, beta, kappa

    def predict(self, x, P):
        Xi, Wm, Wc = unscented_points(x, P, self.alpha, self.beta, self.kappa)
        Xi_pred = np.array([self.dm.g(xi) for xi in Xi])
        x_pred = np.sum(Wm[:, None] * Xi_pred, axis=0)
        P_pred = self.dm.Q.copy()
        for i in range(Xi_pred.shape[0]):
            d = (Xi_pred[i] - x_pred).reshape(-1, 1)
            P_pred += Wc[i] * (d @ d.T)
        return x_pred, ensure_psd(P_pred)

    def update(self, x_pred, P_pred, z):
        Xi, Wm, Wc = unscented_points(x_pred, P_pred, self.alpha, self.beta, self.kappa)
        Zsig = np.array([self.h(xi) for xi in Xi])
        z_pred = np.sum(Wm[:, None] * Zsig, axis=0)
        S = self.R.copy()
        Pxz = np.zeros((P_pred.shape[0], z_pred.shape[0]))
        for i in range(Zsig.shape[0]):
            dz = (Zsig[i] - z_pred).reshape(-1,1)
            dx = (Xi[i] - x_pred).reshape(-1,1)
            S += Wc[i] * (dz @ dz.T)
            Pxz += Wc[i] * (dx @ dz.T)
        K = Pxz @ np.linalg.inv(S)
        y = z - z_pred
        x_upd = x_pred + K @ y
        P_upd = ensure_psd(P_pred - K @ S @ K.T)
        return x_upd, P_upd, y, S

def cubature_points(x, P):
    n = x.shape[0]
    P = ensure_psd(P)
    if not np.all(np.isfinite(P)):
        P = np.eye(n) * 1e-6
    U = safe_cholesky(P)  
    Xi = np.zeros((2*n, n))
    for i in range(n):
        Xi[i] = x + np.sqrt(n) * U[:, i]
        Xi[n + i] = x - np.sqrt(n) * U[:, i]
    W = np.full(2*n, 1.0 / (2*n))
    return Xi, W

class CKF:
    def __init__(self, dm, h_fun, R):
        self.dm = dm
        self.h = h_fun
        self.R = R

    def predict(self, x, P):
        Xi, W = cubature_points(x, P)
        Xi_pred = np.array([self.dm.g(xi) for xi in Xi])
        x_pred = np.sum(W[:, None] * Xi_pred, axis=0)
        P_pred = self.dm.Q.copy()
        for i in range(Xi_pred.shape[0]):
            d = (Xi_pred[i] - x_pred).reshape(-1, 1)
            P_pred += W[i] * (d @ d.T)
        return x_pred, ensure_psd(P_pred)

    def update(self, x_pred, P_pred, z):
        Xi, W = cubature_points(x_pred, P_pred)
        Zsig = np.array([self.h(xi) for xi in Xi])
        z_pred = np.sum(W[:, None] * Zsig, axis=0)
        S = self.R.copy()
        Pxz = np.zeros((P_pred.shape[0], z_pred.shape[0]))
        for i in range(Zsig.shape[0]):
            dz = (Zsig[i] - z_pred).reshape(-1,1)
            dx = (Xi[i] - x_pred).reshape(-1,1)
            S += W[i] * (dz @ dz.T)
            Pxz += W[i] * (dx @ dz.T)
        K = Pxz @ np.linalg.inv(S)
        y = z - z_pred
        x_upd = x_pred + K @ y
        P_upd = ensure_psd(P_pred - K @ S @ K.T)
        return x_upd, P_upd, y, S


import numpy as np
from IDKalman.INFtoCOV import inf_to_cov
from IDKalman.Mupdate import mupdate
from IDKalman.Tupdate import tupdate
from IDKalman.COVtoINF import cov_to_inf


class IDEKF:
    def _num_jacobian_g(self, x, eps=1e-6):
        x = np.asarray(x, dtype=float)
        n = x.size
        fx = np.asarray(self.dm.g(x))
        J = np.zeros((n, n))
        for i in range(n):
            d = np.zeros(n); d[i] = eps
            J[:, i] = (np.asarray(self.dm.g(x + d)) - fx) / eps
        return J

    def _compute_Hk_zhat(self, x_vec: np.ndarray):
        x_col = x_vec.reshape(-1, 1)
        Hk, zhat = None, None
        if callable(self.h):
            out = np.asarray(self.h(x_vec))
            if out.ndim == 2 and out.shape == (self._m, self._n):
                # h == Jacobian
                Hk = out
                zhat = Hk @ x_col
            else:
                # h == measurement function
                zhat = out.reshape(-1, 1)
        if Hk is None:
            Hk = self.H_fun(x_vec) if callable(self.H_fun) else np.asarray(self.H_fun)
        if zhat is None:
            zhat = Hk @ x_col
        return Hk, zhat

    
    def __init__(self, dm, h_fun, H_fun, R):
        self.dm = dm          # DiscreteModel with g(x), F(x), Q
        self.h_fun = h_fun        # measurement function h(x)  (can be None for linear)
        self.H_jac = H_fun    # Jacobian evaluated
        self.R = np.asarray(R)

        # --- dimensions (this is what you keep) ---
        assert self.R.shape[0] == self.R.shape[1], "R must be (m,m)"
        assert dm.Q.shape[0] == dm.Q.shape[1], "dm.Q must be (n,n)"
        self._n = dm.Q.shape[0]      # state dim n
        self._m = self.R.shape[0]    # meas dim m
        # ------------------------------------------

        self.u = None                 # (n,1)
        self.V = None                 # (n,1)
        self.B = None                 # (n,n)
        self._initialized = False

    def _ensure_initialized(self, x: np.ndarray, P: np.ndarray):
        if not self._initialized:
            self.u = x.reshape(-1, 1)
            B, V, _ = cov_to_inf(P, P.shape[0])
            self.B, self.V = B, V
            self._initialized = True

    def predict(self, x: np.ndarray, P: np.ndarray):
        self._ensure_initialized(x, P)

        Phi_k   = self.dm.F(self.u.ravel())
        gamma_k = np.eye(self._n)
        Q_k = self.dm.Q

        u0 = np.asarray(self.u).reshape(-1,1)   
        B0 = np.asarray(self.B).reshape(self._n, self._n)
        V0 = np.asarray(self.V).reshape(-1)
    
        self.u, self.B, self.V = tupdate(u0, B0, V0, Phi_k, gamma_k, Q_k)

        self.u = np.asarray(self.u).reshape(-1, 1)                 # (n,1)
        self.B = np.asarray(self.B).reshape(self._n, self._n)      # (n,n)
        self.V = np.asarray(self.V).reshape(-1)                    # (n,)
        
        x_pred = self.u.ravel()
        P_pred = ensure_psd(inf_to_cov(self.V, self.B, self._n))
        return x_pred, P_pred

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, z):
        z = np.asarray(z).reshape(-1, 1)

        # Evaluate and coerce H to (p,n)
        Hk = self.H_jac(x_pred) if callable(self.H_jac) else self.H_jac
        Hk = np.asarray(Hk).reshape(self._m, self._n)

        h_wrapped = None
        if self.h_fun is not None:
            def h_wrapped(u_vec):
                return np.asarray(self.h_fun(np.asarray(u_vec).reshape(-1))).reshape(-1,1)
            
        out = mupdate(1, z, self.u.reshape(-1,1), self.B, self.V, self.R, Hk, h_wrapped)
        self.u, self.V, self.B = out[0], out[1], out[2]

        self.u = np.asarray(self.u).reshape(-1, 1)
        self.V = np.asarray(self.V).reshape(-1)
        self.B = np.asarray(self.B).reshape(self._n, self._n)

        y = (z - (np.asarray(self.h_fun(x_pred)).reshape(-1,1) if self.h_fun else Hk @ x_pred.reshape(-1,1))).ravel()
        S = ensure_psd(Hk @ P_pred @ Hk.T + self.R)

        x_upd = self.u.ravel()
        P_upd = ensure_psd(inf_to_cov(self.V, self.B, self._n))
        return x_upd, P_upd, y, S
