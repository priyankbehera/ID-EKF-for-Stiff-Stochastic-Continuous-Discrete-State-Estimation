
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
    Wm = np.full(2*n + 1, 1.0 / (2*c))
    Wc = np.full(2*n + 1, 1.0 / (2*c))
    Wm[0] = lam / c
    Wc[0] = lam / c + (1 - alpha**2 + beta)
    U = np.linalg.cholesky((c) * P)
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
    Xi = np.zeros((2*n, n))
    U = np.linalg.cholesky(P)
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

class IDEKF:
    """
    Influence-Diagram Extended Kalman Filter (ID-EKF).

    State is maintained in the Influence Diagram (ID) form:
      - u : (n,1) mean
      - B : (n,n) arc coefficient matrix (often upper-triangular in ID parameterization)
      - V : (n,1) vector of conditional variances

    Measurement model (linear by default): z = H x + v,  v ~ N(0, R)
    Nonlinear measurement supported by passing h(x) to update().
    Time update uses (Phi, gamma, Q) in discrete time.
    """

    def __init__(self, dim_x: int, dim_z: int):
        if dim_x < 1:
            raise ValueError("dim_x must be >= 1")
        if dim_z < 1:
            raise ValueError("dim_z must be >= 1")

        self.dim_x = dim_x
        self.dim_z = dim_z

        # ID-form state
        self.u = np.zeros((dim_x, 1))        # mean
        self.B = np.zeros((dim_x, dim_x))    # arc coefficients (ID form)
        self.V = np.ones((dim_x, 1))         # conditional variances

        # Model matrices
        self.H = np.zeros((dim_z, dim_x))    # measurement matrix
        self.R = np.eye(dim_z)               # measurement noise covariance
        self.Phi = np.eye(dim_x)             # state transition
        self.gamma = np.eye(dim_x)           # process noise mapping
        self.Q = np.eye(dim_x)               # process noise covariance

        self.history_obs = []
        # Form flag retained for compatibility with your code:
        # 0 = ID form native, 1 = allow conversion to covariance form via convert_to_covariance_form()
        self.Form = 0

    # ---------- Core ID-EKF cycle ----------
    def predict(self):
        """
        Time update (ID form): (u, B, V) -> (u+, B+, V+)
        """
        self.u, self.B, self.V = tupdate(self.u, self.B, self.V, self.Phi, self.gamma, self.Q)

    def update(self, z, h=None):
        """
        Measurement update (ID form).
        If z is None, skip the update (useful for missing measurements).
        If h is provided, mupdate is expected to handle the nonlinear measurement internally.
        """
        if z is None:
            self.history_obs.append(None)
            return

        z = np.asarray(z).reshape(-1, 1)
        if z.shape[0] != self.dim_z:
            raise ValueError(f"z must have length {self.dim_z}, got {z.shape[0]}")
        self.history_obs.append(z)

        n = self.B.shape[0]

        # mupdate should accept ID-form arguments and return updated (u, V, B)
        # (If your mupdate returns more outputs like K, S, P1, you can capture them here.)
        self.u, self.V, self.B = mupdate(0, z, self.u, self.B, self.V, self.R, self.H, h)

        # Truncate in case mupdate uses augmentation internally
        self.u = self.u[:n]
        self.V = self.V[:n]
        self.B = self.B[:n, :n]

    def run_filter_step(self, z=None, h=None):
        """
        Convenience: one full step = update(z) then predict().
        """
        self.update(z, h=h)
        self.predict()

    # ---------- Utilities ----------
    def convert_to_covariance_form(self):
        """
        Return the covariance P corresponding to the current (V, B) in ID form.

        Note: Unlike your original version, this method directly returns P regardless of Form.
        If you want to enforce a flag, uncomment the check below.
        """
        # if self.Form != 1:
        #     raise ValueError("Currently in influence diagram form. Set Form=1 to convert.")
        return inf_to_cov(self.V, self.B, self.dim_x)

    # Optional: helpers to set/replace model matrices safely
    def set_measurement_model(self, H: np.ndarray, R: np.ndarray):
        H = np.asarray(H); R = np.asarray(R)
        if H.shape != (self.dim_z, self.dim_x):
            raise ValueError(f"H must be ({self.dim_z},{self.dim_x}), got {H.shape}")
        if R.shape != (self.dim_z, self.dim_z):
            raise ValueError(f"R must be ({self.dim_z},{self.dim_z}), got {R.shape}")
        self.H, self.R = H, R

    def set_process_model(self, Phi: np.ndarray, gamma: np.ndarray, Q: np.ndarray):
        Phi = np.asarray(Phi); gamma = np.asarray(gamma); Q = np.asarray(Q)
        if Phi.shape != (self.dim_x, self.dim_x):
            raise ValueError(f"Phi must be ({self.dim_x},{self.dim_x}), got {Phi.shape}")
        if gamma.shape[0] != self.dim_x:
            raise ValueError(f"gamma must have {self.dim_x} rows, got {gamma.shape[0]}")
        if Q.shape[0] != Q.shape[1] or Q.shape[0] != gamma.shape[1]:
            raise ValueError("Q must be (r,r) matching gamma's column dimension r")
        self.Phi, self.gamma, self.Q = Phi, gamma, Q
