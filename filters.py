
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

class IDEKF(EKF):
    # Placeholder: identical to EKF; replace with your ID logic.
    pass

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
