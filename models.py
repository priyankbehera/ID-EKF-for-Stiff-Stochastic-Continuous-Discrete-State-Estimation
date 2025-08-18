
import numpy as np

def dahlquist_f(mu: float, j: int):
    def f(x: np.ndarray, t: float) -> np.ndarray:
        return np.array([mu * (x[0] ** j)], dtype=float)
    return f

def dahlquist_J(mu: float, j: int):
    def J(x: np.ndarray, t: float) -> np.ndarray:
        val = mu * j * (x[0] ** (j-1)) if j >= 1 else 0.0
        return np.array([[val]], dtype=float)
    return J

def dahlquist_h():
    def h(x: np.ndarray) -> np.ndarray:
        return x.copy()
    return h

def dahlquist_H():
    def H(x: np.ndarray) -> np.ndarray:
        return np.eye(1)
    return H

def vdp_f(mu: float):
    def f(x: np.ndarray, t: float) -> np.ndarray:
        x1, x2 = x
        return np.array([x2, mu * ((1 - x1**2) * x2 - x1)], dtype=float)
    return f

def vdp_J(mu: float):
    def J(x: np.ndarray, t: float) -> np.ndarray:
        x1, x2 = x
        return np.array([[0.0, 1.0],
                         [-mu*(2.0*x1*x2 + 1.0), mu*(1.0 - x1**2)]], dtype=float)
    return J

def vdp_h():
    def h(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] + x[1]], dtype=float)
    return h

def vdp_H():
    def H(x: np.ndarray) -> np.ndarray:
        return np.array([[1.0, 1.0]], dtype=float)
    return H
