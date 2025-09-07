# models_cd.py
# Test problems and measurement models (Dahlquist and Van der Pol)

from __future__ import annotations
import numpy as np
from typing import Callable

# ------------------------------- Dahlquist ----------------------------------

def dahlquist_f(mu: float, j: int) -> Callable[[float, np.ndarray], np.ndarray]:
    def f(t: float, x: np.ndarray) -> np.ndarray:
        return np.array([mu * (x[0] ** j)])
    return f

def dahlquist_J(mu: float, j: int) -> Callable[[float, np.ndarray], np.ndarray]:
    def J(t: float, x: np.ndarray) -> np.ndarray:
        if j == 0:
            return np.array([[0.0]])
        return np.array([[mu * j * (x[0] ** (j - 1))]])
    return J

def dahlquist_h() -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: np.array([x[0]])

def dahlquist_H() -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: np.array([[1.0]])

# Continuous-time diffusion (scalar)
def dahlquist_G() -> Callable[[float], np.ndarray]:
    return lambda t: np.array([[1.0]])  # unit diffusion

def dahlquist_Qc() -> Callable[[float], np.ndarray]:
    return lambda t: np.array([[1.0]])  # unit intensity

# -------------------------------- Van der Pol -------------------------------

def vdp_f(mu: float) -> Callable[[float, np.ndarray], np.ndarray]:
    def f(t: float, x: np.ndarray) -> np.ndarray:
        x1, x2 = x
        return np.array([x2, mu * ((1.0 - x1**2) * x2 - x1)])
    return f

def vdp_J(mu: float) -> Callable[[float, np.ndarray], np.ndarray]:
    def J(t: float, x: np.ndarray) -> np.ndarray:
        x1, x2 = x
        return np.array([[0.0, 1.0],
                         [-2.0 * mu * x1 * x2 - mu, mu * (1.0 - x1**2)]])
    return J

def vdp_h() -> Callable[[np.ndarray], np.ndarray]:
    # measurement: z = x1 + x2
    return lambda x: np.array([x[0] + x[1]])

def vdp_H() -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: np.array([[1.0, 1.0]])

def vdp_G() -> Callable[[float], np.ndarray]:
    # noise only in the second state
    return lambda t: np.array([[0.0, 0.0], [0.0, 1.0]])

def vdp_Qc() -> Callable[[float], np.ndarray]:
    return lambda t: np.eye(2)
