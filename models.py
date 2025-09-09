# models.py
# Continuous-time test problems and measurement models
from __future__ import annotations
import numpy as np
from typing import Callable

# ------------------------------- Dahlquist ----------------------------------

def dahlquist_f(mu: float, j: int) -> Callable[[float, np.ndarray], np.ndarray]:
    """dx/dt = mu * x^j  (scalar)"""
    def f(t: float, x: np.ndarray) -> np.ndarray:
        return np.array([mu * (x[0] ** j)], dtype=float)
    return f

def dahlquist_J(mu: float, j: int) -> Callable[[float, np.ndarray], np.ndarray]:
    """Jacobian of Dahlquist drift wrt x."""
    def J(t: float, x: np.ndarray) -> np.ndarray:
        if j == 0:
            return np.array([[0.0]], dtype=float)
        return np.array([[mu * j * (x[0] ** (j - 1))]], dtype=float)
    return J

def dahlquist_h() -> Callable[[np.ndarray], np.ndarray]:
    """Linear measurement: z = x + v"""
    return lambda x: np.array([x[0]], dtype=float)

def dahlquist_H() -> Callable[[np.ndarray], np.ndarray]:
    """Jacobian of measurement wrt state (for EKF)."""
    return lambda x: np.array([[1.0]], dtype=float)

def dahlquist_G() -> Callable[[float], np.ndarray]:
    """Continuous-time diffusion gain G(t) (scalar)."""
    return lambda t: np.array([[1.0]], dtype=float)

def dahlquist_Qc() -> Callable[[float], np.ndarray]:
    """Continuous-time noise intensity Qc(t) (scalar)."""
    return lambda t: np.array([[1.0]], dtype=float)

# -------------------------------- Van der Pol -------------------------------

def vdp_f(mu: float) -> Callable[[float, np.ndarray], np.ndarray]:
    """Van der Pol oscillator (stiff for large mu).
       x1' = x2
       x2' = mu * ((1 - x1^2) * x2 - x1)
    """
    def f(t: float, x: np.ndarray) -> np.ndarray:
        x1, x2 = float(x[0]), float(x[1])
        return np.array([x2, mu * ((1.0 - x1**2) * x2 - x1)], dtype=float)
    return f

def vdp_J(mu: float) -> Callable[[float, np.ndarray], np.ndarray]:
    """Jacobian of Van der Pol drift wrt x."""
    def J(t: float, x: np.ndarray) -> np.ndarray:
        x1, x2 = float(x[0]), float(x[1])
        return np.array([[0.0, 1.0],
                         [-2.0 * mu * x1 * x2 - mu, mu * (1.0 - x1**2)]], dtype=float)
    return J

# --- Measurement models ---
def vdp_h() -> Callable[[np.ndarray], np.ndarray]:
    """Linear measurement (matches the papers): z = x1 + x2 + v"""
    return lambda x: np.array([x[0] + x[1]], dtype=float)

def vdp_H() -> Callable[[np.ndarray], np.ndarray]:
    """Jacobian for EKF when using the linear measurement above."""
    return lambda x: np.array([[1.0, 1.0]], dtype=float)

# Optional: a nonlinear measurement to accentuate UKF vs CKF differences
def vdp_h_nonlinear(c: float = 0.05) -> Callable[[np.ndarray], np.ndarray]:
    """Nonlinear measurement: z = x1 + x2 + c * x1^4 + v"""
    return lambda x: np.array([x[0] + x[1] + c * (x[0] ** 4)], dtype=float)

def vdp_H_nonlinear(c: float = 0.05) -> Callable[[np.ndarray], np.ndarray]:
    """Jacobian for EKF with the nonlinear measurement above."""
    return lambda x: np.array([[1.0 + 4.0 * c * (float(x[0]) ** 3), 1.0]], dtype=float)

def vdp_G() -> Callable[[float], np.ndarray]:
    """Diffusion only in second state."""
    return lambda t: np.array([[0.0, 0.0],
                               [0.0, 1.0]], dtype=float)

def vdp_Qc() -> Callable[[float], np.ndarray]:
    """Continuous-time noise intensity for Van der Pol."""
    return lambda t: np.eye(2, dtype=float)
