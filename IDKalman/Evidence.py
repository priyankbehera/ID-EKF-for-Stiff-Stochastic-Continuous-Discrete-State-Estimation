import numpy as np
from .Reversal import reversal

def get_Kalman(B, n, j):
    K = np.zeros(n)
    # First initialize the first element
    K[0] = B[n + j, 0]

    # Then, recursively calculate each row of K
    for n_idx in range(1, n):
        # Direct term
        direct_term = B[n + j, n_idx]

        # Summation term
        summation_term = 0
        for i in range(n_idx):  # Summation from 1 to n-1 (n_idx, r_idx are zero-based)
            summation_term += K[i] * B[i, n_idx]

        # Compute K[n, r]
        K[n_idx] = direct_term + summation_term

    return K


def evidence(u, B, V, X1, n0, n1, n2, du):
    """
    Enter evidence in influence diagram.

    Parameters:
    u (numpy.ndarray): A matrix with values including the mean of the state X(k) at discrete time k and the product of this mean and measurement matrix at discrete time k that maps the state X(k).
    B (numpy.ndarray): Matrix composed of covariance matrix of state X(k) and the measurement matrix at discrete time k.
    V (numpy.ndarray): A vector combining conditional variances with entries that are non-negative (including inf) and the measurement noise values.
    X1 (numpy.ndarray): Vector of n1 values with evidence in multivariate Gaussian with Influence Diagram form.
    n0 (int): Size of X0, where X0 is the predecessor of X1.
    n1 (int): Size of X1.
    n2 (int): Size of X2, where X2 is the successor of X1.
    du (numpy.ndarray): The change in u caused by the observed variable.

    Returns:
    u (numpy.ndarray): The updated mean vector of the state X(k) changed by the effect of the observed variable.
    B (numpy.ndarray): The updated matrix with strictly upper triangular submatrices and observed values set to 0.
    V (numpy.ndarray): The updated vector with non-negative entries (including inf) with the observed value set to 0.
    """

    K = np.zeros((n0, n1))
    print(X1)
    print(u)
    for j in range(n1):
        B, V = reversal(B, V, 0, n0 + j, 1, n1 - j + n2)

        du[n0 + j] = X1[j] - u[n0 + j]
        du[:n0] = du[n0 + j] * B[n0 + j, :n0]

        if n0 + n1 + n2 >= n0 + j + 1:
            du[n0 + j + 1: n0 + n1 + n2] = (
                    du[n0 + j] * B[n0 + j, n0 + j + 1: n0 + n1 + n2]
            )

        if n0 >= 2:
            for k in range(1, n0):  # Adjusting for 0-indexing in Python
                du[k] += np.dot(B[:k, k], du[:k])

        u[:n0] += du[:n0]

        if n1 + n2 >= j + 1:
            for k in range(n0 + j + 1, n0 + n1 + n2):
                du[k] += np.dot(B[:n0, k], du[:n0])
                du[k] += np.dot(
                    B[n0 + j + 1: n0 + n1 + n2, k], du[n0 + j + 1: n0 + n1 + n2]
                )
                u[k] += du[k]

        K_col = get_Kalman(B, n0, j)
        K[:, j] = K_col
        u[n0 + j] = 0
        V[n0 + j] = 0
        B[n0 + j, : n0 + n1 + n2] = 0

    return u, B, V, K