import numpy as np
from INFtoCOV import inf_to_cov
from Mupdate import mupdate
from Tupdate import tupdate

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
