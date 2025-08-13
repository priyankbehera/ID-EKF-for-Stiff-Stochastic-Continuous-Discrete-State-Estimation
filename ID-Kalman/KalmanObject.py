import numpy as np
from INFtoCOV import inf_to_cov
from Mupdate import mupdate
from Tupdate import tupdate

class InfluenceDiagramKalman:
    
    def __init__(self, dim_x, dim_z):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z

        self.u = np.zeros((dim_x, 1))        # state mean
        self.B = np.zeros((dim_x, dim_x))    # arc coefficient matrix (upper triangular)
        self.V = np.ones((dim_x, 1))         # conditional variances

        self.H = np.zeros((dim_z, dim_x))    # measurement matrix
        self.R = np.eye(dim_z)               # measurement noise covariance
        self.Phi = np.eye(dim_x)             # state transition matrix
        self.gamma = np.eye(dim_x)           # process noise matrix
        self.Q = np.eye(dim_x)               # process noise covariance

        self.history_obs = []
        self.Form = 0  # 0: influence diagram form, 1: covariance form

    def predict(self):
        self.u, self.B, self.V = tupdate(self.u, self.B, self.V, self.Phi, self.gamma, self.Q)

    def update(self, z, h=None):
        if z is None:
            self.history_obs.append(None)
            return

        z = np.array(z).reshape(-1, 1)
        self.history_obs.append(z)

        domain = self.B.shape[0]

        self.u, self.V, self.B = mupdate(0, z, self.u, self.B, self.V, self.R, self.H, h)

        self.u = self.u[:domain]
        self.V = self.V[:domain]
        self.B = self.B[:domain, :domain]

    def convert_to_covariance_form(self):
        if self.Form == 1:
            return inf_to_cov(self.V, self.B, self.dim_x)
        else:
            raise ValueError("Currently in influence diagram form. Set Form=1 to convert.")

    def run_filter_step(self, z=None):
        self.update(z)
        self.predict()
        