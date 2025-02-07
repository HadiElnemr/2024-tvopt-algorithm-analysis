import numpy as np
import cvxpy as cvx
from .lure import LureComponent


class IQC(LureComponent):
    """
    tbd
    """

    def __init__(self, m=1, L=1, delta_model=False):
        super().__init__(m, L, delta_model)
        self.M = None
        self.lambda_var = cvx.Variable(1, nonneg=True)
        self.rho = 0
        self.algo = None

    def update_external_dependencies(self, rho, algo):
        self.rho  = rho
        self.algo = {"algorithm_output": algo.C,
                     "n_delta": algo.internal_state_dim}

    def cvx_reset(self):
        del self.lambda_var
        self.lambda_var = cvx.Variable(1, nonneg=True)


class SectorIQC(IQC):
    """
    tbd
    """

    def __init__(self, m=1, L=1, delta_model=False):
        super().__init__(m, L, delta_model)
        self.name = 'Sector IQC'
        self.internal_state_dim = 0
        self.M = np.asarray([[0, 1], [1, 0]])

    def _update_state_space(self):
        D_psi = np.asarray([[self.L, -1], 
                            [-self.m, 1]])

        if self.delta_model:
            D_psi = np.block([D_psi, np.zeros((2, self.algo["n_delta"]))])

        self.A = []
        self.B = []
        self.C = []
        self.D = D_psi


class OffByOneIQC(IQC):
    """
    tbd
    """
    
    def __init__(self, m=1, L=1, delta_model=False):
        super().__init__(m, L, delta_model)
        self.name = 'Off-by-one IQC'
        self.internal_state_dim = 1

        self.M = np.asarray([[0, 1], [1, 0]])

    def _update_state_space(self):
        if self.rho is None:
            raise ValueError("rho is not set! Make sure to update external dependencies.")

        A_psi = 0
        B_psi = np.asarray([[-self.L, 1]])
        C_psi = np.asarray([[self.rho**2], [0]])
        D_psi = np.asarray([[self.L, -1], 
                            [-self.m, 1]])

        if self.delta_model:
            if self.algo is None:
                raise ValueError("Algorithm is not set! Make sure to update external dependencies.")

            B_psi = np.block([B_psi, -self.algo["algorithm_output"]])
            D_psi = np.block([D_psi, np.zeros((2, self.algo["n_delta"]))])

        self.A = A_psi
        self.B = B_psi
        self.C = C_psi
        self.D = D_psi
