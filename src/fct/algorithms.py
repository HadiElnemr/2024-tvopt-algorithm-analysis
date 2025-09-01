import numpy as np
import control as ctrl
from .lure import LureComponent


class Algorithm(LureComponent):
    """

    """

    def __init__(self, m=1, L=1, nx=1, delta_model=False):
        super().__init__(m, L, delta_model)
        self.nx = nx
        self.internal_state = None
        self.gradient_function = None
        self.name = None

    def initialize(self, xi_0):
        self.internal_state = xi_0

    def update_gradient(self, gradient_function):
        self.gradient_function = gradient_function

    def step(self):
        if self.internal_state is None:
            raise ValueError("Internal state is not initialized.")
        if self.A is None or self.B is None or self.C is None:
            raise ValueError("State-space matrices are not initialized.")
        if self.gradient_function is None:
            raise ValueError("Gradient callable is not set.")

        xk = self.C @ self.internal_state
        gk = self.gradient_function(xk)
        self.internal_state = self.A @ self.internal_state + self.B @ gk
        return self.internal_state, xk



class GradientDescent(Algorithm):
    """

    """

    def __init__(self, m=1, L=1, nx=1, delta_model=False):
        super().__init__(m, L, nx, delta_model)
        self.name = 'Gradient descent'
        self.internal_state_dim = nx

    def _update_state_space(self):
        alpha = 2 / (self.m + self.L)
        if not self.delta_model:
            self.A = np.eye(self.nx)
            self.B = -alpha * np.eye(self.nx)
            self.C = np.eye(self.nx)
            self.D = np.zeros((self.nx, self.nx))
        else:
            self.A = np.eye(self.nx)
            self.B = np.hstack([-alpha * np.eye(self.nx), np.eye(self.nx)])
            self.C = np.eye(self.nx)
            self.D = np.zeros((self.nx, 2 * self.nx))


class Nesterov(Algorithm):
    """

    """

    def __init__(self, m=1, L=1, nx=1, delta_model=False):
        super().__init__(m, L, nx, delta_model)
        self.name = 'Nesterov'
        self.internal_state_dim = 2*nx

    def _update_state_space(self):
        kappa = self.L / self.m
        alpha = 1 / self.L
        beta = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)

        A_base = np.asarray([[1 + beta, -beta], [1, 0]])
        C_base = np.asarray([[1 + beta, -beta]])

        if not self.delta_model:
            B_base = np.asarray([[-alpha], [0]])
            D_base = np.zeros((1, 1))
        else:
            B_base = np.asarray([[-alpha, 1, 0], [0, 0, 1]])
            D_base = np.zeros((1, 3))

        self.A = np.kron(A_base, np.eye(self.nx))
        self.B = np.kron(B_base, np.eye(self.nx))
        self.C = np.kron(C_base, np.eye(self.nx))
        self.D = np.kron(D_base, np.eye(self.nx))

    def initialize(self, x0):
        super().initialize(np.concatenate([x0, x0]))


class TMM(Algorithm):
    """

    """

    def __init__(self, m=1, L=1, nx=1, delta_model=False):
        super().__init__(m, L, nx, delta_model)
        self.name = 'Triple Momentum'
        self.internal_state_dim = 2*nx

    def _update_state_space(self):
        kappa = self.L / self.m
        rho = 1 - 1 / np.sqrt(kappa)
        alpha = (1 + rho) / self.L
        beta = rho**2 / (2 - rho)
        gamma = rho**2 / ((1 + rho) * (2 - rho))

        A_base = np.asarray([[1 + beta, -beta], [1, 0]])
        C_base = np.asarray([[1 + gamma, -gamma]])

        if not self.delta_model:
            B_base = np.asarray([[-alpha], [0]])
            D_base = np.zeros((1, 1))
        else:
            B_base = np.asarray([[-alpha, 1, 0], [0, 0, 1]])
            D_base = np.zeros((1, 3))

        self.A = np.kron(A_base, np.eye(self.nx))
        self.B = np.kron(B_base, np.eye(self.nx))
        self.C = np.kron(C_base, np.eye(self.nx))
        self.D = np.kron(D_base, np.eye(self.nx))

    def initialize(self, x0):
        super().initialize(np.concatenate([x0, x0]))
