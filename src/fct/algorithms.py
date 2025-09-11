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
            A = 1
            B = -alpha
            C = 1
            D = 0
            self.A = np.kron(A, np.eye(self.nx))
            self.B = np.kron(B, np.eye(self.nx))
            self.C = np.kron(C, np.eye(self.nx))
            self.D = np.kron(D, np.eye(self.nx))
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

        A_base = np.asarray([[1 + beta, -beta],
                             [1,          0]])
        C_base = np.asarray([[1 + beta, -beta]])

        if not self.delta_model:
            B_base = np.asarray([[-alpha],
                                 [0     ]])
            D_base = np.zeros((1, 1))
        else:
            B_base = np.asarray([[-alpha, 1, 0],
                                 [0,      0, 1]])
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

        A_base = np.asarray([[1 + beta, -beta],
                             [1,          0]])
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


class HeavyBall(Algorithm):
    """

    """

    def __init__(self, m=1, L=1, nx=1, delta_model=False):
        super().__init__(m, L, nx, delta_model)
        self.name = 'Heavy ball'
        self.internal_state_dim = 2*nx

    def _update_state_space(self):
        kappa = self.L / self.m
        alpha = (2/(np.sqrt(self.L)+np.sqrt(self.m)))**2
        beta = (np.sqrt(kappa)-1) / (np.sqrt(kappa)+1)**2 # TODO: This should be squared according to Lessard's paper

        A = np.asarray([[1+beta, -beta],
                        [1,          0]])
        B = np.asarray([[-alpha], [0]])
        C = np.asarray([[1, 0]])
        D = 0

        self.A = np.kron(A, np.eye(self.nx))
        self.B = np.kron(B, np.eye(self.nx))
        self.C = np.kron(C, np.eye(self.nx))
        self.D = np.kron(D, np.eye(self.nx))

    def initialize(self, x0):
        super().initialize(np.concatenate([x0, x0]))


class MultiStepGradient(Algorithm):
    """

    """

    def __init__(self, m=1, L=1, K=1, nx=1, delta_model=False):
        super().__init__(m, L, nx, delta_model)
        self.K = K
        self.name = 'Multi-step gradient'
        self.internal_state_dim = nx

    def _update_state_space(self):
        alpha = 2/(self.m+self.L)

        A = 1
        B = np.full((1, self.K), -alpha)
        C = np.full((self.K,1), 1)
        D = np.tril(-alpha * np.ones((self.K, self.K)), k=-1)

        self.A = np.kron(A, np.eye(self.nx))
        self.B = np.kron(B, np.eye(self.nx))
        self.C = np.kron(C, np.eye(self.nx))
        self.D = np.kron(D, np.eye(self.nx)) # TODO: Check Correctness

    def initialize(self, x0):
        super().initialize(np.concatenate([x0, x0]))


class ProximalGradient(Algorithm):
    """

    """

    def __init__(self, m=1, L=1, nx=1, delta_model=False):
        super().__init__(m, L, nx, delta_model)
        self.name = 'Proximal gradient'
        self.internal_state_dim = 2*nx

    def _update_state_space(self):
        alpha = 2/(self.m+self.L)
        A = 1
        B = np.asarray([[-alpha, -alpha]])
        C = np.asarray([[1],
                        [1]])
        D = np.asarray([[0,           0],
                        [-alpha, -alpha]])
        self.A = np.kron(A, np.eye(self.nx))
        self.B = np.kron(B, np.eye(self.nx))
        self.C = np.kron(C, np.eye(self.nx))
        self.D = np.kron(D, np.eye(self.nx))

    def initialize(self, x0):
        super().initialize(np.concatenate([x0, x0])) ## TODO: Check if this is correct


class ProximalHeavyBall(Algorithm):
    """

    """

    def __init__(self, m=1, L=1, nx=1, delta_model=False):
        super().__init__(m, L, nx, delta_model)
        self.name = 'Proximal heavy ball'
        self.internal_state_dim = 2*nx

    def _update_state_space(self):
        alpha = (2/(np.sqrt(self.L)+np.sqrt(self.m)))**2
        beta = (np.sqrt(self.L/self.m)-1) / (np.sqrt(self.L/self.m)+1)
        A = np.asarray([[1+beta, -beta],
                        [1,          0]])
        B = np.asarray([[-alpha, -alpha],
                        [0,       0]])
        C = np.asarray([[1,          0],
                        [1+beta, -beta]])

        D = np.asarray([[0,       0],
                        [-alpha, -alpha]])

        self.A = np.kron(A, np.eye(self.nx))
        self.B = np.kron(B, np.eye(self.nx))
        self.C = np.kron(C, np.eye(self.nx))
        self.D = np.kron(D, np.eye(self.nx))

    def initialize(self, x0):
        super().initialize(np.concatenate([x0, x0])) ## TODO: Check if this is correct


class ProximalNesterov(Algorithm):
    """

    """


    def __init__(self, m=1, L=1, nx=1, delta_model=False):
        super().__init__(m, L, nx, delta_model)
        self.name = 'Proximal Nesterov'
        self.internal_state_dim = 2*nx

    def _update_state_space(self):
        alpha = 1/self.L
        beta = (np.sqrt(self.L/self.m)-1) / (np.sqrt(self.L/self.m)+1)
        A = np.asarray([[1+beta, -beta],
                        [1,          0]])
        B = np.asarray([[-alpha, -alpha],
                        [0,           0]])
        C = np.asarray([[1+beta, -beta],
                        [1+beta, -beta]])
        D = np.asarray([[0,       0],
                        [-alpha, -alpha]])
        self.A = np.kron(A, np.eye(self.nx))
        self.B = np.kron(B, np.eye(self.nx))
        self.C = np.kron(C, np.eye(self.nx))
        self.D = np.kron(D, np.eye(self.nx))

    def initialize(self, x0):
        super().initialize(np.concatenate([x0, x0])) ## TODO: Check if this is correct


class ProximalTripleMomentum(Algorithm):
    """

    """

    def __init__(self, m=1, L=1, nx=1, delta_model=False):
        super().__init__(m, L, nx, delta_model)
        self.name = 'Proximal Triple Momentum'
        self.internal_state_dim = 2*nx

    def _update_state_space(self):
        rho = 1 - 1/np.sqrt(self.L/self.m)
        alpha = (1+rho)/self.L
        beta = rho**2/(2-rho)
        gamma = rho**2/(2-rho)
        A = np.asarray([[1+beta, -beta],
                        [1,          0]])
        B = np.asarray([[-alpha, -alpha],
                        [0,           0]])
        C = np.asarray([[1+gamma, -gamma],
                        [1+gamma, -gamma]])
        D = np.asarray([[0,       0],
                        [-alpha, -alpha]])
        self.A = np.kron(A, np.eye(self.nx))
        self.B = np.kron(B, np.eye(self.nx))
        self.C = np.kron(C, np.eye(self.nx))
        self.D = np.kron(D, np.eye(self.nx))

    def initialize(self, x0):
        super().initialize(np.concatenate([x0, x0])) ## TODO: Check if this is correct


class AcceleratedOGD(Algorithm):
    """

    """

    def __init__(self, m=1, L=1, nx=1, delta_model=False):
        super().__init__(m, L, nx, delta_model)
        self.name = 'Accelerated OGD'
        self.internal_state_dim = 2*nx

    def _update_state_space(self):
        alpha = 1/self.L
        gamma = 1/self.L
        tau = gamma*alpha

        A = np.asarray([[tau, 1-tau],
                        [0,       1]])
        B = np.asarray([[-gamma, -gamma, 0],
                        [-alpha, 0, -alpha]])

        C = np.asarray([[tau, 1-tau],
                        [tau, 1-tau],
                        [0,       1]])

        D = np.asarray([[0, 0, 0],
                        [-gamma, -gamma, 0],
                        [-alpha, 0, -alpha]])

        self.A = np.kron(A, np.eye(self.nx))
        self.B = np.kron(B, np.eye(self.nx))
        self.C = np.kron(C, np.eye(self.nx))
        self.D = np.kron(D, np.eye(self.nx))

    def initialize(self, x0):
        super().initialize(np.concatenate([x0, x0])) ## TODO: Check if this is correct


class MultiStepOGD(Algorithm):
    """

    """

    def __init__(self, m=1, L=1, K=1, nx=1, delta_model=False):
        super().__init__(m, L, nx, delta_model)
        self.K = K
        self.name = 'Multi-step OGD'
        self.internal_state_dim = 2*nx

    def _update_state_space(self):
        alpha = 2/(self.m+self.L)

        block1 = np.tril(-alpha * np.ones((self.K, self.K)), k=-1)
        block2 = np.tril(-alpha * np.ones((self.K, self.K)), k=0)

        A = 1
        B = np.full((1, 2*self.K), -alpha)
        C = np.full((2*self.K,1), 1)
        D = np.block([[block1, block1],
                      [block2, block2]])

        self.A = np.kron(A, np.eye(self.nx))
        self.B = np.kron(B, np.eye(self.nx))
        self.C = np.kron(C, np.eye(self.nx))
        self.D = np.kron(D, np.eye(self.nx))

    def initialize(self, x0):
        super().initialize(np.concatenate([x0, x0])) ## TODO: Check if this is correct