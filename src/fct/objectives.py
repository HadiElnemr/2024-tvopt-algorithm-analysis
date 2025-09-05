from abc import ABC, abstractmethod
import numpy as np


class ObjectiveFunction(ABC):
    def __init__(self, nx):
        self.nx = nx  # Dimension of the input vector x

    @abstractmethod
    def eval(self, x):
        """Evaluate the objective function at given x."""
        pass

    @abstractmethod
    def gradient(self, x):
        """Compute the gradient of the objective function at given x."""
        pass

    @abstractmethod
    def get_objective_info(self):
        """Get information about the objective function: minimum and strong convexity/smoothness parameters."""
        pass

    @abstractmethod
    def update(self, t):
        """Update the objective function for a new time instance t."""
        pass



class PeriodicExample2D(ObjectiveFunction):
    def __init__(self, omega):
        super().__init__(nx=2)  # Periodic function is defined for nx=2
        self.omega = omega
        self.current_t = 0

    def update(self, t):
        self.current_t = t

    def eval(self, x, t=None, prev_t=False):
        t = t if t else self.current_t if not prev_t else self.current_t - 1
        f_star = 0
        f = (x[0] - np.exp(np.cos(self.omega * t)))**2 + (x[1] - x[0] * np.tanh(np.sin(self.omega * t)))**2
        return f - f_star

    def gradient(self, x, prev_t=False):
        t = self.current_t if not prev_t else self.current_t - 1
        grad = np.zeros_like(x)
        grad[0] = 2 * (x[0] - np.exp(np.cos(self.omega * t))) - 2 * np.tanh(np.sin(self.omega * t)) * (x[1] - x[0] * np.tanh(np.sin(self.omega * t)))
        grad[1] = 2 * (x[1] - x[0] * np.tanh(np.sin(self.omega * t)))
        return grad

    def hessian(self, x):
        t = self.current_t
        H = np.zeros((2, 2))
        H[0, 0] = 2 + 2 * np.tanh(np.sin(self.omega * t))**2
        H[0, 1] = -2 * np.tanh(np.sin(self.omega * t))
        H[1, 0] = -2 * np.tanh(np.sin(self.omega * t))
        H[1, 1] = 2
        return H

    def get_objective_info(self):
        t = self.current_t
        x_star = np.array([
            np.exp(np.cos(self.omega * t)),
            np.exp(np.cos(self.omega * t)) * np.tanh(np.sin(self.omega * t))
        ])
        m, L = self._sector_constraints(t)
        return x_star, m, L

    def _sector_constraints(self, t):
        A = np.array([
            [2 + 2 * np.tanh(np.sin(self.omega * t))**2, -2 * np.tanh(np.sin(self.omega * t))],
            [-2 * np.tanh(np.sin(self.omega * t)), 2]
        ])
        eigs = np.linalg.eigvals(A)
        return min(eigs), max(eigs) # or return ( 2+y^2 +- |y| * sqrt(y^2+4)) ) ; y = tanh(sin(omega*t))




class WindowedLeastSquares(ObjectiveFunction):
    def __init__(self, nx=10, n_data=100, noise=0.1):
        super().__init__(nx)
        self.n_data = n_data
        self.noise = noise
        self.At_list = []
        self.bt_list = []
        self._initialize_window()

    def _initialize_window(self):
        """Initialize the window with n_data At and bt using negative time indices."""
        nx = self.nx
        n_data = self.n_data
        for t in range(-n_data, 0):
            A0 = -1 + 2 * np.random.rand(n_data, nx)
            b0 = np.zeros((n_data, 1))
            At = 0.5 * np.sin(0.1 * t) + A0 + self.noise * np.random.randn(n_data, nx)
            bt = 1.5 * np.sin(0.1 * t) + b0 + self.noise * np.random.randn(n_data, 1)
            self.At_list.append(At)
            self.bt_list.append(bt)

    def update(self, t):
        """Update At and bt for a new time instance t."""
        nx = self.nx
        n_data = self.n_data
        A0 = -1 + 2 * np.random.rand(n_data, nx)
        b0 = np.zeros((n_data, 1))

        new_At = 0.5 * np.sin(0.1 * t) + A0 + self.noise * np.random.randn(n_data, nx)
        new_bt = 1.5 * np.sin(0.1 * t) + b0 + self.noise * np.random.randn(n_data, 1)

        self.At_list.append(new_At)
        self.bt_list.append(new_bt)

        # Maintain the window size
        if len(self.At_list) > self.n_data:
            self.At_list.pop(0)
            self.bt_list.pop(0)

        self._AtA = sum(A.T @ A for A in self.At_list)
        self._Atb = sum(A.T @ b for A, b in zip(self.At_list, self.bt_list))

    def eval(self, x):
        x_star = self._optimum()
        f = 0.5 * sum(np.linalg.norm(self.At_list[i] @ x - self.bt_list[i])**2 for i in range(len(self.At_list)))
        f_star = 0.5 * sum(np.linalg.norm(self.At_list[i] @ x_star - self.bt_list[i])**2 for i in range(len(self.At_list)))
        return f - f_star

    def gradient(self, x):
        grad = np.zeros_like(x)
        grad += (self._AtA @ x - self._Atb.flatten())
        return grad

    def get_objective_info(self):
        x_star = self._optimum()
        m, L = self._sector_constraints()
        return x_star, m, L

    def _optimum(self):
        return np.linalg.solve(self._AtA, self._Atb)

    def _sector_constraints(self):
        eigs = np.linalg.eigvals(self._AtA)
        return min(eigs), max(eigs)