import cvxpy as cvx
import numpy as np
from itertools import combinations_with_replacement


class PolynomialLyapunovMatrix:
    def __init__(self, param_dim, poly_degree, n_eta):
        self.param_dim = param_dim      # Dimension of p
        self.poly_degree = poly_degree  # Degree of polynomial
        self.n_eta = n_eta              # Dimension of matrix
        
        # Generate polynomial basis terms
        self.basis_terms = self.generate_polynomial_basis()
        self.num_basis = len(self.basis_terms)
        
        # Create SDP variables for each polynomial basis term
        self.lyap_basis = [cvx.Variable((n_eta, n_eta), symmetric=True) for _ in range(self.num_basis)]
    

    def generate_polynomial_basis(self):
        """Generates polynomial basis terms up to the specified degree."""
        basis_terms = []
        for deg in range(self.poly_degree + 1):
            for term in combinations_with_replacement(range(self.param_dim), deg):
                basis_terms.append(term)
        return basis_terms
    

    def P(self, p):
        """Constructs the polynomial Lyapunov matrix as a cvxpy expression."""
        P_p = sum(self.lyap_basis[i] * np.prod([p[j]**term.count(j) for j in range(self.param_dim)])
                   for i, term in enumerate(self.basis_terms))
        return P_p
    

    def P_numeric(self, p):
        """Evaluates P(p) numerically given the values of the SDP variables."""
        P_p = sum(self.lyap_basis[i].value * np.prod([p[j]**term.count(j) for j in range(self.param_dim)])
                   for i, term in enumerate(self.basis_terms))
        return P_p

    
    def cvx_reset(self):
        del self.lyap_basis
        self.lyap_basis = [cvx.Variable((self.n_eta, self.n_eta), symmetric=True) for _ in range(self.num_basis)]
    

    def min_max_eigval(self, p_grid):
        """Computes the minimum and maximum eigenvalues of P(p) over a grid."""
        min_eig, max_eig = np.inf, -np.inf
        
        for p in p_grid:
            P_p = self.P_numeric(p)
            eigvals = np.linalg.eigvalsh(P_p)
            
            min_eig = min(min_eig, np.min(eigvals))
            max_eig = max(max_eig, np.max(eigvals))
        
        return min_eig, max_eig
    
    
    def condition_P(self, p_grid):
        """Computes the condition number of P(p) over the grid."""
        min_eig, max_eig = self.min_max_eigval(p_grid)
        
        if min_eig <= 0:
            return np.inf  # P(p) is not positive definite
        
        return np.sqrt(max_eig / min_eig)



class Solver:
    def __init__(self, algorithm, delta_model, rho_max, consistent_polytope, eps=1e-6):
        self.algorithm = algorithm
        self.delta_model = delta_model
        self.rho_max = rho_max
        self.consistent_polytope = consistent_polytope
        self.eps = eps
        self.IQCs = []
    
    def add_iqc(self, iqc):
        """Add an IQC to the solver."""
        self.IQCs.append(iqc)
    
    def setup_LMI(self, rho):
        """Sets up the LMI system based on rho and IQCs."""
        LMI_system = []
        n_xi = self.algorithm.internal_state_dim
        n_x, n_g = self.algorithm.nx, self.algorithm.nx
        n_eta = n_xi + (1 if self.delta_model else 0)
        I_n_eta = np.eye(n_eta)
        
        lyap = PolynomialLyapunovMatrix(param_dim=1, poly_degree=2, n_eta=n_eta)
        t = cvx.Variable(1, nonneg=True)
        t_I = cvx.multiply(t, I_n_eta)
        
        for p_k, delta_p in self.consistent_polytope:
            p_kp1 = p_k + delta_p
            P_k = lyap.P(p_k)
            P_kp1 = lyap.P(p_kp1)
            
            self.algorithm.update_algorithm(m=1, L=p_k[0])
            G = self.algorithm.get_state_space(delta_model=self.delta_model)
            
            M_combined = []
            lambda_combined = []
            
            for iqc in self.IQCs:
                iqc.update(p_k)
                M_combined.append(iqc.lambda_var * iqc.M)
                lambda_combined.append(iqc.lambda_var)
            
            M_total = cvx.bmat([[M_combined[i] if i == j else np.zeros_like(M_combined[i]) for j in range(len(M_combined))] for i in range(len(M_combined))])
            
            LMI_inner = cvx.bmat([
                [-rho**2 * P_k, np.zeros((n_eta, n_eta)), np.zeros((n_eta, M_total.shape[0]))],
                [np.zeros((n_eta, n_eta)), P_kp1, np.zeros((n_eta, M_total.shape[0]))],
                [np.zeros((M_total.shape[0], n_eta)), np.zeros((M_total.shape[0], n_eta)), M_total]
            ])
            
            LMI_system.append(LMI_inner << 0)
            LMI_system.append(P_k >> self.eps * I_n_eta)
            LMI_system.append(P_kp1 >> self.eps * I_n_eta)
            LMI_system.append(P_k << t_I)
            LMI_system.append(P_kp1 << t_I)
            LMI_system.append(cvx.bmat([[P_k, I_n_eta], [I_n_eta, t_I]]) >> 0)
            LMI_system.append(cvx.bmat([[P_kp1, I_n_eta], [I_n_eta, t_I]]) >> 0)
        
        return LMI_system, t, lambda_combined
    
    def optimize_for_rho(self):
        """Performs bisection on rho to find the optimal value."""
        rho_min = 0
        rho_tol = 1e-3
        sol = (np.nan, np.nan, np.nan, np.nan, np.nan)
        
        while (self.rho_max - rho_min > rho_tol):
            rho = (rho_min + self.rho_max) / 2
            LMI_system, _, _ = self.setup_LMI(rho)
            
            problem = cvx.Problem(cvx.Minimize(0), LMI_system)
            try:
                problem.solve(solver=cvx.MOSEK)
            except cvx.SolverError:
                pass
            
            if problem.status == cvx.OPTIMAL:
                self.rho_max = rho
            else:
                rho_min = rho
        
        return self.rho_max
    
    def optimize_for_bound(self, epsilon=1e-4, k1=1e-3, k_lambdas=1e-3):
        """Optimizes for bound given the best rho from optimize_for_rho."""
        rho_opt = self.optimize_for_rho() + epsilon
        LMI_system, t, lambda_combined = self.setup_LMI(rho_opt)
        gamma = cvx.Variable(1, nonneg=True)
        
        objective = cvx.Minimize(t + k1 * gamma + sum(k_lambdas * l for l in lambda_combined))
        problem = cvx.Problem(objective, LMI_system)
        
        try:
            problem.solve(solver=cvx.MOSEK)
        except cvx.SolverError:
            pass
        
        return t.value, [l.value for l in lambda_combined], gamma.value


