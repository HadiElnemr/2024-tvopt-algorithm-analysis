import numpy as np
from fct.objectives import PeriodicExample2D, WindowedLeastSquares
from lib.utils import consistent_polytope_nd
from lib.tracking_analysis import bisection_thm1, bisection_thm2
from fct.algorithms import GradientDescent, Nesterov, TMM, Algorithm
from lib.algorithms_unconstrained import gradient_descent, nesterov, triple_momentum

def calculate_L_m_bounds(objective, n_points=1000):
    """
    Calculate L_max, L_min, delta_L_max and delta_m_max for a given function

    Parameters:
    - objective: An instance of a function class (e.g., PeriodicExample2D, WindowedLeastSquares)
    - n_points: Number of points to sample in the domain for estimating bounds

    Returns:
    - L_min: Minimum Lipschitz constant
    - L_max: Maximum Lipschitz constant
    - delta_L_max: Maximum change in Lipschitz constant between consecutive points
    - delta_m_max: Maximum change in strong convexity constant between consecutive points
    """
    L_max, L_min, delta_L_max, delta_m_max
    t = 200
    for k in range(t+1):
        objective.update(k)


    # Sample points in the domain
    x_samples = np.random.uniform(low=[-100]*objective.nx, high=[100]*objective.nx, size=(n_points, objective.nx))
    print(f"x_samples shape: {x_samples.shape}")
    print(f"x_samples: {x_samples}")

    # Evaluate gradients and Hessians at sampled points
    gradients = np.array([objective.gradient(x) for x in x_samples])
    hessians = np.array([objective.hessian(x) for x in x_samples])

    # Compute eigenvalues of Hessians to estimate m and L
    eigenvalues = np.array([np.linalg.eigvalsh(hess) for hess in hessians])
    m_values = np.min(eigenvalues, axis=1)
    L_values = np.max(eigenvalues, axis=1)

    # Calculate L_min, L_max, delta_L_max, delta_m_max
    L_min = np.min(L_values)
    L_max = np.max(L_values)
    delta_L = np.abs(np.diff(L_values))
    delta_m = np.abs(np.diff(m_values))

    delta_L_max = np.max(delta_L) if len(delta_L) > 0 else 0
    delta_m_max = np.max(delta_m) if len(delta_m) > 0 else 0

    return L_min, L_max, delta_L_max, delta_m_max

def calculate_consistent_polytope(objective, n_points=1000):
    """
    Calculate the consistent polytope for the given objective function.

    Parameters:
    - objective: An instance of a function class (e.g., PeriodicExample2D, WindowedLeastSquares)
    - n_points: Number of points to sample in the domain for estimating the polytope

    Returns:
    - consistent_polytope: A list of tuples representing the consistent polytope
    """
    # Sample points in the domain
    x_samples = np.random.uniform(low=[-100]*objective.nx, high=[100]*objective.nx, size=(n_points, objective.nx))

    # Evaluate gradients at sampled points
    gradients = np.array([objective.gradient(x) for x in x_samples])

    # Compute the consistent polytope using the sampled gradients
    consistent_polytope = consistent_polytope_nd(gradients)

    return consistent_polytope

def compute_algorithm_parameters(algo_name, m, L, consistent_polytope):
    """
    Compute algorithm parameters using bisection methods for the given algorithm.

    Parameters:
    - algo_name: Name of the algorithm ('gradient_descent', 'nesterov', 'tmm')
    - m: Strong convexity constant
    - L: Lipschitz constant
    - consistent_polytope: A list of tuples representing the consistent polytope

    Returns:
    - algorithm: An instance of the algorithm class with updated parameters
    """
    # Select the appropriate algorithm class
    if algo_name == 'gradient_descent':
        algorithm = GradientDescent(m=m, L=L, nx=1, delta_model=False)
        algo = gradient_descent
    elif algo_name == 'nesterov':
        algorithm = Nesterov(m=m, L=L, nx=1, delta_model=False)
        algo = nesterov
    elif algo_name == 'tmm':
        algorithm = TMM(m=m, L=L, nx=1, delta_model=False)
        algo = triple_momentum
    else:
        raise ValueError("Unsupported algorithm name. Choose from 'gradient_descent', 'nesterov', or 'tmm'.")

    # Compute parameters for sector IQC
    rho_sec, sol_sec = bisection_thm1(algo=algo, consistent_polytope=consistent_polytope)
    algorithm.rho_sec = rho_sec
    algorithm.c_sec = sol_sec[0]
    algorithm.lambd = sol_sec[1]

    # Compute parameters for off-by-1 IQC
    rho, sol = bisection_thm2(algo=algo, consistent_polytope=consistent_polytope, optimize_bound=True)
    algorithm.rho = rho
    algorithm.c1, algorithm.c2 = sol[0]
    algorithm.lambd = sol[1]
    algorithm.gamma_xi = sol[2]
    algorithm.gamma_delta = sol[3]

    return algorithm

# Example usage:
if __name__ == "__main__":
    # Define the objective function
    objective = PeriodicExample2D(omega=0.1)

    # Calculate L bounds
    # L_min, L_max, delta_L_max, delta_m_max = calculate_L_m_bounds(objective)
    # print(f"L_min: {L_min}, L_max: {L_max}, delta_L_max: {delta_L_max}, delta_m_max: {delta_m_max}")

    L = 10
    m = 10
    n_grid = 2
    L_min, L_max = L * 0.8, L
    m_min, m_max = m * 0.4, m
    grid_step = (L_max - L_min) / n_grid

    params = np.array([np.linspace(L_min, L_max, n_grid + 1),
               np.linspace(m_min, m_max, n_grid + 1)])
    print(f"Parameter grid: {params}")
    delta_L_max = lambda rate_bound: rate_bound * (L_max - L_min)
    delta_m_max = lambda rate_bound: rate_bound * (m_max - m_min)

    grid_points = consistent_polytope_nd(params, np.array([-delta_L_max(rate_bound=0.05),-delta_m_max(rate_bound=0.05)]), np.array([-delta_L_max(rate_bound=0.05),-delta_m_max(rate_bound=0.05)]), step_size=grid_step)
    print(f"Grid points: {grid_points}")
    rho, sol = bisection_thm2(algo=gradient_descent, consistent_polytope=grid_points, optimize_bound=True)
    print(rho, sol)
    # rhos_ogd.append((rho, sol))
    # Calculate consistent polytope
    # consistent_polytope = calculate_consistent_polytope(objective)
    # print(f"Consistent Polytope: {consistent_polytope}")

    # Compute algorithm parameters for Nesterov's method
    # algorithm = compute_algorithm_parameters('nesterov', m=1, L=L_max, consistent_polytope=consistent_polytope)
    # print(f"Algorithm Parameters - rho: {algorithm.rho}, c1: {algorithm.c1}, c2: {algorithm.c2}, lambda: {algorithm.lambd}, gamma_xi: {algorithm.gamma_xi}, gamma_delta: {algorithm.gamma_delta}")