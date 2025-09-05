import numpy as np
from fct.objectives import ObjectiveFunction
import matplotlib.pyplot as plt

def consistent_polytope_nd(params, delta_params_min, delta_params_max, step_size=0.1):
    """
    Generates grid points such that p_min <= p + delta_p <= p_max
    and delta_p_min <= delta_p <= delta_p_max for each parameter vector.

    Parameters:
        params (numpy.ndarray): A 2D array where each row represents a parameter vector.
        delta_params_min (numpy.ndarray): Minimum allowable deltas for each parameter.
        delta_params_max (numpy.ndarray): Maximum allowable deltas for each parameter.
        step_size (float): Step size for generating grid points in the delta ranges.

    Returns:
        list: A list of tuples, each containing (p_k, delta_p) where p_k is the parameter vector
              and delta_p is the corresponding delta vector satisfying the constraints.
    """
    # Handle 1D input by converting it to 2D for uniform processing
    if params.ndim == 1:
        params = params[None, :]  # Convert to shape (1, n)
        delta_params_min = np.array([delta_params_min])
        delta_params_max = np.array([delta_params_max])

    # Calculate global min and max for params directly using numpy
    p_min = np.min(params, axis=1)
    p_max = np.max(params, axis=1)

    # Initialize the grid points for (p, delta_p)
    grid_points = []

    # Loop through each parameter vector
    for k in range(params.shape[1]):
        p_k = params[:, k]

        # Determine feasible ranges for deltas for each dimension
        delta_min_k = np.maximum(delta_params_min, np.maximum(p_min - p_k, -delta_params_max))
        delta_max_k = np.minimum(delta_params_max, np.minimum(p_max - p_k, delta_params_max))

        # Generate grid points for all dimensions
        delta_ranges = []
        for d in range(params.shape[0]):
            delta_min = delta_min_k[d]
            delta_max = delta_max_k[d]
            num_dp_points = int((delta_max - delta_min) / step_size) + 1
            range_points = [
                min(delta_min + j * step_size, delta_max)  # Ensure boundary inclusion
                for j in range(num_dp_points)
            ]
            if range_points[-1] < delta_max:
                range_points.append(delta_max)

            delta_ranges.append(np.array(range_points))

        # Create a meshgrid of delta ranges and iterate through combinations
        delta_mesh = np.meshgrid(*delta_ranges, indexing="ij")
        delta_combinations = np.stack([delta.ravel() for delta in delta_mesh], axis=-1)

        for delta_p in delta_combinations:
            delta_p = np.clip(delta_p, delta_params_min, delta_params_max)
            grid_points.append((p_k, delta_p))

    return grid_points

def calculate_L_m_bounds(objective:ObjectiveFunction):
    """
    Calculate L_max, L_min, delta_L_max and delta_m_max for a given function

    Parameters:
    - objective: An instance of a function class (e.g., PeriodicExample2D, WindowedLeastSquares)

    Returns:
    - L_min: Minimum Lipschitz constant
    - L_max: Maximum Lipschitz constant
    - m_min: Minimum strong convexity constant
    - m_max: Maximum strong convexity constant
    - delta_L_max: Maximum change in Lipschitz constant between consecutive points
    - delta_m_max: Maximum change in strong convexity constant between consecutive points
    - delta_L_min: Minimum change in Lipschitz constant between consecutive points
    - delta_m_min: Minimum change in strong convexity constant between consecutive points
    """

    t = 200
    delta_L_max = 0
    delta_m_max = 0
    _, m_k, L_k = objective.get_objective_info()
    delta_L_min = L_k
    delta_m_min = m_k
    m_prev = m_k
    L_prev = L_k
    m_min = m_k
    m_max = m_k
    L_min = L_k
    L_max = L_k

    # Update the objective function over time
    for k in range(t+1):
        objective.update(k)
        _, m_k, L_k = objective.get_objective_info()

        if m_k - m_prev > delta_m_max:
            delta_m_max = m_k - m_prev
        if L_k - L_prev > delta_L_max:
            delta_L_max = L_k - L_prev
        if m_k - m_prev < delta_m_min:
            delta_m_min = m_k - m_prev
        if L_k - L_prev < delta_L_min:
            delta_L_min = L_k - L_prev
        if m_k < m_min:
            m_min = m_k
        if m_k > m_max:
            m_max = m_k
        if L_k < L_min:
            L_min = L_k
        if L_k > L_max:
            L_max = L_k
        m_prev = m_k
        L_prev = L_k

    return L_min, L_max, m_min, m_max, delta_L_max, delta_m_max, delta_L_min, delta_m_min

def visualize(grid_points, param_dim: int = None):
        """
        Visualizes the grid points in scatter plots.
        """

        p_values = np.array([p_k for p_k, _ in grid_points])
        delta_values = np.array([delta_p for _, delta_p in grid_points])
        num_dims = p_values.shape[1]  # Number of dimensions

        if param_dim is not None:
            if param_dim >= num_dims or param_dim < 0:
                print(f"Invalid dimension for visualization: {param_dim}")
                return

            # Single dimension plot
            plt.figure(figsize=(6, 4))
            plt.scatter(p_values[:, param_dim], delta_values[:, param_dim], alpha=0.5, label=f"Delta Values (Dim {param_dim+1})")
            plt.xlabel(f"Parameter Values (Dim {param_dim+1})")
            plt.ylabel(f"Delta Values (Dim {param_dim+1})")
            plt.title(f"Grid Point Visualization for Dimension {param_dim+1}")
            plt.show()
        else:
            # Multiple subplots for all dimensions
            fig, axes = plt.subplots(num_dims, 1, figsize=(6, 4 * num_dims))
            if num_dims == 1:
                axes = [axes]  # Ensure axes is always iterable

            for dim in range(num_dims):
                ax = axes[dim]
                ax.scatter(p_values[:, dim], delta_values[:, dim], alpha=0.5, label=f"Dim {dim}")
                ax.set_xlabel(f"Parameter Values (Dim {dim+1})")
                ax.set_ylabel(f"Delta Values (Dim {dim+1})")
                ax.set_title(f"Grid Point Visualization for Dimension {dim+1}")

            plt.tight_layout()
            plt.show()