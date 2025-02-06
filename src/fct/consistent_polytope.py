import numpy as np
import matplotlib.pyplot as plt

class ConsistentPolytope:
    def __init__(self, params, delta_params_min, delta_params_max, step_size=0.1):
        """
        Initializes the class and precomputes the grid points.

        Parameters:
            params (numpy.ndarray): A 2D array where each row represents a parameter vector.
            delta_params_min (numpy.ndarray): Minimum allowable deltas for each parameter.
            delta_params_max (numpy.ndarray): Maximum allowable deltas for each parameter.
            step_size (float): Step size for generating grid points in the delta ranges.
        """
        self.grid_points = self._generate_grid(params, delta_params_min, delta_params_max, step_size)

    def _generate_grid(self, params, delta_params_min, delta_params_max, step_size):
        """Computes the grid points satisfying the constraints."""
        if params.ndim == 1:
            params = params[None, :]  # Convert to shape (1, n)
            delta_params_min = np.array([delta_params_min])
            delta_params_max = np.array([delta_params_max])

        p_min = np.min(params, axis=1)
        p_max = np.max(params, axis=1)
        grid_points = []

        for k in range(params.shape[1]):
            p_k = params[:, k]
            delta_min_k = np.maximum(delta_params_min, np.maximum(p_min - p_k, -delta_params_max))
            delta_max_k = np.minimum(delta_params_max, np.minimum(p_max - p_k, delta_params_max))
            delta_ranges = []

            for d in range(params.shape[0]):
                delta_min = delta_min_k[d]
                delta_max = delta_max_k[d]
                num_dp_points = int((delta_max - delta_min) / step_size) + 1
                range_points = [
                    min(delta_min + j * step_size, delta_max)
                    for j in range(num_dp_points)
                ]
                if range_points[-1] < delta_max:
                    range_points.append(delta_max)

                delta_ranges.append(np.array(range_points))

            delta_mesh = np.meshgrid(*delta_ranges, indexing="ij")
            delta_combinations = np.stack([delta.ravel() for delta in delta_mesh], axis=-1)

            for delta_p in delta_combinations:
                delta_p = np.clip(delta_p, delta_params_min, delta_params_max)
                grid_points.append((p_k, delta_p))

        return grid_points

    def __iter__(self):
        """Allows iteration over the grid points."""
        return iter(self.grid_points)

    def visualize(self, param_dim: int = None):
        """
        Visualizes the grid points in scatter plots.

        Parameters:
            param_dim (int, optional): The index of the dimension to plot.
                                       If None, plots all dimensions separately.
        """
        if len(self.grid_points) == 0:
            print("No grid points to visualize.")
            return
        
        p_values = np.array([p_k for p_k, _ in self.grid_points])
        delta_values = np.array([delta_p for _, delta_p in self.grid_points])
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
