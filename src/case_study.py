import numpy as np
import matplotlib.pyplot as plt
from fct.algorithms import GradientDescent, Nesterov, TMM
from fct.objectives import PeriodicExample2D, WindowedLeastSquares


def run_simulation():

    # Seed and time horizon
    np.random.seed(3)
    nk = 200

    # Select objective function
    omega = 0.1
    obj = PeriodicExample2D(omega=omega)
    # obj = WindowedLeastSquares()

    # Optimization algorithms
    algo_names = ['gradient', 'nesterov', 'tmm']
    algo_classes = {
        'gradient': GradientDescent,
        'nesterov': Nesterov,
        'tmm': TMM
    }

    # Initial condition
    nx = obj.nx
    x0 = np.ones(nx) * 5

    for algo_name in algo_names:
 
        ### Initialize the algorithm with the starting point
        algorithm = algo_classes[algo_name](m=1, L=1, nx=nx)
        algorithm.initialize(x0)
        xi_0 = algorithm.internal_state
        # algorithm.compute_bounds(p_set, delta_p_min, delta_p_max, @m(p), @L(p))

        ### initialize storage lists
        xi_tilde_norm_list = list()
        epsilon_list = list()
        Delta_xi_list = list()

        error_bound_sector = list()
        error_bound_offby1 = list()

        ### initialize values
        xi_tilde_0 = 0
        h_km1_xkm1 = 0  # old function evaluate at old iterate
        h_k_xk     = 0  # current function evaluated at current iterate

        for k in range(nk):

            h_km1_xkm1 = h_k_xk  # old function evaluated at old iterate

            ### Update objective function, compute epsilon
            obj.update(k)

            ### Update algorithm parameters and perform step
            x_star_k, m_k, L_k = obj.get_objective_info()
            gradient_function = lambda x: obj.gradient(x)

            algorithm.update_sectors(m_k, L_k)
            algorithm.update_gradient(gradient_function)

            xi_k, x_k = algorithm.step()

            xi_tilde_norm_list.append(np.linalg.norm(x_k - x_star_k))

            ### calculate xi_star
            if algo_name == 'gradient':
                xi_star_k = x_star_k
            elif algo_name in ['nesterov','tmm']:
                xi_star_k = np.concatenate([x_star_k, x_star_k])

            if k == 0:
                x_km1 = x_star_k
                xi_star_km1 = xi_star_k
                xi_tilde_0 = xi_0 - xi_star_k

            ### Determine the optimizer speed
            Delta_xi_km1 = xi_star_km1 - xi_star_k
            Delta_xi_list.append(Delta_xi_km1)

            ### Determine tau-induced residual epsilon
            h_k_xk = obj.eval(x_k)
            h_k_xkm1 = obj.eval(x_km1)
            epsilon_k = algorithm.rho**2 * (h_k_xkm1 - h_km1_xkm1)
            epsilon_list.append(epsilon_k)

            ### Compute error bound based on sector IQC
            bound_sec = algorithm.c_sec * algorithm.rho_sec**k * np.linalg.norm(xi_tilde_0)
            if k >= 1:
                for jdx, Delta_xi_jm1 in enumerate(Delta_xi_list):
                    bound_sec += algorithm.c_sec * algorithm.rho_sec**(k - (jdx+1)) * np.linalg.norm(Delta_xi_jm1)

            error_bound_sector.append(bound_sec)

            ### Compute error bound based on off-by-1 IQC
            bound_off = (algorithm.c2 / algorithm.c1) * algorithm.rho**(2*k) * np.linalg.norm(xi_tilde_0)**2
            if k >= 1:
                for jdx, (Delta_xi_jm1, epsilon_jm1) in enumerate(zip(Delta_xi_list, epsilon_list)):
                    bound_off += (1/algorithm.c1) * algorithm.rho**(2*(k - (jdx+1))) * (algorithm.gamma * np.linalg.norm(Delta_xi_jm1)**2 + algorithm.lambd * epsilon_jm1)

            error_bound_offby1.append(np.sqrt(bound_off))

            ### update values
            xi_star_km1 = xi_star_k
            x_km1 = x_k

        ### Plot results
        plt.figure()
        plt.semilogy(xi_tilde_norm_list, label=f'{algo_name}')
        plt.semilogy(error_bound_sector, label=f'sector bound: {algo_name}')
        plt.semilogy(error_bound_offby1, label=f'off-by-1 bound: {algo_name}')
        plt.grid()
        plt.legend()

    plt.show()


if __name__ == "__main__":
    run_simulation()
