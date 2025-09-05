import numpy as np
import matplotlib.pyplot as plt
from fct.algorithms import GradientDescent, Nesterov, TMM, Algorithm # To be removed later
from fct.objectives import PeriodicExample2D, WindowedLeastSquares
from lib.tracking_analysis import bisection_thm1, bisection_thm2, bisection_thm3
from lib.utils import consistent_polytope_nd, calculate_L_m_bounds, visualize
from lib.algorithms_unconstrained import gradient_descent, nesterov, heavy_ball, triple_momentum
from tqdm import tqdm


def run_simulation(algo_names=['gradient', 'nesterov', 'tmm'], obj=None, x0=None, T=None):
    """Run simulation for given algorithms and objective function.

    Parameters:
    - algo_names: List of algorithm names to simulate. Options are 'gradient', 'nesterov', 'tmm'.
    - obj: Objective function instance. If None, defaults to PeriodicExample2D.
    - x0: Initial point. If None, defaults to a vector of fives.
    - T: Time horizon. If None, defaults to 200.
    """
    # Seed and time horizon
    np.random.seed(3)
    if T is None:
        T = 200
    nk = T

    # Select objective function
    omega = 0.1
    obj = PeriodicExample2D(omega=omega)
    # obj = WindowedLeastSquares()

    # Optimization algorithms
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

        algorithm:Algorithm = algo_classes[algo_name](m=1, L=1, nx=nx)
        algorithm.initialize(x0)
        algorithm.p = 1
        xi_0 = algorithm.internal_state
        # algorithm.compute_bounds(p_set, delta_p_min, delta_p_max, @m(p), @L(p))

        ### initialize storage lists

        x_k_list = list()
        x_star_k_list = list()

        xi_tilde_norm_list = list()
        Delta_xi_list = list()
        Delta_delta_list = list()
        Delta_f_hat_list = list()

        error_bound_sector = list()
        error_bound_offby1 = list()

        ### initialize values
        xi_delta_0 = 0
        grad_km1_s_km1 = 0 # old gradient evaluate at old iterate
        grad_k_s_k = 0 # current gradient evaluate at current iterate

        m_k_list, L_k_list = list(), list()
        m_k, L_k = 1, 1

        # compute algorithm parameters for off-by-1 IQC
        algo = nesterov if algo_name == 'nesterov' else triple_momentum if algo_name == 'tmm' else gradient_descent if algo_name == 'gradient' else heavy_ball if algo_name == 'heavy_ball' else None
        if algo is None:
            raise ValueError(f"Algorithm {algo_name} not recognized.")

        # Calculate L, m bounds
        L_min, L_max, m_min, m_max, delta_L_max, delta_m_max, delta_L_min, delta_m_min = calculate_L_m_bounds(obj)
        # print("L, m bounds:", L_min, L_max, m_min, m_max, delta_L_max, delta_m_max, delta_L_min, delta_m_min)

        n_grid = 2
        grid_step = (L_max - L_min) / n_grid

        params = np.array([np.linspace(L_min, L_max, n_grid + 1),
            np.linspace(m_min, m_max, n_grid + 1)])
        # Grid points consisting of p_k and delta_p === p(theta) Δtheta
        grid_points = consistent_polytope_nd(params, np.array([ delta_L_min, delta_m_min]),
                                                     np.array([ delta_L_max, delta_m_max]),
                                                     step_size=grid_step)
        # visualize(grid_points)
        # print(grid_points)

        # rho_sec, sol_sec = bisection_thm1(algo=algo, consistent_polytope=[(np.array([L_k]), 0)], optimize_bound=True)

        rho_sec, sol_sec = bisection_thm1(algo=algo, consistent_polytope=grid_points, optimize_bound=True)
        algorithm.rho_sec = rho_sec
        algorithm.c_sec = sol_sec

        rho, sol = bisection_thm2(algo=algo, consistent_polytope=grid_points, optimize_bound=True)
        algorithm.rho = rho
        algorithm.c1,algorithm.c2 = sol[0]
        algorithm.lambd = sol[1]
        algorithm.gamma_xi = np.asarray(sol[2]).item()
        algorithm.gamma_delta = np.asarray(sol[3]).item()

        # print(f"Algorithm parameters for {algo_name}: rho, {algorithm.rho}, c1, {algorithm.c1}, c2, {algorithm.c2}, lambda, {algorithm.lambd}, gamma_xi, {algorithm.gamma_xi}, gamma_delta, {algorithm.gamma_delta}")

        # Delta_delta_km1_s_km1 = [] # TODO to be used if p>1

        ## TODO: put this just in case but check
        x_km1 = np.array([0, 0])

        print("Starting obj function simulation")

        ### calculate initial xi_star
        obj.update(0)
        x_star_0, _, _ = obj.get_objective_info()
        if algo_name == 'gradient':
            xi_star_0 = x_star_0
        elif algo_name in ['nesterov','tmm']: # TODO: check else case
            xi_star_0 = np.concatenate([x_star_0, x_star_0])

        # Compute initial Xi delta value
        xi_delta_0 = xi_0 - xi_star_0 # initial difference between state and xi star

        for k in tqdm(range(nk)):

            ### Update objective function
            obj.update(k)

            ### Update algorithm parameters and perform step
            x_star_k, m_k, L_k = obj.get_objective_info()

            x_star_k_list.append(x_star_k)
            m_k_list.append(m_k)
            L_k_list.append(L_k)

            gradient_function = lambda x: obj.gradient(x)

            algorithm.update_sectors(m_k, L_k)
            algorithm.update_gradient(gradient_function)

            xi_k, x_k = algorithm.step()

            x_k_list.append(x_k)

            # compute current gradient
            grad_k_s_k = obj.gradient(x_k) # TODO: gradient_function(.)
            grad_k_s_km1 = obj.gradient(x_km1) # TODO: gradient_function(.)

            # compute current gradient variation
            Delta_delta_km1_s_km1 = grad_km1_s_km1 - grad_k_s_km1 # since p=1, compute manually for now # TODO: to be modified if p>1
            Delta_delta_list.append(Delta_delta_km1_s_km1)

            ### calculate current xi_star_k
            if algo_name == 'gradient':
                xi_star_k = x_star_k
            elif algo_name in ['nesterov','tmm']: # TODO: check else case
                xi_star_k = np.concatenate([x_star_k, x_star_k])

            ### Determine the xi error
            xi_tilde_norm_list.append(np.linalg.norm(xi_k - xi_star_k)**2)

            if k >= 1:
                ### Determine the optimizer speed
                Delta_xi_km1 = xi_star_km1 - xi_star_k
                Delta_xi_list.append(Delta_xi_km1)

                ## Determine f_hat function variation when k>0
                f_hat_km1_s_km1 = obj.eval(x_km1, prev_t=True) - obj.eval(x_star_km1, prev_t=True)
                f_hat_k_s_km1   = obj.eval(x_km1)   - obj.eval(x_star_k) ## TODO: check: x_star_km1 -> x_star_k
                Delta_f_hat = (L_km1 - m_km1) * f_hat_km1_s_km1 - (L_k - m_k) * f_hat_k_s_km1
                Delta_f_hat_list.append(Delta_f_hat)


            ############################## Bound Error based on Sector IQC ##############################

            ### Compute error bound based on sector IQC
            bound_sec = algorithm.c_sec * algorithm.rho_sec**k * np.linalg.norm(xi_delta_0)
            if k >= 1:
                for jdx, Delta_xi_jm1 in enumerate(Delta_xi_list):
                    bound_sec += algorithm.c_sec * algorithm.rho_sec**(k - (jdx+1)) * np.linalg.norm(Delta_xi_jm1)

            error_bound_sector.append(bound_sec)

            ############################## Bound Error based on Off-by-1 IQC ##############################
            ### Compute error bound based on off-by-1 IQC
            # initial term: c1 * rho^{2k} * ||Δxi_0||^2
            bound_off = algorithm.c1 * algorithm.rho**(2*k) * np.linalg.norm(xi_delta_0)**2

            if k >= 1:
                if algorithm.p == 1:
                    # implement if p==1 the running algorithm
                    for jdx in range(1, k+1): # from `1` to `k`
                        Delta_xi_jm1 = Delta_xi_list[jdx-1]
                        delta_xi_term = algorithm.gamma_xi * np.linalg.norm(Delta_xi_jm1)**2

                        delta_delta_term_sum = 0
                        for idx in range(algorithm.p):
                            # Delta_delta_km1_s_km1.append(
                            # inner_sum_1 += np.linalg.norm(Delta_delta_km1_s_km1[idx])**2
                            # since p=1, compute manually for now
                            delta_delta_term_sum += np.linalg.norm(Delta_delta_list[jdx-1])**2

                        delta_delta_term = algorithm.gamma_delta * delta_delta_term_sum

                        bound_off += algorithm.c2 * algorithm.rho**(2*(k - (jdx+1))) * (
                                    delta_xi_term + delta_delta_term
                                    )
                    delta_f_hat_term_sum = 0
                    for idx in range(algorithm.p):
                        for t in range(k): # `0 to k-1`
                            # TODO: Extend to p>1
                                # USE ~ delta_f_hat_term_sum += np.linalg.norm(Delta_delta_t_s_list[t][idx])**2
                                # AND ~ algorithm.lambd[idx]
                            delta_f_hat_term_sum += algorithm.rho**(2*(k - t)) * algorithm.lambd * (
                                Delta_f_hat_list[t]
                            )
                    delta_f_hat_term = delta_f_hat_term_sum
                    bound_off += delta_f_hat_term

            error_bound_offby1.append(np.sqrt(bound_off))

            ### update xi_star_km1 and x_km1 values
            xi_star_km1 = xi_star_k
            x_km1 = x_k

            ### update grad_km1_s_km1 value
            grad_km1_s_km1 =  grad_k_s_k # old gradient evaluated at old iterate

            ### update x_star_km1 value
            x_star_km1 = x_star_k # old optimal point

            ### update previous L, m values
            m_km1, L_km1 = m_k, L_k


        ### Plot results
        plt.figure()
        plt.semilogy(xi_tilde_norm_list, label=f'{algo_name}')
        plt.semilogy(error_bound_sector, label=f'sector bound: {algo_name}')
        plt.semilogy(error_bound_offby1, label=f'off-by-1 bound: {algo_name}')
        plt.grid()
        plt.legend()

    plt.show()


if __name__ == "__main__":
    run_simulation(algo_names=['gradient', 'nesterov', 'tmm'], T=200)
