import numpy as np
import matplotlib.pyplot as plt
from fct.algorithms import GradientDescent, Nesterov, TMM, Algorithm # To be removed later
from fct.objectives import PeriodicExample2D, WindowedLeastSquares
from lib.tracking_analysis import bisection_thm1, bisection_thm2, bisection_thm3
from lib.utils import consistent_polytope_nd, calculate_L_m_bounds
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
        # define variable as Integer

        algorithm:Algorithm = algo_classes[algo_name](m=1, L=1, nx=nx)
        algorithm.initialize(x0)
        algorithm.p = 1
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
        f_hat_km1_xm1 = 0  # old function evaluate at old iterate
        f_hat_k_x     = 0  # current function evaluated at current iterate
        grad_km1_sm1 = 0 # old gradient evaluate at old iterate
        grad_k_s = 0 # current gradient evaluate at current iterate

        Delta_delta_t_s_list = []
        Delta_delta_t_s = []
        m_k, L_k = 1, 1

        # compute algorithm parameters for off-by-1 IQC
        algo = nesterov if algo_name == 'nesterov' else triple_momentum if algo_name == 'tmm' else gradient_descent if algo_name == 'gradient' else heavy_ball if algo_name == 'heavy_ball' else None
        if algo is None:
            raise ValueError(f"Algorithm {algo_name} not recognized.")

        # Calculate L bounds
        L_min, L_max, m_min, m_max, delta_L_max, delta_m_max = calculate_L_m_bounds(obj)

        L_range = np.logspace(np.log10(1/0.8 + 1e-3), 2, 10)
        m_range = L_range * 0.4
        # L = 10
        # m = 10
        n_grid = 2
        grid_step = (L_max - L_min) / n_grid

        params = np.array([np.linspace(L_min, L_max, n_grid + 1),
            np.linspace(m_min, m_max, n_grid + 1)])

        delta_L_max = lambda rate_bound: rate_bound * (L_max - L_min)
        delta_m_max = lambda rate_bound: rate_bound * (m_max - m_min)

        grid_points = consistent_polytope_nd(params, np.array([-delta_L_max(rate_bound=0.05),-delta_m_max(rate_bound=0.05)]),
                                                np.array([-delta_L_max(rate_bound=0.05),-delta_m_max(rate_bound=0.05)]),
                                                step_size=grid_step)

        rho_sec, sol_sec = bisection_thm1(algo=algo, consistent_polytope=[(np.array([L_k]), 0)], optimize_bound=True)
        algorithm.rho_sec = rho_sec
        algorithm.c_sec = sol_sec

        rho, sol = bisection_thm2(algo=algo, consistent_polytope=[(np.array([L_k]), 0)], optimize_bound=True)
        algorithm.rho = rho
        algorithm.c1,algorithm.c2 = sol[0]
        algorithm.lambd = sol[1]
        algorithm.gamma_xi = float(sol[2])
        algorithm.gamma_delta = float(sol[3])

        # print(f"Algorithm parameters for {algo_name}: rho, {algorithm.rho}, c1, {algorithm.c1}, c2, {algorithm.c2}, lambda, {algorithm.lambd}, gamma_xi, {algorithm.gamma_xi}, gamma_delta, {algorithm.gamma_delta}")

        # Delta_delta_tm1_sm1 = [] # TODO to be used if p>1

        for k in tqdm(range(nk)):

            h_km1_xkm1 = h_k_xk  # old function evaluated at old iterate

            f_hat_km1_xm1 = f_hat_k_x # old function evaluated at old iterate

            grad_km1_sm1 =  grad_k_s # old gradient evaluated at old iterate

            x_star_km1 = x_star_k if k > 0 else x0 # old optimal point

            ### Update objective function, compute epsilon
            obj.update(k)

            ### Update previous L, m values
            m_km1, L_km1 = m_k, L_k
            # print(f"Time step {k}, objective parameters: m = {m_k}, L = {L_k}, ratio = {L_k/m_k}")

            ### Update algorithm parameters and perform step
            x_star_k, m_k, L_k = obj.get_objective_info()
            gradient_function = lambda x: obj.gradient(x)
            gradient_function_at_tm1 = lambda x,t: obj.gradient(x, prev_t=True)
            old_gradient_function = algorithm.gradient_function

            algorithm.update_sectors(m_k, L_k)
            algorithm.update_gradient(gradient_function)

            xi_k, x_k = algorithm.step()

            # compute current function value variation
            f_hat_k_x = obj.eval(x_k) - obj.eval(x_star_k)

            # compute current gradient
            grad_k_s = obj.gradient(x_k)

            # compute current gradient variation
            Delta_delta_tm1_sm1 = grad_km1_sm1 - grad_k_s # ?????

            # Compute xi_tilde
            # xi_tilde_norm_list.append(np.linalg.norm(x_k - x_star_k)) # Tilde or delta?

            # xi_tilde_k = x_k - x_star_k
            # xi_tilde_norm_list.append(np.linalg.norm(x_k - x_star_k))
            # xi_tilde_norm_list.append(np.linalg.norm(xi_tilde_k)**2)

            ### calculate Δxi_star
            if algo_name == 'gradient':
                xi_star_k = x_star_k
                # xi_star_k_vec
            elif algo_name in ['nesterov','tmm']:
                xi_star_k = np.concatenate([x_star_k, x_star_k])

            xi_tilde_norm_list.append(np.linalg.norm(xi_k - xi_star_k))  # <<< CHANGED

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
            # explain: epsilon_k is h_k(x_{k-1}) - h_{k-1}(x_{k-1}), no extra rho factor   # <<< CHANGED
            epsilon_k = (h_k_xkm1 - h_km1_xkm1)                                   # <<< CHANGED

            epsilon_list.append(epsilon_k)

            ############################## Bound Error based on Sector IQC ##############################

            ### Compute error bound based on sector IQC
            bound_sec = algorithm.c_sec * algorithm.rho_sec**k * np.linalg.norm(xi_tilde_0)
            if k >= 1:
                for jdx, Delta_xi_jm1 in enumerate(Delta_xi_list):
                    bound_sec += algorithm.c_sec * algorithm.rho_sec**(k - (jdx+1)) * np.linalg.norm(Delta_xi_jm1)

            error_bound_sector.append(bound_sec)

            ################################ For testing purposes only
            # test_old = not True
            # if test_old:
            #     bound_off = (algorithm.c2 / algorithm.c1) * algorithm.rho**(2*k) * np.linalg.norm(xi_tilde_0)**2
            #     if k >= 1:
            #         for jdx, (Delta_xi_jm1, epsilon_jm1) in enumerate(zip(Delta_xi_list, epsilon_list)):
            #             bound_off += (1/algorithm.c1) * algorithm.rho**(2*(k - (jdx+1))) * (algorithm.gamma * np.linalg.norm(Delta_xi_jm1)**2 + algorithm.lambd * epsilon_jm1)
            #     error_bound_offby1.append(np.sqrt(bound_off))
            #     continue
            ################################

            ############################## Bound Error based on Off-by-1 IQC ##############################
            ### Compute error bound based on off-by-1 IQC
            # initial term: c1 * rho^{2k} * ||\tilde{\xi}_0||^2
            bound_off = algorithm.c1 * algorithm.rho**(2*k) * np.linalg.norm(xi_tilde_0)**2

            if k >= 1:
                for jdx, (Delta_xi_jm1, epsilon_jm1) in enumerate(zip(Delta_xi_list, epsilon_list)):
                    # bound_off += (1/algorithm.c1) * algorithm.rho**(2*(k - (jdx+1))) * (algorithm.gamma * np.linalg.norm(Delta_xi_jm1)**2 + algorithm.lambd * epsilon_jm1)
                    # sum1 = c2 * rho^{2(k-t)} * [ gamma_xi * ||\Delta \xi_{t-1}||^2 + gamma_delta * inner_sum_1 ]
                    delta_xi_term = algorithm.gamma_xi * np.linalg.norm(Delta_xi_jm1)**2

                    inner_sum_1 = 0
                    for idx in range(algorithm.p):
                        # Delta_delta_tm1_sm1.append(
                        # inner_sum_1 += np.linalg.norm(Delta_delta_tm1_sm1[idx])**2
                        # since p=1, compute manually for now
                        inner_sum_1 += np.linalg.norm(Delta_delta_tm1_sm1)**2

                    delta_delta_term = algorithm.gamma_delta * inner_sum_1

                    bound_off += algorithm.c2 * algorithm.rho**(2*(k - (jdx+1))) * (
                                delta_xi_term + delta_delta_term
                                )
                inner_sum_2 = 0
                for idx in range(algorithm.p):
                    for t in range(k):
                        # inner_sum_2 += np.linalg.norm(Delta_delta_t_s_list[t][idx])**2
                        f_hat_tm1_s_tm1 = obj.eval(x_km1, prev_t=True) - obj.eval(x_star_km1, prev_t=True)
                        f_hat_t_s_tm1   = obj.eval(x_km1)   - obj.eval(x_star_km1)

                        inner_sum_2 += algorithm.rho**(2*(k - t)) * algorithm.lambd * (
                            (L_km1 - m_km1) * f_hat_tm1_s_tm1 -
                            (L_k - m_k) * f_hat_t_s_tm1
                        )

                bound_off += inner_sum_2

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
    run_simulation(algo_names=['gradient', 'nesterov', 'tmm'], T=200)
