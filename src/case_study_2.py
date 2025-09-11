import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from dataclasses import dataclass
from tqdm import tqdm

from fct.objectives import PeriodicExample2D
from fct.algorithms import GradientDescent, Nesterov, TMM #, HeavyBall, MultiStepGradient, AcceleratedOGD, MultiStepOGD, ProximalGradient, ProximalHeavyBall, ProximalNesterov, ProximalTripleMomentum

from lib.tracking_analysis import bisection_thm1, \
                                  bisection_thm1_with_intermediate_rho_sens, \
                                  calculate_sensitivities_over_rhos_sectional, \
                                  bisection_thm2, \
                                  bisection_thm2_with_intermediate_rho_sens, \
                                  calculate_sensitivities_over_rhos

from lib.utils import consistent_polytope_nd, calculate_L_m_bounds
from lib.algorithms_unconstrained import gradient_descent, nesterov, triple_momentum, heavy_ball, multi_step_gradient
from lib.algorithms_constrained import accelerated_ogd, multi_step_ogd, proximal_gradient, proximal_heavy_ball, proximal_nesterov, proximal_triple_momentum

# ---------- wiring ----------
ALGO_MAP = {
    'gradient': gradient_descent,
    'nesterov': nesterov,
    'tmm':      triple_momentum,
    'heavy_ball': heavy_ball,
    'multi_step_gradient': multi_step_gradient,
    'accelerated_ogd': accelerated_ogd,
    'multi_step_ogd': multi_step_ogd,
    'proximal_gradient': proximal_gradient,
    'proximal_heavy_ball': proximal_heavy_ball,
    'proximal_nesterov': proximal_nesterov,
    'proximal_triple_momentum': proximal_triple_momentum,
}

ALGO_MAP_fct = {
    'gradient': GradientDescent,
    'nesterov': Nesterov,
    'tmm':TMM,
    # 'heavy_ball': HeavyBall,
    # 'multi_step_gradient': MultiStepGradient,
    # 'accelerated_ogd': AcceleratedOGD,
    # 'multi_step_ogd': MultiStepOGD,
    # 'proximal_gradient': ProximalGradient,
    # 'proximal_heavy_ball': ProximalHeavyBall,
    # 'proximal_nesterov': ProximalNesterov,
    # 'proximal_triple_momentum': ProximalTripleMomentum,
}


@dataclass
class Sens:
    rho: float
    c1: float
    c2: float
    sensitivity_f: float
    sensitivity_x: float
    sensitivity_g: float
    # lambd: float         # sensitivity_f
    # gamma_xi: float      # sensitivity_x
    # gamma_delta: float   # sensitivity_g

@dataclass
class Sens_sec:
    rho: float
    c: float # Cond(P)

def get_sensitivity_candidates(obj, algo_name, rate_bound=0.05, n_grid=2):
    algo = ALGO_MAP[algo_name]
    L_min, L_max, m_min, m_max, delta_L_max, delta_m_max, delta_L_min, delta_m_min = calculate_L_m_bounds(obj)
    grid_step = (L_max - L_min) / n_grid
    params = np.array([
        np.linspace(L_min, L_max, n_grid + 1),
        np.linspace(m_min, m_max, n_grid + 1),
    ])
    gp = consistent_polytope_nd(
        params,
        np.array([ delta_L_min, delta_m_min]),
        np.array([ delta_L_max, delta_m_max]),
        step_size=grid_step
    )

    # rho_sec, sol_sec = bisection_thm1(algo=algo, consistent_polytope=gp, optimize_bound=True) # TODO: add sectional iqc parameters
    # rho_sens_list_sec = [{'rho': rho_sec, 'c': sol_sec}]

    # rho_sec, sol_sec, rho_sens_list_sec = bisection_thm1_with_intermediate_rho_sens(algo=algo, consistent_polytope=gp, optimize_bound=True) # TODO: add sectional iqc parameters

    rho_sec, sol_sec, rho_sens_list_sec = calculate_sensitivities_over_rhos_sectional(algo=algo, consistent_polytope=gp, optimize_bound=True) # TODO: add sectional iqc parameters

    # rho_max, sol = bisection_thm2(
    #     algo=algo,
    #     consistent_polytope=gp,
    #     optimize_bound=True
    # )
    # rho_sens_list = [{'rho': rho_max, 'c1': sol[0][0], 'c2': sol[0][1], 'sensitivity_f': sol[1], 'sensitivity_x': sol[2], 'sensitivity_g': sol[3]}]

    # rho_max, sol_rho_max, rho_sens_list = bisection_thm2_with_intermediate_rho_sens(
    #     algo=algo,             # your bisection uses internal algo mapping; pass if needed
    #     consistent_polytope=gp,
    #     optimize_bound=True
    # )

    # Get all sensitivities between rho=0.2 to rho=1.0 in steps of 0.01
    rho_sens_list = calculate_sensitivities_over_rhos(
        algo=algo,             # your bisection uses internal algo mapping; pass if needed
        consistent_polytope=gp,
        optimize_bound=True
    )

    # Get all sensitivities with new objective functions
    # rho_sens_list = bisection_thm6(
    #     algo=algo,             # your bisection uses internal algo mapping; pass if needed
    #     consistent_polytope=gp,
    #     optimize_bound=True
    # )

    # rho_sens_list = bisection_thmx(
    #     algo=algo,
    #     consistent_polytope=gp,
    #     target='x',
    #     optimize_bound=True
    # )

    # normalize to Sens list
    out = []
    for d in rho_sens_list:
        out.append(Sens(
            rho=float(d['rho']),
            c1=float(d['c1']),
            c2=float(d['c2']),
            sensitivity_f=float(d['sensitivity_f']),
            sensitivity_x=float(d['sensitivity_x']),
            sensitivity_g=float(d['sensitivity_g']),
        ))

    out_sec = []
    for d in rho_sens_list_sec:
        out_sec.append(Sens_sec(
            rho=float(d['rho']),
            c=float(d['c'])
        ))
    return out, out_sec

# ---------- core sim for ONE sensitivity set ----------
def simulate_once(algo_name, obj, x0, T, sens: Sens):
    algo = ALGO_MAP_fct[algo_name]
    # algo = ALGO_MAP[algo_name] # TODO: Lets try to use this version

    ### Initialize the algorithm with the starting point
    algorithm = algo(m=1, L=1, nx=obj.nx)   # dummy init, will update per-step
    algorithm.initialize(x0)
    algorithm.p = 1

    if isinstance(sens, Sens):
        ### set sensitivities
        algorithm.rho       = sens.rho
        algorithm.c1        = sens.c1
        algorithm.c2        = sens.c2
        algorithm.sensitivity_f = sens.sensitivity_f
        algorithm.sensitivity_x = sens.sensitivity_x
        algorithm.sensitivity_g = sens.sensitivity_g
    elif isinstance(sens, Sens_sec):
        ### set sensitivities
        algorithm.rho_sec = sens.rho
        algorithm.c_sec = sens.c

    xi_0  = algorithm.internal_state
    x_km1 = x0.copy()

    ### initialize storage lists

    x_k_list = list()
    x_star_k_list = list()

    ### initialize rolling terms
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

    ### TODO: to be used if p>1
    Delta_delta_km1_s_km1 = []

    ### TODO: put this just in case but check
    x_km1 = np.array([0, 0])

    ### calculate initial xi_star
    obj.update(0)
    x_star_0, _, _ = obj.get_objective_info()
    if algo_name == 'gradient':
        xi_star_0 = x_star_0
    elif algo_name in ['nesterov','tmm']: # TODO: check else case
        xi_star_0 = np.concatenate([x_star_0, x_star_0])

    # Compute initial Xi delta value
    xi_delta_0 = xi_0 - xi_star_0 # initial difference between state and xi star

    for k in range(T):
        ### Update time-varying objective function
        obj.update(k)

        ### Update algorithm parameters and perform step
        x_star_k, m_k, L_k = obj.get_objective_info()

        x_star_k_list.append(x_star_k)
        m_k_list.append(m_k)
        L_k_list.append(L_k)


        algorithm.update_sectors(m_k, L_k)
        algorithm.update_gradient(lambda x: obj.gradient(x))

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


        if isinstance(sens, Sens_sec):
            ############################## Bound Error based on Sector IQC ##############################
            ### Compute error bound based on sector IQC
            bound_sec = algorithm.c_sec * algorithm.rho_sec**k * np.linalg.norm(xi_delta_0)
            if k >= 1:
                for jdx, Delta_xi_jm1 in enumerate(Delta_xi_list):
                    bound_sec += algorithm.c_sec * algorithm.rho_sec**(k - (jdx+1)) * np.linalg.norm(Delta_xi_jm1)

            error_bound_sector.append(bound_sec)
        else:
            ############################## Bound Error based on Off-by-1 IQC ##############################
            ### Compute error bound based on off-by-1 IQC
            # initial term: c1 * rho^{2k} * ||Δxi_0||^2
            bound_off = algorithm.c1 * algorithm.rho**(2*k) * np.linalg.norm(xi_delta_0)**2

            if k >= 1:
                if algorithm.p == 1:
                    # implement if p==1 the running algorithm
                    for jdx in range(1, k+1): # from `1` to `k`
                        Delta_xi_jm1 = Delta_xi_list[jdx-1]
                        delta_xi_term = algorithm.sensitivity_x * np.linalg.norm(Delta_xi_jm1)**2

                        delta_delta_term_sum = 0
                        for idx in range(algorithm.p):
                            # Delta_delta_km1_s_km1.append(
                            # inner_sum_1 += np.linalg.norm(Delta_delta_km1_s_km1[idx])**2
                            # since p=1, compute manually for now
                            delta_delta_term_sum += np.linalg.norm(Delta_delta_list[jdx-1])**2

                        delta_delta_term = algorithm.sensitivity_g * delta_delta_term_sum

                        bound_off += algorithm.c2 * algorithm.rho**(2*(k - (jdx+1))) * (
                                    delta_xi_term + delta_delta_term
                                    )
                    delta_f_hat_term_sum = 0
                    for idx in range(algorithm.p):
                        for t in range(k): # `0 to k-1`
                            # TODO: Extend to p>1
                                # USE ~ delta_f_hat_term_sum += np.linalg.norm(Delta_delta_t_s_list[t][idx])**2
                                # AND ~ algorithm.lambd[idx]
                            delta_f_hat_term_sum += algorithm.rho**(2*(k - t)) * algorithm.sensitivity_f * (
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

    return xi_tilde_norm_list, error_bound_offby1 if isinstance(sens, Sens) else error_bound_sector

# ---------- compare across sensitivities ----------
def compare_sensitivities(algo_name, obj, x0, T, sens_list, score="sum"):
    results = []
    idx = 0
    tracking_error = []
    
    for s in tqdm(sens_list, desc=f"{algo_name} sensitivities"):
        # Use a cache to avoid recomputation of simulate_once for the same parameters
        cache_dir = "sim_cache"
        os.makedirs(cache_dir, exist_ok=True)
        # Create a unique filename based on algo_name, T, and sensitivity parameters
        def sens_to_tuple(s):
            # Handles both Sens and Sens_sec dataclasses
            if hasattr(s, '__dataclass_fields__'):
                return tuple(getattr(s, f) for f in s.__dataclass_fields__)
            else:
                return tuple(s)
        cache_key = (algo_name, T, tuple(x0), sens_to_tuple(s))
        cache_file = os.path.join(cache_dir, f"sim_{hash(cache_key)}.pkl")

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                tracking_error, error_bound = pickle.load(f)
        else:
            tracking_error, error_bound = simulate_once(algo_name, obj, x0, T, s)
            print("idx:", idx, len(tracking_error))
            idx += 1
            with open(cache_file, "wb") as f:
                pickle.dump((tracking_error, error_bound), f)
        if score == "max":
            val = float(np.max(error_bound))
        elif score == "mean":
            val = float(np.mean(error_bound))
        elif score == "sum":
            val = float(np.sum(error_bound))
        else:
            raise ValueError("score ∈ {'max','mean','sum'}")
        results.append((val, s, error_bound))
    
    print("outside         idx:", idx, len(tracking_error))
    results.sort(key=lambda t: t[0])
    
    # Fixed: Check if sens_list is empty before accessing its first element
    if len(sens_list) > 0:
        print("algo_name:", algo_name, f"sens_list length {type(sens_list[0])}: {len(sens_list)}")
    else:
        print("algo_name:", algo_name, f"sens_list length: {len(sens_list)} (empty)")
    
    return tracking_error, results  # ranked: sorted best→worst according to val which could be the sum

# ---------- driver ----------
def run_comparison(algo_name, T=200, seed=3, score='sum'):
    np.random.seed(seed)
    obj = PeriodicExample2D(omega=0.1)
    x0 = np.ones(obj.nx) * 5

    sens_list, sens_list_sec = get_sensitivity_candidates(obj, algo_name, n_grid=2)
    if len(sens_list) == 0:
        print(f"No sensitivity candidates found for {algo_name}.")
        return

    tracking_error, ranked = compare_sensitivities(algo_name, obj, x0, T, sens_list, score=score)
    tracking_error_sec, ranked_sec = compare_sensitivities(algo_name, obj, x0, T, sens_list_sec, score=score)

    best_val, best_s, best_bound = ranked[0] # Take best result from off-by-1, which contains `score value`, `sensitivity`, and `error bound`
    
    if len(ranked_sec) == 0:
        print(f"[{algo_name}] No sectional sensitivity candidates found.")
        return
    
    best_val_sec, best_s_sec, best_bound_sec = ranked_sec[0] # Take best result from sectional

    print(f"[{algo_name}] best {score} bound = {best_val:.6g} with Sens("
          f"rho={best_s.rho:.4g}, c1={best_s.c1:.4g}, c2={best_s.c2:.4g}, "
          f"λ={best_s.sensitivity_f:.4g}, γξ={best_s.sensitivity_x:.4g}, γδ={best_s.sensitivity_g:.4g})")
    print(f"[{algo_name}] best {score} bound = {best_val_sec:.6g} with Sens_sec(ρ={best_s_sec.rho:.4g}, c={best_s_sec.c:.4g})")

    # plot for context
    plt.figure()

    plt.semilogy(tracking_error , label=f"{algo_name}")
    for idx, (v, s, y) in enumerate(ranked[:5]):
        plt.semilogy(y, label=f"Off-by-1: #{idx+1}: {v:.3g}, ρ={s.rho:.3g}")
    plt.grid(True); plt.legend();
    plt.title(f"{algo_name}")
    # plt.title(f"{algo_name}: off-by-1 bound")
    # plt.show()

    # plt.semilogy(tracking_error_sec, label=f"sec")
    for idx, (v, s, y) in enumerate(ranked_sec[:5]):
        plt.semilogy(y, label=f"Sectional: #{idx+1}: {v:.3g}, ρ={s.rho:.3g}")
    plt.grid(True); plt.legend();
    # plt.title(f"{algo_name}: sectional bound")
    plt.show()


if __name__ == "__main__":
    run_comparison('gradient', T=200, score='sum')
    run_comparison('nesterov',  T=200, score='sum')
    run_comparison('tmm',       T=200, score='sum')
    # run_comparison('heavy_ball',T=200, score='sum')
