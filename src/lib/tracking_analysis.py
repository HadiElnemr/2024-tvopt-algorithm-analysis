import numpy as np
import cvxpy as cvx
import control as ctrl
import scipy.linalg as linalg

from lib.lyapunov_matrix import PolynomialLyapunovMatrix
from lib.lure import lti_stack


def bisection_thm1(algo, consistent_polytope, rho_max=1.5, eps=1e-6):
    # will be provided
    pass


def bisection_thm2(algo, consistent_polytope, optimize_sensitivities=False, rho_max=1.5, eps=1e-6):

    sol = None

    # Get algorithm dimensions
    G, p, q = algo(1,1)
    n_xi = G.nstates
    n_g  = G.ninputs

    ### get dimensions ###
    n_zeta = 4
    n_eta0 = n_xi + n_zeta

    # zeros and identities
    Z_1xi = np.zeros((1,n_xi))
    I_n_eta = np.eye(n_eta0)

    ### start bisection ###
    rho_min = 0
    rho_tol = 1e-3

    while (rho_max-rho_min > rho_tol):

        rho = (rho_min + rho_max)/2

        ### SDP variables ###
        LMI_system = list()

        lyap = PolynomialLyapunovMatrix(param_dim=1, poly_degree=2, n_eta=n_eta0)

        lambd_sector = cvx.Variable(1, nonneg=True)
        lambd_offby1 = cvx.Variable(1, nonneg=True)

        gamm_xi = cvx.Variable(1, nonneg=True)
        gamm_d = cvx.Variable(1, nonneg=True)
        t      = cvx.Variable(1, nonneg=True)
        t_I    = cvx.multiply(t, I_n_eta)

        ### grid over parameter space ###
        for p_k, delta_p in consistent_polytope:

            p_kp1 = p_k + delta_p

            P_k   = lyap.P(p_k)
            P_kp1 = lyap.P(p_kp1)

            ### algorithm ###
            m, L = 1, p_k[0]
            G, p, q = algo(m,L)
            AG, BG, CG, DG = ctrl.ssdata(G)

            ### augment plant with delta models ###
            BG_aug = np.block([[BG, np.eye(n_xi), np.zeros((n_xi,n_g))]])
            DG_aug = np.block([[DG, np.zeros((n_g, n_xi)), np.zeros((n_g,n_g))]])
            G_aug  = ctrl.ss(AG, BG_aug, CG, DG_aug, dt=1)

            ### Sector IQC ###
            D_psi = np.asarray([[L, -1],
                                [-m, 1]])
            D_psi = np.block([[D_psi, np.zeros((2,n_xi+1))]])

            Psi_sector = ctrl.ss([],[],[],D_psi,dt=1)

            ### Off-by-1 IQC ###
            a = np.sqrt(m*(L-m)/2)
            A_psi = np.zeros((4, 4))
            B_psi = np.block([[1,  0,    CG,  0],
                              [0,  1, Z_1xi, -1],
                              [a,  0, Z_1xi,  0],
                              [-m, 1, Z_1xi,  0]])
            C_psi = np.asarray([[-L*rho**2, rho**2, 0,    0],
                                [0,         0,      0,    0],
                                [0,         0,      rho,  0],
                                [rho*a,     0,      0,    0],
                                [0,         0,      0,  rho],
                                [-m*rho,  rho,      0,    0]])
            D_psi = np.block([[L, -1, Z_1xi, 0],
                              [-m, 1, Z_1xi, 0],
                              [np.zeros((4,3+n_xi))]])

            Psi_offby1 = ctrl.ss(A_psi, B_psi, C_psi, D_psi, dt=1)

            Psi = lti_stack(Psi_sector, Psi_offby1)

            # build Lur'e system
            n_in = G_aug.ninputs
            G_I = lti_stack(G_aug, np.eye(n_in))
            G_hat = ctrl.series(G_I, Psi)
            A_hat, B_hat, C_hat, D_hat = ctrl.ssdata(G_hat)

            # get dimensions
            n_eta = A_hat.shape[0]
            n_psi = C_hat.shape[0]

            ### Multiplier ###
            M_sector = np.asarray([[0,1],
                                   [1,0]])
            M_offby1 = linalg.block_diag(1/2*M_sector,
                                         np.asarray([[1,0],
                                                     [0,-1]]),
                                         1/2*np.asarray([[1,0],
                                                         [0,-1]]))
            Multiplier = cvx.bmat([[lambd_sector * M_sector,         np.zeros((2,6))],
                                   [np.zeros((6,2)),         lambd_offby1 * M_offby1]])

            LMI_inner = cvx.bmat([
                [-rho**2 * P_k,              np.zeros((n_eta, n_eta)), np.zeros((n_eta, n_psi)), np.zeros((n_eta, n_xi)),             np.zeros((n_eta, 1))],
                [np.zeros((n_eta, n_eta)),   P_kp1,                    np.zeros((n_eta, n_psi)), np.zeros((n_eta, n_xi)),             np.zeros((n_eta, 1))],
                [np.zeros((n_psi, n_eta)),   np.zeros((n_psi, n_eta)), Multiplier,               np.zeros((n_psi, n_xi)),             np.zeros((n_psi, 1))],
                [np.zeros((n_xi, n_eta)),    np.zeros((n_xi, n_eta)),  np.zeros((n_xi, n_psi)), -cvx.multiply(gamm_xi, np.eye(n_xi)), np.zeros((n_xi, 1)) ],
                [np.zeros((1, n_eta)),       np.zeros((1, n_eta)),     np.zeros((1, n_psi)),     np.zeros((1, n_xi)),                -cvx.multiply(gamm_d, np.eye(1))]
            ])


            LMI_outer = cvx.bmat([
                [np.eye(n_eta), np.zeros((n_eta, n_in))],
                [cvx.bmat([[A_hat, B_hat]])],
                [cvx.bmat([[C_hat, D_hat]])],
                [np.zeros((n_xi, n_eta + 1)), np.eye(n_xi), np.zeros((n_xi,1))],
                [np.zeros((1,n_eta+1+n_xi)), np.eye(1)]
            ])

            LMI = LMI_outer.T @ LMI_inner @ LMI_outer

            LMI_system.append(P_k   >> eps*np.eye(n_eta))
            LMI_system.append(P_kp1 >> eps*np.eye(n_eta))
            LMI_system.append(LMI << 0)

            LMI_system.append(P_k   << t_I)
            LMI_system.append(P_kp1 << t_I)
            LMI_system.append(cvx.bmat([[P_k,   I_n_eta], [I_n_eta, t_I]]) >> 0)
            LMI_system.append(cvx.bmat([[P_kp1, I_n_eta], [I_n_eta, t_I]]) >> 0)

        # solve problem
        cost = cvx.Minimize(t + lambd_offby1 + gamm_xi + gamm_d) if optimize_sensitivities else cvx.Minimize(0)
        problem = cvx.Problem(cost, LMI_system)

        try:
            problem.solve(solver=cvx.MOSEK)
        except cvx.SolverError as e:
            # print("MOSEK failed:", e)
            # problem.solve(solver=cvx.SCS)
            pass
    
        if problem.status == cvx.OPTIMAL:
            ### solution found, decrease rho, save solution
            rho_max = rho

            eig_min, eig_max = lyap.min_max_eigval(list(zip(*consistent_polytope))[0])
            c1 = eig_max / eig_min
            c2 = 1 / eig_min

            lambd_bar, gamm_xi, gamm_d = lambd_offby1.value, gamm_xi.value, gamm_d.value

            sol = ((c1,c2), lambd_bar, gamm_xi, gamm_d)
        else:
            ### infeasible, increase rho
            rho_min = rho

        del lyap

    return rho_max, sol