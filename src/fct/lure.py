import numpy as np
import control as ctrl
import cvxpy as cvx
import scipy.linalg as linalg
from abc import ABC, abstractmethod


class LureComponent(ABC):
    """ 
    Component of a Lur'e system. Can be an algorithm or an IQC.
    """

    def __init__(self, m=1, L=1, delta_model=False):
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.internal_state_dim = None
        self.delta_model = delta_model
        self.m = m   # lower sector constant
        self.L = L   # upper sector constant
    
    @abstractmethod
    def _update_state_space(self):
        pass

    def update_sectors(self, m, L):
        self.m = m
        self.L = L
        self._update_state_space()
    
    def get_state_space(self):
        return ctrl.ss(self.A, self.B, self.C, self.D, dt=1)



class LureSystem():
    """ 
    Lur'e system consisting of the algorithm and IQCs 
    """
    
    def __init__(self, algo, IQCs):
        self.algo = algo
        self.IQCs = IQCs

        # check that delta_model setting is unanimous
        assert len(set([algo.delta_model] + [iqc.delta_model for iqc in IQCs])) <= 1

        self.n_xi   = algo.internal_state_dim
        self.n_zeta = np.sum([iqc.internal_state_dim for iqc in IQCs])
        self.n_z    = np.sum([iqc.M.shape[0] for iqc in IQCs])

        self.n_eta = self.n_xi + self.n_zeta
        self.n_g   = algo.nx + self.n_xi if algo.delta_model else algo.nx


    def _lti_stack(self, sys1, sys2):
        
        A1, B1, C1, D1 = ctrl.ssdata(sys1)
        A2, B2, C2, D2 = ctrl.ssdata(sys2)

        if B1.shape[1] != B2.shape[1] or D1.shape[1] != D2.shape[1]:
            raise ValueError('Error in system stacking: number of inputs must be the same for both subsystems!')

        A = linalg.block_diag(A1, A2)
        B = np.vstack((B1, B2))
        C = linalg.block_diag(C1, C2)
        D = np.vstack((D1, D2))

        return ctrl.ss(A, B, C, D, dt=1)


    def interconnect(self, m, L, rho):

        # get algorithm realization
        self.algo.update_sectors(m, L)
        G = self.algo.get_state_space()

        # get IQC realization
        for idx, iqc in enumerate(self.IQCs):
            iqc.update_external_dependencies(rho=rho, algo=self.algo)
            iqc.update_sectors(m, L)
            
            if idx==0:
                Psi = iqc.get_state_space()
            else:
                Psi = self._lti_stack(Psi, iqc.get_state_space())

        # build extended plant
        G_I   = self._lti_stack(G, np.eye(self.n_g))
        G_hat = ctrl.series(G_I, Psi)
        A, B, C, D = ctrl.ssdata(G_hat)

        return A, B, C, D


    def build_IQC_multiplier(self):
        block_rows = []
    
        for i, iqc_i in enumerate(self.IQCs):
            row_blocks = []
            for j, iqc_j in enumerate(self.IQCs):
                if i == j:
                    row_blocks.append(iqc_i.lambda_var * iqc_i.M)
                else:
                    row_blocks.append(np.zeros((iqc_i.M.shape[0], iqc_j.M.shape[1])))
            block_rows.append(row_blocks)
        
        return cvx.bmat(block_rows)
