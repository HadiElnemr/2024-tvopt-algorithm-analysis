import numpy as np
import control as ctrl
from scipy import linalg


def lti_stack(sys1, sys2):
    """
    Stacks two Linear Time-Invariant (LTI) systems into a single system.

    Returns:
    - A new LTI system that represents the stacked combination of sys1 and sys2.
    """
    
    A1, B1, C1, D1 = ctrl.ssdata(sys1)
    A2, B2, C2, D2 = ctrl.ssdata(sys2)

    if B1.shape[1] != B2.shape[1] or D1.shape[1] != D2.shape[1]:
        raise ValueError('Error in system stacking: number of inputs must be the same for both subsystems!')

    A = linalg.block_diag(A1, A2)
    B = np.vstack((B1, B2))
    C = linalg.block_diag(C1, C2)
    D = np.vstack((D1, D2))

    return ctrl.ss(A, B, C, D, dt=1)