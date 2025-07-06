import numpy as np
from scipy.sparse import hstack, csc_matrix, block_diag
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def spectraShadow(I):
    """
    Converts an interval to a spectrahedral shadow.
    """
    from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow

    # Check if not a matrix set
    if len(I.inf.shape) > 1 and I.inf.shape[1] > 1:
        raise CORAerror('CORA:wrongValue', 'first', 'Interval must not be an n-d array with n > 1.')

    n = I.dim()
    lower = I.inf.flatten()
    upper = I.sup.flatten()

    A0_diagonals = []
    Ai = []

    for i in range(n):
        # Construct D_A0 for the i-th dimension
        D_A0 = np.array([[upper[i], 0], [0, -lower[i]]])
        if np.isinf(upper[i]): D_A0[0, 0] = 1 if upper[i] > 0 else -1
        if np.isinf(lower[i]): D_A0[1, 1] = -1 if lower[i] > 0 else 1
        A0_diagonals.append(csc_matrix(D_A0))

        # Construct D_Ai for the i-th dimension
        D_Ai = np.array([[-1, 0], [0, 1]])
        if np.isinf(upper[i]): D_Ai[0, 0] = 0
        if np.isinf(lower[i]): D_Ai[1, 1] = 0
        
        # Construct Ai matrix using block diagonals
        D_Ai_sparse = csc_matrix(D_Ai)
        A_i_mat = block_diag((
            csc_matrix((2 * i, 2 * i)), 
            D_Ai_sparse, 
            csc_matrix((2 * (n - i - 1), 2 * (n - i - 1)))
        )).tocsc()
        Ai.append(A_i_mat)

    A0 = block_diag(A0_diagonals).tocsc()
    
    # Concatenate [A0, A1, ..., An]
    A_matrices = [A0] + Ai
    A = hstack(A_matrices)

    SpS = SpectraShadow(A)

    # Additional properties
    SpS.bounded = I.is_bounded()
    SpS.emptySet = I.is_empty()
    SpS.fullDim = I.isFullDim()
    SpS.center = I.center()

    return SpS 