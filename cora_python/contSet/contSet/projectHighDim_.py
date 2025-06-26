import numpy as np

def projectHighDim_(S, N, proj, *varargin):
    """
    projectHighDim_ - project a set to a higher-dimensional space,
    having the new dimensions bounded at 0 (core implementation).

    Syntax:
        S_new = projectHighDim_(S, N, proj)

    Inputs:
        S (contSet): A contSet object.
        N (int): Dimension of the higher-dimensional space.
        proj (list of int): 0-indexed states of the high-dimensional space
            that correspond to the states of the low-dimensional set.

    Outputs:
        contSet: A contSet object in the higher-dimensional space.
    """
    from cora_python.contSet.contSet.contSet import ContSet
    # Polymorphic dispatch
    if type(S).projectHighDim_ is not ContSet.projectHighDim_:
        return type(S).projectHighDim_(S, N, proj, *varargin)

    # --- Primary Method Body ---
    
    # Create identity matrix for projection
    I = np.eye(N)
    
    # Select columns to form the projection matrix
    T = I[:, proj]
    
    # Project the set by matrix multiplication
    # This relies on the __matmul__ (mtimes) operator being defined
    # for the specific contSet subclass.
    return T @ S 