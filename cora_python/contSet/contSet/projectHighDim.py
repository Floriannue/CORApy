import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def projectHighDim(S, N, proj=None):
    """
    projectHighDim - project a set to a higher-dimensional space,
    having the new dimensions bounded at 0.

    This function serves as a wrapper that performs input validation before
    calling the actual implementation in projectHighDim_.

    Syntax:
        S_new = projectHighDim(S, N)
        S_new = projectHighDim(S, N, proj)

    Inputs:
        S (contSet): A contSet object.
        N (int): Dimension of the higher-dimensional space.
        proj (list of int, optional): States of the high-dimensional space
            that correspond to the states of the low-dimensional set.
            Defaults to range(1, S.dim() + 1).

    Outputs:
        contSet: A contSet object in the higher-dimensional space.

    See also:
        contSet.project, contSet.lift
    """
    from cora_python.contSet.contSet.contSet import ContSet
    # Polymorphic dispatch
    if type(S).projectHighDim is not ContSet.projectHighDim:
        return type(S).projectHighDim(S, N, proj=proj)

    # --- Primary Method Body ---

    # Input parsing
    if proj is None:
        proj = list(range(1, S.dim() + 1))

    # Input validation
    if not isinstance(N, int) or N < 0:
        raise CORAerror('nonnegative_integer', 'N', N)
    
    if not isinstance(proj, list) or not all(isinstance(i, int) for i in proj):
         raise CORAerror('integer_vector', 'proj', proj)

    s_dim = S.dim()
    if s_dim > N:
        raise CORAerror('wrongValue', 'second', 'Dimension of higher-dimensional space must be larger than or equal to the dimension of the given set.')
    elif s_dim != len(proj):
        raise CORAerror('wrongValue', 'third', 'Number of dimensions in higher-dimensional space must match the dimension of the given set.')
    elif proj and max(proj) > N:
        raise CORAerror('wrongValue', 'third', 'Specified dimensions exceed dimension of high-dimensional space.')

    # Convert 1-based proj to 0-based for internal function
    proj_0_based = [p - 1 for p in proj]

    # Call subfunction
    return S.projectHighDim_(N, proj_0_based) 