import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def projectOnHyperplane(S, hyp):
    """
    projectOnHyperplane - projects a set onto a hyperplane.

    Syntax:
        S_proj = projectOnHyperplane(S, hyp)

    Inputs:
        S (contSet): A contSet object.
        hyp (polytope): A polytope object representing a hyperplane.

    Outputs:
        contSet: The projected set (of the same dimension as the input set).

    Example:
        Z = zonotope(np.array([2, 2]), np.array([[1, -1], [0, 1]]))
        # from cora_python.contSet.polytope.polytope import polytope
        # hyp = polytope.from_constraint_eq(np.array([[1, 1]]), np.array([1]))
        # res = Z.projectOnHyperplane(hyp)
        # ... plotting ...
    """
    from cora_python.contSet.contSet.contSet import ContSet
    # Polymorphic dispatch
    if type(S).projectOnHyperplane is not ContSet.projectOnHyperplane:
        return type(S).projectOnHyperplane(S, hyp)

    # --- Primary Method Body ---
    
    # Ensure that the polytope represents a hyperplane
    if not hyp.representsa_('conHyperplane', 1e-12):
        raise CORAerror('wrongValue', 'second', 'must represent a hyperplane.')
        
    # Dimension
    n = S.dim()
    
    # Normalize hyperplane constraints
    hyp_norm = hyp.normalizeConstraints('A')
    
    # Extract hyperplane normal c and offset d
    # Assuming hyp_norm.Ae and hyp_norm.be are (1, n) and (1,) respectively
    c = hyp_norm.Ae.T
    d = hyp_norm.be
    
    # Linear map A*x + b for the projection
    A = np.eye(n) - c @ c.T
    b = (d * c).flatten() # Ensure b is a 1D vector
    
    # Project the set
    S_proj = A @ S + b
    
    return S_proj 