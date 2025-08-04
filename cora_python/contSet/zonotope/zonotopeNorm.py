"""
zonotopeNorm - computes the norm of the point p w.r.t. the zonotope-norm induced by the zonotope Z (see [1, Definition 4])

Syntax:
    res, minimizer = zonotopeNorm(Z, p)

Inputs:
    Z - zonotope
    p - nx1-array, with n the dimension of Z

Outputs:
    res - zonotope-norm of the point p
    minimizer - (optional) returns a solution x s.t. Gx = p and for which norm(x,inf) = zonotopeNorm(Z,p)

Example:
    from cora_python.contSet.zonotope import Zonotope, zonotopeNorm
    import numpy as np
    c = np.array([[0], [0]])
    G = np.array([[2, 3, 0], [2, 0, 3]])
    Z = Zonotope(c, G)
    p = np.random.rand(2, 1)
    d, minimizer = zonotopeNorm(Z, p)

References:
    [1] A. Kulmburg, M. Althoff. "On the co-NP-Completeness of the Zonotope Containment Problem", European Journal of Control 2021

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Adrian Kulmburg (MATLAB)
               Python translation by AI Assistant
Written:       14-May-2021 (MATLAB)
Last update:   28-March-2025 (TL, return minimizer) (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple, Union
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from scipy.sparse import eye as speye

if TYPE_CHECKING:
    from .zonotope import Zonotope

def zonotopeNorm(Z: 'Zonotope', p: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes the norm of the point p w.r.t. the zonotope-norm induced by the zonotope Z.
    """
    
    # Check number of input arguments (Python equivalent of narginchk(2,2))
    # This is implicitly handled by Python function signature
    
    # Input arguments check
    if not hasattr(Z, '__class__') or Z.__class__.__name__ != 'Zonotope':
        raise ValueError("First argument must be a zonotope")
    
    if not isinstance(p, np.ndarray):
        raise ValueError("Second argument must be a numeric array")
    
    # Ensure p is a column vector
    p = np.array(p, dtype=float)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    elif p.ndim == 2:
        if p.shape[1] == 0:
            # Handle empty case: (n, 0) array
            pass  # Keep as is for empty point
        elif p.shape[1] != 1:
            raise ValueError("Point p must be a column vector")
    
    # Handle empty point case first
    if p.size == 0 or (p.ndim == 2 and p.shape[1] == 0):
        # Empty point - return 0 norm regardless of zonotope
        return 0.0, np.array([])
    
    # Check dimension of Z and p
    equal_dim_check(Z, p)
    
    # Empty set
    if Z.representsa_('emptySet', np.finfo(float).eps):
        if p.size == 0 or (p.ndim == 2 and p.shape[1] == 0):
            return 0.0, np.array([])
        else:
            return np.inf, np.array([])
    
    # Retrieve generator-representation of Z
    if Z.G is None or Z.G.size == 0:
        if not np.any(p):
            return 0.0, np.array([])
        else:
            return np.inf, np.array([])
    
    # Retrieve dimensions of the generator matrix of Z
    n, numGen = Z.G.shape
    
    # Set up objective and constraints of the linear program as defined in
    # [1, Equation (8)]
    problem = {}
    problem['f'] = np.concatenate([[1.0], np.zeros(numGen)])
    
    problem['Aeq'] = np.hstack([np.zeros((n, 1)), Z.G])
    problem['beq'] = p.flatten()
    
    # Inequality constraints: -t <= beta_i <= t for all i
    # This is equivalent to: beta_i - t <= 0 and -beta_i - t <= 0
    # In matrix form: [-1, I; -1, -I] * [t; beta] <= 0
    Aineq_1 = np.hstack([-np.ones((numGen, 1)), speye(numGen).toarray()])
    Aineq_2 = np.hstack([-np.ones((numGen, 1)), -speye(numGen).toarray()])
    problem['Aineq'] = np.vstack([Aineq_1, Aineq_2])
    problem['bineq'] = np.zeros(2 * numGen)
    
    # Bounds integrated in inequality constraints
    problem['lb'] = []
    problem['ub'] = []
    
    # Solve the linear program: If the problem is infeasible, this means that
    # the zonotope must be degenerate, and that the point can not be realized
    # as a linear combination of the generators of the zonotope. In that case,
    # the norm is defined as Inf. The same goes when the problem is 'unbounded'
    minimizer_p, res, exitflag, output, lambda_out = CORAlinprog(problem)
    
    if exitflag == -2 or exitflag == -3:
        res = np.inf
        minimizer = np.array([])
    elif exitflag != 1:
        # In case anything else went wrong, throw out an error
        raise CORAerror('CORA:solverIssue', 'Linear programming solver failed')
    else:
        # Extract the minimizer (beta values, excluding t)
        if minimizer_p is not None:
            minimizer = minimizer_p.flatten()[1:]  # Skip first element (t)
        else:
            minimizer = np.array([])
    
    # Always return both values - caller can choose to ignore minimizer
    return res, minimizer 