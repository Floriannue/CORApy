"""
normalizeConstraints - normalizes all inequality/equality constraints

Syntax:
    P_out = normalizeConstraints(P, type)

Inputs:
    P - polytope object
    type - (optional) 'b', 'be' (default): normalize offset vectors b and be
                      'A', 'Ae': normalize norm of constraints in A and Ae to 1

Outputs:
    P_out - polytope object

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       01-December-2022 (MATLAB)
Last update:   13-December-2022 (MW, add A normalization)
Python translation: 2025
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.private.priv_normalize_constraints import priv_normalize_constraints


def normalizeConstraints(P, *varargin):
    """
    Normalizes all inequality/equality constraints of a polytope.
    
    Args:
        P: Polytope object
        *varargin: Optional arguments
            type: 'b'/'be' (default) or 'A'/'Ae' normalization type
            
    Returns:
        Polytope: Normalized polytope object
    """
    # Can only normalize constraints if constraints are given
    if not P.isHRep:
        raise CORAerror('CORA:specialError',
                       'Halfspace representation not computed... nothing to normalize!')
    
    # Set default type
    defaults, _ = setDefaultValues(['b'], list(varargin))
    type_ = defaults[0]
    
    # Check input arguments
    inputArgsCheck([[P, 'att', 'polytope'],
                    [type_, 'str', ['b', 'be', 'A', 'Ae']]])
    
    # Copy polytope including properties (copy constructor)
    P_out = Polytope(P)
    
    # Normalize constraints
    A_norm, b_norm, Ae_norm, be_norm = priv_normalize_constraints(
        P.A, P.b, P.Ae, P.be, type_)
    
    # Update the normalized constraints
    P_out._A = A_norm
    P_out._b = b_norm
    P_out._Ae = Ae_norm
    P_out._be = be_norm
    
    return P_out 