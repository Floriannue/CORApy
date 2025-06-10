"""
contains_ - internal implementation of containment checking for contSet

This function should be overridden by subclasses to provide specific
containment checking algorithms.

Syntax:
    res = contains_(S1, S2, method, tol, maxEval, cert_toggle, scaling_toggle)

Inputs:
    S1 - contSet object
    S2 - contSet object or numeric array
    method - method for computation
    tol - tolerance
    maxEval - maximal number of iterations
    cert_toggle - whether to compute certification
    scaling_toggle - whether to compute scaling

Outputs:
    res - containment result
    cert - (optional) certification result
    scaling - (optional) scaling factor

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       18-August-2022
Last update:   ---
Last revision: ---
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError

def contains_(S1, S2, method='exact', tol=1e-12, maxEval=200, cert_toggle=False, scaling_toggle=False):
    """
    Internal implementation of containment checking.
    
    This base implementation throws an error - to be overridden in subclasses.
    
    Args:
        S1: contSet object
        S2: contSet object or numeric array
        method: method for computation
        tol: tolerance
        maxEval: maximal number of iterations
        cert_toggle: whether to compute certification
        scaling_toggle: whether to compute scaling
        
    Returns:
        bool or tuple: containment result, optionally with certification and scaling
        
    Raises:
        CORAError: This method should be overridden in subclasses
    """
    # is overridden in subclass if implemented; throw error
    raise CORAError("CORA:noops", f"Function contains_ not implemented for class {type(S1).__name__}") 