"""
representsa_ - checks if a set can also be represented by a different set,
    e.g., a special case
    (internal use, see also contSet/representsa)

Syntax:
    res = representsa_(S, type, tol)
    res, S = representsa_(S, type, tol)

Inputs:
    S - contSet object
    type - other set representation or 'origin', 'point', 'hyperplane'
    tol - (optional) tolerance

Outputs:
    res - true/false whether S can be represented by 'type'
    S - contSet object (optional converted set)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/representsa

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       12-September-2023
Last update:   ---
Last revision: ---
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError

def representsa_(S, set_type, tol=1e-12, method='linearize', iter=1, splits=0):
    """
    Checks if a set can also be represented by a different set type.
    
    This function delegates to the object's representsa_ method if available,
    otherwise raises an error.
    
    Args:
        S: contSet object
        set_type: string indicating the target set type
        tol: tolerance (default: 1e-12)
        method: method for computation (default: 'linearize')
        iter: number of iterations (default: 1) 
        splits: number of splits (default: 0)
    
    Returns:
        bool or tuple: Whether S can be represented by set_type, optionally with converted set
        
    Raises:
        CORAError: If representsa_ is not implemented for the specific set type
    """
    # Check if the object has a representsa_ method and use it
    if hasattr(S, 'representsa_') and callable(getattr(S, 'representsa_')):
        return S.representsa_(set_type, tol, method=method, iter=iter, splits=splits)
    
    # Fallback error
    raise CORAError("CORA:noops", f"Function representsa_ not implemented for class {type(S).__name__}") 