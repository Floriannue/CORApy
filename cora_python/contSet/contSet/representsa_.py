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

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def representsa_(S: 'ContSet', set_type: str, tol: float = 1e-12, method: str = 'linearize', iter_val: int = 1, splits: int = 0, **kwargs):
    """
    Checks if a set can also be represented by a different set type.
    
    This function uses polymorphic dispatch to call the appropriate subclass
    implementation of representsa_, or provides the base implementation.
    
    Args:
        S: contSet object
        set_type: string indicating the target set type
        tol: tolerance (default: 1e-12)
        method: method for computation (default: 'linearize')
        iter_val: number of iterations (default: 1) 
        splits: number of splits (default: 0)
    
    Returns:
        bool or tuple: Whether S can be represented by set_type, optionally with converted set
        
    Raises:
        CORAerror: If representsa_ is not implemented for the specific set type
    """
    base_class = type(S).__bases__[0] if type(S).__bases__ else None
    if (hasattr(type(S), 'representsa_') and 
        base_class and hasattr(base_class, 'representsa_') and
        type(S).representsa_ is not base_class.representsa_):
        # All implementations accept **kwargs, so pass everything via kwargs for consistency
        return type(S).representsa_(S, set_type, tol, method=method, iter_val=iter_val, splits=splits, **kwargs)
    else:
        # Base implementation - throw error as this method should be overridden
        raise CORAerror("CORA:noops", f"Function representsa_ not implemented for class {type(S).__name__}")