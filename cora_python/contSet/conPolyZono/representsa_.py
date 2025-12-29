"""
representsa_ - checks if a constrained polynomial zonotope can also be
   represented by a different set, e.g., a special case

Syntax:
    res = representsa_(cPZ,type,tol,method,iter,splits)
    [res,S] = representsa_(cPZ,type,tol,method,iter,splits)

Inputs:
    cPZ - conPolyZono object
    type - other set representation or 'origin', 'point', 'hyperplane'
    tol - tolerance
    method - algorithm used for contraction ('forwardBackward',
             'linearize', 'polynomial', 'interval', or 'all')
    iter - number of iteration (integer > 0 or 'fixpoint')
    splits - number of recursive splits (integer > 0)

Outputs:
    res - true/false
    S - converted set

Example:
    c = [0;0];
    G = [1 0 1;0 1 1];
    E = [1 0 2;0 1 1];
    A = [1 -1 0; 0 -1 1];
    b1 = [0; 1]; b2 = [0; 0];
    EC = [2 0 1; 0 1 0];
    cPZ1 = conPolyZono(c,G,E,A,b1,EC);
    cPZ2 = conPolyZono(c,G,E,A,b2,EC);

    res1 = representsa_(cPZ1,'emptySet',eps,'linearize',3,7)
    res2 = representsa_(cPZ2,'emptySet',eps,'linearize',3,7)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/representsa

Authors:       Mark Wetzlinger, Niklas Kochdumper
Written:       19-July-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple, Optional, Any, Union

if TYPE_CHECKING:
    from cora_python.contSet.conPolyZono.conPolyZono import ConPolyZono


def representsa_(cPZ: 'ConPolyZono', type_: str, tol: float = 1e-9, 
                 method: str = 'linearize', iter_val: Union[int, str] = 1, 
                 splits: int = 0, **kwargs) -> Union[bool, Tuple[bool, Optional[Any]]]:
    """
    Checks if a constrained polynomial zonotope can also be represented by a different set type.
    
    Args:
        cPZ: conPolyZono object
        type_: other set representation or 'origin', 'point', 'hyperplane'
        tol: tolerance
        method: algorithm used for contraction ('forwardBackward', 'linearize', 
                'polynomial', 'interval', or 'all')
        iter_val: number of iteration (integer > 0 or 'fixpoint')
        splits: number of recursive splits (integer > 0)
        **kwargs: Additional arguments, including 'return_set' to control return format
        
    Returns:
        bool or tuple: Whether cPZ can be represented by type_, optionally with converted set
    """
    # Import here to avoid circular imports
    from cora_python.contSet.contSet.representsa_emptyObject import representsa_emptyObject
    from cora_python.contSet.emptySet.emptySet import EmptySet
    
    # Helper function to handle return value consistently
    def _return_result(res_val, set_val=None):
        """Return result based on return_set parameter"""
        if kwargs.get('return_set', False):
            return res_val, set_val
        return res_val
    
    # Check empty object case
    try:
        empty, res, S_conv = representsa_emptyObject(cPZ, type_)
        if empty:
            return _return_result(res, S_conv)
    except:
        empty, res = representsa_emptyObject(cPZ, type_, return_conv=False)
        if empty:
            return _return_result(res, None)
    
    # Dimension
    n = cPZ.dim()
    
    # Init second output argument (covering all cases with res = false)
    S = None
    
    # Switch case equivalent
    if type_ == 'point':
        # Todo: check constraints
        # MATLAB: res = isempty(cPZ.G) && isempty(cPZ.GI);
        res = (cPZ.G.size == 0 if hasattr(cPZ, 'G') else True) and \
              (cPZ.GI.size == 0 if hasattr(cPZ, 'GI') else True)
        if res:
            S = cPZ.c
        return _return_result(res, S)
        
    elif type_ == 'conPolyZono':
        # obviously true
        res = True
        if kwargs.get('return_set', False):
            S = cPZ
        return _return_result(res, S)
    
    elif type_ == 'probZonotope':
        # cannot be true
        res = False
        return _return_result(res, None)
    
    elif type_ == 'hyperplane':
        # constrained polynomial zonotopes cannot be unbounded (unless 1D,
        # where hyperplane is also bounded)
        res = (n == 1)
        return _return_result(res, None)
    
    elif type_ == 'emptySet':
        res = _aux_isEmptySet(cPZ, tol, method, splits, iter_val)
        if res and kwargs.get('return_set', False):
            S = EmptySet.empty(n)
        return _return_result(res, S)
    
    elif type_ == 'fullspace':
        # constrained polynomial zonotopes cannot be unbounded
        res = False
        return _return_result(res, None)
    
    else:
        from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
        raise CORAerror('CORA:notSupported',
            f'Comparison of conPolyZono to {type_} not supported.')


# Auxiliary functions -----------------------------------------------------

def _aux_isEmptySet(cPZ: 'ConPolyZono', tol: float, method: str, splits: int, iter_val: Union[int, str]) -> bool:
    """
    Check if a constrained polynomial zonotope represents an empty set.
    
    Args:
        cPZ: conPolyZono object
        tol: tolerance
        method: algorithm used for contraction
        splits: number of recursive splits
        iter_val: number of iterations or 'fixpoint'
        
    Returns:
        bool: True if the set is empty, False otherwise
    """
    # Check if independent generators are empty
    if hasattr(cPZ, 'GI') and cPZ.GI.size > 0:
        return False
    
    # Check if constraints exist
    if not hasattr(cPZ, 'A') or cPZ.A.size == 0:
        return False
    
    # Try to contract the domain to the empty set -> set is empty
    # MATLAB: temp = ones(length(cPZ.id),1);
    # MATLAB: dom = interval(-temp,temp);
    from cora_python.contSet.interval.interval import Interval
    
    if hasattr(cPZ, 'id') and cPZ.id.size > 0:
        # MATLAB: length(cPZ.id) gives number of elements in id vector
        id_len = cPZ.id.shape[0] if cPZ.id.ndim > 0 else 1
        temp = np.ones((id_len, 1))
    else:
        # If id is empty, determine dimension from EC (number of factors)
        if hasattr(cPZ, 'EC') and cPZ.EC.size > 0:
            # EC is (h × r) where h is number of factors
            temp = np.ones((cPZ.EC.shape[0], 1))
        elif hasattr(cPZ, 'A') and cPZ.A.size > 0:
            # A is (m × r) where r is number of constraint factors
            # We need the number of factors, which should match EC columns or A columns
            temp = np.ones((cPZ.A.shape[1], 1))
        else:
            # Fallback: use dimension of cPZ (though this may not be correct)
            temp = np.ones((cPZ.dim(), 1))
    
    # Create interval from -temp to temp
    dom = Interval(-temp, temp)
    
    # MATLAB: D = contractPoly(-cPZ.b,cPZ.A,[],cPZ.EC,dom,method,iter,splits);
    from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractPoly import contractPoly
    D = contractPoly(-cPZ.b, cPZ.A, np.array([]), cPZ.EC, dom, method, iter_val, splits)
    
    # MATLAB: res = representsa_(D,'emptySet',tol);
    if hasattr(D, 'representsa_'):
        res, _ = D.representsa_('emptySet', tol, return_set=True)
        return res
    else:
        # Fallback: check if D is empty
        return D.isemptyobject() if hasattr(D, 'isemptyobject') else False

