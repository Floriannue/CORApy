"""
representsa - checks if a set can also be represented by a different set,
    e.g., a special case

Syntax:
    res = representsa(S,type)
    res = representsa(S,type,tol)
    res, S_conv = representsa(S,type)
    res, S_conv = representsa(S,type,tol)

Inputs:
    S - contSet object
    type - char array
    tol - (optional) tolerance
    method - (only conPolyZono) algorithm used for contraction
             ('forwardBackward', 'linearize', 'polynomial', 'interval', or 'all')
    iter - (only conPolyZono) number of iteration (integer > 0 or 'fixpoint')
    splits - (only conPolyZono) number of recursive splits (integer > 0)

Outputs:
    res - true/false
    S_conv - converted set

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       19-July-2023
Last update:   ---
Last revision: ---
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet


def representsa(S: 'ContSet', set_type: str, *args, **kwargs):
    """
    Checks if a set can also be represented by a different set type.
    
    Args:
        S: contSet object
        set_type: string indicating target set type
        *args: additional arguments (tol, method, iter, splits)
        **kwargs: keyword arguments
        
    Returns:
        bool or tuple: Whether S can be represented by set_type, optionally with converted set
    """
    # Default values
    tol = 1e-12
    method = 'linearize'
    iter_val = 1
    splits = 0
    
    # Parse arguments
    if len(args) >= 1:
        tol = args[0]
    if len(args) >= 2:
        method = args[1]
    if len(args) >= 3:
        iter_val = args[2]
    if len(args) >= 4:
        splits = args[3]
        
    # Override with keyword arguments
    tol = kwargs.get('tol', tol)
    method = kwargs.get('method', method)
    iter_val = kwargs.get('iter', iter_val)
    splits = kwargs.get('splits', splits)
    
    # All admissible comparisons
    admissible_types = [
        'capsule', 'conPolyZono', 'conHyperplane', 'conZonotope', 'ellipsoid',
        'halfspace', 'interval', 'levelSet', 'polygon', 'polytope', 'polyZonotope',
        'probZonotope', 'zonoBundle', 'zonotope',  # contSet classes
        'origin', 'point', 'hyperplane', 'parallelotope', 'convexSet',  # special types
        'emptySet', 'fullspace'
    ]
    
    # Check if type is admissible
    if set_type not in admissible_types:
        raise ValueError(f"Unknown set type: {set_type}")
    
    # Input validation
    if not isinstance(tol, (int, float)) or tol < 0:
        raise ValueError("Tolerance must be a non-negative number")
    
    # Call subfunction
    # The extra arguments (method, iter_val, splits) are only relevant for conPolyZono. 
    # For other types, representsa_ expects only (set_type, tol).
    # We need to ensure that the method representsa_ of the specific class can handle these arguments.
    # For now, we will pass only set_type and tol to avoid TypeError, as most implementations only expect those.
    # When conPolyZono is implemented, this logic will need to be refined based on `S`'s type.
    return S.representsa_(set_type, tol, method, iter_val, splits) 