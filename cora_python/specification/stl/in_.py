"""
in_ - set containment operator for Signal Temporal Logic

TRANSLATED FROM: cora_matlab/specification/@stl/in.m

Note: Named 'in_' because 'in' is a reserved keyword in Python

Syntax:
    res = in_(obj, S)

Inputs:
    obj - logic formula (class stl)
    S - set (class interval, zonotope, polytope, conZonotope,
             zonoBundle, halfspace, or polygon)

Outputs:
    res - resulting stl formula (class stl)

Example: 
    x = stl('x',2);
    pgon = polygon.generateRandom();
    eq = finally(in(x,pgon),interval(0.1,0.2))

Authors:       Niklas Kochdumper, Benedikt Seidl
Written:       09-November-2022
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .stl import Stl
    from cora_python.contSet.polytope import Polytope

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.specification.stl.polytope2stl import polytope2stl


def in_(obj: 'Stl', S: Any) -> 'Stl':
    """
    Set containment operator for Signal Temporal Logic
    
    Args:
        obj: logic formula (class stl)
        S: set (interval, zonotope, polytope, conZonotope, zonoBundle, halfspace, or polygon)
    
    Returns:
        Stl: resulting stl formula
    """
    # Check input arguments
    if obj.type not in ['variable', 'concat']:
        raise CORAerror('CORA:notSupported',
                      'This operation is not supported for stl objects!')
    
    # Get dimension of S
    from cora_python.contSet.contSet.contSet import ContSet
    
    if isinstance(S, ContSet):
        S_dim = S.dim()
    else:
        # Try to get dimension from S
        try:
            S_dim = len(S) if hasattr(S, '__len__') else None
        except:
            S_dim = None
    
    # Check dimension compatibility
    if obj.type == 'variable':
        obj_dim = len(obj.variables) if hasattr(obj, 'variables') else 1
    elif obj.type == 'concat':
        obj_dim = len(obj.var) if hasattr(obj, 'var') else 1
    else:
        obj_dim = 1
    
    if S_dim is not None and S_dim != obj_dim:
        raise CORAerror('CORA:wrongValue',
                      'dimensions of set and stl object have to match!')
    
    # Convert S to polytope and create STL formula
    from cora_python.contSet.polytope import Polytope
    
    # Convert S to polytope if needed
    if not isinstance(S, Polytope):
        try:
            S = Polytope(S)
        except:
            raise CORAerror('CORA:notSupported',
                          'This operation is not supported for this type of set representation!')
    
    # Use polytope2stl to create the STL formula
    return polytope2stl(obj, S)

