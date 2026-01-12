"""
convGuard - converts the guard sets to the set representation that is
   required for the selected guard-intersection method

TRANSLATED FROM: cora_matlab/hybridDynamics/@transition/convGuard.m

Syntax:
    trans = convGuard(trans,inv,options)

Inputs:
    trans - transition object
    inv - invariant set of the location
    options - struct/dict containing algorithm settings

Outputs:
    trans - modified transition object

Example:
    -

Authors:       Niklas Kochdumper (MATLAB)
Written:       16-May-2018 (MATLAB)
Last update:   19-June-2022 (MW, error message, use switch-case, MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Any, Union, Dict
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .transition import Transition


def convGuard(trans: 'Transition', inv: Any, options: Union[Dict, Any]) -> 'Transition':
    """
    Converts the guard sets to the set representation required for the
    selected guard-intersection method.
    
    Args:
        trans: transition object
        inv: invariant set of the location
        options: dict or object containing algorithm settings with 'guardIntersect' key/attribute
    
    Returns:
        Transition: modified transition object
    """
    from cora_python.contSet.polytope.polytope import Polytope
    from cora_python.contSet.conZonotope.conZonotope import ConZonotope
    from cora_python.contSet.contSet.and_ import and_
    
    # Get guardIntersect method from options
    if isinstance(options, dict):
        guard_intersect = options.get('guardIntersect', 'polytope')
    else:
        guard_intersect = getattr(options, 'guardIntersect', 'polytope')
    
    # MATLAB: switch options.guardIntersect
    if guard_intersect == 'polytope':
        # MATLAB: if ~isa(trans.guard,'polytope')
        if not isinstance(trans.guard, Polytope):
            # MATLAB: trans.guard = polytope(trans.guard);
            # Convert to polytope using the set's polytope() method if available
            if hasattr(trans.guard, 'polytope'):
                trans.guard = trans.guard.polytope()
            else:
                # Fallback: try direct conversion (for other set types)
                trans.guard = Polytope(trans.guard)
        
        # MATLAB: trans.guard = and_(trans.guard,inv,'exact');
        # Call and_ method on the guard polytope (polytope.and_ takes 2 args: self and other)
        trans.guard = trans.guard.and_(inv)
    
    elif guard_intersect in ['conZonotope', 'conZonotopeFast']:
        # MATLAB: if isa(trans.guard,'polytope')
        if isinstance(trans.guard, Polytope):
            # MATLAB: trans.guard = conZonotope(trans.guard);
            trans.guard = ConZonotope(trans.guard)
    
    else:
        # MATLAB: throw(CORAerror('CORA:wrongFieldValue','options.guardIntersect',...))
        raise CORAerror('CORA:wrongFieldValue', 'options.guardIntersect',
                       ['zonoGirard', 'hyperplaneMap', 'pancake', 'nondetGuard'])
    
    return trans

