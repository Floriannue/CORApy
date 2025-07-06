"""
representsa_emptyObject - checks if a contSet class object is fully empty
   and converts the set to an instance of class 'type'

Syntax:
   [empty,res] = representsa_emptyObject(S,type)
   [empty,res,S_conv] = representsa_emptyObject(S,type)

Inputs:
   S - contSet object
   type - contSet class

Outputs:
   empty - true/false whether S is fully empty
   res - true/false whether S can be represented by 'type'
   S_conv - object of class 'type', converted from S

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       24-July-2023
Last update:   01-January-2024 (MW, update fully empty polytopes)
               14-July-2024 (MW, support polytopes in V representation)
Last revision: ---
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple, Optional, Any

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def representsa_emptyObject(S: 'ContSet', type_: str, return_conv: bool = True) -> Tuple[bool, Optional[bool], Optional[Any]]:
    """
    Checks if a contSet class object is fully empty and converts the set to an instance of class 'type'
    
    Args:
        S: contSet object
        type_: contSet class name as string
        return_conv: whether to return conversion object (simulates nargout == 3)
        
    Returns:
        tuple: (empty, res, S_conv) where:
            - empty: true/false whether S is fully empty
            - res: true/false whether S can be represented by 'type', or None if not fully empty
            - S_conv: object of class 'type', converted from S (or None)
    """
    from cora_python.contSet.fullspace import Fullspace
    from cora_python.contSet.interval import Interval
    from cora_python.contSet.polytope import Polytope
    from cora_python.contSet.emptySet import EmptySet
    from cora_python.contSet.zonotope import Zonotope
    from cora_python.contSet.ellipsoid import Ellipsoid
    from cora_python.contSet.capsule import Capsule
    from cora_python.contSet.conZonotope import ConZonotope
    from cora_python.contSet.polyZonotope import PolyZonotope
    from cora_python.contSet.zonoBundle import ZonoBundle
    from cora_python.contSet.probZonotope import ProbZonotope
    from cora_python.contSet.spectraShadow import SpectraShadow
    from cora_python.contSet.taylm import Taylm

    # check if set is a fully empty object
    empty = S.isemptyobject()
    
    # default: no converted set
    res = None
    S_conv = None
    
    # dimension
    n = S.dim()
    
    # only further information if S is fully empty
    if empty:
        # self-checking
        if S.__class__.__name__.lower() == type_.lower():
            res = True
            if return_conv:
                # Create empty set using eval equivalent: type.empty(n)
                if hasattr(S.__class__, 'empty'):
                    S_conv = S.__class__.empty(n)
            return empty, res, S_conv
        
        # Switch case equivalent
        if type_.lower() in ['origin', 'point', 'parallelotope']:
            # fully empty objects cannot be any single point or a parallelotope
            res = False
            
        elif type_.lower() == 'fullspace':
            # fully empty polytopes in halfspace representation represent
            # R^n (=fullspace); ensure that V representation is not given
            res = (S.__class__.__name__.lower() == 'polytope' and 
                   ((hasattr(S, 'isHRep') and S.isHRep and 
                     hasattr(S, 'b') and hasattr(S, 'be') and
                     (S.b is None or len(S.b) == 0) and 
                     (S.be is None or len(S.be) == 0)) or
                    (hasattr(S, 'isVRep') and S.isVRep and 
                     hasattr(S, 'V') and S.V is not None and 
                     len(S.V) > 0 and np.all(np.isinf(S.V)))))
            
            if return_conv and res:
                # Create fullspace object: fullspace(dim(S))
                try:
                    S_conv = Fullspace(S.dim())
                except ImportError:
                    S_conv = None
                    
        elif type_.lower() == 'hyperplane':
            res = True
            
        elif type_.lower() == 'interval':
            res = True
            if return_conv:
                if (S.__class__.__name__.lower() == 'polytope' and 
                    ((hasattr(S, 'isHRep') and S.isHRep and 
                      hasattr(S, 'b') and hasattr(S, 'be') and
                      (S.b is None or len(S.b) == 0) and 
                      (S.be is None or len(S.be) == 0)) or
                     (hasattr(S, 'isVRep') and S.isVRep and 
                      hasattr(S, 'V') and S.V is not None and 
                      len(S.V) > 0 and np.all(np.isinf(S.V))))):
                    # fullspace: interval(-Inf(dim(S),1),Inf(dim(S),1))
                    try:
                        S_conv = Interval(-np.inf * np.ones((S.dim(), 1)), 
                                         np.inf * np.ones((S.dim(), 1)))
                    except ImportError:
                        S_conv = None
                else:
                    # empty set: interval(zeros(dim(S),0))
                    try:
                        S_conv = Interval(np.zeros((S.dim(), 0)))
                    except ImportError:
                        S_conv = None
                        
        elif type_.lower() == 'polytope':
            res = True
            if return_conv:
                # all other fully empty set reps represent the empty set
                try:
                    if hasattr(Polytope, 'empty'):
                        S_conv = Polytope.empty(S.dim())
                except ImportError:
                    S_conv = None
                    
        else:
            # All other fully empty set representations can be converted to the
            # general empty set representation of all other set classes
            # (e.g. empty zonotope, empty ellipsoid, etc.), except for
            # polytopes and spectrahedral shadows.
            
            # MATLAB logic: res = dim(S) == 0 || (~isa(S,'polytope') && ~isa(S,'spectraShadow')) || ...
            #                     (isa(S,'polytope') && S.isVRep.val && isempty(S.V));
            
            # check if S is a polytope
            is_polytope = S.__class__.__name__.lower() == 'polytope'
            
            # check if S is a spectrahedral shadow
            is_spectrashadow = S.__class__.__name__.lower() == 'spectrashadow'
            
            # check if polytope in V-rep is truly empty
            poly_vrep_empty = (is_polytope and 
                             hasattr(S, 'isVRep') and S.isVRep and 
                             hasattr(S, 'V') and (S.V is None or S.V.size == 0))
            
            # Apply MATLAB logic exactly
            res = (n == 0 or 
                   (not is_polytope and not is_spectrashadow) or 
                   poly_vrep_empty)

            if return_conv and res:
                # Create empty set using eval equivalent: type.empty(n)
                try:
                    if type_.lower() == 'emptyset':
                        if hasattr(EmptySet, 'empty'):
                            S_conv = EmptySet.empty(n)
                    elif type_.lower() == 'zonotope':
                        if hasattr(Zonotope, 'empty'):
                            S_conv = Zonotope.empty(n)
                    elif type_.lower() == 'ellipsoid':
                        if hasattr(Ellipsoid, 'empty'):
                            S_conv = Ellipsoid.empty(n)
                    elif type_.lower() == 'capsule':
                        if hasattr(Capsule, 'empty'):
                            S_conv = Capsule.empty(n)
                    elif type_.lower() == 'conzonotope':
                        if hasattr(ConZonotope, 'empty'):
                            S_conv = ConZonotope.empty(n)
                    elif type_.lower() == 'polyzonotope':
                        if hasattr(PolyZonotope, 'empty'):
                            S_conv = PolyZonotope.empty(n)
                    elif type_.lower() == 'zonobundle':
                        if hasattr(ZonoBundle, 'empty'):
                            S_conv = ZonoBundle.empty(n)
                    elif type_.lower() == 'probzonotope':
                        if hasattr(ProbZonotope, 'empty'):
                            S_conv = ProbZonotope.empty(n)
                    elif type_.lower() == 'spectrashadow':
                        if hasattr(SpectraShadow, 'empty'):
                            S_conv = SpectraShadow.empty(n)
                    elif type_.lower() == 'taylm':
                        if hasattr(Taylm, 'empty'):
                            S_conv = Taylm.empty(n)
                    # For any other type, try to dynamically create
                    # This mimics eval([type, '.empty(', num2str(n), ')'])
                except (ImportError, AttributeError):
                    # Type not available or doesn't have empty method
                    S_conv = None
    
    return empty, res, S_conv 