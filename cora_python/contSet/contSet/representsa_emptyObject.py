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
"""

import numpy as np
from typing import Tuple, Union, Optional, Any


def representsa_emptyObject(S, type_: str) -> Tuple[bool, bool, Optional[Any]]:
    """
    Checks if a contSet class object is fully empty and converts the set to an instance of class 'type'
    
    Args:
        S: contSet object
        type_: contSet class name as string
        
    Returns:
        tuple: (empty, res, S_conv) where:
            - empty: true/false whether S is fully empty
            - res: true/false whether S can be represented by 'type'
            - S_conv: object of class 'type', converted from S (or None)
    """
    # check if set is a fully empty object
    empty = S.isemptyobject()
    
    # default: no converted set
    res = False
    S_conv = None
    
    # dimension
    n = S.dim()
    
    # only further information if S is fully empty
    if empty:
        # self-checking
        if S.__class__.__name__.lower() == type_.lower():
            res = True
            # Create empty set of same type
            if hasattr(S.__class__, 'empty'):
                S_conv = S.__class__.empty(n)
            return empty, res, S_conv
        
        if type_.lower() in ['origin', 'point', 'parallelotope']:
            # fully empty objects cannot be any single point or a parallelotope
            res = False
            
        elif type_.lower() == 'fullspace':
            # fully empty polytopes in halfspace representation represent
            # R^n (=fullspace); ensure that V representation is not given
            if S.__class__.__name__.lower() == 'polytope':
                # This would need polytope-specific logic
                res = False  # Simplified for now
            else:
                res = False
                
        elif type_.lower() == 'hyperplane':
            res = True
            
        elif type_.lower() == 'interval':
            res = True
            if S.__class__.__name__.lower() == 'polytope':
                # This would need polytope-specific logic for fullspace vs empty
                from cora_python.contSet.interval import Interval
                S_conv = Interval(np.zeros((n, 0)))
            else:
                # empty set
                from cora_python.contSet.interval import Interval
                S_conv = Interval(np.zeros((n, 0)))
                
        elif type_.lower() == 'polytope':
            res = True
            # all other fully empty set reps represent the empty set
            from cora_python.contSet.polytope import Polytope
            if hasattr(Polytope, 'empty'):
                S_conv = Polytope.empty(n)
                
        else:
            # all fully empty objects represent the empty set (except for
            # polytopes and spectrahedral shadow); all sets can represent
            # the empty set
            is_polytope = S.__class__.__name__.lower() == 'polytope'
            is_spectra = S.__class__.__name__.lower() == 'spectrashadow'
            
            res = (n == 0 or 
                   (not is_polytope and not is_spectra) or
                   (is_polytope))  # Simplified polytope logic
            
            if res:
                # Try to create empty set of target type
                try:
                    if type_.lower() == 'emptyset':
                        from cora_python.contSet.emptySet import EmptySet
                        S_conv = EmptySet(n)
                    elif type_.lower() == 'zonotope':
                        from cora_python.contSet.zonotope import Zonotope
                        if hasattr(Zonotope, 'empty'):
                            S_conv = Zonotope.empty(n)
                    elif type_.lower() == 'ellipsoid':
                        from cora_python.contSet.ellipsoid import Ellipsoid
                        if hasattr(Ellipsoid, 'empty'):
                            S_conv = Ellipsoid.empty(n)
                    # Add more types as needed
                except ImportError:
                    # Type not available, just return None for S_conv
                    pass
    
    return empty, res, S_conv 