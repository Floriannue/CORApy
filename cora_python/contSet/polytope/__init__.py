"""
Polytope module for CORA

This module contains the Polytope class for representing
convex polytopes defined by linear constraints.
"""

from .polytope import Polytope
from .zonotope import zonotope
from .contains_ import contains_
from .dim import dim
from .center import center
from .isBounded import isBounded
from .isemptyobject import isemptyobject
from .interval import interval

__all__ = [
    'Polytope',
    'zonotope',
    'contains_',
    'dim',
    'center', 
    'isBounded',
    'isemptyobject',
    'interval'
] 