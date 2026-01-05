"""
location - Location package for hybrid automata

This package contains the Location class and all its methods.
Each method is implemented in its own file following the MATLAB structure.
"""

# Import the main Location class
from .location import Location

# Import method implementations
from .calcBasis import calcBasis
from .checkFlow import checkFlow
from .guardIntersect import guardIntersect
from .guardIntersect_zonoGirard import guardIntersect_zonoGirard
from .guardIntersect_nondetGuard import guardIntersect_nondetGuard
from .guardIntersect_levelSet import guardIntersect_levelSet
from .guardIntersect_polytope import guardIntersect_polytope
from .guardIntersect_conZonotope import guardIntersect_conZonotope
from .guardIntersect_hyperplaneMap import guardIntersect_hyperplaneMap
from .guardIntersect_pancake import guardIntersect_pancake
from .potInt import potInt
from .potOut import potOut
from .reach import reach
from .isequal import isequal
from .isemptyobject import isemptyobject
from .display import display, display_

# Attach methods to the Location class
Location.calcBasis = calcBasis
Location.checkFlow = checkFlow
Location.guardIntersect = guardIntersect
Location.guardIntersect_zonoGirard = guardIntersect_zonoGirard
Location.guardIntersect_nondetGuard = guardIntersect_nondetGuard
Location.guardIntersect_levelSet = guardIntersect_levelSet
Location.guardIntersect_polytope = guardIntersect_polytope
Location.guardIntersect_conZonotope = guardIntersect_conZonotope
Location.guardIntersect_hyperplaneMap = guardIntersect_hyperplaneMap
Location.guardIntersect_pancake = guardIntersect_pancake
Location.potInt = potInt
Location.potOut = potOut
Location.reach = reach
Location.isequal = isequal
Location.isemptyobject = isemptyobject
Location.display = display
Location.display_ = display_

# Attach display_ to __str__
Location.__str__ = lambda self: display_(self)

__all__ = ['Location', 'calcBasis', 'checkFlow', 'guardIntersect', 'guardIntersect_zonoGirard', 
           'guardIntersect_nondetGuard', 'guardIntersect_levelSet', 'guardIntersect_polytope', 
           'guardIntersect_conZonotope', 'guardIntersect_hyperplaneMap', 'guardIntersect_pancake', 
           'potInt', 'potOut', 'reach', 'isequal', 'isemptyobject', 'display', 'display_']

