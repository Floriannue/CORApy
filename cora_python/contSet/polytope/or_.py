"""
or_ - Union (convex hull) of two polytopes (MATLAB parity)

Implements logical OR as convex hull: P or Q := convHull_(P, Q).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polytope import Polytope


def or_(P: 'Polytope', Q: 'Polytope') -> 'Polytope':
    from .convHull_ import convHull_
    return convHull_(P, Q)


