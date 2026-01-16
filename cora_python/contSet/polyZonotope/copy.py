"""
copy - copies the polyZonotope object
"""
import numpy as np
from .polyZonotope import PolyZonotope


def copy(pZ: PolyZonotope) -> PolyZonotope:
    """Return a deep copy of a PolyZonotope."""
    # Use the class copy constructor to preserve internal consistency.
    return PolyZonotope(pZ)
