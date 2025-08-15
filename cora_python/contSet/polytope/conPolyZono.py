"""
conPolyZono - converts a polytope object to a constrained polynomial zonotope (Polytope -> ConPolyZono)

Syntax:
   cPZ = conPolyZono(P)

Inputs:
   P - polytope object

Outputs:
   cPZ - conPolyZono object

Authors:       Niklas Kochdumper (MATLAB)
Automatic python translation: Florian Nüssel BA 2025
"""

from cora_python.contSet.conZonotope import ConZonotope
from cora_python.contSet.conPolyZono import ConPolyZono
from .polytope import Polytope


def conPolyZono(P: Polytope):
    """
    Converts a polytope to a constrained polynomial zonotope.

    Args:
        P: Polytope object.

    Returns:
        ConPolyZono object.
    """
    # The MATLAB function simply calls: cPZ = conPolyZono(conZonotope(P));
    # This means the ConZonotope constructor should handle Polytope input,
    # and the ConPolyZono constructor should handle ConZonotope input.
    # The necessary logic for ConZonotope(Polytope) is added in conZonotope.py.
    # The necessary logic for ConPolyZono(ConZonotope) will be added in conPolyZono.py (the constructor).

    # First, convert the Polytope to a ConZonotope
    cz = ConZonotope(P)

    # Then, convert the ConZonotope to a ConPolyZono
    cPZ = ConPolyZono(cz)

    return cPZ
