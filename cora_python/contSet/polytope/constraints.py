from cora_python.contSet.polytope.polytope import Polytope
from .private.priv_V_to_H import priv_V_to_H

def constraints(P):
    """
    Computes the half-space representation (A, b) and equality constraints (Ae, be)
    of the polytope if it is not already available.
    
    If the polytope is defined by vertices, it converts them to half-spaces.
    
    Returns:
        Polytope: The same polytope object with A, b, Ae, and be properties populated.
    """
    if P.A is None or P.b is None:
        if P.V is not None:
            # Convert vertex representation to half-space representation
            P.A, P.b, P.Ae, P.be = priv_V_to_H(P.V)
        else:
            raise ValueError("Polytope is not defined by vertices or half-spaces.")
    
    return P 