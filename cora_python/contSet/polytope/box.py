"""
box - computes an enclosing axis-aligned box represented as a polytope in
   halfspace representation

Syntax:
   P_out = box(P)

Inputs:
   P - polytope object

Outputs:
   P_out - polytope object 

Example:
   A = np.array([[1, 2], [-2, 1], [-2, -2], [3, -1]])
   b = np.array([1, 1, 1, 1])
   P = Polytope(A, b)
   B = box(P)

Authors: Viktor Kotsev, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 16-May-2022 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.polytope.polytope import Polytope

def box(P: 'Polytope') -> 'Polytope':
    """
    Computes an enclosing axis-aligned box represented as a polytope.
    
    Args:
        P: polytope object
        
    Returns:
        P_out: polytope object representing the enclosing axis-aligned box
    """
    from cora_python.contSet.polytope.polytope import Polytope
    
    # fullspace case
    if P.representsa_('fullspace', tol=0):
        return Polytope.Inf(P.dim())
    
    # the computation of the box outer approximation is much faster for the
    # vertex representation, so we first check for that
    if P.isVRep:
        # --- V representation: take the min and max along each dimension
        P_out = _aux_box_V(P)
    else:
        # --- H representation: compute the support function in the direction
        # of all 2n plus/minus axis-aligned basis vectors
        P_out = _aux_box_H(P)
    
    return P_out


def _aux_box_V(P: 'Polytope') -> 'Polytope':
    """
    Computation of box enclosure for a polytope in vertex representation
    """
    from cora_python.contSet.polytope.polytope import Polytope
    from cora_python.contSet.polytope.private.priv_box_V import priv_box_V
    
    # compute halfspace representation
    A, b, empty, fullDim, bounded = priv_box_V(P.V, P.dim())
    
    if empty:
        # add properties to input polytope
        P._emptySet_val = True
        P._bounded_val = True
        P._fullDim_val = False
        P._V = np.zeros((P.dim(), 0))
        P.isVRep = True
        P._minVRep_val = True
        # output empty polytope
        P_out = Polytope.empty(P.dim())
        return P_out
    
    # instantiate polytope (note that this eliminates all constraints where the
    # offset b is +-Inf)
    P_out = Polytope(A, b)
    
    # set properties
    P_out._minHRep_val = True
    P_out._emptySet_val = empty
    P_out._fullDim_val = fullDim
    P_out._bounded_val = bounded
    
    return P_out


def _aux_box_H(P: 'Polytope') -> 'Polytope':
    """
    Computation of box enclosure for a polytope in halfspace representation
    """
    from cora_python.contSet.polytope.polytope import Polytope
    from cora_python.contSet.polytope.private.priv_box_H import priv_box_H
    
    # compute halfspace representation
    A, b, empty, fullDim, bounded = priv_box_H(P.A, P.b, P.Ae, P.be, P.dim())
    
    if empty:
        # add properties to input polytope
        P._emptySet_val = True
        P._bounded_val = True
        P._fullDim_val = False
        P._V = np.zeros((P.dim(), 0))
        P.isVRep = True
        P._minVRep_val = True
        # output empty polytope
        P_out = Polytope.empty(P.dim())
        return P_out
    
    # set properties: input polytope
    P._emptySet_val = False
    P._fullDim_val = fullDim
    P._bounded_val = bounded
    
    # construct box
    P_out = Polytope(A, b)
    # set properties: output polytope
    P_out._minHRep_val = True
    P_out._emptySet_val = empty
    P_out._fullDim_val = fullDim
    P_out._bounded_val = bounded
    
    return P_out