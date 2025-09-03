"""
splitLongestGen - Splits the longest generator dependent generator with a 
   polynomial order of 1 for a polynomial zonotope

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Niklas Kochdumper
         Python: AI Assistant
"""

import numpy as np
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope


def splitLongestGen(pZ: PolyZonotope, polyOrd=None):
    """
    Split the longest generator dependent generator with a polynomial order of 1 for a polynomial zonotope
    
    Syntax:
        pZsplit = splitLongestGen(pZ)
        pZsplit = splitLongestGen(pZ, polyOrd)
    
    Inputs:
        pZ - polyZonotope object
        polyOrd - maximum number of polynomial terms that are splitted exactly
                  (without an over-approximation)
    
    Outputs:
        pZsplit - list of split polyZonotopes
        factor - identifier of the dependent factor that is split
    
    Example: 
        pZ = polyZonotope([0;0],[2 0 1;0 2 1],[0;0],[1 0 3;0 1 1]);
        temp = splitLongestGen(pZ);
        pZsplit1 = splitLongestGen(temp[0]);
        pZsplit2 = splitLongestGen(temp[1]);
    """
    
    # Check if there are any generators
    if pZ.G.shape[1] == 0:
        raise ValueError("Cannot split polynomial zonotope with no generators")
    
    # Determine longest generator
    len_gen = np.sum(pZ.G**2, axis=0)
    ind = np.argmax(len_gen)
    
    # Find factor with the largest exponent
    factor = np.argmax(pZ.E[:, ind])
    factor = pZ.id[factor, 0]  # Access the scalar value from 2D array
    
    # Split the zonotope at the determined generator
    return pZ.splitDepFactor(factor, polyOrd)
