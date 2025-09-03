"""
splitDepFactor - Splits one dependent factor of a polynomial zonotope

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Niklas Kochdumper, Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope


def splitDepFactor(pZ: PolyZonotope, ind, polyOrd=None):
    """
    Split one dependent factor of a polynomial zonotope
    
    Syntax:
        pZsplit = splitDepFactor(pZ, ind)
        pZsplit = splitDepFactor(pZ, ind, polyOrd)
    
    Inputs:
        pZ - polyZonotope object
        ind - identifier of the dependent factor that is splitted
        polyOrd - maximum number of polynomial terms that are splitted exactly
                  (without an over-approximation)
    
    Outputs:
        pZsplit - list of split polyZonotopes
    
    Example: 
        pZ = polyZonotope([0;0],[2 0 1;0 2 1],[0;0],[1 0 3;0 1 1],[1;2]);
        pZsplit = splitDepFactor(pZ, 1, 5);
    
    Reference:
      [1] Kochdumper, Niklas. Extensions of Polynomial Zonotopes and their 
          Application to Verification of Cyber-Physical Systems. Diss. 
          Technische Universität München, 2022.
    """
    
    # Find selected dependent factor
    ind_mask = (pZ.id == ind).flatten()  # Flatten to 1D boolean array
    if np.sum(ind_mask) != 1:
        raise ValueError("Given value for 'ind' should be contained in identifiers of polynomial zonotope")
    
    E = pZ.E
    E_ind = E[ind_mask, :]
    
    # Parse input arguments
    if polyOrd is None:
        polyOrd = int(np.max(E_ind))
    else:
        polyOrd = int(polyOrd)
    
    # Determine all generators in which the selected dependent factor occurs
    genInd = ((E_ind > 0) & (E_ind <= polyOrd)).flatten()  # Flatten to 1D boolean array
    
    # [1, Prop 3.1.43/44] using bounds [0,1] and [-1,0]
    # Create coeffs for i=1...polyOrd:
    # (0.5 + 0.5 * a_ind)^i and (-0.5 + 0.5 * a_ind)^i
    polyCoeff1 = [None] * max(2, polyOrd)
    polyCoeff2 = [None] * max(2, polyOrd)
    hout = np.sum(~genInd)
    
    # Create pascal triangle
    P = [[1, 1]]
    for i in range(2, polyOrd + 1):
        prev_row = P[i-2]
        new_row = [1]
        for j in range(1, i):
            new_row.append(prev_row[j-1] + prev_row[j])
        new_row.append(1)
        P.append(new_row)
    
    for i in range(1, polyOrd + 1):
        Pi = P[i-1]
        polyCoeff1[i-1] = (0.5**i) * np.array(Pi)
        # MATLAB: (-mod(i:-1:0, 2)*2+1) creates [-1, 1, -1, 1, ...]
        # For i=1: [-1, 1], for i=2: [1, -1, 1], etc.
        alternating_signs = np.array([(-1)**(i-j) for j in range(i+1)])
        polyCoeff2[i-1] = (0.5**i) * np.array(Pi) * alternating_signs
        
        numExpi = np.sum(E_ind == i)
        hout = hout + len(Pi) * numExpi
    
    # Construct the modified generators for the splitted zonotopes
    c1 = pZ.c.copy()
    c2 = pZ.c.copy()
    
    G1 = np.full((len(c1), hout), np.nan)
    G2 = np.full((len(c2), hout), np.nan)
    Eout = np.full((E.shape[0], hout), np.nan)  # identical for splitted sets
    
    h = 0  # 0-based indexing in Python
    dh = np.sum(~genInd)
    if dh > 0:
        G1[:, h:h+dh] = pZ.G[:, ~genInd]
        G2[:, h:h+dh] = pZ.G[:, ~genInd]
        Eout[:, h:h+dh] = E[:, ~genInd]
        h = h + dh
    
    for i in range(1, polyOrd + 1):
        coef1 = polyCoeff1[i-1]
        coef2 = polyCoeff2[i-1]
        
        expi = (E_ind == i).flatten()  # Flatten to 1D boolean array
        
        dh = len(coef1) * np.sum(expi)
        
        # Kronecker product equivalent
        G1_cols = []
        G2_cols = []
        for j in range(len(coef1)):
            G1_cols.append(coef1[j] * pZ.G[:, expi])
            G2_cols.append(coef2[j] * pZ.G[:, expi])
        
        if G1_cols:
            G1[:, h:h+dh] = np.hstack(G1_cols)
            G2[:, h:h+dh] = np.hstack(G2_cols)
        
        # Repeat E columns
        E_repeated = np.tile(E[:, expi], (1, i+1))
        Eout[:, h:h+dh] = E_repeated
        Eout[ind_mask, h:h+dh] = np.kron(np.arange(i+1), np.ones(np.sum(expi)))  # fix a_ind
        
        h = h + dh
        
        genInd = np.logical_xor(genInd, expi)
    
    # Over-approximate all selected generators that did not get splitted
    if np.sum(genInd) > 0:
        Eout = np.vstack([Eout, np.zeros((1, Eout.shape[1]))])
        
        Eout[-1, genInd] = Eout[ind_mask, genInd]
        Eout[ind_mask, genInd] = 0
    
    # Add every generator with all-zero exponent matrix to the zonotope center
    temp = np.sum(Eout, axis=0)
    genInd_zero = temp == 0
    
    c1 = c1 + np.sum(G1[:, genInd_zero], axis=1, keepdims=True)
    G1 = G1[:, ~genInd_zero]
    Eout = Eout[:, ~genInd_zero]
    
    c2 = c2 + np.sum(G2[:, genInd_zero], axis=1, keepdims=True)
    G2 = G2[:, ~genInd_zero]
    
    # Construct the resulting polynomial zonotopes
    pZsplit = [None, None]
    pZsplit[0] = PolyZonotope(c1, G1, pZ.GI, Eout)
    pZsplit[1] = PolyZonotope(c2, G2, pZ.GI, Eout)
    
    return pZsplit
