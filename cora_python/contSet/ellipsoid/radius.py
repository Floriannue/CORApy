"""
radius - computes the radius of an enclosing hyperball of an ellipsoid

Syntax:
    r = radius(E)
    r = radius(E,i)

Inputs:
    E - ellipsoid object
    i - number of largest radii to be returned

Outputs:
    r - radius/vector of radii of enclosing hyperball

Example: 
    E = ellipsoid([1,0.5;0.5,3],[1;-1])
    r = radius(E)

    figure hold on
    plot(E,[1,2],'r')
    E_circ = ellipsoid(r^2 * eye(2),center(E))
    plot(E_circ,[1,2],'b')

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff, Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       05-March-2021 (MATLAB)
Last update:   19-March-2021 (VG, empty case added, MATLAB)
               24-March-2022 (VG, change input argument, MATLAB)
               04-July-2022 (VG, input checks, MATLAB)
Python translation: 2025
"""

import numpy as np
from scipy.sparse.linalg import eigs
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck

# Removed debug logging utility

def radius(E, *args):
    """
    Computes the radius of an enclosing hyperball of an ellipsoid
    
    Args:
        E: ellipsoid object
        i: number of largest radii to be returned (default: 1)
        
    Returns:
        r: radius/vector of radii of enclosing hyperball
    """
    # Default: only largest radius considered
    i_val = setDefaultValues([1], args)[0]
    if isinstance(i_val, list) and len(i_val) == 1:
        i = i_val[0]
    else:
        i = i_val
    
    # Check input arguments
    inputArgsCheck([
        [E, 'att', 'ellipsoid'],
        [i, 'att', 'numeric', ['integer', 'positive', lambda x: x <= E.dim()]] # Reverted lambda: x is now guaranteed scalar
    ])
    
    # Quick check for empty set
    if E.representsa_('emptySet', E.TOL):
        return np.zeros((E.dim(), 0))
    
    # Compute eigenvalues of Q (PSD). Radii are sqrt of eigenvalues along principal axes.
    # Use dense eig for small matrices to avoid scipy eigs corner issues.
    vals = np.linalg.eigvalsh(E.Q)
    # sort descending
    vals_sorted = np.sort(vals)[::-1]
    take = min(i, vals_sorted.size)
    d = vals_sorted[:take]
    
    # Ensure real
    if np.iscomplexobj(d):
        d = np.real(d)
    
    # Compute radius
    r = np.sqrt(d)
    # Ensure shape: for i==1 return scalar (numpy scalar) to match tests expecting size==1
    if take == 1:
        return np.array([r.item()])
    
    return r