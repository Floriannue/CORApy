"""
polytope - computes an enclosing polytope for the polynomial zonotope

Syntax:
    P = polytope(pZ)
    P = polytope(pZ,'template',dirs)

Inputs:
    pZ - polyZonotope object
    dirs - directions for the template polyhedron

Outputs:
    P - polytope object

Example: 
    pZ = polyZonotope(np.array([[0], [0]]), np.array([[1, 0, -1, 1], [1, 1, 1, 1]]), [], np.array([[1, 0, 1, 2], [0, 1, 1, 0]]))

    P1 = polytope(pZ)
    P2 = polytope(pZ, 'template', np.array([[1, 0, 1, 1], [0, 1, 1, -1]]))
    
Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: interval, zonotope

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       10-November-2018 (MATLAB)
Last update:   ---
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from scipy.spatial import ConvexHull

def polytope(pZ, *varargin):
    """
    Computes an enclosing polytope for the polynomial zonotope
    
    Args:
        pZ: polyZonotope object
        *varargin: template and directions arguments (optional)
        
    Returns:
        P: polytope object
    """
    # Check if a template polyhedron should be computed
    if len(varargin) >= 2 and varargin[0] == 'template':
        
        # Initialize variables
        dir = varargin[1]
        
        C = np.vstack([dir.T, dir.T])
        d = np.zeros((C.shape[0], 1))
        
        counter = 1
        
        # Loop over all directions
        for i in range(dir.shape[1]):
            
            dirTemp = dir[:, i] / np.linalg.norm(dir[:, i])
            I = pZ.supportFunc_(dirTemp, 'range', 'split', 8, 1e-3)
            
            C[counter-1, :] = dirTemp
            d[counter-1] = I.supremum()
            
            C[counter, :] = -dirTemp
            d[counter] = -I.infimum()
            
            counter = counter + 2
        
        P = polytope(C, d)
        
    else:
        
        Tol = 1e-14
        
        # Over-approximate all terms where factors with exponents greater than
        # one appear with a zonotope
        ind = np.where(np.max(pZ.E, axis=0) > 1)[0]
        
        E_ = pZ.E[:, ind]
        G_ = pZ.G[:, ind]
        c_ = np.zeros_like(pZ.c)
        
        # Remove the high-order terms
        pZ.E = np.delete(pZ.E, ind, axis=1)
        pZ.G = np.delete(pZ.G, ind, axis=1)
        
        # Convert to zonotope
        from cora_python.contSet.zonotope import Zonotope
        Z = Zonotope(pZ.__class__(c_, G_, [], E_))
        
        # Add over-approximation to the polynomial zonotope
        pZ.c = pZ.c + Z.c
        pZ.GI = np.hstack([pZ.GI, Z.G]) if pZ.GI.size > 0 else Z.G
        
        # Determine all potential vertices and remove redundant points
        points = pZ.randPoint_('all', 'extreme')
        points = points[:, np.lexsort(points.T)]
        
        points_ = np.zeros_like(points)
        points_[:, 0] = points[:, 0]
        counter = 1
        
        for i in range(1, points.shape[1]):
            if not np.all(withinTol(points[:, i], points_[:, counter-1], Tol)):
                counter = counter + 1
                points_[:, counter-1] = points[:, i]
        
        points = points_[:, :counter]
        
        # Determine vertices with the n-dimensional convex hull
        if points.shape[1] > points.shape[0]:
            try:
                hull = ConvexHull(points.T)
                ind = hull.vertices
                vert = points[:, ind]
            except:
                # Fallback if convex hull fails
                vert = points
        else:
            vert = points
        
        # Construct the resulting polytope
        from cora_python.contSet.polytope import Polytope
        P = Polytope(vert)
    
    # Set properties
    P._bounded_val = True
    
    return P
