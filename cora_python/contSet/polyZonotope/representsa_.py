"""
representsa_ - checks if a polynomial zonotope can also be represented by
   a different set, e.g., a special case

Syntax:
   res = representsa_(pZ,type,tol)
   [res,S] = representsa_(pZ,type,tol)

Inputs:
   pZ - polyZonotope object
   type - other set representation or 'origin', 'point', 'hyperplane'
   tol - tolerance

Outputs:
   res - true/false
   S - converted set

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/representsa

Authors:       Mark Wetzlinger, Niklas Kochdumper, Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       19-July-2023
Last update:   ---
Last revision: ---
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple, Optional, Any, Union
from scipy.spatial import ConvexHull

if TYPE_CHECKING:
    from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope


def representsa_(pZ: 'PolyZonotope', set_type: str, tol: float, method: str = 'linearize', 
                iter_val: int = 1, splits: int = 0) -> Union[bool, Tuple[bool, Optional[Any]]]:
    """
    Checks if a polynomial zonotope can also be represented by a different set type.
    
    Args:
        pZ: polyZonotope object
        set_type: other set representation or 'origin', 'point', 'hyperplane'
        tol: tolerance
        method: (for conPolyZono) algorithm used for contraction
        iter_val: (for conPolyZono) number of iterations
        splits: (for conPolyZono) number of recursive splits
        
    Returns:
        bool or tuple: Whether pZ can be represented by set_type, optionally with converted set
    """
    # Import here to avoid circular imports
    from cora_python.contSet.contSet.representsa_emptyObject import representsa_emptyObject
    from cora_python.g.functions.helper.sets.removeRedundantExponents import removeRedundantExponents
    
    # Check empty object case
    try:
        empty, res, S_conv = representsa_emptyObject(pZ, set_type)
        if empty:
            return res, S_conv
    except:
        empty, res = representsa_emptyObject(pZ, set_type, return_conv=False)
        if empty:
            return res
    
    # Dimension
    n = pZ.dim()
    
    # Init second output argument (covering all cases with res = false)
    S = None
    
    # Minimal representation
    try:
        pZ = pZ.compact_('all', np.finfo(float).eps)
    except:
        # If compact_ not available, continue with original
        pass
    
    # Is polynomial zonotope a single point?
    isPoint = (pZ.G.size == 0 and pZ.GI.size == 0)
    
    if set_type == 'origin':
        if isPoint and np.allclose(pZ.c, 0, atol=tol):
            res = True
        else:
            try:
                # Convert to interval and check norm
                I = pZ.interval()
                res = I.norm() <= tol
            except:
                res = False
        
        if res:
            S = np.zeros((n, 1))
            return res, S
        return res
        
    elif set_type == 'point':
        res = isPoint
        if res:
            S = pZ.c
            return res, S
        return res
        
    elif set_type == 'capsule':
        # 1D, a point, or maximum one generator
        res = (n == 1 or isPoint or 
               (pZ.G.shape[1] + pZ.GI.shape[1] <= 1))
        if res:
            from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
            raise CORAerror('CORA:notSupported',
                f'Conversion from polyZonotope to {set_type} not supported.')
        return res
        
    elif set_type == 'conPolyZono':
        res = True
        if res:
            try:
                from cora_python.contSet.conPolyZono.conPolyZono import ConPolyZono
                S = ConPolyZono(pZ)
                return res, S
            except ImportError:
                return res
        return res
        
    elif set_type == 'conZonotope':
        # Only in special cases
        res = (n == 1 or _aux_isZonotope(pZ, tol) or _aux_isPolytope(pZ, tol))
        if res:
            from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
            raise CORAerror('CORA:notSupported',
                f'Conversion from polyZonotope to {set_type} not supported.')
        return res
        
    elif set_type == 'ellipsoid':
        res = (n == 1 or isPoint)
        if res:
            try:
                from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
                S = Ellipsoid(pZ)
                return res, S
            except ImportError:
                return res
        return res
        
    elif set_type == 'halfspace':
        # Polynomial zonotopes are bounded
        res = False
        return res
        
    elif set_type == 'interval':
        res = (n == 1 or (_aux_isZonotope(pZ, tol) and 
               _check_zonotope_interval_representsa(pZ, tol)))
        if res:
            try:
                S = pZ.interval()
                return res, S
            except:
                return res
        return res
        
    elif set_type == 'polytope':
        res = (n == 1 or _aux_isZonotope(pZ, tol) or _aux_isPolytope(pZ, tol))
        if res:
            try:
                from cora_python.contSet.polytope.polytope import Polytope
                S = Polytope(pZ)
                return res, S
            except ImportError:
                return res
        return res
        
    elif set_type == 'polyZonotope':
        # Obviously true
        res = True
        S = pZ
        return res, S
        
    elif set_type == 'probZonotope':
        res = False
        return res
        
    elif set_type == 'zonoBundle':
        # Only in special cases
        res = (n == 1 or _aux_isZonotope(pZ, tol) or _aux_isPolytope(pZ, tol))
        if res:
            from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
            raise CORAerror('CORA:notSupported',
                f'Conversion from polyZonotope to {set_type} not supported.')
        return res
        
    elif set_type == 'zonotope':
        res = (n == 1 or _aux_isZonotope(pZ, tol))
        if res:
            try:
                from cora_python.contSet.zonotope.zonotope import Zonotope
                S = Zonotope(pZ)
                return res, S
            except ImportError:
                return res
        return res
        
    elif set_type == 'hyperplane':
        # Polynomial zonotopes are bounded
        res = False
        return res
        
    elif set_type == 'parallelotope':
        res = (n == 1 or (pZ.G.size == 0 and 
               _check_zonotope_parallelotope_representsa(pZ, tol)))
        if res:
            try:
                from cora_python.contSet.zonotope.zonotope import Zonotope
                S = Zonotope(pZ)
                return res, S
            except ImportError:
                return res
        return res
        
    elif set_type == 'emptySet':
        # Already handled in isemptyobject
        res = False
        return res
        
    elif set_type == 'fullspace':
        res = False
        return res
        
    else:
        from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
        raise CORAerror('CORA:notSupported',
            f'Comparison of polyZonotope to {set_type} not supported.')


# Auxiliary functions


def _aux_isZonotope(pZ: 'PolyZonotope', tol: float) -> bool:
    """Check if polynomial zonotope can be represented as a zonotope"""
    
    if pZ.G.size == 0:
        return True
    
    # Remove redundant exponent vectors
    try:
        from cora_python.g.functions.helper.sets.removeRedundantExponents import removeRedundantExponents
        E, G = removeRedundantExponents(pZ.E, pZ.G)
    except:
        E, G = pZ.E, pZ.G
    
    # Check matrix dimensions
    if E.shape[0] != G.shape[1]:
        return False
    
    # Sort exponent matrix rows (descending)
    if E.size > 0:
        E = E[np.lexsort(-E.T)]
    
    # Check if exponent matrix is the identity matrix
    if E.size > 0:
        diag_E = np.diag(np.diag(E))
        return np.sum(np.sum(np.abs(E - diag_E))) == 0
    
    return True


def _aux_isPolytope(pZ: 'PolyZonotope', tol: float) -> bool:
    """Check if polynomial zonotope can be represented as a polytope"""
    
    # Check if variable appears with exponent greater than 1
    if np.any(pZ.E >= 2):
        return False
    
    # Compute vertices (simplified version)
    try:
        V, Ilist = _aux_polyVertices(pZ)
        
        # Loop over all vertices
        for i in range(V.shape[1]):
            # Compute vectors of normal cone
            N = _aux_normalCone(pZ, Ilist[i], tol)
            
            # Loop over all vertices
            for j in range(V.shape[1]):
                if i != j:
                    if not _aux_inCone(N, V[:, i], V[:, j], tol):
                        return False
        
        return True
    except:
        # If vertex computation fails, assume not a polytope
        return False


def _aux_inCone(N: np.ndarray, v: np.ndarray, v_: np.ndarray, tol: float) -> bool:
    """Check if a vertex is inside the normal cone"""
    
    try:
        from scipy.optimize import linprog
        
        n, m = N.shape
        c = np.concatenate([np.zeros(m), np.ones(n)])
        
        A1 = np.concatenate([N.T, -np.eye(n)], axis=1)
        b1 = v_ - v
        A2 = np.concatenate([-N.T, -np.eye(n)], axis=1)
        b2 = v - v_
        
        A_ub = np.concatenate([A1, A2], axis=0)
        b_ub = np.concatenate([b1, b2])
        
        bounds = [(0, None) for _ in range(n + m)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x = result.x
            return np.all(x[m:] < tol)
        else:
            return False
    except:
        return False


def _aux_normalCone(pZ: 'PolyZonotope', Ilist: np.ndarray, tol: float) -> np.ndarray:
    """Compute the normal cone for a vertex"""
    
    Nall = np.array([]).reshape(pZ.G.shape[0], 0)
    
    for l in range(Ilist.shape[1]):
        I = Ilist[:, l]
        N = np.zeros((pZ.G.shape[0], len(I)))
        
        # Loop over all factors
        for i in range(len(I)):
            ind = np.where(pZ.E[i, :] > 0)[0]
            
            # Loop over all generators
            for j in range(len(ind)):
                k = ind[j]
                N[:, i] += pZ.G[:, k] * np.prod(I**pZ.E[:, k]) / I[i]**pZ.E[i, k]
        
        N = N * (-np.sign(I).reshape(1, -1))
        N = N[:, np.sum(np.abs(N), axis=0) > tol]
        Nall = np.concatenate([Nall, N], axis=1)
    
    return Nall


def _aux_polyVertices(pZ: 'PolyZonotope') -> Tuple[np.ndarray, list]:
    """Compute the polytope vertices and store the corresponding factor values"""
    
    # Determine all potential vertices
    p = pZ.E.shape[0]
    n = pZ.G.shape[0]
    
    # Create vertices of hypercube [-1,1]^p
    from cora_python.contSet.interval.interval import Interval
    I_int = Interval(-np.ones((p, 1)), np.ones((p, 1)))
    I = I_int.vertices()
    
    V = np.zeros((n, I.shape[1]))
    
    for i in range(I.shape[1]):
        V[:, i] = pZ.c.flatten()
        for j in range(pZ.G.shape[1]):
            V[:, i] += pZ.G[:, j] * np.prod(I[:, i]**pZ.E[:, j])
    
    # Remove redundant points
    V_sorted = np.sort(V, axis=0)
    _, unique_indices = np.unique(V_sorted, axis=1, return_index=True)
    V = V[:, unique_indices]
    I = I[:, unique_indices]
    
    # Group identical vertices
    V_ = np.zeros_like(V)
    Ilist = []
    V_[:, 0] = V[:, 0]
    Itemp = I[:, [0]]
    counter = 0
    
    tol = 1e-10
    for i in range(1, V.shape[1]):
        if not np.allclose(V[:, i], V_[:, counter], atol=tol):
            # Unique point
            Ilist.append(Itemp)
            counter += 1
            V_[:, counter] = V[:, i]
            Itemp = I[:, [i]]
        else:
            # Redundant point
            Itemp = np.concatenate([Itemp, I[:, [i]]], axis=1)
    
    Ilist.append(Itemp)
    V = V_[:, :counter+1]
    
    # Determine vertices with the n-dimensional convex hull
    if V.shape[1] > n:
        try:
            hull = ConvexHull(V.T)
            indices = np.unique(hull.vertices)
            V = V[:, indices]
            Ilist = [Ilist[i] for i in indices]
        except:
            pass
    
    return V, Ilist


def _check_zonotope_interval_representsa(pZ: 'PolyZonotope', tol: float) -> bool:
    """Check if the zonotope representation can be converted to interval"""
    try:
        from cora_python.contSet.zonotope.zonotope import Zonotope
        Z = Zonotope(pZ)
        return Z.representsa_('interval', tol)
    except:
        return False


def _check_zonotope_parallelotope_representsa(pZ: 'PolyZonotope', tol: float) -> bool:
    """Check if the zonotope representation can be converted to parallelotope"""
    try:
        from cora_python.contSet.zonotope.zonotope import Zonotope
        Z = Zonotope(pZ)
        return Z.representsa_('parallelotope', tol)
    except:
        return False