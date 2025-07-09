"""
split method for zonotope class
"""

import numpy as np
from typing import Union, List, Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def split(Z: Zonotope, *args) -> Union[List[Zonotope], List[List[Zonotope]]]:
    """
    Splits a zonotope into two or more enclosing zonotopes
    
    Args:
        Z: zonotope object
        *args: Variable arguments
               - split(Z): split all dimensions
               - split(Z, N): split dimension N
               - split(Z, dir): split halfway in direction dir
               - split(Z, hs): split according to halfspace hs
               - split(Z, dir, 'bundle'): split using zonotope bundle
               - split(Z, N, maxOrder): split with reduction
               - split(Z, origDir, auxDir): split in perpendicular direction
        
    Returns:
        List of split zonotopes or list of lists for bundle case
        
    Example:
        Z = Zonotope(np.random.rand(2, 4))
        Zsplit = split(Z)
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Initialize
    if len(args) == 0:
        # Split all dimensions
        from .interval import interval
        Z = Zonotope(interval(Z))
        Zsplit = []
        for N in range(len(Z.c)):
            # Split one dimension
            Zsplit.append(_aux_split_one_dim(Z, N))
        return Zsplit
    
    elif len(args) == 1:
        # No splitting halfspace is passed
        if isinstance(args[0], (int, np.integer)):
            N = args[0]
            from .interval import interval
            Z = Zonotope(interval(Z))
            return _aux_split_one_dim(Z, N)
        elif isinstance(args[0], np.ndarray):
            # Split halfway in a direction
            dir_vec = args[0]
            return _aux_direction_split(Z, dir_vec)
        else:
            # Split according to halfspace
            h = args[0]
            return _aux_halfspace_split(Z, h)
    
    elif len(args) == 2:
        if args[1] == 'bundle':
            # Split halfway in a direction using a zonotope bundle
            dir_vec = args[0]
            return _aux_direction_split_bundle(Z, dir_vec)
        elif isinstance(args[0], (int, np.integer)):
            N = args[0]
            maxOrder = args[1]
            # Use reduce instead of reduceGirard
            from .reduce import reduce
            Z = reduce(Z, 'girard', maxOrder)
            return _aux_split_one_dim(Z, N)
        else:
            # Compute split in perpendicular direction
            origDir = args[0]
            auxDir = args[1]
            
            # Perpendicular direction
            projVal = auxDir.T @ origDir / np.linalg.norm(origDir)
            perpDir = auxDir - projVal * origDir / np.linalg.norm(origDir)
            
            # Split halfway in perpendicular direction
            return _aux_direction_split(Z, perpDir)
    
    else:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       f'Invalid number of arguments: {len(args) + 1}')


def _aux_split_one_dim(Z: Zonotope, dim: int) -> List[Zonotope]:
    """
    Split zonotope along one dimension
    
    Args:
        Z: zonotope object
        dim: dimension to split along
        
    Returns:
        List of two split zonotopes
    """
    # Center and generator matrix
    c = Z.c
    G = Z.G
    
    # Compute centers of split parallelpiped
    c1 = c - G[:, dim:dim+1] / 2
    c2 = c + G[:, dim:dim+1] / 2
    
    # Compute new set of generators
    Gnew = G.copy()
    Gnew[:, dim:dim+1] = Gnew[:, dim:dim+1] / 2
    
    # Generate split parallelpipeds
    Zsplit = [Zonotope(c1, Gnew), Zonotope(c2, Gnew)]
    
    return Zsplit


def _aux_direction_split(Z: Zonotope, dir_vec: np.ndarray) -> List[Zonotope]:
    """
    Split zonotope along a direction
    
    Args:
        Z: zonotope object
        dir_vec: direction vector
        
    Returns:
        List of two split zonotopes
    """
    # Center and generator matrix
    c = Z.c
    G = Z.G
    
    # Aligned generator
    alignedVal = 0
    dirUnitVec = dir_vec / np.linalg.norm(dir_vec)
    
    for i in range(G.shape[1]):
        # Aligned part
        alignedPart = dirUnitVec.T @ G[:, i:i+1]
        # Enlarge aligned generator
        alignedVal = alignedVal + abs(alignedPart[0, 0])
        # Update generator
        G[:, i:i+1] = G[:, i:i+1] - alignedPart * dirUnitVec
    
    # New generator
    newGen = alignedVal * dirUnitVec
    
    # Beta value of aligned generator
    beta = 0
    
    # Check if intersection is possible
    if abs(beta) < 1:
        # New generators and centers
        g_1 = 0.5 * (beta + 1) * newGen
        g_2 = 0.5 * (beta - 1) * newGen
        c_1 = c + beta * newGen - g_1
        c_2 = c + beta * newGen - g_2
        
        # Update zonotope
        Zsplit = [Zonotope(c_1, np.hstack([g_1, G])), 
                  Zonotope(c_2, np.hstack([g_2, G]))]
    else:
        Zsplit = [Z]
    
    return Zsplit


def _aux_direction_split_bundle(Z: Zonotope, dir_vec: np.ndarray) -> List[List[Zonotope]]:
    """
    Split zonotope along direction using zonotope bundle
    
    Args:
        Z: zonotope object
        dir_vec: direction vector
        
    Returns:
        List of zonotope bundles
    """
    # Obtain dimension
    N = len(dir_vec)
    
    # Obtain rotation matrix
    newDir = np.array([[1]] + [[0]] * (N-1))
    rotMat = _aux_rotation_matrix(dir_vec, newDir)
    
    # Obtain enclosing interval
    from .interval import interval
    IH = interval(Zonotope(rotMat @ Z.c, rotMat @ Z.G))
    
    # Get intervals (assuming interval has intervals property)
    # This is a simplified version - in practice, you'd need to access interval bounds
    intervals1 = np.array([[-1, 1]])  # Placeholder
    intervals2 = intervals1.copy()
    
    # Split intervals
    intervals1[0, 1] = 0.5 * (intervals1[0, 0] + intervals1[0, 1])
    intervals2[0, 0] = 0.5 * (intervals2[0, 0] + intervals2[0, 1])
    
    # Create intervals (simplified)
    from cora_python.contSet.interval import Interval
    IH1 = Interval(intervals1[:, 0:1], intervals1[:, 1:2])
    IH2 = Interval(intervals2[:, 0:1], intervals2[:, 1:2])
    
    # Zonotopes for zonotope bundle
    Z1 = [Z, Zonotope(rotMat.T @ IH1.c, rotMat.T @ IH1.G)]
    Z2 = [Z, Zonotope(rotMat.T @ IH2.c, rotMat.T @ IH2.G)]
    
    # Instantiate zonotope bundles (simplified - return as lists for now)
    Zsplit = [Z1, Z2]
    
    return Zsplit


def _aux_halfspace_split(Z: Zonotope, hs) -> List[Zonotope]:
    """
    Split zonotope according to halfspace
    
    Args:
        Z: zonotope object
        hs: halfspace object
        
    Returns:
        List of split zonotopes
    """
    # Halfspace values
    dir_vec = hs.A.T
    d = hs.b
    
    # Center and generator matrix
    c = Z.c
    G = Z.G
    
    # Aligned generator
    alignedVal = 0
    dirUnitVec = dir_vec / np.linalg.norm(dir_vec)
    
    for i in range(G.shape[1]):
        # Aligned part
        alignedPart = dirUnitVec.T @ G[:, i:i+1]
        # Enlarge aligned generator
        alignedVal = alignedVal + abs(alignedPart[0, 0])
        # Update generator
        G[:, i:i+1] = G[:, i:i+1] - alignedPart * dirUnitVec
    
    # New generator
    newGen = alignedVal * dirUnitVec
    
    # Beta value of aligned generator
    beta = (d - dir_vec.T @ c) / (dir_vec.T @ newGen)
    
    # Check if intersection is possible
    if abs(beta) < 1:
        # New generators and centers
        g_1 = 0.5 * (beta + 1) * newGen
        g_2 = 0.5 * (beta - 1) * newGen
        c_1 = c + beta * newGen - g_1
        c_2 = c + beta * newGen - g_2
        
        # Update zonotope
        Zsplit = [Zonotope(c_1, np.hstack([g_1, G])), 
                  Zonotope(c_2, np.hstack([g_2, G]))]
    else:
        Zsplit = [Z]
    
    return Zsplit


def _aux_rotation_matrix(dir_vec: np.ndarray, newDir: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix between two directions
    
    Args:
        dir_vec: original direction
        newDir: target direction
        
    Returns:
        Rotation matrix
    """
    # Get dimension
    N = len(dir_vec)
    
    if abs(dir_vec.T @ newDir) != 1:
        # Normalize normal vectors
        n = dir_vec / np.linalg.norm(dir_vec)
        newDir = newDir / np.linalg.norm(newDir)
        
        # Create mapping matrix
        B = np.zeros((N, N))
        B[:, 0:1] = n
        
        # Find orthonormal basis for n, uVec
        indVec = newDir - (newDir.T @ n) * n
        B[:, 1:2] = indVec / np.linalg.norm(indVec)
        
        # Complete mapping matrix B
        if N > 2:
            from scipy.linalg import null_space
            B[:, 2:] = null_space(B[:, :2].T)
        
        # Compute angle between uVec and n
        angle = np.arccos(newDir.T @ n)
        
        # Rotation matrix
        R = np.eye(N)
        R[0, 0] = np.cos(angle)
        R[0, 1] = -np.sin(angle)
        R[1, 0] = np.sin(angle)
        R[1, 1] = np.cos(angle)
        
        # Final rotation matrix
        rotMat = B @ R @ np.linalg.inv(B)
    else:
        if np.isclose(dir_vec.T @ newDir, 1):
            rotMat = np.eye(N)
        else:
            rotMat = -np.eye(N)
    
    return rotMat 