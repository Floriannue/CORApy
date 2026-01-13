"""
dependentTerms - computes exact Taylor terms of a matrix zonotope square
   and a matrix zonotope exponential

Syntax:
    [zSq, zH] = dependentTerms(matZ, r)

Inputs:
    matZ - matZonotope object
    r - time step size

Outputs:
    zSq - exact square matrix
    zH - exact Taylor terms up to second order

Authors:       Matthias Althoff, Tobias Ladner (MATLAB)
              Python translation: 2025
"""

import numpy as np
from typing import Tuple
from .matZonotope import matZonotope
from .dim import dim
from .numgens import numgens
from .pagemtimes import pagemtimes

# Try to import combinator, fallback if not available
try:
    from cora_python.g.functions.matlab.validate.check.auxiliary.combinator import combinator
except ImportError:
    # Fallback combinator implementation
    from itertools import combinations
    def combinator(n, k, mode='c'):
        """Simple combinator for combinations"""
        if mode == 'c':
            # Return combinations as numpy array with MATLAB 1-indexing
            comb_list = list(combinations(range(1, n + 1), k))
            return np.array(comb_list) if comb_list else np.array([]).reshape(0, k)
        else:
            raise NotImplementedError(f"Combinator mode '{mode}' not implemented")


def dependentTerms(matZ: matZonotope, r: float) -> Tuple[matZonotope, matZonotope]:
    """
    Computes exact Taylor terms of a matrix zonotope square and exponential
    
    Args:
        matZ: matZonotope object
        r: time step size
        
    Returns:
        zSq: Exact square matrix zonotope
        zH: Exact Taylor terms up to second order (matrix zonotope)
    """
    # Load data from object structure
    C = matZ.C
    G = matZ.G
    n = dim(matZ)[0]  # Get row dimension
    gens = numgens(matZ)
    
    # Square computation
    # New center
    # MATLAB: sqC = C^2*r^2 + 0.5*sum(pagemtimes(G,G),3)*r^2;
    sqC = C @ C * r ** 2 + 0.5 * np.sum(pagemtimes(G, G), axis=2) * r ** 2
    
    # Get generators
    # MATLAB: sqG = cat(3, (pagemtimes(C,G) + pagemtimes(G,C))*r^2, 0.5*pagemtimes(G,G)*r^2 ...);
    sqG_list = [
        (pagemtimes(C, G) + pagemtimes(G, C)) * r ** 2,
        0.5 * pagemtimes(G, G) * r ** 2
    ]
    
    # Get indices for 3rd set of generators
    # MATLAB: if (gens>=2)
    if gens >= 2:
        # MATLAB: ind = combinator(gens,2,'c');
        ind = combinator(gens, 2, 'c')
        # MATLAB: sqG = cat(3, sqG, ...);
        gen_pairs = []
        for i in range(ind.shape[0]):
            idx1 = int(ind[i, 0]) - 1  # MATLAB 1-indexed, Python 0-indexed
            idx2 = int(ind[i, 1]) - 1
            G1 = G[:, :, idx1]
            G2 = G[:, :, idx2]
            gen_pairs.append((G1 @ G2 + G2 @ G1) * r ** 2)
        
        if len(gen_pairs) > 0:
            gen_pairs_array = np.stack(gen_pairs, axis=2)
            sqG_list.append(gen_pairs_array)
    
    # Concatenate all generators
    sqG = np.concatenate(sqG_list, axis=2)
    
    # H computation
    # New center
    # MATLAB: HC = eye(n) + C*r + 0.5*sqC;
    HC = np.eye(n) + C * r + 0.5 * sqC
    
    # Get generators
    # MATLAB: HG = 0.5 * sqG;
    # MATLAB: HG(:,:,1:gens) = HG(:,:,1:gens) + G*r;
    HG = 0.5 * sqG
    if gens > 0:
        HG[:, :, :gens] = HG[:, :, :gens] + G * r
    
    # Write as matrix zonotopes
    zSq = matZonotope(sqC, sqG)
    zH = matZonotope(HC, HG)
    
    return zSq, zH
