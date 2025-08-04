"""
randPoint_ - generates random points within a full-dimensional space
   case R^0: only point is 0 (not representable in MATLAB)

Syntax:
   p = randPoint_(fs)
   p = randPoint_(fs,N)
   p = randPoint_(fs,N,type)
   p = randPoint_(fs,'all','extreme')

Inputs:
   fs - fullspace object
   N - number of random points
   type - type of the random point ('standard', 'extreme')

Outputs:
   p - random point in R^n

Example: 
   O = fullspace(2);
   p = randPoint(O);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/randPoint

Authors:       Mark Wetzlinger
Written:       05-April-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def randPoint_(fs, N=None, type_='standard', *args):
    """
    Generates random points within a full-dimensional space
    case R^0: only point is 0 (not representable in MATLAB)
    
    Args:
        fs: fullspace object
        N: number of random points
        type_: type of the random point ('standard', 'extreme')
        *args: additional arguments
        
    Returns:
        p: random point in R^n
    """
    if fs.dimension == 0:
        raise CORAerror('CORA:notSupported', 'Sampling of R^0 not supported')
    
    if type_ == 'standard':
        # use built-in random sampling
        p = np.random.randn(fs.dimension, N)
    
    elif type_ == 'extreme':
        if isinstance(N, str) and N == 'all':
            # sample all 2^n 'vertices', i.e., -Inf/+Inf
            # may throw an error if the dimension is too large...
            I = fs.interval()
            p = I.vertices()
        else:
            # init with random points...
            p = np.random.randn(fs.dimension, N)
            # at least one entry has to be -Inf/+Inf
            pos_inf = np.floor(np.random.rand(1, N) * fs.dimension).astype(int)
            s = np.sign(np.random.randn(1, N))
            linear_idx = (np.arange(N) * fs.dimension) + pos_inf
            p.flat[linear_idx] = s * np.inf
    
    return p

# ------------------------------ END OF CODE ------------------------------ 