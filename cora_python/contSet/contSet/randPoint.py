"""
randPoint - generates a random point within a given continuous set

This function generates random points within a continuous set using various
sampling methods including standard, extreme, gaussian, and uniform sampling.

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-November-2020 (MATLAB)
Last update: 22-May-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Union, Optional
import numpy as np
from scipy.stats import chi2, multivariate_normal
from scipy.linalg import sqrtm, svd
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def randPoint(S: 'ContSet', N: Union[int, str] = 1, type_: str = 'standard', pr: float = 0.7) -> np.ndarray:
    """
    Generates a random point within a given continuous set
    
    Syntax:
        x = randPoint(S)
        x = randPoint(S, N)
        x = randPoint(S, N, type)
        x = randPoint(S, N, type, pr)
    
    Args:
        S: contSet object
        N: Number of random points or 'all'
        type_: Type of random point generation:
               - 'standard': Standard random sampling
               - 'extreme': Extreme points (vertices)
               - 'gaussian': Gaussian distribution sampling
               - 'uniform': Uniform sampling
               - 'uniform:hitAndRun': Hit-and-run uniform sampling
               - 'uniform:ballWalk': Ball walk uniform sampling
               - 'uniform:billiardWalk': Billiard walk uniform sampling
               - 'radius': Radius-based sampling
               - 'boundary': Boundary sampling
        pr: Probability for gaussian sampling (0 <= pr <= 1)
        
    Returns:
        np.ndarray: Random points (each column is a point)
        
    Raises:
        CORAerror: If method not supported for set type
        ValueError: If invalid parameters
        
    Example:
        >>> S = interval([1, 2], [3, 4])
        >>> points = randPoint(S, 10, 'standard')
        >>> # points is a 2x10 array of random points
    """
    # N can be numeric or 'all'
    if isinstance(N, (int, float)):
        N = int(N)
        checkN = [N, 'att', 'numeric', ['scalar', 'integer', 'positive']]
    else:
        checkN = [N, 'str', 'all']
    
    # Check input arguments
    inputArgsCheck([
        [S, 'att', 'contSet'],
        checkN,
        [type_, 'str', ['standard', 'extreme', 'gaussian', 'uniform', 
                       'uniform:hitAndRun', 'uniform:ballWalk', 'uniform:billiardWalk', 
                       'radius', 'boundary']],
        [pr, 'att', 'numeric', [lambda x: 0 <= x <= 1]]
    ])
    
    # if N = 'all', then type has to be 'extreme'
    if isinstance(N, str) and N == 'all' and type_ != 'extreme':
        raise CORAerror('CORA:wrongValue', 
                       "If the number of points is 'all', the type has to be 'extreme'.")
    
    # type = 'gaussian' implemented in contSet, other types in subclass methods
    if type_ != 'gaussian':
        try:
            x = S.randPoint_(N, type_)
        except Exception as ME:
            if S.representsa_('emptySet', np.finfo(float).eps):
                return np.empty((S.dim(), 0))
            elif S.representsa_('origin', np.finfo(float).eps):
                return np.tile(np.zeros((S.dim(), 1)), (1, N if isinstance(N, int) else 1))
            else:
                raise ME
        return x
    
    # Handle gaussian sampling
    return _randPoint_gaussian(S, N, pr)


def _randPoint_gaussian(S: 'ContSet', N: Union[int, str], pr: float) -> np.ndarray:
    """
    Generate random points using Gaussian distribution sampling
    
    This method is currently supported only for zonotope, interval, 
    ellipsoid, and polytope classes.
    """
    # Check if subclass is supported for Gaussian sampling
    if not (type(S).__name__.lower() in ['zonotope', 'interval', 'ellipsoid', 'polytope']):
        raise CORAerror('CORA:notSupported',
                       f"The function randPoint for {type(S).__name__} does not support type = 'gaussian'.")
    
    # Handle N = 'all' case
    if isinstance(N, str) and N == 'all':
        N = 1  # For gaussian with 'all', generate one point
    
    # zonotope: set does not have any generators
    if type(S).__name__.lower() == 'zonotope':
        compact_S = S.compact_('zeros', np.finfo(float).eps)
        if compact_S.generators().size == 0:
            return np.tile(S.c.reshape(-1, 1), (1, N))
    
    # generates a random vector according to Gaussian distribution within a
    # given set enclose set by ellipsoid
    if type(S).__name__.lower() != 'ellipsoid':
        from cora_python.contSet.ellipsoid import Ellipsoid
        E = Ellipsoid(S)
    else:
        # degeneracy handling for ellipsoid: projection on proper subspace,
        # back-transformation after sampling of points
        E = S
        
        # read center, shift ellipsoid
        c = E.q.copy()
        E = E + (-c)
        
        # compute rank and dimension
        r = np.linalg.matrix_rank(E.Q)
        n = E.dim()
        
        # determine degeneracy: if so, project on proper subspace (via SVD)
        n_rem = n - r
        if n_rem > 0:
            T, _, _ = svd(E.Q)
            E = T.T @ E
            E = E.project(list(range(r)))
            G = np.linalg.inv(sqrtm(E.Q))
            E = G @ E
    
    # obtain center
    c = E.center()
    
    # quantile function for probability pr of the chi-squared distribution
    quantile_value = chi2.ppf(pr, E.dim())
    
    # obtain covariance matrix
    Sigma = E.Q / quantile_value
    
    # create N samples
    x = np.zeros((S.dim(), N))
    remaining_samples = N
    idx = 0
    
    while remaining_samples > 0:
        # create remaining number of samples of normal distribution
        # Flatten c to 1D array for multivariate_normal.rvs
        c_flat = c.flatten()
        pt = multivariate_normal.rvs(mean=c_flat, cov=Sigma, size=remaining_samples)
        if pt.ndim == 1:
            pt = pt.reshape(-1, 1)
        else:
            pt = pt.T
        
        # check containment
        # Use the public contains method to get the boolean result
        pt_inside = S.contains(pt)
        
        # contains returns arrays in the format [1, n_points] for multiple points, so flatten for indexing
        if isinstance(pt_inside, np.ndarray):
            pt_inside_flat = pt_inside.flatten()
        else:
            # Single point case
            pt_inside_flat = np.array([pt_inside])
        
        n_inside = np.sum(pt_inside_flat)
        remaining_samples = remaining_samples - n_inside
        
        # store the ones that are contained, repeat loop for remaining number
        if n_inside > 0:
            x[:, idx:idx+n_inside] = pt[:, pt_inside_flat]
            idx += n_inside
    
    # degenerate ellipsoids: stack again, backtransform and shift
    if type(S).__name__.lower() == 'ellipsoid' and 'n_rem' in locals() and n_rem > 0:
        x = T @ np.vstack([np.linalg.inv(G) @ x, np.zeros((n_rem, N))]) + c.reshape(-1, 1)
    
    return x 