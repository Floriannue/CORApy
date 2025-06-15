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

from typing import Union, Optional
import numpy as np
from scipy.stats import chi2, multivariate_normal
from .dim import dim
from .representsa_ import representsa_
from .randPoint_ import randPoint_
from .contains import contains
from .center import center
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def randPoint(S: 'ContSet', 
              N: Union[int, str] = 1, 
              type_: str = 'standard',
              pr: float = 0.7) -> np.ndarray:
    """
    Generates a random point within a given continuous set
    
    Args:
        S: contSet object
        N: Number of random points or 'all' for extreme points
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
        CORAError: If method not supported for set type
        ValueError: If invalid parameters
        
    Example:
        >>> S = interval([1, 2], [3, 4])
        >>> points = randPoint(S, 10, 'standard')
        >>> # points is a 2x10 array of random points
    """
    # Validate inputs
    if isinstance(N, str):
        if N != 'all':
            raise ValueError("If N is string, it must be 'all'")
        if type_ != 'extreme':
            raise ValueError("If N is 'all', type must be 'extreme'")
    elif not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer or 'all'")
    
    valid_types = ['standard', 'extreme', 'gaussian', 'uniform', 
                   'uniform:hitAndRun', 'uniform:ballWalk', 'uniform:billiardWalk',
                   'radius', 'boundary']
    if type_ not in valid_types:
        raise ValueError(f"Invalid type '{type_}'. Use one of {valid_types}")
    
    if not (0 <= pr <= 1):
        raise ValueError("pr must be between 0 and 1")
    
    # Handle gaussian sampling (implemented in base class)
    if type_ == 'gaussian':
        return _randPoint_gaussian(S, N, pr)
    
    # For other types, delegate to subclass
    try:
        x = randPoint_(S, N, type_)
    except Exception as ME:
        # Handle special cases
        if representsa_(S, 'emptySet', 1e-15):
            return np.empty((dim(S), 0))
        elif representsa_(S, 'origin', 1e-15):
            return np.tile(np.zeros((dim(S), 1)), (1, N if isinstance(N, int) else 1))
        else:
            raise ME
    
    return x


def _randPoint_gaussian(S: 'ContSet', N: Union[int, str], pr: float) -> np.ndarray:
    """
    Generate random points using Gaussian distribution sampling
    
    This method is currently supported only for zonotope, interval, 
    ellipsoid, and polytope classes.
    """
    # Check if subclass is supported for Gaussian sampling
    supported_classes = ['Zonotope', 'Interval', 'Ellipsoid', 'Polytope']
    class_name = type(S).__name__
    
    if class_name not in supported_classes:
        raise CORAError('CORA:notSupported',
                       f"The function randPoint for {class_name} does not support type = 'gaussian'.")
    
    # Handle degenerate zonotope case
    if class_name == 'Zonotope':
        # Check if zonotope has no generators (would need compact_ and generators methods)
        # For now, assume it's handled in the subclass
        pass
    
    # Enclose set by ellipsoid (if not already an ellipsoid)
    if class_name != 'Ellipsoid':
        # Would need ellipsoid constructor that takes a set
        # For now, this is a placeholder - actual implementation would depend on ellipsoid class
        raise NotImplementedError("Ellipsoid enclosure not yet implemented")
    else:
        E = S
    
    # Get center and covariance matrix
    c = center(E)
    
    # For ellipsoid, we would need access to the Q matrix
    # This is a placeholder implementation
    if hasattr(E, 'Q'):
        Q = E.Q
    else:
        # Fallback - would need proper ellipsoid implementation
        Q = np.eye(dim(E))
    
    # Quantile function for probability pr of the chi-squared distribution
    quantile_value = chi2.ppf(pr, dim(E))
    
    # Obtain covariance matrix
    Sigma = Q / quantile_value
    
    # Create N samples
    if isinstance(N, str):  # N == 'all'
        N = 1  # For 'all' with gaussian, generate one point
    
    x = np.zeros((dim(S), N))
    remaining_samples = N
    idx = 0
    
    while remaining_samples > 0:
        # Create remaining number of samples from normal distribution
        pt = multivariate_normal.rvs(mean=c.flatten(), cov=Sigma, size=remaining_samples)
        if pt.ndim == 1:
            pt = pt.reshape(-1, 1)
        else:
            pt = pt.T
        
        # Check containment
        pt_inside = contains(S, pt)
        if isinstance(pt_inside, bool):
            pt_inside = [pt_inside]
        
        n_inside = np.sum(pt_inside)
        remaining_samples -= n_inside
        
        # Store the ones that are contained
        if n_inside > 0:
            x[:, idx:idx+n_inside] = pt[:, pt_inside]
            idx += n_inside
    
    return x 