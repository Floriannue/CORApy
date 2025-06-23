"""
randPoint_ - computes random point in interval

This function generates random points within an interval using various
sampling methods including standard and extreme point sampling.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 17-September-2019 (MATLAB)
Last update: 27-March-2023 (MATLAB)
Python translation: 2025
"""

from typing import Union, TYPE_CHECKING
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .interval import Interval

def randPoint_(I: 'Interval', N: Union[int, str] = 1, type_: str = 'standard') -> np.ndarray:
    """
    Computes random point in interval
    
    Args:
        I: Interval object
        N: Number of random points or 'all' for extreme points
        type_: Type of random point ('standard', 'extreme', 'uniform')
        
    Returns:
        np.ndarray: Random points (each column is a point)
        
    Raises:
        CORAerror: If interval matrix is provided
        
    Example:
        >>> I = Interval([-2, 1], [3, 2])
        >>> p = randPoint_(I, 20)
    """
    # Get object properties
    c = I.center()
    r = I.rad()
    n = I.dim()
    
    # Handle empty interval
    if I.isemptyobject():
        if isinstance(N, str):
            N = 0
        elif N == 1:
            N = 0  # For empty intervals, return empty array even if N=1
        return np.zeros((n, N))
    
    # Check for interval matrix (not supported)
    if r.ndim > 1 and r.shape[0] > 1 and r.shape[1] > 1:
        raise CORAerror('CORA:wrongInputInConstructor',
                       'interval/randPoint not defined for interval matrices!')
    
    # Generate different types of points
    if type_ == 'standard' or type_.startswith('uniform'):
        
        if r.ndim > 1 and r.shape[1] > 1:
            if r.shape[0] > 1:
                # Both dimensions larger than 1 -> interval matrix
                raise CORAerror('CORA:wrongInputInConstructor',
                               'interval/randPoint not defined for interval matrices!')
            else:
                # Row interval
                p = c + (-1 + 2 * np.random.rand(N, len(r))) * r
        else:
            # Column interval
            if isinstance(N, str):
                N = 1
            p = c.reshape(-1, 1) + (-1 + 2 * np.random.rand(len(r), N)) * r.reshape(-1, 1)
            
    elif type_ == 'extreme':
        
        # Consider degenerate case
        ind = np.where(r > 0)[0]
        if len(ind) < n:
            I_proj = I.project(ind)
            temp = randPoint_(I_proj, N, type_)
            if isinstance(N, str):
                N = temp.shape[1]
            p = np.tile(c, (1, N))
            p[ind, :] = temp
            return p
        
        # Return all extreme points
        if isinstance(N, str) and N == 'all':
            p = I.vertices_()
            
        elif isinstance(N, int):
            if 10 * N < 2**n:
                # Generate random vertices
                p = np.zeros((n, N))
                cnt = 0
                generated_points = set()
                
                while cnt < N:
                    temp = np.sign(-1 + 2 * np.random.rand(n))
                    temp_tuple = tuple(temp)
                    if temp_tuple not in generated_points:
                        generated_points.add(temp_tuple)
                        p[:, cnt] = temp
                        cnt += 1
                
                p = c.reshape(-1, 1) + p * r.reshape(-1, 1)
                
            elif N <= 2**n:
                # Select random vertices
                V = I.vertices_()
                ind = np.random.permutation(V.shape[1])
                V = V[:, ind]
                p = V[:, :N]
                
            else:
                # Compute vertices and additional points on the boundary
                V = I.vertices_()
                p = np.zeros((n, N))
                p[:, :V.shape[1]] = V
                
                for i in range(V.shape[1], N):
                    temp = np.sign(-1 + 2 * np.random.rand(n))
                    ind_rand = np.random.randint(0, n)
                    temp[ind_rand] = -1 + 2 * np.random.rand()
                    p[:, i] = c.flatten() + temp * r.flatten()
        else:
            raise ValueError("N must be an integer or 'all'")
    
    else:
        raise ValueError(f"Unknown type: {type_}")
    
    return p 