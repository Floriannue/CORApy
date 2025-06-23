"""
enclosePoints - enclose a point cloud with an interval

Syntax:
    I = Interval.enclosePoints(points)

Inputs:
    points - matrix storing point cloud (dimension: [n,p] for p points)

Outputs:
    I - Interval object

Example: 
    p = -1 + 2*np.random.rand(2,10)
    I = Interval.enclosePoints(p)
    
    # figure; hold on;
    # I.plot()
    # plt.plot(p[0,:],p[1,:],'k.')

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
"""

import numpy as np

from .interval import Interval

def enclosePoints(points: np.ndarray) -> 'Interval':
    """
    enclosePoints - enclose a point cloud with an interval
    
    Args:
        points: matrix storing point cloud (dimension: [n,p] for p points)
        
    Returns:
        Interval object that encloses all the points
    """
    # Convert to numpy array if needed
    points = np.array(points, dtype=float)
    
    # Handle edge cases
    if points.size == 0:
        raise ValueError("Empty point cloud cannot be enclosed")
    
    # If points is 1D, treat as 1 dimension with multiple points
    if points.ndim == 1:
        # This is 1 dimension with multiple points
        min_vals = np.array([np.min(points)])
        max_vals = np.array([np.max(points)])
    else:
        # Multiple dimensions, compute min and max along axis=1 (for each dimension)
        min_vals = np.min(points, axis=1)
        max_vals = np.max(points, axis=1)
    
    # Create interval from min and max bounds
    return Interval(min_vals, max_vals) 