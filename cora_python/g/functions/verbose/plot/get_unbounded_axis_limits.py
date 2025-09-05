"""
get_unbounded_axis_limits - get axis limits for unbounded sets

This function provides reasonable axis limits for plotting unbounded sets,
similar to MATLAB's getUnboundedAxisLimits functionality.

Syntax:
    xlim, ylim = get_unbounded_axis_limits(V)
    xlim, ylim, zlim = get_unbounded_axis_limits(V)

Inputs:
    V - vertex matrix (n x m) where n is dimension

Outputs:
    xlim - x-axis limits [xmin, xmax]
    ylim - y-axis limits [ymin, ymax]  
    zlim - z-axis limits [zmin, zmax] (for 3D)

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union


def get_unbounded_axis_limits(V: np.ndarray = None) -> Union[Tuple[np.ndarray, np.ndarray], 
                                                    Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Get reasonable axis limits for plotting unbounded sets
    
    Args:
        V: Vertex matrix (n x m) where n is dimension
        
    Returns:
        Axis limits for plotting
    """
    # Allow being called without V (tests call with missing arg). When V is None,
    # fall back to current axis limits; if none, use default ranges.
    if V is None:
        ax = plt.gca()
        if hasattr(ax, 'get_zlim'):
            return np.array(ax.get_xlim()), np.array(ax.get_ylim()), np.array(ax.get_zlim())
        else:
            return np.array(ax.get_xlim()), np.array(ax.get_ylim())

    n, m = V.shape
    
    # Get current axis limits if they exist
    ax = plt.gca()
    
    if n == 1:
        # 1D case - extend to 2D for plotting
        if hasattr(ax, 'get_xlim'):
            xlim = np.array(ax.get_xlim())
            ylim = np.array([-1, 1])  # Default y range for 1D plots
        else:
            # No current plot
            xlim = _get_limits_from_vertices(V[0, :])
            ylim = np.array([-1, 1])
        return xlim, ylim
        
    elif n == 2:
        # 2D case
        if hasattr(ax, 'get_xlim') and hasattr(ax, 'get_ylim'):
            xlim = np.array(ax.get_xlim())
            ylim = np.array(ax.get_ylim())
        else:
            xlim = _get_limits_from_vertices(V[0, :])
            ylim = _get_limits_from_vertices(V[1, :])
        return xlim, ylim
        
    elif n == 3:
        # 3D case
        if hasattr(ax, 'get_xlim') and hasattr(ax, 'get_ylim') and hasattr(ax, 'get_zlim'):
            xlim = np.array(ax.get_xlim())
            ylim = np.array(ax.get_ylim()) 
            zlim = np.array(ax.get_zlim())
        else:
            xlim = _get_limits_from_vertices(V[0, :])
            ylim = _get_limits_from_vertices(V[1, :])
            zlim = _get_limits_from_vertices(V[2, :])
        return xlim, ylim, zlim
    
    else:
        raise ValueError(f"Cannot handle {n}-dimensional plotting")


def _get_limits_from_vertices(coords: np.ndarray) -> np.ndarray:
    """
    Get axis limits from coordinate array
    
    Args:
        coords: 1D array of coordinates
        
    Returns:
        Limits [min, max]
    """
    # Filter out infinite values
    finite_coords = coords[np.isfinite(coords)]
    
    if len(finite_coords) == 0:
        # All infinite - use default range
        return np.array([-10, 10])
    
    # Get range of finite coordinates
    coord_min = np.min(finite_coords)
    coord_max = np.max(finite_coords)
    
    if coord_min == coord_max:
        # Single point - add some padding
        center = coord_min
        padding = 1.0 if center == 0 else abs(center) * 0.1
        return np.array([center - padding, center + padding])
    
    # Add 10% padding to range
    range_size = coord_max - coord_min
    padding = range_size * 0.1
    
    return np.array([coord_min - padding, coord_max + padding]) 