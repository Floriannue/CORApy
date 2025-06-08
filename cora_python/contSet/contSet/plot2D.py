"""
plot2D - plots a 2D projection of a contSet

This function visualizes a 2D projection of a continuous set by computing
vertices and plotting them as a polygon.

Syntax:
    handle = plot2D(S)
    handle = plot2D(S, plot_kwargs)
    handle = plot2D(S, plot_kwargs, nvpairs_polygon)

Inputs:
    S - projected contSet object (2D)
    plot_kwargs - (optional) plot settings as dictionary
    nvpairs_polygon - (optional) polygon computation settings

Outputs:
    handle - matplotlib graphics object handle

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 14-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Dict, Any, List, Optional
from cora_python.g import set_default_values, plot_polygon


def plot2D(S, plot_kwargs: Optional[Dict[str, Any]] = None, 
           nvpairs_polygon: Optional[List[Any]] = None):
    """
    Plot a 2D projection of a contSet
    
    Args:
        S: Projected contSet object (2D)
        plot_kwargs: Plot settings as dictionary
        nvpairs_polygon: Polygon computation settings
        
    Returns:
        Matplotlib graphics object handle
    """
    # Set default values
    if plot_kwargs is None:
        plot_kwargs = {}
    if nvpairs_polygon is None:
        nvpairs_polygon = []
    
    # Check if set is bounded or is an interval
    is_unbounded = False
    is_interval = False
    
    # Check if it's an interval first
    if hasattr(S, '__class__') and 'interval' in S.__class__.__name__.lower():
        is_interval = True
    
    # Check if set is unbounded
    if hasattr(S, 'is_bounded') and not is_interval:
        is_unbounded = not S.is_bounded()
    
    if is_unbounded or is_interval:
        # Compute vertices directly for unbounded sets or intervals
        # This provides faster computation for degenerate intervals
        if hasattr(S, 'vertices_'):
            V = S.vertices_()
        elif hasattr(S, 'vertices'):
            V = S.vertices()
        else:
            raise ValueError("Cannot extract vertices from set for plotting")
    else:
        # Compute vertices via polygon for better approximation
        # This obtains tighter results for some set representations due to splits
        # or provides results for sets where direct vertex computation is not feasible
        
        if hasattr(S, 'polygon'):
            # Convert to polygon first
            pgon = S.polygon(*nvpairs_polygon)
            
            # Read vertices from polygon
            if hasattr(pgon, 'vertices_'):
                V = pgon.vertices_()
            elif hasattr(pgon, 'vertices'):
                V = pgon.vertices()
            else:
                raise ValueError("Cannot extract vertices from polygon")
        else:
            # Fallback to direct vertex computation
            if hasattr(S, 'vertices'):
                V = S.vertices(*nvpairs_polygon)
            elif hasattr(S, 'vertices_'):
                V = S.vertices_()
            else:
                raise ValueError("Cannot extract vertices from set for plotting")
    
    # Close the polygon if needed
    if V.shape[1] > 1 and not np.any(np.isnan(V)):
        # Check if polygon is already closed
        if not (np.isclose(V[0, 0], V[0, -1]) and np.isclose(V[1, 0], V[1, -1])):
            # Add first vertex at the end to close the polygon
            V = np.column_stack([V, V[:, 0]])
    else:
        # For multiple regions or NaN values, let plot_polygon handle closing
        plot_kwargs['CloseRegions'] = True
    
    # Plot vertices using polygon plotting
    handle = plot_polygon(V, **plot_kwargs)
    
    return handle 