"""
plot1D - plots a 1D projection of a contSet

This function visualizes a 1D projection of a continuous set by converting
it to an interval and plotting its vertices.

Syntax:
    handle = plot1D(S)
    handle = plot1D(S, plot_kwargs)
    handle = plot1D(S, plot_kwargs, nvpairs_interval)

Inputs:
    S - projected contSet object (1D)
    plot_kwargs - (optional) plot settings as dictionary
    nvpairs_interval - (optional) interval hull computation settings

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


def plot1D(S, plot_kwargs: Optional[Dict[str, Any]] = None, 
           nvpairs_interval: Optional[List[Any]] = None):
    """
    Plot a 1D projection of a contSet
    
    Args:
        S: Projected contSet object (1D)
        plot_kwargs: Plot settings as dictionary
        nvpairs_interval: Interval hull computation settings
        
    Returns:
        Matplotlib graphics object handle
    """
    # Set default values
    if plot_kwargs is None:
        plot_kwargs = {}
    if nvpairs_interval is None:
        nvpairs_interval = []
    
    # Convert to interval
    if hasattr(S, 'interval'):
        I = S.interval(*nvpairs_interval)
    else:
        # Fallback: assume S is already an interval-like object
        I = S
    
    # Get vertices from interval
    if hasattr(I, 'vertices'):
        V = I.vertices()
    elif hasattr(I, 'vertices_'):
        V = I.vertices_()
    else:
        # Fallback: create vertices from bounds
        if hasattr(I, 'inf') and hasattr(I, 'sup'):
            # Create 1D interval vertices
            V = np.array([[I.inf[0], I.sup[0]]])
        else:
            raise ValueError("Cannot extract vertices from set for 1D plotting")
    
    # Plot using polygon plotting (which handles 1D as 2D)
    handle = plot_polygon(V, **plot_kwargs)
    
    return handle 