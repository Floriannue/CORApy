"""
plot3D - plots a 3D projection of a contSet

This function visualizes a 3D projection of a continuous set by computing
vertices and plotting them as a 3D polytope.

Syntax:
    handle = plot3D(S)
    handle = plot3D(S, plot_kwargs)
    handle = plot3D(S, plot_kwargs, nvpairs_vertices)

Inputs:
    S - projected contSet object (3D)
    plot_kwargs - (optional) plot settings as dictionary
    nvpairs_vertices - (optional) vertex computation settings

Outputs:
    handle - matplotlib graphics object handle

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 14-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Dict, Any, List, Optional
from cora_python.g.functions.matlab.validate.preprocessing.set_default_values import set_default_values
from cora_python.g.functions.verbose.plot import plot_polytope_3d


def plot3D(S, plot_kwargs: Optional[Dict[str, Any]] = None, 
           nvpairs_vertices: Optional[List[Any]] = None):
    """
    Plot a 3D projection of a contSet
    
    Args:
        S: Projected contSet object (3D)
        plot_kwargs: Plot settings as dictionary
        nvpairs_vertices: Vertex computation settings
        
    Returns:
        Matplotlib graphics object handle
    """
    # Set default values
    if plot_kwargs is None:
        plot_kwargs = {}
    if nvpairs_vertices is None:
        nvpairs_vertices = []
    
    # Compute vertices
    if hasattr(S, 'vertices'):
        V = S.vertices(*nvpairs_vertices)
    elif hasattr(S, 'vertices_'):
        V = S.vertices_()
    else:
        raise ValueError("Cannot extract vertices from set for 3D plotting")
    
    # Plot vertices using 3D polytope plotting
    handle = plot_polytope_3d(V, **plot_kwargs)
    
    return handle 