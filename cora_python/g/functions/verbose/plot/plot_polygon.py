"""
plot_polygon - plot a polygon defined by its vertices

This function mimics MATLAB's plotPolygon functionality for plotting polygons
using matplotlib.

Syntax:
    handle = plot_polygon(V, **kwargs)

Inputs:
    V - matrix storing the polygon vertices (n x m) where n is dimension
    kwargs - plot settings as keyword arguments

Outputs:
    handle - matplotlib graphics object handle

Authors: Niklas Kochdumper, Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 2020 (MATLAB)
Python translation: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from typing import Dict, Any, Optional, Union
from .read_plot_options import read_plot_options


def plot_polygon(V: np.ndarray, *args, **kwargs) -> Any:
    """
    Plot a polygon defined by its vertices
    
    Args:
        V: Vertex matrix (n x m) where n is dimension, m is number of vertices
        *args: Additional positional arguments (for linespec compatibility)
        **kwargs: Keyword arguments for plotting options
        
    Returns:
        Matplotlib graphics object handle
    """
    # Convert args to list for processing
    args_list = list(args)
    
    # Process plotting options
    plot_options = read_plot_options(args_list)
    plot_options.update(kwargs)
    
    # Extract special options
    conv_hull = plot_options.pop('ConvHull', False)
    close_regions = plot_options.pop('CloseRegions', False)
    plot_background = plot_options.pop('PlotBackground', False)
    
    # Handle positioning (XPos, YPos, ZPos)
    V = _position_at_xyz(V, plot_options)
    
    # Cut infinity values at reasonable limits
    V = _cut_infinity_at_limits(V)
    
    # Check if we need face color (filled plot)
    has_face_color = 'facecolor' in plot_options and plot_options['facecolor'] != 'none'
    
    # Get current axes
    ax = plt.gca()
    
    # Plot based on data
    if V.size == 0:
        # Plot empty set
        handle = _plot_empty(has_face_color, plot_options)
        
    elif V.shape[1] == 1:
        # Single point
        handle = _plot_single_point(V, plot_options, has_face_color)
        
    elif V.shape[0] == 2:
        # 2D polygon
        handle = _plot_2d(V, plot_options, conv_hull, has_face_color, close_regions)
        
    elif V.shape[0] == 3:
        # 3D polygon
        handle = _plot_3d(V, plot_options, conv_hull, has_face_color)
        
    else:
        raise ValueError(f"Cannot plot {V.shape[0]}-dimensional vertices directly")
    
    # Move to background if requested
    if plot_background and hasattr(handle, 'set_zorder'):
        handle.set_zorder(-1)
    
    return handle


def _position_at_xyz(V: np.ndarray, plot_options: Dict[str, Any]) -> np.ndarray:
    """Position vertices at specified X/Y/Z positions"""
    x_pos = plot_options.pop('XPos', None)
    y_pos = plot_options.pop('YPos', None) 
    z_pos = plot_options.pop('ZPos', None)
    
    dims, n_points = V.shape
    
    # Add dimensions as needed for positioning
    if dims == 1 and z_pos is not None:
        if x_pos is None and y_pos is None:
            y_pos = 0
    if dims == 1 and x_pos is None and y_pos is None:
        y_pos = 0
    
    # Build new vertex matrix
    new_V = []
    
    if x_pos is not None:
        new_V.append(np.full(n_points, x_pos))
    if dims >= 1:
        new_V.append(V[0, :])
    
    if y_pos is not None:
        if len(new_V) == 1:  # Only x_pos was added
            new_V.append(np.full(n_points, y_pos))
        else:
            new_V.insert(-1, np.full(n_points, y_pos))
    if dims >= 2:
        new_V.append(V[1, :])
    
    if z_pos is not None:
        if dims <= 2:
            new_V.append(np.full(n_points, z_pos))
        else:
            new_V.append(V[2, :])
    if dims >= 3:
        new_V.append(V[2, :])
    
    return np.array(new_V) if new_V else V


def _cut_infinity_at_limits(V: np.ndarray) -> np.ndarray:
    """Replace infinity values with reasonable limits"""
    V_copy = V.copy()
    
    # Get current axis limits
    ax = plt.gca()
    
    if V.shape[0] >= 1:
        xlim = ax.get_xlim() if hasattr(ax, 'get_xlim') else [-10, 10]
        inf_mask = np.isinf(V_copy[0, :])
        V_copy[0, np.logical_and(inf_mask, V_copy[0, :] > 0)] = xlim[1] * 2
        V_copy[0, np.logical_and(inf_mask, V_copy[0, :] < 0)] = xlim[0] * 2
        
    if V.shape[0] >= 2:
        ylim = ax.get_ylim() if hasattr(ax, 'get_ylim') else [-10, 10]
        inf_mask = np.isinf(V_copy[1, :])
        V_copy[1, np.logical_and(inf_mask, V_copy[1, :] > 0)] = ylim[1] * 2
        V_copy[1, np.logical_and(inf_mask, V_copy[1, :] < 0)] = ylim[0] * 2
        
    if V.shape[0] >= 3:
        zlim = ax.get_zlim() if hasattr(ax, 'get_zlim') else [-10, 10]
        inf_mask = np.isinf(V_copy[2, :])
        V_copy[2, np.logical_and(inf_mask, V_copy[2, :] > 0)] = zlim[1] * 2
        V_copy[2, np.logical_and(inf_mask, V_copy[2, :] < 0)] = zlim[0] * 2
    
    return V_copy


def _plot_empty(has_face_color: bool, plot_options: Dict[str, Any]) -> Any:
    """Plot empty set (NaN values for visibility in legend)"""
    if np.any(has_face_color):
        return plt.fill([np.nan], [np.nan], **plot_options)
    else:
        return plt.plot([np.nan], [np.nan], **plot_options)[0]


def _plot_single_point(V: np.ndarray, plot_options: Dict[str, Any], has_face_color: bool) -> Any:
    """Plot single point"""
    if V.shape[0] == 2:
        x, y = V[0, 0], V[1, 0]
        if has_face_color:
            return plt.scatter([x], [y], **plot_options)
        else:
            plot_options['marker'] = plot_options.get('marker', 'o')
            return plt.plot([x], [y], **plot_options)[0]
    elif V.shape[0] == 3:
        ax = plt.gca(projection='3d')
        x, y, z = V[0, 0], V[1, 0], V[2, 0]
        if has_face_color:
            return ax.scatter([x], [y], [z], **plot_options)
        else:
            plot_options['marker'] = plot_options.get('marker', 'o')
            return ax.plot([x], [y], [z], **plot_options)[0]


def _plot_2d(V: np.ndarray, plot_options: Dict[str, Any], conv_hull: bool, 
             has_face_color: bool, close_regions: bool) -> Any:
    """Plot 2D polygon"""
    x_coords = V[0, :]
    y_coords = V[1, :]
    
    # Compute convex hull if requested
    if conv_hull and len(x_coords) > 2:
        try:
            points = np.column_stack((x_coords, y_coords))
            hull = ConvexHull(points)
            hull_indices = hull.vertices
            x_coords = x_coords[hull_indices]
            y_coords = y_coords[hull_indices]
        except:
            pass  # Use original points if convex hull fails
    
    # Close polygon if needed
    if close_regions and len(x_coords) > 1:
        if x_coords[0] != x_coords[-1] or y_coords[0] != y_coords[-1]:
            x_coords = np.append(x_coords, x_coords[0])
            y_coords = np.append(y_coords, y_coords[0])
    
    # Plot
    if np.any(has_face_color):
        # Use polygon patch for filled plotting
        vertices = np.column_stack((x_coords, y_coords))
        polygon = Polygon(vertices, **plot_options)
        ax = plt.gca()
        ax.add_patch(polygon)
        return polygon
    else:
        # Line plot
        return plt.plot(x_coords, y_coords, **plot_options)[0]


def _plot_3d(V: np.ndarray, plot_options: Dict[str, Any], conv_hull: bool, has_face_color: bool) -> Any:
    """Plot 3D polygon"""
    from mpl_toolkits.mplot3d import Axes3D
    
    # Ensure we have 3D axes
    ax = plt.gca()
    if not hasattr(ax, 'zaxis'):
        ax = plt.figure().add_subplot(111, projection='3d')
    
    x_coords = V[0, :]
    y_coords = V[1, :]
    z_coords = V[2, :]
    
    if has_face_color:
        # For 3D filled plots, we need to compute triangulation
        try:
            points = np.column_stack((x_coords, y_coords, z_coords))
            hull = ConvexHull(points)
            
            # Plot each face of the convex hull
            for simplex in hull.simplices:
                triangle = points[simplex]
                ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], **plot_options)
            
            return ax.collections[-1] if ax.collections else None
        except:
            # Fallback to line plot
            return ax.plot(x_coords, y_coords, z_coords, **plot_options)[0]
    else:
        # Line plot in 3D
        return ax.plot(x_coords, y_coords, z_coords, **plot_options)[0] 