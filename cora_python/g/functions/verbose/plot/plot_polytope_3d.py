"""
plot_polytope_3d - plot a polytope defined by its vertices in 3D

This function mimics MATLAB's plotPolytope3D functionality for plotting 3D polytopes
using matplotlib.

Syntax:
    handle = plot_polytope_3d(V, **kwargs)

Inputs:
    V - matrix storing the polytope vertices (3 x m)
    kwargs - plot settings as keyword arguments

Outputs:
    handle - matplotlib graphics object handle

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 2020 (MATLAB)
Python translation: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from typing import Dict, Any, List, Optional
from .read_plot_options import read_plot_options
from .plot_polygon import plot_polygon


def plot_polytope_3d(V: np.ndarray, *args, **kwargs) -> Any:
    """
    Plot a polytope defined by its vertices in 3D
    
    Args:
        V: Vertex matrix (3 x m) where m is number of vertices
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
    
    # Check if vertex array is empty
    if V.size == 0:
        # Plot dummy set (NaN values for legend visibility)
        ax = _ensure_3d_axes()
        return ax.plot([np.nan], [np.nan], [np.nan], **plot_options)[0]
    
    # Transpose for consistency with scipy.spatial
    vertices = V.T  # Now shape (m, 3)
    
    # Check dimensionality
    if vertices.shape[1] != 3:
        # Not really 3D, try with plotPolygon
        try:
            return plot_polygon(V, *args, ConvHull=True, **kwargs)
        except Exception as e:
            raise ValueError(f"Cannot plot {vertices.shape[1]}D vertices as 3D polytope") from e
    
    # Compute convex hull
    try:
        hull = ConvexHull(vertices)
    except Exception as e:
        # Fallback for degenerate cases
        ax = _ensure_3d_axes()
        return ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], **plot_options)
    
    # Get face color information
    face_color = plot_options.get('facecolor', plot_options.get('color', 'blue'))
    has_face_color = face_color is not None
    
    # Check if face_color is 'none' (handle both string and array cases)
    if isinstance(face_color, str):
        has_face_color = has_face_color and face_color != 'none'
    elif isinstance(face_color, np.ndarray):
        # Array colors are always valid
        has_face_color = True
    
    # Ensure we have 3D axes
    ax = _ensure_3d_axes()
    
    if has_face_color:
        # Plot filled polytope using faces
        return _plot_filled_polytope(ax, vertices, hull, plot_options)
    else:
        # Plot wireframe
        return _plot_wireframe_polytope(ax, vertices, hull, plot_options)


def _ensure_3d_axes():
    """Ensure we have 3D axes"""
    ax = plt.gca()
    if not hasattr(ax, 'zaxis'):
        fig = plt.gcf()
        ax = fig.add_subplot(111, projection='3d')
    return ax


def _plot_filled_polytope(ax, vertices: np.ndarray, hull: ConvexHull, plot_options: Dict[str, Any]) -> Any:
    """Plot filled polytope using face patches"""
    
    # Extract face color and edge color
    face_color = plot_options.get('facecolor', plot_options.get('color', 'blue'))
    edge_color = plot_options.get('edgecolor', 'black')
    alpha = plot_options.get('alpha', 0.7)
    
    # Group faces by normal vector to merge coplanar faces
    faces = _group_coplanar_faces(vertices, hull)
    
    # Create 3D polygon collection
    poly3d = []
    for face_vertices in faces:
        poly3d.append(face_vertices)
    
    # Create the collection
    collection = Poly3DCollection(poly3d, 
                                 facecolors=face_color,
                                 edgecolors=edge_color,
                                 alpha=alpha)
    
    # Add other properties
    if 'linewidth' in plot_options:
        collection.set_linewidth(plot_options['linewidth'])
    if 'linestyle' in plot_options:
        collection.set_linestyle(plot_options['linestyle'])
    
    ax.add_collection3d(collection)
    
    # Update axis limits
    _update_3d_limits(ax, vertices)
    
    return collection


def _plot_wireframe_polytope(ax, vertices: np.ndarray, hull: ConvexHull, plot_options: Dict[str, Any]) -> Any:
    """Plot wireframe polytope using edges"""
    
    # Get unique edges from the hull
    edges = set()
    for simplex in hull.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                edges.add(edge)
    
    # Plot edges
    lines = []
    for edge in edges:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        line = ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], **plot_options)[0]
        lines.append(line)
        
        # Only set plot options for first line to avoid duplicate legend entries
        if len(lines) == 1:
            plot_options = {k: v for k, v in plot_options.items() if k != 'label'}
    
    # Update axis limits
    _update_3d_limits(ax, vertices)
    
    return lines[0] if lines else None


def _group_coplanar_faces(vertices: np.ndarray, hull: ConvexHull, tol: float = 1e-14) -> List[np.ndarray]:
    """Group coplanar faces together"""
    
    # Compute normal vectors for each face
    normals = []
    face_vertices = []
    
    for simplex in hull.simplices:
        # Get vertices of this face
        face_verts = vertices[simplex]
        face_vertices.append(face_verts)
        
        # Compute normal vector
        if len(face_verts) >= 3:
            v1 = face_verts[1] - face_verts[0]
            v2 = face_verts[2] - face_verts[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else normal
            
            # Ensure consistent orientation
            if np.sum(normal) < 0:
                normal = -normal
                
            normals.append(normal)
        else:
            normals.append(np.array([0, 0, 1]))  # Default normal
    
    # Group faces with similar normals
    normals = np.array(normals)
    grouped_faces = []
    used = set()
    
    for i, normal in enumerate(normals):
        if i in used:
            continue
            
        # Find all faces with similar normal
        group = [face_vertices[i]]
        used.add(i)
        
        for j in range(i + 1, len(normals)):
            if j not in used and np.allclose(normal, normals[j], atol=tol):
                group.append(face_vertices[j])
                used.add(j)
        
        # Merge coplanar faces if possible
        if len(group) == 1:
            grouped_faces.append(group[0])
        else:
            # For simplicity, just add all faces separately
            # In a more sophisticated implementation, we would merge coplanar faces
            for face in group:
                grouped_faces.append(face)
    
    return grouped_faces


def _update_3d_limits(ax, vertices: np.ndarray) -> None:
    """Update 3D axis limits to include all vertices"""
    if vertices.size > 0:
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        z_coords = vertices[:, 2]
        
        # Get current limits
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim() 
        z_lim = ax.get_zlim()
        
        # Expand limits if necessary
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        
        ax.set_xlim(min(x_lim[0], x_min), max(x_lim[1], x_max))
        ax.set_ylim(min(y_lim[0], y_min), max(y_lim[1], y_max))
        ax.set_zlim(min(z_lim[0], z_min), max(z_lim[1], z_max)) 