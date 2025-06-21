"""
plot - plots a projection of the reachable set

Syntax:
    han = plot(R)
    han = plot(R, dims)
    han = plot(R, dims, **kwargs)

Inputs:
    R - reachSet object
    dims - (optional) dimensions for projection (default: [0, 1])
    **kwargs - (optional) plot settings including:
        'Set' - which set to plot ('ti', 'tp', 'y')
        'Unify' - whether to unify sets (default: False)
        Other matplotlib plotting options

Outputs:
    han - handle to the graphics object

Authors: Niklas Kochdumper, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 02-June-2020 (MATLAB)
Last update: 17-December-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
from typing import List, Optional, Any, Union
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
import warnings


def plot(R, dims: Optional[List[int]] = None, **kwargs) -> Any:
    """
    Plot a projection of the reachable set
    
    Args:
        R: reachSet object
        dims: Dimensions for projection (default: [0, 1])
        **kwargs: Plot settings including:
            'Set': which set to plot ('ti', 'tp', 'y') 
            'Unify': whether to unify sets (default: False)
            'FaceColor': face color for filled polygons
            'EdgeColor': edge color for polygon boundaries
            'alpha': transparency level
            'label': legend label
        
    Returns:
        Handle to the graphics object
    """
    # Set default dimensions
    if dims is None:
        dims = [0, 1]
    
    # Validate inputs
    if len(dims) < 2:
        raise ValueError("At least 2 dimensions required for plotting")
    elif len(dims) > 3:
        raise ValueError("At most 3 dimensions supported for plotting")
    
    # Apply reachSet specific plot options like MATLAB does:
    # NVpairs = readPlotOptions(varargin(2:end),'reachSet');
    try:
        from ...functions.verbose.plot.read_plot_options import read_plot_options
        # Convert kwargs to list format like MATLAB varargin
        plot_options_list = []
        for k, v in kwargs.items():
            plot_options_list.extend([k, v])
        plot_options = read_plot_options(plot_options_list, 'reachSet')
        
        # Extract MATLAB-style options from the processed options
        whichset = plot_options.pop('Set', 'ti')  # 'ti', 'tp', or 'y'
        unify = plot_options.pop('Unify', plot_options.pop('unify', False))  # Check both cases
        face_color = plot_options.pop('facecolor', plot_options.pop('FaceColor', None))
        edge_color = plot_options.pop('edgecolor', plot_options.pop('EdgeColor', None))
        alpha = plot_options.pop('alpha', 0.7)
        label = plot_options.pop('label', plot_options.pop('DisplayName', None))
        
        # Use processed options
        kwargs = plot_options
        
    except ImportError:
        # Fallback: manual processing
        whichset = kwargs.pop('Set', 'ti')  # 'ti', 'tp', or 'y'
        unify = kwargs.pop('Unify', False)
        face_color = kwargs.pop('FaceColor', kwargs.pop('facecolor', None))
        edge_color = kwargs.pop('EdgeColor', kwargs.pop('edgecolor', None))
        alpha = kwargs.pop('alpha', 0.7)
        label = kwargs.pop('label', kwargs.pop('DisplayName', None))
        
        # Set default colors for reachable sets using CORA colors
        if face_color is None:
            try:
                from ...functions.verbose.plot.color.cora_color import cora_color
                face_color = cora_color('CORA:reachSet')
            except ImportError:
                face_color = 'blue'  # Fallback
        if edge_color is None:
            edge_color = face_color  # Use same color for edges
    
    # Set default z-order for reachable sets (lower than initial sets)
    if 'zorder' not in kwargs:
        kwargs['zorder'] = 1  # Low z-order so initial sets can be on top
    
    # Check if reachSet is empty
    from .isemptyobject import isemptyobject
    if isemptyobject(R):
        # Plot empty set
        if len(dims) == 2:
            return plt.plot([], [], **kwargs)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            return ax.plot([], [], [], **kwargs)
    
    # Get the sets to plot based on whichset parameter
    sets_to_plot = _get_sets_to_plot(R, whichset)
    

    
    if not sets_to_plot:
        # Plot empty set

        if len(dims) == 2:
            return plt.plot([], [], **kwargs)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            return ax.plot([], [], [], **kwargs)
    
    # If Unify is True, create unified polygon
    if unify:
        return _plot_unified(sets_to_plot, dims, face_color, edge_color, alpha, label, **kwargs)
    else:
        return _plot_individual(sets_to_plot, dims, face_color, edge_color, alpha, label, **kwargs)


def _get_sets_to_plot(R, whichset: str) -> List:
    """Get the appropriate sets to plot from the reachSet object"""
    try:
        if whichset == 'ti':
            if hasattr(R, 'timeInterval') and hasattr(R.timeInterval, 'set'):
                return R.timeInterval.set
            elif hasattr(R, 'timeInterval') and isinstance(R.timeInterval, dict) and 'set' in R.timeInterval:
                return R.timeInterval['set']
        elif whichset == 'tp':
            if hasattr(R, 'timePoint') and hasattr(R.timePoint, 'set'):
                return R.timePoint.set
            elif hasattr(R, 'timePoint') and isinstance(R.timePoint, dict) and 'set' in R.timePoint:
                return R.timePoint['set']
        elif whichset == 'y':
            if hasattr(R, 'timeInterval') and hasattr(R.timeInterval, 'algebraic'):
                return R.timeInterval.algebraic
            elif hasattr(R, 'timeInterval') and isinstance(R.timeInterval, dict) and 'algebraic' in R.timeInterval:
                return R.timeInterval['algebraic']
        
        # Fallback: try to get any available sets
        if hasattr(R, 'timeInterval'):
            if hasattr(R.timeInterval, 'set'):
                return R.timeInterval.set
            elif isinstance(R.timeInterval, dict) and 'set' in R.timeInterval:
                return R.timeInterval['set']
        
        if hasattr(R, 'timePoint'):
            if hasattr(R.timePoint, 'set'):
                return R.timePoint.set
            elif isinstance(R.timePoint, dict) and 'set' in R.timePoint:
                return R.timePoint['set']
        
        return []
    except:
        return []


def _plot_unified(sets_to_plot: List, dims: List[int], face_color: str, edge_color: str, 
                 alpha: float, label: str, **kwargs) -> Any:
    """Plot unified reachable sets using polygon union"""
    

    
    # Collect all polygons
    polygons = []
    
    for i, s in enumerate(sets_to_plot):
        try:
            # Project set to desired dimensions
            if hasattr(s, 'project'):
                s_proj = s.project(dims)
            else:
                s_proj = s
            
            # Get vertices of the projected set
            if hasattr(s_proj, 'vertices'):
                vertices = s_proj.vertices()
            elif hasattr(s_proj, 'polytope'):
                # For zonotopes, convert to polytope first
                poly = s_proj.polytope()
                if hasattr(poly, 'vertices'):
                    vertices = poly.vertices()
                else:
                    continue
            elif hasattr(s_proj, 'c') and hasattr(s_proj, 'G'):
                # Direct zonotope vertices computation
                vertices = _zonotope_vertices(s_proj, dims)
            else:
                continue
            
            if vertices is not None and vertices.size > 0:
                # Ensure vertices is 2D and in correct format
                if vertices.ndim == 1:
                    vertices = vertices.reshape(-1, 1)
                
                # Convert to (N, 2) format for shapely (transpose if needed)
                if vertices.shape[0] == 2 and vertices.shape[1] > vertices.shape[0]:
                    vertices = vertices.T  # Transpose from (2, N) to (N, 2)
                elif vertices.shape[1] == 2:
                    pass  # Already in (N, 2) format
                else:
                    continue  # Skip invalid vertex arrays
                
                # Create shapely polygon for union operations
                if vertices.shape[0] >= 3:  # Need at least 3 points for a polygon
                    try:
                        # Compute convex hull to ensure proper vertex ordering
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(vertices)
                        hull_vertices = vertices[hull.vertices]
                        
                        poly = ShapelyPolygon(hull_vertices)
                        if poly.is_valid:
                            polygons.append(poly)
                    except Exception:
                        continue
                        
        except Exception as e:
            # Skip problematic sets
            continue
    
    if not polygons:
        # No valid polygons found, fallback to individual plotting
        return _plot_individual(sets_to_plot, dims, face_color, edge_color, alpha, label, **kwargs)
    
    # Filter out empty or invalid polygons
    valid_polygons = [p for p in polygons if p.is_valid and not p.is_empty and p.area > 1e-12]
    
    if not valid_polygons:
        # No valid polygons after filtering, fallback to individual plotting
        return _plot_individual(sets_to_plot, dims, face_color, edge_color, alpha, label, **kwargs)
    
    try:
        # Compute union of all polygons
        if len(valid_polygons) == 1:
            unified_poly = valid_polygons[0]
        else:
            unified_poly = unary_union(valid_polygons)
        
        # Plot the unified polygon
        ax = plt.gca()
        
        # Extract z-order from kwargs (default to 1 for reachable sets)
        zorder = kwargs.pop('zorder', 1)
        
        if hasattr(unified_poly, 'geoms'):
            # MultiPolygon result
            patches = []
            for geom in unified_poly.geoms:
                if hasattr(geom, 'exterior'):
                    coords = np.array(geom.exterior.coords)
                    patches.append(Polygon(coords, closed=True))
            
            collection = PolyCollection(patches, facecolors=face_color, 
                                      edgecolors=edge_color, alpha=alpha, zorder=zorder, **kwargs)
            # Set label after creation for proper legend registration
            if label is not None:
                collection.set_label(label)
            handle = ax.add_collection(collection)
        else:
            # Single Polygon result
            if hasattr(unified_poly, 'exterior'):
                coords = np.array(unified_poly.exterior.coords)
                patch = Polygon(coords, closed=True, facecolor=face_color, 
                              edgecolor=edge_color, alpha=alpha, zorder=zorder, **kwargs)
                # Set label after creation for proper legend registration
                if label is not None:
                    patch.set_label(label)
                handle = ax.add_patch(patch)
            else:
                # Fallback
                return _plot_individual(sets_to_plot, dims, face_color, edge_color, alpha, label, **kwargs)
        
        # Update axis limits
        ax.relim()
        ax.autoscale_view()
        
        return handle
        
    except Exception as e:

        warnings.warn(f"Unify failed: {e}. Falling back to individual plotting.")
        return _plot_individual(sets_to_plot, dims, face_color, edge_color, alpha, label, **kwargs)


def _plot_individual(sets_to_plot: List, dims: List[int], face_color: str, edge_color: str,
                    alpha: float, label: str, **kwargs) -> Any:
    """Plot individual reachable sets"""
    
    handles = []
    
    for i, s in enumerate(sets_to_plot):
        try:
            # Project set to desired dimensions
            if hasattr(s, 'project'):
                s_proj = s.project(dims)
            else:
                s_proj = s
            
            # Plot the projected set
            current_kwargs = kwargs.copy()
            current_kwargs['FaceColor'] = face_color
            current_kwargs['EdgeColor'] = edge_color  
            current_kwargs['alpha'] = alpha
            
            # Only set label for the first set
            if i == 0 and label is not None:
                current_kwargs['label'] = label
                current_kwargs['DisplayName'] = label  # Also set MATLAB-style DisplayName
            
            if hasattr(s_proj, 'plot'):
                # Set has its own plot method
                handle = s_proj.plot(dims, **current_kwargs)
                handles.append(handle)
            elif isinstance(s_proj, np.ndarray):
                # Plot as points or lines
                if len(dims) == 2:
                    if s_proj.ndim == 1:
                        handle = plt.plot(s_proj[0], s_proj[1], 'o', **current_kwargs)
                    else:
                        handle = plt.plot(s_proj[0, :], s_proj[1, :], **current_kwargs)
                    handles.append(handle)
        except Exception as e:
            # Skip problematic sets
            continue
    
    # Return the first handle or all handles
    if len(handles) == 1:
        return handles[0]
    elif len(handles) > 1:
        return handles
    else:
        return None


def _zonotope_vertices(zonotope, dims: List[int]) -> np.ndarray:
    """Compute vertices of a zonotope projection efficiently"""
    try:
        # Project center and generators
        c = zonotope.c[dims]
        G = zonotope.G[dims, :]
        
        if len(dims) == 2 and G.shape[1] > 0:
            # For efficiency, limit the number of generators to avoid exponential explosion
            max_generators = min(G.shape[1], 15)  # Limit to 2^15 = 32k vertices max
            
            if G.shape[1] > max_generators:
                # Use only the most significant generators (largest magnitude)
                gen_magnitudes = np.linalg.norm(G, axis=0)
                top_indices = np.argsort(gen_magnitudes)[-max_generators:]
                G = G[:, top_indices]
            
            # Get all possible combinations of +/- for generators
            n_gen = G.shape[1]
            combinations = np.array(np.meshgrid(*[[-1, 1] for _ in range(n_gen)])).T.reshape(-1, n_gen)
            
            # Compute all vertices
            vertices = c.reshape(1, -1) + (combinations @ G.T)
            
            # Compute convex hull to get boundary vertices in correct order
            try:
                from scipy.spatial import ConvexHull
                if len(vertices) > 3:
                    hull = ConvexHull(vertices)
                    hull_vertices = vertices[hull.vertices]
                    # Sort vertices in counter-clockwise order
                    center_vertex = np.mean(hull_vertices, axis=0)
                    angles = np.arctan2(hull_vertices[:, 1] - center_vertex[1], 
                                      hull_vertices[:, 0] - center_vertex[0])
                    sorted_indices = np.argsort(angles)
                    return hull_vertices[sorted_indices]
                else:
                    return vertices
            except Exception as e:
                # Fallback: return all vertices
                return vertices
        else:
            # For 1D or higher dimensions, return interval bounds
            if G.shape[1] > 0:
                # Compute interval bounds
                radius = np.sum(np.abs(G), axis=1)
                min_bounds = c - radius
                max_bounds = c + radius
                
                if len(dims) == 1:
                    # 1D case: return min and max points
                    return np.array([[min_bounds[0]], [max_bounds[0]]])
                else:
                    # Multi-D case: return box corners
                    corners = np.array([[min_bounds[0], min_bounds[1]],
                                      [max_bounds[0], min_bounds[1]], 
                                      [max_bounds[0], max_bounds[1]],
                                      [min_bounds[0], max_bounds[1]]])
                    return corners
            else:
                # No generators: just return center
                return c.reshape(1, -1)
            
    except Exception as e:
        # Return empty array on error
        return np.array([]) 