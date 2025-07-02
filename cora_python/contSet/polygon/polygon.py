"""
Polygon class

A polygon represents a 2D polytope with vertices.

Properties:
    set: underlying shape representation
    x: x-coordinates of vertices
    y: y-coordinates of vertices
    V: vertices matrix (2 × n)

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 30-October-2020 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, Optional, Any, Tuple, List
from ..contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.macros import CHECKS_ENABLED

# Import geometric computation capabilities
try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry import Point, MultiPolygon, LineString
    from shapely.ops import unary_union
    from shapely import convex_hull, buffer
    import shapely
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    ShapelyPolygon = None


class Polygon(ContSet):
    """
    Polygon class
    
    A polygon represents a 2D polytope with vertices.
    """
    
    TOL = 1e-12  # tolerance for polygon operations
    
    def __init__(self, *varargin, **kwargs):
        """
        Constructor for polygon objects
        
        Args:
            *varargin: Variable arguments
                     - polygon(): empty polygon
                     - polygon(V): vertices as 2×n matrix
                     - polygon(x, y): x and y coordinates
                     - polygon(other_polygon): copy constructor
        """
        # 1. parse input arguments
        V, x, y, other_polygon = _aux_parseInputArgs(*varargin, **kwargs)

        # 2. check input arguments
        _aux_checkInputArgs(V, x, y, other_polygon, len(varargin))

        # 3. compute object
        poly, dim, v_matrix = _aux_computeProperties(V, x, y, other_polygon, len(varargin))
        
        self.set = poly
        self.dim = dim
        self._V = v_matrix

        # 4. set precedence
        super().__init__()
        self.precedence = 10

    @property
    def V(self):
        """Get vertices of the polygon"""
        return self._V

    @property
    def x(self):
        """Get x coordinates of vertices"""
        if self.V.size > 0:
            return self.V[0, :]
        return np.array([])
    
    @property
    def y(self):
        """Get y coordinates of vertices"""
        if self.V.size > 0:
            return self.V[1, :]
        return np.array([])
    
    @property
    def nrOfRegions(self):
        """Get number of regions"""
        if SHAPELY_AVAILABLE and hasattr(self.set, 'geoms'):
            # MultiPolygon case
            return len(self.set.geoms)
        elif SHAPELY_AVAILABLE and isinstance(self.set, ShapelyPolygon):
            return 1 if not self.set.is_empty else 0
        else:
            return 1 if self.V.size > 0 else 0
    
    @property
    def nrOfHoles(self):
        """Get number of holes"""
        if SHAPELY_AVAILABLE and isinstance(self.set, ShapelyPolygon) and not self.set.is_empty:
            return len(self.set.interiors)
        else:
            return 0

    def __repr__(self):
        """String representation"""
        if self.V.size > 0:
            return f"Polygon(vertices={self.V.shape[1]})"
        else:
            return "Polygon(empty)" 


# Auxiliary functions -----------------------------------------------------

def _aux_parseInputArgs(*varargin, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional['Polygon']]:
    """Parse input arguments from user and assign to variables"""
    
    V = np.array([])
    x = np.array([])
    y = np.array([])
    other_polygon = None
    
    if not varargin:
        return V, x, y, other_polygon
        
    # copy constructor
    if isinstance(varargin[0], Polygon):
        other_polygon = varargin[0]
    # from vertices
    elif len(varargin) == 1:
        V = np.asarray(varargin[0])
    # from x and y coordinates
    elif len(varargin) == 2:
        x, y = np.asarray(varargin[0]), np.asarray(varargin[1])
        
    return V, x, y, other_polygon


def _aux_checkInputArgs(V: np.ndarray, x: np.ndarray, y: np.ndarray, other_polygon: Optional['Polygon'], n_in: int):
    """Check correctness of input arguments"""
    
    # only check if macro set to true
    if CHECKS_ENABLED:
        if n_in == 1 and not isinstance(other_polygon, Polygon):
            if V.ndim != 2 or V.shape[0] != 2:
                raise CORAerror('CORA:wrongInputInConstructor', 
                    'Input must be a 2xV matrix, where V is the number of vertices.')
        elif n_in == 2:
            if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
                raise CORAerror('CORA:wrongInputInConstructor', 
                    'Inputs x and y must be vectors of the same length.')

def _aux_computeProperties(V, x, y, other_polygon, n_in) -> Tuple[Any, int, np.ndarray]:
    """Compute properties according to given user inputs"""
    
    # from other polygon object (copy constructor)
    if other_polygon is not None:
        poly = other_polygon.set
        v_matrix = other_polygon._V
    # from vertices
    elif V.size > 0:
        poly = ShapelyPolygon(V.T) if SHAPELY_AVAILABLE else {'vertices': V}
        v_matrix = V
    # from x and y coordinates
    elif x.size > 0:
        v_matrix = np.vstack((x, y))
        poly = ShapelyPolygon(v_matrix.T) if SHAPELY_AVAILABLE else {'vertices': v_matrix}
    # empty polygon
    else:
        poly = ShapelyPolygon() if SHAPELY_AVAILABLE else None
        v_matrix = np.array([]).reshape(2,0)
        
    dim = 2 # Polygons are always 2D
        
    return poly, dim, v_matrix