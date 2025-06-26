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
    from shapely.geometry import Point, MultiPolygon
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
        self.poly, self.dim = _aux_computeProperties(V, x, y, other_polygon, len(varargin))

        # 4. set precedence
        super().__init__()
        self.precedence = 10

    def _postprocess(self, set_, x, y, NVpairs):
        """Postprocessing step to finalize polygon construction"""
        
        # copy constructor
        if isinstance(set_, Polygon):
            self.set = set_.set
            self.V = set_.V.copy() if hasattr(set_, 'V') else np.array([]).reshape(2, 0)
            self.TOL = set_.TOL if hasattr(set_, 'TOL') else 1e-12
            return
        
        # shapely polygon given?
        if SHAPELY_AVAILABLE and isinstance(set_, ShapelyPolygon):
            self.set = set_
            coords = np.array(set_.exterior.coords)
            if coords.size > 0:
                self.V = coords[:-1, :].T  # Remove duplicate last point and transpose
            else:
                self.V = np.array([]).reshape(2, 0)
            return
        
        # vertices matrix given?
        if isinstance(set_, np.ndarray) and set_.size > 0:
            if set_.shape[0] == 2:
                # 2×n matrix
                x = set_[0, :]
                y = set_[1, :]
            else:
                raise CORAerror("CORA:wrongValue", 'first', 'Given vertices should be two dimensional.')
        
        # empty x,y?
        if x is None or len(x) == 0:
            if SHAPELY_AVAILABLE:
                self.set = ShapelyPolygon()
            else:
                self.set = None
            self.V = np.array([]).reshape(2, 0)
            return
        
        # x,y given
        x = np.array(x).reshape(-1)
        y = np.array(y).reshape(-1)
        
        if len(x) != len(y):
            raise CORAerror("CORA:wrongInputInConstructor", 
                          'Given vertices x,y need to be vectors of the same length.')
        
        # Store vertices
        self.V = np.vstack([x, y])
        
        if SHAPELY_AVAILABLE:
            # Use Shapely for robust polygon operations
            try:
                # Create polygon from vertices
                if len(x) >= 3:  # Need at least 3 points for a polygon
                    coords = list(zip(x, y))
                    self.set = ShapelyPolygon(coords)
                    
                    # If invalid, try to fix it
                    if not self.set.is_valid:
                        # Try buffering with small value to fix topology issues
                        self.set = self.set.buffer(0)
                        
                        # If still invalid, create convex hull
                        if not self.set.is_valid:
                            points = [Point(xi, yi) for xi, yi in zip(x, y)]
                            if len(points) >= 3:
                                self.set = convex_hull(unary_union(points))
                            else:
                                self.set = ShapelyPolygon()
                                
                elif len(x) == 2:
                    # Line segment - create very thin rectangle
                    line_points = np.array([[x[0], y[0]], [x[1], y[1]]])
                    # Create a small buffer around the line
                    from shapely.geometry import LineString
                    line = LineString(line_points)
                    self.set = line.buffer(self.TOL)
                    
                elif len(x) == 1:
                    # Single point - create small circle
                    point = Point(x[0], y[0])
                    self.set = point.buffer(self.TOL)
                    
                else:
                    # Empty
                    self.set = ShapelyPolygon()
                    
            except Exception as e:
                # Fallback: create simple polygon or handle degeneracies
                if len(x) >= 3:
                    try:
                        # Simple polygon creation without validation
                        coords = list(zip(x, y))
                        self.set = ShapelyPolygon(coords)
                    except:
                        # Ultimate fallback
                        self.set = ShapelyPolygon()
                else:
                    self.set = ShapelyPolygon()
                    
            # Update vertices from the processed polygon
            if self.set.is_empty:
                self.V = np.array([]).reshape(2, 0)
            else:
                try:
                    coords = np.array(self.set.exterior.coords)
                    if coords.size > 0:
                        self.V = coords[:-1, :].T  # Remove duplicate last point and transpose
                except:
                    # Keep original vertices if extraction fails
                    pass
                    
        else:
            # Fallback without Shapely - simple vertex storage
            self.set = {'vertices': self.V}

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

def _aux_parseInputArgs(*varargin) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional['Polygon']]:
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
        V = varargin[0]
    # from x and y coordinates
    elif len(varargin) == 2:
        x, y = varargin[0], varargin[1]
        
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
            if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
                raise CORAerror('CORA:wrongInputInConstructor', 
                    'Inputs x and y must be vectors of the same length.')

def _aux_computeProperties(V, x, y, other_polygon, n_in) -> Tuple[ShapelyPolygon, int]:
    """Compute properties according to given user inputs"""
    
    # from other polygon object (copy constructor)
    if other_polygon is not None:
        poly = other_polygon.poly
    # from vertices
    elif V.size > 0:
        poly = ShapelyPolygon(V.T)
    # from x and y coordinates
    elif x.size > 0:
        poly = ShapelyPolygon(np.vstack((x, y)).T)
    # empty polygon
    else:
        poly = ShapelyPolygon()
        
    dim = len(poly.exterior.coords.xy[0]) if not poly.is_empty else 0
        
    return poly, dim 