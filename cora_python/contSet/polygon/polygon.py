"""
polygon - object constructor for polygon objects

Description:
    This class represents polygon objects based on vertices of a nonconvex set in 2D,
    similar to MATLAB's polyshape class.

Syntax:
    obj = polygon(x, y, **kwargs)
    obj = polygon(V, **kwargs)
    obj = polygon(set, **kwargs)

Inputs:
    x - vector with x coordinates of the polygon vertices
    y - vector with y coordinates of the polygon vertices
    V - vertices (2-dimensional matrix)
    set - polygon object or polyshape-like object
    **kwargs - additional arguments

Outputs:
    obj - polygon object

Example:
    x = np.array([0, 1, 1, 0])
    y = np.array([0, 0, 1, 1])
    pgon = polygon(x, y)

Authors: Niklas Kochdumper, Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 13-March-2020 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, Optional, Any, Tuple, List
from ..contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

# Import geometric computation capabilities

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point, MultiPolygon, LineString
from shapely.ops import unary_union
from shapely import convex_hull, buffer



class Polygon(ContSet):
    """
    Polygon class
    
    A polygon represents a 2D polytope with vertices based on MATLAB's polyshape.
    """
    
    def __init__(self, *varargin, **kwargs):
        """
        Constructor for polygon objects
        
        Args:
            *varargin: Variable arguments matching MATLAB constructor patterns:
                     - polygon(): empty polygon
                     - polygon(V): vertices as 2×n matrix
                     - polygon(x, y): x and y coordinates
                     - polygon(other_polygon): copy constructor
            **kwargs: Additional keyword arguments (name-value pairs)
        """
        # Call parent constructor
        super().__init__()
        
        # Parse input arguments
        set_obj, x, y, nvpairs = self._aux_parseInputArgs(*varargin, **kwargs)

        # Check input arguments
        self._aux_checkInputArgs(set_obj, x, y, nvpairs)

        # Postprocessing
        self._aux_postprocessing(set_obj, x, y, nvpairs)

        # Set precedence (fixed)
        self.precedence = 15

    def _aux_parseInputArgs(self, *varargin, **kwargs):
        """Parse input arguments from user and assign to variables"""
        # Default values
        set_obj = None
        x = np.array([])
        y = np.array([])
        nvpairs = kwargs

        # Empty constructor
        if len(varargin) == 0:
            set_obj = ShapelyPolygon()
            return set_obj, x, y, nvpairs
        
        # polygon(set)
        if len(varargin) == 1:
            set_obj = varargin[0]
            return set_obj, x, y, nvpairs

        # len(varargin) > 1
        # Check if set or (x,y) pair is given
        x = varargin[0]
        y = varargin[1]

        # Check if second parameter is numeric or first name-value pair
        if isinstance(y, (int, float, np.number)) or (isinstance(y, np.ndarray) and y.dtype.kind in 'biufc'):
            # (x,y) given - numeric second argument
            nvpairs.update(kwargs)  # Add any additional kwargs
        else:
            # 'y' is first name of first name-value pair
            set_obj = x
            x = np.array([])
            y = np.array([])
            # Combine varargin[1:] with kwargs as nvpairs
            nvpairs.update(kwargs)

        return set_obj, x, y, nvpairs

    def _aux_checkInputArgs(self, set_obj, x, y, nvpairs):
        """Check correctness of input arguments"""
        # Check set/V
        if set_obj is not None:
            if isinstance(set_obj, Polygon):
                # Valid polygon object
                pass
            elif isinstance(set_obj, ShapelyPolygon):
                # Valid shapely polygon
                pass
            elif isinstance(set_obj, np.ndarray):
                # Numeric array - should be 2D vertices
                if set_obj.ndim != 2 or set_obj.shape[0] != 2:
                    raise CORAerror("CORA:wrongValue", 'first', 'Given vertices should be two dimensional.')
            else:
                # Try to convert to numpy array
                try:
                    set_obj = np.asarray(set_obj)
                    if set_obj.ndim != 2 or set_obj.shape[0] != 2:
                        raise CORAerror("CORA:wrongValue", 'first', 'Given vertices should be two dimensional.')
                except:
                    raise CORAerror("CORA:wrongInputInConstructor", 'Invalid input type for polygon constructor.')

        # Check (x,y)
        if x.size > 0 or y.size > 0:
            x = np.asarray(x)
            y = np.asarray(y)
            if x.shape != y.shape or x.ndim != 1:
                raise CORAerror("CORA:wrongInputInConstructor", 
                    'Given vertices x,y need to be vectors of the same length.')

        # Check if finite (but nan is ok)
        if x.size > 0 and np.any(np.isinf(x)):
            raise CORAerror("CORA:wrongInputInConstructor", 'Given vertices cannot be infinite.')
        if y.size > 0 and np.any(np.isinf(y)):
            raise CORAerror("CORA:wrongInputInConstructor", 'Given vertices cannot be infinite.')
        if isinstance(set_obj, np.ndarray) and np.any(np.isinf(set_obj)):
            raise CORAerror("CORA:wrongInputInConstructor", 'Given vertices cannot be infinite.')

    def _aux_postprocessing(self, set_obj, x, y, nvpairs):
        """Postprocessing to create the polygon object"""
        
        # Initialize properties
        self._V = np.array([]).reshape(2, 0)
        self.TOL = 1e-8
        
        # Shapely polygon given?
        if isinstance(set_obj, ShapelyPolygon):
            self.set = set_obj
            self.poly = set_obj  # For compatibility with tests
            if not set_obj.is_empty:
                coords = np.array(set_obj.exterior.coords)
                if coords.shape[0] > 1:
                    self._V = coords[:-1].T  # Remove duplicate last point
            return

        # Polygon given?
        if isinstance(set_obj, Polygon):
            self.set = set_obj.set
            self.poly = set_obj.poly if hasattr(set_obj, 'poly') else set_obj.set
            self._V = set_obj._V.copy() if hasattr(set_obj, '_V') else np.array([]).reshape(2, 0)
            self.TOL = set_obj.TOL if hasattr(set_obj, 'TOL') else 1e-8
            return

        # 'set_obj' is a matrix
        if isinstance(set_obj, np.ndarray) and set_obj.size > 0:
            # Rewrite to x and y
            x = set_obj[0, :]
            y = set_obj[1, :]

        # Empty x,y?
        if x.size == 0:
            self.set = ShapelyPolygon()
            self.poly = self.set


        # x,y given
        x = x.reshape(-1)
        y = y.reshape(-1)

        # Create shapely polygon
        try:
            # Try to create polygon directly
            coords = list(zip(x, y))
            poly = ShapelyPolygon(coords)
            
            # Check if valid
            if poly.is_valid and not poly.is_empty:
                self.set = poly
                self.poly = poly
                self._V = np.vstack([x, y])
            else:
                # Try with convex hull
                poly = convex_hull(ShapelyPolygon(coords))
                self.set = poly
                self.poly = poly
                self._V = np.vstack([x, y])
                
        except Exception:
            # Fallback: create empty polygon
            self.set = ShapelyPolygon()
            self.poly = self.set
            self._V = np.array([]).reshape(2, 0)

    @property
    def V(self):
        """Get vertices of the polygon"""
        return self._V

    @property
    def x(self):
        """Get x coordinates of vertices"""
        if self._V.size > 0:
            return self._V[0, :]
        return np.array([])
    
    @property
    def y(self):
        """Get y coordinates of vertices"""
        if self._V.size > 0:
            return self._V[1, :]
        return np.array([])
    
    @property
    def nrOfRegions(self):
        """Get number of regions"""
        if hasattr(self.set, 'geoms'):
            # MultiPolygon case
            return len(self.set.geoms)
        elif isinstance(self.set, ShapelyPolygon):
            return 1 if not self.set.is_empty else 0
        else:
            return 1 if self._V.size > 0 else 0
    
    @property
    def nrOfHoles(self):
        """Get number of holes"""
        if isinstance(self.set, ShapelyPolygon) and not self.set.is_empty:
            return len(self.set.interiors)
        else:
            return 0

    def __repr__(self):
        """String representation"""
        if self._V.size > 0:
            return f"Polygon(vertices={self._V.shape[1]})"
        else:
            return "Polygon(empty)"
    
    def vertices_(self, method='convHull'):
        """
        Compute vertices of the polygon
        
        Args:
            method: string - method for vertex computation (currently ignored for polygons)
            
        Returns:
            np.ndarray: 2×n matrix with vertices as columns
        """
        return self._V 