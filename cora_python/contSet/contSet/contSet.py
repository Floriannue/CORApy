"""
contSet - abstract superclass for continuous sets

This class represents the abstract base class for all continuous set representations
in CORA. It provides common functionality and defines the interface that all
continuous sets must implement.

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 02-May-2007 (MATLAB)
Last update: 22-September-2024 (MATLAB)
Python translation: 2025
"""

from abc import ABC, abstractmethod
from typing import Any, Union, Optional, List, Tuple
import numpy as np
try:
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError
except ImportError:
    # Fallback for when running from within the cora_python directory
    from g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class ContSet(ABC):
    """
    Abstract superclass for continuous sets
    
    This class provides the base functionality for all continuous set representations.
    It defines common operations and maintains precedence ordering for set operations.
    
    Properties:
        precedence: Ordering of set representation (lower value = higher precedence)
                   0: emptySet, 10: fullspace, 15: polygon, 20: levelSet,
                   30: conPolyZono, 40: spectraShadow, 50: ellipsoid, 60: capsule,
                   70: polyZonotope, 80: polytope, 90: conZonotope, 100: zonoBundle,
                   110: zonotope, 120: interval, 1000: contSet
    """
    
    def __init__(self, other: Optional['ContSet'] = None):
        """
        Constructor for contSet
        
        Args:
            other: Another contSet object for copy construction
        """
        if other is not None and isinstance(other, ContSet):
            # Copy constructor
            self.precedence = other.precedence
        else:
            # Default constructor
            self.precedence = 1000
    
    # Prevent class array operations (similar to MATLAB's approach)
    def vertcat(self, *args):
        """Vertical concatenation - not supported"""
        raise CORAError('CORA:notSupported',
                       'Given subclass of contSet does not support class arrays.')
    
    def horzcat(self, *args):
        """Horizontal concatenation - not supported"""
        raise CORAError('CORA:notSupported',
                       'Given subclass of contSet does not support class arrays.')
    
    def cat(self, *args):
        """Concatenation along different dimensions - not supported"""
        raise CORAError('CORA:notSupported',
                       'Given subclass of contSet does not support class arrays.')
    
    def subsasgn(self, *args):
        """Assignment to index - not supported for arrays"""
        raise CORAError('CORA:notSupported',
                       'Given subclass of contSet does not support class arrays.')
    
    # Static methods
    @staticmethod
    def enclosePoints(*args, **kwargs):
        """Encloses a point cloud with a set"""
        raise NotImplementedError("enclosePoints not implemented")
    
    @staticmethod
    def generateRandom(*args, **kwargs):
        """Generates a random contSet"""
        from .generateRandom import generateRandom
        return generateRandom(*args, **kwargs)
    
    @staticmethod
    def initEmptySet(set_type: str):
        """Instantiates an empty set of a contSet class"""
        raise NotImplementedError("initEmptySet not implemented")
    
    @staticmethod
    def empty(n: int = 0):
        """Instantiates an empty set of a contSet class"""
        raise NotImplementedError("empty not implemented")
    
    @staticmethod
    def Inf(n: int):
        """Instantiates a fullspace set of a contSet class"""
        raise NotImplementedError("Inf not implemented")
    
    # Protected methods (method signatures only)
    def _representsa_emptyObject(self, set_type: str):
        """Check if object represents empty set of given type"""
        raise NotImplementedError("_representsa_emptyObject not implemented")
    
    def _getPrintSetInfo(self):
        """Get abbreviation and print order for set"""
        raise NotImplementedError("_getPrintSetInfo not implemented")
    
    # Plotting methods (method signatures only)
    def _plot1D(self, *args, **kwargs):
        """1D plotting method"""
        raise NotImplementedError("_plot1D not implemented")
    
    def _plot2D(self, *args, **kwargs):
        """2D plotting method"""
        raise NotImplementedError("_plot2D not implemented")
    
    def _plot3D(self, *args, **kwargs):
        """3D plotting method"""
        raise NotImplementedError("_plot3D not implemented")
    
    def plot(self, *args, **kwargs):
        """Main plotting method - delegates to appropriate dimension"""
        from .plot import plot
        return plot(self, *args, **kwargs)
    
    # Common utility methods
    def dim(self) -> int:
        """Get dimension of the set"""
        from .dim import dim
        return dim(self)
    
    def is_empty(self) -> bool:
        """Check if set is empty"""
        from .isemptyobject import isemptyobject
        return isemptyobject(self)
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if set contains given point(s)"""
        from .contains import contains
        return contains(self, point)
    
    def center(self) -> np.ndarray:
        """Get center of the set"""
        from .center import center
        return center(self)
    
    def representsa(self, set_type: str, tol: float = 1e-9) -> bool:
        """Check if set represents a specific type"""
        from .representsa import representsa
        return representsa(self, set_type, tol)
    
    def representsa_(self, set_type: str, tol: float = 1e-9) -> bool:
        """Internal check if set represents a specific type"""
        from .representsa_ import representsa_
        return representsa_(self, set_type, tol)
    
    def __eq__(self, other) -> bool:
        """Equality comparison"""
        from .eq import eq
        return eq(self, other)
    
    def __ne__(self, other) -> bool:
        """Inequality comparison"""
        from .ne import ne
        return ne(self, other)
    
    def isequal(self, other, tol: float = None) -> bool:
        """Check if sets are equal"""
        from .isequal import isequal
        return isequal(self, other, tol)
    
    def isempty(self) -> bool:
        """Check if set is empty (deprecated)"""
        from .isempty import isempty
        return isempty(self)
    
    def display(self) -> None:
        """Display set information"""
        from .display import display
        return display(self)
    
    def copy(self) -> 'ContSet':
        """Create a copy of the set"""
        from .copy import copy
        return copy(self)
    
    def __add__(self, other) -> 'ContSet':
        """Addition operator (+)"""
        from .plus import plus
        return plus(self, other)
    
    def __radd__(self, other) -> 'ContSet':
        """Right addition operator"""
        from .plus import plus
        return plus(other, self)
    
    def __sub__(self, other) -> 'ContSet':
        """Subtraction operator (-)"""
        from .minus import minus
        return minus(self, other)
    
    def __rsub__(self, other) -> 'ContSet':
        """Right subtraction operator"""
        from .minus import minus
        return minus(other, self)
    
    def __neg__(self) -> 'ContSet':
        """Unary minus operator"""
        from .uminus import uminus
        return uminus(self)
    
    def __pos__(self) -> 'ContSet':
        """Unary plus operator"""
        from .uplus import uplus
        return uplus(self)
    
    def __mul__(self, other) -> 'ContSet':
        """Multiplication operator (*)"""
        from .mtimes import mtimes
        return mtimes(other, self)  # Note: order is reversed for matrix multiplication
    
    def __rmul__(self, other) -> 'ContSet':
        """Right multiplication operator"""
        from .mtimes import mtimes
        return mtimes(other, self)
    
    def is_bounded(self) -> bool:
        """Check if set is bounded"""
        from .isBounded import isBounded
        return isBounded(self)
    
    def vertices(self, method: str = None, *args, **kwargs) -> np.ndarray:
        """Get vertices of the set"""
        from .vertices import vertices
        return vertices(self, method, *args, **kwargs)
    
    def vertices_(self, method: str = 'convHull', *args, **kwargs) -> np.ndarray:
        """Get vertices of the set (internal version)"""
        from .vertices_ import vertices_
        return vertices_(self, method, *args, **kwargs)
    
    def volume(self, method: str = 'exact', order: int = 5) -> float:
        """Compute volume of the set"""
        from .volume import volume
        return volume(self, method, order)
    
    def volume_(self, method: str = 'exact', order: int = 5) -> float:
        """Compute volume of the set (internal version)"""
        from .volume_ import volume_
        return volume_(self, method, order)
    
    def norm(self, norm_type: Union[int, float, str] = 2, mode: str = 'ub'):
        """Compute norm of the set"""
        from .norm import norm
        return norm(self, norm_type, mode)
    
    def norm_(self, norm_type: Union[int, float, str] = 2, mode: str = 'ub'):
        """Compute norm of the set (internal version)"""
        from .norm_ import norm_
        return norm_(self, norm_type, mode)
    
    def __and__(self, other) -> 'ContSet':
        """Intersection operator (&)"""
        from .and_op import and_op
        return and_op(self, other)
    
    def __or__(self, other) -> 'ContSet':
        """Union operator (|)"""
        from .or_op import or_op
        return or_op(self, other)
    
    def and_(self, other: 'ContSet', method: str = 'exact') -> 'ContSet':
        """Set intersection (internal)"""
        from .and_ import and_
        return and_(self, other, method)
    
    def convHull(self, other: 'ContSet' = None, method: str = 'exact') -> 'ContSet':
        """Compute convex hull"""
        from .convHull import convHull
        return convHull(self, other, method)
    
    def convHull_(self, other: 'ContSet' = None, method: str = 'exact') -> 'ContSet':
        """Compute convex hull (internal version)"""
        from .convHull_ import convHull_
        return convHull_(self, other, method)
    
    # Methods needed for plotting (to be implemented by subclasses)
    def project(self, dims: Union[List[int], np.ndarray]) -> 'ContSet':
        """Project set to lower-dimensional subspace"""
        from .project import project
        return project(self, dims)
    
    def decompose(self, num_sets: int) -> List['ContSet']:
        """Decompose set into multiple sets"""
        from .decompose import decompose
        return decompose(self, num_sets)
    
    def times(self, other) -> 'ContSet':
        """Element-wise multiplication"""
        from .times import times
        return times(self, other)
    
    def interval(self, *args) -> 'ContSet':
        """Convert to interval representation"""
        raise NotImplementedError("interval not implemented")
    
    def polygon(self, *args) -> 'ContSet':
        """Convert to polygon representation"""
        raise NotImplementedError("polygon not implemented")
    
    # Support function operations
    def supportFunc(self, direction: np.ndarray, type_: str = 'upper', 
                   method: str = 'interval', max_order_or_splits: int = 8, 
                   tol: float = 1e-3):
        """Evaluate support function"""
        from .supportFunc import supportFunc
        return supportFunc(self, direction, type_, method, max_order_or_splits, tol)
    
    def supportFunc_(self, direction: np.ndarray, type_: str = 'upper',
                    method: str = 'interval', max_order_or_splits: int = 8,
                    tol: float = 1e-3):
        """Evaluate support function (internal)"""
        from .supportFunc_ import supportFunc_
        return supportFunc_(self, direction, type_, method, max_order_or_splits, tol)
    
    # Random point generation
    def randPoint(self, N: Union[int, str] = 1, type_: str = 'standard', pr: float = 0.7) -> np.ndarray:
        """Generate random points"""
        from .randPoint import randPoint
        return randPoint(self, N, type_, pr)
    
    def randPoint_(self, N: Union[int, str] = 1, type_: str = 'standard') -> np.ndarray:
        """Generate random points (internal)"""
        from .randPoint_ import randPoint_
        return randPoint_(self, N, type_)
    
    # Intersection checking
    def isIntersecting(self, other: Union['ContSet', np.ndarray], 
                      type_: str = 'exact', tol: float = 1e-8) -> bool:
        """Check if sets intersect"""
        from .isIntersecting import isIntersecting
        return isIntersecting(self, other, type_, tol)
    
    def isIntersecting_(self, other: Union['ContSet', np.ndarray],
                       type_: str = 'exact', tol: float = 1e-8) -> bool:
        """Check if sets intersect (internal)"""
        from .isIntersecting_ import isIntersecting_
        return isIntersecting_(self, other, type_, tol)
    
    # Property checking functions
    def isFullDim(self) -> bool:
        """Check if set is full-dimensional"""
        from .isFullDim import isFullDim
        return isFullDim(self)
    
    def isZero(self) -> bool:
        """Check if set represents origin (deprecated)"""
        from .isZero import isZero
        return isZero(self)
    
    # Geometric operations
    def enlarge(self, factor: np.ndarray) -> 'ContSet':
        """Enlarge set by given factor"""
        from .enlarge import enlarge
        return enlarge(self, factor)
    
    def lift(self, N: int, proj: Optional[np.ndarray] = None) -> 'ContSet':
        """Lift set to higher-dimensional space"""
        from .lift import lift
        return lift(self, N, proj)
    
    def lift_(self, N: int, proj: np.ndarray) -> 'ContSet':
        """Lift set to higher-dimensional space (internal)"""
        from .lift_ import lift_
        return lift_(self, N, proj)
    
    # Compactification operations
    def compact(self, method: Optional[str] = None, tol: Optional[float] = None) -> 'ContSet':
        """Remove redundancies in set representation"""
        from .compact import compact
        return compact(self, method, tol)
    
    def compact_(self, method: Optional[str] = None, tol: Optional[float] = None) -> 'ContSet':
        """Remove redundancies in set representation (internal)"""
        from .compact_ import compact_
        return compact_(self, method, tol)
    
    # Additional operations
    def reduce(self, *args, **kwargs) -> 'ContSet':
        """Reduce the order of a set"""
        from .reduce import reduce
        return reduce(self, *args, **kwargs)
    
    @staticmethod
    def origin(n: int) -> 'ContSet':
        """Create a set representing only the origin"""
        from .origin import origin
        return origin(n)
    
    def minkDiff(self, other: Union['ContSet', np.ndarray], method: Optional[str] = None) -> 'ContSet':
        """Compute Minkowski difference"""
        from .minkDiff import minkDiff
        return minkDiff(self, other, method)
    
    def linComb(self, other: 'ContSet') -> 'ContSet':
        """Compute linear combination"""
        from .linComb import linComb
        return linComb(self, other)
    
    def quadMap(self, other: 'ContSet', Q: List[np.ndarray]) -> 'ContSet':
        """Compute quadratic map"""
        from .quadMap import quadMap
        return quadMap(self, other, Q)
    
    def cubMap(self, S2: Optional['ContSet'] = None, S3: Optional['ContSet'] = None,
               T: Optional[np.ndarray] = None, ind: Optional[List] = None) -> 'ContSet':
        """Compute cubic map"""
        from .cubMap import cubMap
        return cubMap(self, S2, S3, T, ind) 