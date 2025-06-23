"""
contSet - Base class for all continuous sets

This module contains the abstract base class ContSet which defines the interface
for all continuous set representations in CORA.

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 02-May-2007 (MATLAB)
Python translation: 2025
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        from g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class ContSet(ABC):
    """
    Abstract base class for all continuous sets
    
    This class defines the common interface and functionality for all
    continuous set representations in CORA.
    """
    
    def __init__(self):
        """Initialize the base ContSet"""
        self.precedence = 50  # Default precedence for operator overloading
    
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def dim(self) -> int:
        """Get dimension of the set"""
        pass
    
    @abstractmethod
    def is_empty(self) -> bool:
        """Check if set is empty"""
        pass
    
    # String representation methods following Python best practices
    def __repr__(self) -> str:
        """
        Official string representation for programmers.
        Should be unambiguous and ideally allow object reconstruction.
        """
        # Default implementation - subclasses should override
        class_name = self.__class__.__name__
        try:
            dim_val = self.dim()
            return f"{class_name}(dim={dim_val})"
        except:
            return f"{class_name}()"
    
    def __str__(self) -> str:
        """
        Informal string representation for users.
        Should be readable and user-friendly.
        """
        # For most contSet objects, use the display method if available
        if hasattr(self, 'display') and callable(getattr(self, 'display')):
            try:
                return self.display()
            except:
                pass
        
        # Fallback to repr if display is not available or fails
        return self.__repr__()
    
    def display(self) -> str:
        """
        Display method that mirrors MATLAB behavior.
        Default implementation - subclasses should override for specific formatting.
        """
        # Use MATLAB-style display format
        class_name = self.__class__.__name__.lower()
        try:
            dim_val = self.dim()
            return f"{class_name}:\n- dimension: {dim_val}"
        except:
            return f"{class_name}:\n- dimension: unknown"
    
    # Operator overloading with proper polymorphic dispatch
    def subsasgn(self, *args):
        """Assignment to index - not supported for arrays"""
        from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError
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
        # Import here to avoid circular imports
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
        # Use polymorphic dispatch
        if hasattr(self, 'plot') and callable(getattr(self, 'plot')):
            return self.plot(*args, **kwargs)
        else:
            from .plot import plot
            return plot(self, *args, **kwargs)
    
    # Common utility methods with polymorphic dispatch
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if set contains given point(s)"""
        # Check if subclass has overridden contains_ method
        if hasattr(self, 'contains_') and callable(getattr(self, 'contains_')):
            # Call contains_ and return just the boolean result
            result = self.contains_(point)
            if isinstance(result, tuple):
                return result[0]  # Return just the boolean part
            else:
                return result
        else:
            from .contains import contains
            return contains(self, point)
    
    def center(self) -> np.ndarray:
        """Get center of the set"""
        if hasattr(self, 'center') and callable(getattr(self, 'center')):
            return self.center()
        else:
            from .center import center
            return center(self)
    
    def representsa(self, set_type: str, tol: float = 1e-9) -> bool:
        """Check if set represents a specific type"""
        if hasattr(self, 'representsa') and callable(getattr(self, 'representsa')):
            return self.representsa(set_type, tol)
        else:
            from .representsa import representsa
            return representsa(self, set_type, tol)
    
    def representsa_(self, set_type: str, tol: float = 1e-9) -> bool:
        """Internal check if set represents a specific type"""
        if hasattr(self, 'representsa_') and callable(getattr(self, 'representsa_')):
            return self.representsa_(set_type, tol)
        else:
            from .representsa_ import representsa_
            return representsa_(self, set_type, tol)
    
    def __eq__(self, other) -> bool:
        """Equality comparison"""
        # Check if subclass has overridden __eq__ method
        if type(self).__eq__ is not ContSet.__eq__:
            return type(self).__eq__(self, other)
        else:
            from .eq import eq
            return eq(self, other)
    
    def __ne__(self, other) -> bool:
        """Inequality comparison"""
        # Check if subclass has overridden __ne__ method
        if type(self).__ne__ is not ContSet.__ne__:
            return type(self).__ne__(self, other)
        else:
            from .ne import ne
            return ne(self, other)
    
    def isequal(self, other, tol: float = None) -> bool:
        """Check if sets are equal"""
        if hasattr(self, 'isequal') and callable(getattr(self, 'isequal')):
            return self.isequal(other, tol)
        else:
            from .isequal import isequal
            return isequal(self, other, tol)
    
    def isempty(self) -> bool:
        """Check if set is empty (deprecated)"""
        # Check if subclass has overridden isempty method
        if type(self).isempty is not ContSet.isempty:
            return type(self).isempty(self)
        else:
            from .isempty import isempty
            return isempty(self)
    
    def copy(self) -> 'ContSet':
        """Create a copy of the set"""
        # Check if subclass has overridden copy method
        if type(self).copy is not ContSet.copy:
            return type(self).copy(self)
        else:
            from .copy import copy
            return copy(self)

    def __add__(self, other) -> 'ContSet':
        """Addition operator (+)"""
        # Check if subclass has overridden __add__ method
        if type(self).__add__ is not ContSet.__add__:
            return type(self).__add__(self, other)
        else:
            from .plus import plus
            return plus(self, other)

    def __radd__(self, other) -> 'ContSet':
        """Right addition operator"""
        # Check if subclass has overridden __radd__ method
        if type(self).__radd__ is not ContSet.__radd__:
            return type(self).__radd__(self, other)
        else:
            from .plus import plus
            return plus(other, self)

    def __sub__(self, other) -> 'ContSet':
        """Subtraction operator (-)"""
        # Check if subclass has overridden __sub__ method
        if type(self).__sub__ is not ContSet.__sub__:
            return type(self).__sub__(self, other)
        else:
            from .minus import minus
            return minus(self, other)

    def __rsub__(self, other) -> 'ContSet':
        """Right subtraction operator"""
        # Check if subclass has overridden __rsub__ method
        if type(self).__rsub__ is not ContSet.__rsub__:
            return type(self).__rsub__(self, other)
        else:
            from .minus import minus
            return minus(other, self)

    def __neg__(self) -> 'ContSet':
        """Unary minus operator"""
        # Check if subclass has overridden __neg__ method
        if type(self).__neg__ is not ContSet.__neg__:
            return type(self).__neg__(self)
        else:
            from .uminus import uminus
            return uminus(self)

    def __pos__(self) -> 'ContSet':
        """Unary plus operator"""
        # Check if subclass has overridden __pos__ method
        if type(self).__pos__ is not ContSet.__pos__:
            return type(self).__pos__(self)
        else:
            from .uplus import uplus
            return uplus(self)

    def __mul__(self, other) -> 'ContSet':
        """Multiplication operator (*)"""
        # Check if subclass has overridden __mul__ method
        if type(self).__mul__ is not ContSet.__mul__:
            return type(self).__mul__(self, other)
        else:
            from .mtimes import mtimes
            return mtimes(other, self)  # Note: order is reversed for matrix multiplication

    def __rmul__(self, other) -> 'ContSet':
        """Right multiplication operator"""
        # Check if subclass has overridden __rmul__ method
        if type(self).__rmul__ is not ContSet.__rmul__:
            return type(self).__rmul__(self, other)
        else:
            from .mtimes import mtimes
            return mtimes(other, self)

    def __and__(self, other) -> 'ContSet':
        """Intersection operator (&)"""
        return self.and_(other)

    def __or__(self, other) -> 'ContSet':
        """Union operator (|)"""
        return self.or_(other)

    def is_bounded(self) -> bool:
        """Check if set is bounded"""
        if hasattr(self, 'is_bounded') and callable(getattr(self, 'is_bounded')):
            return self.is_bounded()
        else:
            from .isBounded import isBounded
            return isBounded(self)
    
    def vertices(self) -> np.ndarray:
        """Get vertices of the set"""
        # Check if subclass has overridden vertices method
        if hasattr(self, 'vertices_') and callable(getattr(self, 'vertices_')):
            return self.vertices_()
        else:
            from .vertices import vertices
            return vertices(self)
    
    def vertices_(self) -> np.ndarray:
        """Get vertices of the set (internal)"""
        # Check if subclass has overridden vertices_ method
        if type(self).vertices_ is not ContSet.vertices_:
            return type(self).vertices_(self)
        else:
            from .vertices_ import vertices_
            return vertices_(self)
    
    def and_(self, other: 'ContSet') -> 'ContSet':
        """Intersection operator (internal)"""
        # Check if subclass has overridden and_ method
        if type(self).and_ is not ContSet.and_:
            return type(self).and_(self, other)
        else:
            from .and_ import and_
            return and_(self, other)
    
    def or_(self, other: 'ContSet') -> 'ContSet':
        """Union operator (internal)"""
        # Check if subclass has overridden or_ method
        if type(self).or_ is not ContSet.or_:
            return type(self).or_(self, other)
        else:
            from .or_ import or_
            return or_(self, other)
    
    def convHull(self, other: 'ContSet' = None) -> 'ContSet':
        """Compute convex hull"""
        if hasattr(self, 'convHull') and callable(getattr(self, 'convHull')):
            return self.convHull(other)
        else:
            from .convHull import convHull
            return convHull(self, other)
    
    def convHull_(self, other: 'ContSet' = None, method: str = 'exact') -> 'ContSet':
        """Compute convex hull (internal version)"""
        if hasattr(self, 'convHull_') and callable(getattr(self, 'convHull_')):
            return self.convHull_(other, method)
        else:
            from .convHull_ import convHull_
            return convHull_(self, other, method)
    
    # Methods needed for plotting (to be implemented by subclasses)
    def project(self, dims: Union[List[int], np.ndarray]) -> 'ContSet':
        """Project set to lower-dimensional subspace"""
        if hasattr(self, 'project') and callable(getattr(self, 'project')):
            return self.project(dims)
        else:
            from .project import project
            return project(self, dims)
    
    def decompose(self, num_sets: int) -> List['ContSet']:
        """Decompose set into multiple sets"""
        if hasattr(self, 'decompose') and callable(getattr(self, 'decompose')):
            return self.decompose(num_sets)
        else:
            from .decompose import decompose
            return decompose(self, num_sets)
    
    def times(self, other) -> 'ContSet':
        """Element-wise multiplication"""
        if hasattr(self, 'times') and callable(getattr(self, 'times')):
            return self.times(other)
        else:
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
        if hasattr(self, 'supportFunc') and callable(getattr(self, 'supportFunc')):
            return self.supportFunc(direction, type_, method, max_order_or_splits, tol)
        else:
            from .supportFunc import supportFunc
            return supportFunc(self, direction, type_, method, max_order_or_splits, tol)
    
    def supportFunc_(self, direction: np.ndarray, type_: str = 'upper',
                    method: str = 'interval', max_order_or_splits: int = 8,
                    tol: float = 1e-3):
        """Evaluate support function (internal)"""
        if hasattr(self, 'supportFunc_') and callable(getattr(self, 'supportFunc_')):
            return self.supportFunc_(direction, type_, method, max_order_or_splits, tol)
        else:
            from .supportFunc_ import supportFunc_
            return supportFunc_(self, direction, type_, method, max_order_or_splits, tol) 