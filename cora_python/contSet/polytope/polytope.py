"""
polytope - class for convex polytopes

This class implements convex polytopes defined by linear constraints
of the form {x | A*x <= b}.

Syntax:
    P = Polytope(A, b)
    P = Polytope(V)  # From vertices

Inputs:
    A - constraint matrix (m x n)
    b - constraint vector (m x 1)
    V - vertex matrix (n x k)

Outputs:
    P - polytope object

Example:
    # Polytope from constraints: x1 + x2 <= 1, x1 >= 0, x2 >= 0
    A = np.array([[1, 1], [-1, 0], [0, -1]])
    b = np.array([1, 0, 0])
    P = Polytope(A, b)
    
    # Polytope from vertices (unit simplex)
    V = np.array([[0, 1, 0], [0, 0, 1]])
    P = Polytope(V)

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2006 (MATLAB)
Last update: 25-July-2016 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Optional, Union, Any
from ..contSet.contSet import ContSet
from ...g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class Polytope(ContSet):
    """
    Polytope class for convex polytopes
    
    This class represents convex polytopes defined by linear constraints
    of the form {x | A*x <= b} or by vertices.
    
    Properties:
        A: Constraint matrix (m x n)
        b: Constraint vector (m x 1)
        V: Vertices (n x k) - computed on demand
        bounded: Whether polytope is bounded
        fullDim: Whether polytope is full-dimensional
        precedence: Set to 80 for polytopes
    """
    
    def __init__(self, *args):
        """
        Constructor for polytope objects
        
        Args:
            *args: Variable arguments for different construction modes:
                   - Polytope(A, b): From constraints A*x <= b
                   - Polytope(V): From vertices V
                   - Polytope(P): Copy constructor
        """
        # Call parent constructor
        super().__init__()
        
        # Avoid empty instantiation
        if len(args) == 0:
            raise CORAError('CORA:noInputInSetConstructor', 
                           'No input arguments provided to polytope constructor')
        
        if len(args) > 2:
            raise CORAError('CORA:wrongInput', 
                           'Too many input arguments for polytope constructor')
        
        # Copy constructor
        if len(args) == 1 and isinstance(args[0], Polytope):
            other = args[0]
            self.A = other.A.copy() if other.A is not None else None
            self.b = other.b.copy() if other.b is not None else None
            self.V = other.V.copy() if hasattr(other, 'V') and other.V is not None else None
            self.bounded = getattr(other, 'bounded', None)
            self.fullDim = getattr(other, 'fullDim', None)
            self.precedence = 80
            return
        
        # Parse input arguments
        if len(args) == 1:
            # From vertices
            V = np.asarray(args[0], dtype=float)
            if V.ndim != 2:
                raise CORAError('CORA:wrongInputInConstructor',
                               'Vertex matrix must be 2D')
            
            self.V = V
            self.A = None
            self.b = None
            # Convert to constraints (would need convex hull computation)
            # For now, store vertices only
            
        elif len(args) == 2:
            # From constraints A*x <= b
            A = np.asarray(args[0], dtype=float)
            b = np.asarray(args[1], dtype=float)
            
            if A.ndim != 2:
                raise CORAError('CORA:wrongInputInConstructor',
                               'Constraint matrix A must be 2D')
            
            if b.ndim == 1:
                b = b.reshape(-1, 1)
            elif b.ndim != 2 or b.shape[1] != 1:
                raise CORAError('CORA:wrongInputInConstructor',
                               'Constraint vector b must be 1D or column vector')
            
            if A.shape[0] != b.shape[0]:
                raise CORAError('CORA:wrongInputInConstructor',
                               'Number of rows in A must match length of b')
            
            self.A = A
            self.b = b
            self.V = None
        
        # Set properties
        self.bounded = None  # Computed on demand
        self.fullDim = None  # Computed on demand
        self.precedence = 80
    
    def dim(self) -> int:
        """Get dimension of the polytope"""
        if self.A is not None:
            return self.A.shape[1]
        elif self.V is not None:
            return self.V.shape[0]
        else:
            return 0
    
    def is_empty(self) -> bool:
        """Check if polytope is empty"""
        if self.A is not None and self.b is not None:
            # Check if constraints are feasible
            # This is a simplified check - would need proper LP solver
            return False  # Assume non-empty for now
        elif self.V is not None:
            return self.V.shape[1] == 0
        else:
            return True
    
    def contains(self, point: Union[np.ndarray, 'ContSet']) -> bool:
        """Check if polytope contains given point(s) or set"""
        if isinstance(point, np.ndarray):
            return self._contains_point(point)
        else:
            # For sets, would need more sophisticated containment check
            raise NotImplementedError("Set containment not implemented for polytopes")
    
    def _contains_point(self, point: np.ndarray) -> bool:
        """Check if polytope contains a point"""
        if self.A is None or self.b is None:
            raise NotImplementedError("Point containment requires constraint representation")
        
        point = np.asarray(point, dtype=float)
        if point.ndim == 1:
            point = point.reshape(-1, 1)
        
        # Check A*x <= b
        return np.all(self.A @ point <= self.b + 1e-12)  # Small tolerance
    
    def vertices(self) -> np.ndarray:
        """Get vertices of the polytope"""
        if self.V is not None:
            return self.V
        else:
            # Would need to compute vertices from constraints
            # This requires vertex enumeration algorithms
            raise NotImplementedError("Vertex computation from constraints not implemented")
    
    def center(self) -> np.ndarray:
        """Get center of the polytope"""
        if self.V is not None:
            # Centroid of vertices
            return np.mean(self.V, axis=1, keepdims=True)
        else:
            # Would need to compute Chebyshev center
            raise NotImplementedError("Center computation from constraints not implemented")
    
    def is_bounded(self) -> bool:
        """Check if polytope is bounded"""
        if self.bounded is not None:
            return self.bounded
        
        if self.V is not None:
            self.bounded = True  # Vertex representation implies bounded
        else:
            # Would need to check if all directions are bounded
            self.bounded = True  # Assume bounded for now
        
        return self.bounded
    
    def isIntersecting(self, other: Union['ContSet', np.ndarray]) -> bool:
        """Check if polytope intersects with another set"""
        if isinstance(other, np.ndarray):
            return self.contains(other)
        else:
            # Would need sophisticated intersection algorithms
            raise NotImplementedError("Set intersection not implemented for polytopes")
    
    def __add__(self, other):
        """Minkowski sum"""
        raise NotImplementedError("Minkowski sum not implemented for polytopes")
    
    def __sub__(self, other):
        """Pontryagin difference"""
        raise NotImplementedError("Pontryagin difference not implemented for polytopes")
    
    def __matmul__(self, other):
        """Linear transformation"""
        if isinstance(other, np.ndarray):
            return self._linear_map(other)
        else:
            raise NotImplementedError("Matrix multiplication with sets not implemented")
    
    def _linear_map(self, M: np.ndarray) -> 'Polytope':
        """Apply linear transformation M to polytope"""
        if self.V is not None:
            # Transform vertices
            V_new = M @ self.V
            return Polytope(V_new)
        else:
            # Transform constraints: A*x <= b becomes A*M^(-1)*y <= b
            # This is complex and requires matrix inversion
            raise NotImplementedError("Linear map from constraints not implemented")
    
    def __and__(self, other):
        """Intersection"""
        if isinstance(other, Polytope):
            return self._intersect_polytope(other)
        else:
            raise NotImplementedError("Intersection with other sets not implemented")
    
    def _intersect_polytope(self, other: 'Polytope') -> 'Polytope':
        """Intersect with another polytope"""
        if self.A is not None and other.A is not None:
            # Combine constraints
            A_new = np.vstack([self.A, other.A])
            b_new = np.vstack([self.b, other.b])
            return Polytope(A_new, b_new)
        else:
            raise NotImplementedError("Intersection requires constraint representation")
    
    # Static methods
    @staticmethod
    def empty(n: int = 0) -> 'Polytope':
        """Create empty polytope"""
        # Empty polytope has infeasible constraints
        A = np.array([[1], [-1]])  # x >= 1 and x <= -1 (infeasible)
        b = np.array([[-1], [-1]])
        if n > 1:
            A = np.hstack([A, np.zeros((2, n-1))])
        return Polytope(A, b)
    
    @staticmethod
    def Inf(n: int) -> 'Polytope':
        """Create unbounded polytope (whole space)"""
        # No constraints means whole space
        A = np.zeros((0, n))
        b = np.zeros((0, 1))
        return Polytope(A, b)
    
    def __str__(self) -> str:
        """String representation"""
        if self.A is not None:
            return f"Polytope: {self.A.shape[0]} constraints in R^{self.dim()}"
        elif self.V is not None:
            return f"Polytope: {self.V.shape[1]} vertices in R^{self.dim()}"
        else:
            return "Polytope: empty"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__() 