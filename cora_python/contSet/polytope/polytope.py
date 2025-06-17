"""
polytope - class for polytope objects

This class represents polytope objects defined as (halfspace representation)
{ x | A*x <= b }. 
For convenience, equality constraints { x | A*x <= b, Ae*x == be }
can be added, too.
Alternatively, polytopes can be defined as (vertex representation)
{ sum_i a_i v_i | sum_i a_i = 1, a_i >= 0 }

Syntax:
    P = polytope(V)
    P = polytope(A,b)
    P = polytope(A,b,Ae,be)

Inputs:
    V - (n x p) array of vertices
    A - (n x m) matrix for the inequality representation
    b - (n x 1) vector for the inequality representation
    Ae - (k x l) matrix for the equality representation
    be - (k x 1) vector for the equality representation

Outputs:
    obj - generated polytope object

Example: 
    A = [1 0 -1 0 1; 0 1 0 -1 1]';
    b = [3; 2; 3; 2; 1];
    P = polytope(A,b);

Authors: Viktor Kotsev, Mark Wetzlinger, Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 25-April-2022 (MATLAB)
Last update: 01-December-2022 (MW, add CORAerrors, checks) (MATLAB)
             16-July-2024 (MW, allow separate usage of VRep/HRep) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, Optional, Any
from cora_python.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class Polytope(ContSet):
    """
    Polytope class for convex polytope objects
    
    This class represents polytopes in either halfspace representation 
    (A*x <= b) or vertex representation (convex hull of vertices).
    """
    
    def __init__(self, *args):
        """
        Constructor for polytope objects

        Args:
            *args: Variable arguments for different construction modes:
                   - Polytope(A, b): From constraints A*x <= b
                   - Polytope(A, b, Ae, be): From constraints with equalities
                   - Polytope(V): From vertices V
                   - Polytope(P): Copy constructor
                   - Polytope(Z): From zonotope Z
        """
        # Call parent constructor
        super().__init__()
        
        # Avoid empty instantiation
        if len(args) == 0:
            raise CORAError('CORA:noInputInSetConstructor',
                           'No input arguments provided to polytope constructor')

        if len(args) > 4:
            raise CORAError('CORA:wrongInput',
                           'Too many input arguments for polytope constructor')

        # Copy constructor
        if len(args) == 1 and isinstance(args[0], Polytope):
            other = args[0]
            self.A = other.A.copy() if other.A is not None else None
            self.b = other.b.copy() if other.b is not None else None
            self.Ae = getattr(other, 'Ae', None)
            self.be = getattr(other, 'be', None)
            self.V = other.V.copy() if hasattr(other, 'V') and other.V is not None else None
            self.bounded = getattr(other, 'bounded', None)
            self.fullDim = getattr(other, 'fullDim', None)
            self.precedence = 80
            return

        # Handle zonotope conversion
        if len(args) == 1 and hasattr(args[0], '__class__') and args[0].__class__.__name__ == 'Zonotope':
            # Convert zonotope to polytope
            from .zonotope import zonotope as poly_from_zonotope
            P = poly_from_zonotope(args[0])
            self.A = P.A
            self.b = P.b
            self.Ae = getattr(P, 'Ae', None)
            self.be = getattr(P, 'be', None)
            self.V = getattr(P, 'V', None)
            self.bounded = getattr(P, 'bounded', True)  # Zonotopes are always bounded
            self.fullDim = getattr(P, 'fullDim', None)
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
            self.Ae = None
            self.be = None
            
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
            self.Ae = None
            self.be = None
            
        elif len(args) == 4:
            # From constraints A*x <= b, Ae*x == be
            A = np.asarray(args[0], dtype=float)
            b = np.asarray(args[1], dtype=float)
            Ae = np.asarray(args[2], dtype=float)
            be = np.asarray(args[3], dtype=float)
            
            # Validate A, b
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
                               
            # Validate Ae, be
            if Ae.ndim != 2:
                raise CORAError('CORA:wrongInputInConstructor',
                               'Equality constraint matrix Ae must be 2D')
            
            if be.ndim == 1:
                be = be.reshape(-1, 1)
            elif be.ndim != 2 or be.shape[1] != 1:
                raise CORAError('CORA:wrongInputInConstructor',
                               'Equality constraint vector be must be 1D or column vector')
            
            if Ae.shape[0] != be.shape[0]:
                raise CORAError('CORA:wrongInputInConstructor',
                               'Number of rows in Ae must match length of be')
                               
            if A.shape[1] != Ae.shape[1]:
                raise CORAError('CORA:wrongInputInConstructor',
                               'Number of columns must match between A and Ae')
            
            self.A = A
            self.b = b
            self.Ae = Ae
            self.be = be
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
        result = np.all(self.A @ point <= self.b + 1e-12)  # Small tolerance
        
        # Check equality constraints if present
        if self.Ae is not None and self.be is not None:
            result = result and np.allclose(self.Ae @ point, self.be, atol=1e-12)
        
        return result
    
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
            
            # Handle equality constraints
            Ae_new = None
            be_new = None
            if self.Ae is not None or other.Ae is not None:
                Ae_list = []
                be_list = []
                if self.Ae is not None:
                    Ae_list.append(self.Ae)
                    be_list.append(self.be)
                if other.Ae is not None:
                    Ae_list.append(other.Ae)
                    be_list.append(other.be)
                if Ae_list:
                    Ae_new = np.vstack(Ae_list)
                    be_new = np.vstack(be_list)
            
            if Ae_new is not None:
                return Polytope(A_new, b_new, Ae_new, be_new)
            else:
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
            constraints_str = f"{self.A.shape[0]} constraints"
            if self.Ae is not None:
                constraints_str += f", {self.Ae.shape[0]} equalities"
            return f"Polytope: {constraints_str} in R^{self.dim()}"
        elif self.V is not None:
            return f"Polytope: {self.V.shape[1]} vertices in R^{self.dim()}"
        else:
            return f"Polytope: empty in R^{self.dim()}"
    
    def contains_(self, S, method='exact', tol=1e-12, maxEval=0, certToggle=True, scalingToggle=False):
        """Check if polytope contains another set or points"""
        from .contains_ import contains_
        return contains_(self, S, method, tol, maxEval, certToggle, scalingToggle)
    
    def dim(self) -> int:
        """Get dimension of polytope"""
        from .dim import dim
        return dim(self)
    
    def center(self):
        """Get center of polytope"""
        from .center import center
        return center(self)
    
    def isBounded(self) -> bool:
        """Check if polytope is bounded"""
        from .isBounded import isBounded
        return isBounded(self)
    
    def isemptyobject(self) -> bool:
        """Check if polytope is empty"""
        from .isemptyobject import isemptyobject
        return isemptyobject(self)
    
    def interval(self):
        """Enclose polytope by an interval"""
        from .interval import interval
        return interval(self)
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__() 