"""
zonotope - object constructor for zonotope objects

Description:
    This class represents zonotopes objects defined as
    {c + ∑_{i=1}^p beta_i * g^(i) | beta_i ∈ [-1,1]}.

Syntax:
    obj = zonotope(c, G)
    obj = zonotope(Z)

Inputs:
    c - center vector
    G - generator matrix  
    Z - center vector and generator matrix Z = [c,G]

Outputs:
    obj - generated zonotope object

Example:
    c = np.array([1, 1])
    G = np.array([[1, 1, 1], [1, -1, 0]])
    Z = zonotope(c, G)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: polyZonotope, conZonotope, zonoBundle, conPolyZono

Authors:       Matthias Althoff, Niklas Kochdumper, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       14-September-2006 (MATLAB)
Last update:   05-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, Optional, Any
from ..contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class Zonotope(ContSet):
    """
    Zonotope class for representing zonotopic sets
    
    A zonotope Z ⊂ R^n is defined as:
    Z := {c + ∑_{i=1}^p β_i * g^(i) | β_i ∈ [-1,1]}
    
    where c ∈ R^n is the center and g^(i) ∈ R^n are the generators.
    
    Properties:
        c: center vector (numpy array)
        G: generator matrix (numpy array)
        precedence: precedence for set operations (110)
    """
    
    def __init__(self, *args):
        """
        Constructor for zonotope objects
        
        Args:
            *args: Variable arguments
                   - zonotope(c, G): center and generator matrix
                   - zonotope(Z): combined matrix [c, G]
                   - zonotope(other_zonotope): copy constructor
        
        Raises:
            CORAError: If no input arguments provided or invalid arguments
        """
        # Call parent constructor
        super().__init__()
        
        # Set precedence for zonotope
        self.precedence = 110
        
        # Avoid empty instantiation
        if len(args) == 0:
            raise CORAError('CORA:noInputInSetConstructor',
                          'No input arguments provided to zonotope constructor')
        
        # Copy constructor
        if len(args) == 1 and isinstance(args[0], Zonotope):
            other = args[0]
            self.c = other.c.copy() if other.c is not None else None
            self.G = other.G.copy() if other.G is not None else None
            return
        
        # Parse input arguments
        c, G = self._parse_input_args(*args)
        
        # Check correctness of input arguments
        self._check_input_args(c, G, len(args))
        
        # Compute properties
        c, G = self._compute_properties(c, G)
        
        # Assign properties
        self.c = c
        self.G = G
    
    def _parse_input_args(self, *args):
        """Parse input arguments from user and assign to variables"""
        
        if len(args) == 0:
            return np.zeros((0, 0)), np.zeros((0, 0))
        
        if len(args) == 1:
            # Check if input is an Interval object
            if hasattr(args[0], 'inf') and hasattr(args[0], 'sup'):
                # Convert interval to zonotope: c = center, G = diag(radius)
                interval_obj = args[0]
                
                # Get center and radius
                from ..interval.center import center
                from ..interval.rad import rad
                
                c = center(interval_obj)
                r = rad(interval_obj)
                
                # Create generator matrix as diagonal matrix of radii
                # Remove zero generators (where radius is 0)
                nonzero_indices = r != 0
                if np.any(nonzero_indices):
                    G = np.diag(r)[:, nonzero_indices]
                else:
                    G = np.zeros((len(c), 0))
                
                return c, G
            
            # Handle numeric array input
            Z = np.asarray(args[0])
            if Z.size == 0:
                return Z, np.array([]).reshape(Z.shape[0], 0)
            elif Z.ndim == 1:
                return Z, np.array([]).reshape(len(Z), 0)
            else:
                c = Z[:, 0]
                G = Z[:, 1:] if Z.shape[1] > 1 else np.array([]).reshape(Z.shape[0], 0)
                return c, G
        
        elif len(args) == 2:
            c = np.asarray(args[0]) if args[0] is not None else np.array([])
            G = np.asarray(args[1]) if args[1] is not None else np.array([])
            return c, G
        
        else:
            raise CORAError('CORA:wrongInputInConstructor',
                          'Too many input arguments')
    
    def _check_input_args(self, c, G, n_in):
        """Check correctness of input arguments"""
        
        # Convert to numpy arrays if not already
        if not isinstance(c, np.ndarray):
            c = np.asarray(c)
        if not isinstance(G, np.ndarray):
            G = np.asarray(G)
        
        # Check for NaN values
        if c.size > 0 and np.any(np.isnan(c)):
            raise CORAError('CORA:wrongInputInConstructor',
                          'Center contains NaN values')
        if G.size > 0 and np.any(np.isnan(G)):
            raise CORAError('CORA:wrongInputInConstructor',
                          'Generator matrix contains NaN values')
        
        if n_in == 2:
            # Check dimensions
            if c.size == 0 and G.size > 0:
                raise CORAError('CORA:wrongInputInConstructor',
                              'Center is empty')
            elif c.size > 0 and c.ndim > 1 and min(c.shape) > 1:
                raise CORAError('CORA:wrongInputInConstructor',
                              'Center is not a vector')
            elif G.size > 0 and c.size > 0 and len(c) != G.shape[0]:
                raise CORAError('CORA:wrongInputInConstructor',
                              'Dimension mismatch between center and generator matrix')
    
    def _compute_properties(self, c, G):
        """Compute and fix properties to ensure correct dimensions"""
        
        # Ensure c is a column vector
        if c.size > 0:
            c = c.reshape(-1, 1)
        
        # If G is empty, set correct dimension
        if G.size == 0 and c.size > 0:
            G = np.zeros((len(c), 0))
        elif G.size > 0:
            # Ensure G is 2D
            if G.ndim == 1:
                G = G.reshape(-1, 1)
        
        return c, G
    
    def dim(self) -> int:
        """Get dimension of the zonotope"""
        from .dim import dim
        return dim(self)
    
    def is_empty(self) -> bool:
        """Check if zonotope is empty"""
        from .isemptyobject import isemptyobject
        return isemptyobject(self)
    
    @staticmethod
    def empty(n: int = 0) -> 'Zonotope':
        """Create an empty zonotope of dimension n"""
        from .empty import empty
        return empty(n)
    
    @staticmethod
    def origin(n: int) -> 'Zonotope':
        """Create a zonotope representing the origin in dimension n"""
        from .origin import origin
        return origin(n)
    
    def display(self):
        """Display zonotope properties"""
        from .display import display
        return display(self)
    
    def randPoint_(self, N=1, type_='standard'):
        """Generate random points within the zonotope (internal version)"""
        from .randPoint_ import randPoint_
        return randPoint_(self, N, type_)
    
    def vertices_(self, method='convHull', *args):
        """Get vertices of the zonotope (internal version)"""
        from .vertices_ import vertices_
        return vertices_(self, method, *args)
    
    def project(self, dims):
        """Project zonotope to lower-dimensional subspace"""
        from .project import project
        return project(self, dims)
    
    def norm_(self, norm_type=2, mode='ub', return_vertex=False):
        """Compute maximum norm value of the zonotope"""
        from .norm_ import norm_
        return norm_(self, norm_type, mode, return_vertex)
    
    def zonotopeNorm(self, p, return_minimizer=False):
        """Compute zonotope norm of point p with respect to this zonotope"""
        from .zonotopeNorm import zonotopeNorm
        return zonotopeNorm(self, p, return_minimizer)
    
    def __repr__(self) -> str:
        """String representation of zonotope"""
        if self.is_empty():
            return f"zonotope (empty, dimension: {self.dim()})"
        else:
            return f"zonotope (dimension: {self.dim()}, generators: {self.G.shape[1] if self.G.size > 0 else 0})"
    
    def __str__(self) -> str:
        """String representation for display"""
        return self.__repr__()
    
    # Operator overloading
    def __add__(self, other):
        """Addition operator (+)"""
        from .plus import plus
        return plus(self, other)
    
    def __radd__(self, other):
        """Right addition operator (other + self)"""
        from .plus import plus
        return plus(other, self)
    
    def __mul__(self, other):
        """Multiplication operator (*)"""
        from .mtimes import mtimes
        return mtimes(self, other)
    
    def __rmul__(self, other):
        """Right multiplication operator (other * self)"""
        from .mtimes import mtimes
        return mtimes(other, self)
    
    def __matmul__(self, other):
        """Matrix multiplication operator (@)"""
        from .mtimes import mtimes
        return mtimes(self, other)
    
    def __rmatmul__(self, other):
        """Right matrix multiplication operator (other @ self)"""
        from .mtimes import mtimes
        return mtimes(other, self)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle numpy universal functions"""
        if ufunc == np.add:
            if method == '__call__':
                # Handle addition with numpy arrays
                if len(inputs) == 2:
                    if inputs[0] is self:
                        return self.__add__(inputs[1])
                    else:
                        return self.__radd__(inputs[0])
        elif ufunc == np.multiply:
            if method == '__call__':
                # Handle multiplication with numpy arrays
                if len(inputs) == 2:
                    if inputs[0] is self:
                        return self.__mul__(inputs[1])
                    else:
                        return self.__rmul__(inputs[0])
        elif ufunc == np.matmul:
            if method == '__call__':
                # Handle matrix multiplication with numpy arrays
                if len(inputs) == 2:
                    if inputs[0] is self:
                        return self.__matmul__(inputs[1])
                    else:
                        return self.__rmatmul__(inputs[0])
        
        # For other ufuncs, return NotImplemented to let numpy handle it
        return NotImplemented 