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
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


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
            CORAerror: If no input arguments provided or invalid arguments
        """
        # Call parent constructor
        super().__init__()
        
        # Set precedence for zonotope
        self.precedence = 110
        
        # Avoid empty instantiation
        if len(args) == 0:
            raise CORAerror('CORA:noInputInSetConstructor',
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
        
        if len(args) == 1:
            # Check if it's an Interval object
            if hasattr(args[0], '__class__') and args[0].__class__.__name__ == 'Interval':
                # Convert interval to zonotope using interval's zonotope method
                from cora_python.contSet.interval.zonotope import zonotope
                Z_from_interval = zonotope(args[0])
                return Z_from_interval.c, Z_from_interval.G
            
            Z = np.asarray(args[0])
            if Z.ndim <= 1:
                # Input is a vector (point)
                c = Z
                G = np.array([])
            else:
                # Input is a matrix [c, G]
                c = Z[:, 0]
                G = Z[:, 1:]
        elif len(args) == 2:
            c, G = args
        else:
            # This case should ideally be caught earlier, but as a safeguard:
            raise CORAerror('CORA:wrongInputInConstructor', 'Invalid number of arguments for zonotope.')

        return np.asarray(c), np.asarray(G)
    
    def _check_input_args(self, c, G, n_in):
        """Check correctness of input arguments"""
        
        # Convert to numpy arrays if not already
        if not isinstance(c, np.ndarray):
            c = np.asarray(c)
        if not isinstance(G, np.ndarray):
            G = np.asarray(G)
        
        # Check for NaN values
        if c.size > 0 and np.any(np.isnan(c)):
            raise CORAerror('CORA:wrongInputInConstructor',
                          'Center contains NaN values')
        if G.size > 0 and np.any(np.isnan(G)):
            raise CORAerror('CORA:wrongInputInConstructor',
                          'Generator matrix contains NaN values')
        
        if n_in == 2:
            # Check dimensions
            if c.size == 0 and G.size > 0:
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Center is empty')
            elif c.size > 0 and c.ndim > 1 and min(c.shape) > 1:
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Center is not a vector')
            elif G.size > 0 and c.size > 0:
                c1d = np.atleast_1d(c)
                if len(c1d) != G.shape[0]:
                    raise CORAerror('CORA:wrongInputInConstructor',
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
    
    def __str__(self) -> str:
        """String representation of zonotope"""
        if hasattr(self, 'display') and callable(self.display):
            return self.display()
        return f"Zonotope with center {self.c.flatten()} and {self.G.shape[1]} generators"
    
    def __repr__(self) -> str:
        """Representation of zonotope"""
        return self.__str__()
    
    @property
    def center(self):
        """Center of the zonotope"""
        return self.c

    @property
    def generators(self):
        """Generators of the zonotope"""
        return self.G
    
    # Legacy properties with deprecation warnings (matching MATLAB behavior)
    @property
    def Z(self):
        """Legacy property: concatenated center and generator matrix [c, G]"""
        import warnings
        warnings.warn(
            "Property 'zonotope.Z' is deprecated since CORA v2024. "
            "Please use zonotope.c and zonotope.G instead. "
            "This change was made to be consistent with the notation in papers.",
            DeprecationWarning,
            stacklevel=2
        )
        return np.hstack([self.c, self.G])
    
    @Z.setter
    def Z(self, Z):
        """Legacy setter for Z property"""
        import warnings
        warnings.warn(
            "Property 'zonotope.Z' is deprecated since CORA v2024. "
            "Please use zonotope.c and zonotope.G instead. "
            "This change was made to be consistent with the notation in papers.",
            DeprecationWarning,
            stacklevel=2
        )
        if Z is not None and Z.size > 0:
            self.c = Z[:, 0:1]  # Keep as column vector
            self.G = Z[:, 1:]
    
    @property
    def halfspace(self):
        """Legacy property: halfspace representation (deprecated)"""
        import warnings
        warnings.warn(
            "Property 'zonotope.halfspace' is deprecated since CORA v2025. "
            "Please call polytope(obj) instead. "
            "This change was made to avoid code redundancy.",
            DeprecationWarning,
            stacklevel=2
        )
        return None  # Always returns None as per MATLAB implementation
    
    @halfspace.setter
    def halfspace(self, hs):
        """Legacy setter for halfspace property"""
        import warnings
        warnings.warn(
            "Property 'zonotope.halfspace' is deprecated since CORA v2025. "
            "Please use polytope objects instead. "
            "This change was made to avoid code redundancy.",
            DeprecationWarning,
            stacklevel=2
        )
        # Do nothing, just like MATLAB implementation
        pass
    
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
                # Handle multiplication with numpy arrays - element-wise
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

    def contains_(self, S, method='exact', tol=1e-12, maxEval=200, certToggle=True, scalingToggle=True, *args, **kwargs):
        """
        Instance method for containment check, matching MATLAB usage.
        Calls the module-level contains_ function.
        """
        from .contains_ import contains_ as zonotope_contains_
        return zonotope_contains_(self, S, method, tol, maxEval, certToggle, scalingToggle, *args) 

    def cubMap(self, *args):
        """
        Instance method for cubic multiplication (cubMap), matching MATLAB usage.
        Calls the module-level cubMap function.
        """
        from .cubMap import cubMap as zonotope_cubMap
        return zonotope_cubMap(self, *args)

    @staticmethod
    def empty(dim):
        """Return an empty zonotope of given dimension (no generators, center at 0)"""
        c = np.zeros((dim, 1))
        G = np.zeros((dim, 0))
        return Zonotope(c, G) 

    # Attach MATLAB-like isequal as a method
    def isequal(self, S, tol=None):
        from cora_python.contSet.zonotope.isequal import isequal as isequal_func
        return isequal_func(self, S, tol) 