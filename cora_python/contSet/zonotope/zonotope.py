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
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
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
        c: center vector as column vector (n, 1) where n is dimension
        G: generator matrix (n, p) where n is dimension, p is number of generators
        precedence: precedence for set operations (110)
    
    Unified representation convention:
        All zonotope operations assume and maintain:
        - c: column vector shape (n, 1)
        - G: matrix shape (n, p)
        This matches MATLAB's representation and ensures consistency across operations.
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
        
        # Validate number of input arguments (MATLAB: assertNarginConstructor(1:2,nargin))
        from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
        assertNarginConstructor([1, 2], len(args))
        
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
                if Z.shape[1] == 0:
                    # Empty matrix case: create empty center and generators
                    c = np.zeros((Z.shape[0], 0))
                    G = np.zeros((Z.shape[0], 0))
                else:
                    c = Z[:, 0]
                    G = Z[:, 1:]
        elif len(args) == 2:
            c, G = args
        # Note: More than 2 arguments should be caught by assertNarginConstructor

        return np.asarray(c), np.asarray(G)
    
    def _check_input_args(self, c, G, n_in):
        """Check correctness of input arguments"""
        
        # Convert to numpy arrays if not already
        if not isinstance(c, np.ndarray):
            c = np.asarray(c)
        if not isinstance(G, np.ndarray):
            G = np.asarray(G)
        
        # Check for NaN values - use inputArgsCheck to match MATLAB behavior
        # MATLAB uses inputArgsCheck with 'nonnan' attribute which raises CORA:wrongValue
        from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
        if n_in == 1:
            # For single argument, check combined [c, G] (after parsing, c and G are already split)
            Z_combined = np.column_stack([c, G]) if c.size > 0 and G.size > 0 else (c if c.size > 0 else G)
            if Z_combined.size > 0:
                inputArgsCheck([[Z_combined, 'att', 'numeric', 'nonnan']])
        elif n_in == 2:
            # For two arguments, check separately
            # MATLAB: inputArgsCheck({ {c, 'att', {'numeric','gpuArray'}, 'nonnan'}; {G, 'att', {'numeric','gpuArray'}, 'nonnan'}; })
            if c.size > 0:
                inputArgsCheck([[c, 'att', 'numeric', 'nonnan']])
            if G.size > 0:
                inputArgsCheck([[G, 'att', 'numeric', 'nonnan']])
        
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
        """
        Compute and fix properties to ensure correct dimensions
        
        Unified representation convention (matching MATLAB):
        - c: center vector as column vector (n, 1) where n is dimension
        - G: generator matrix (n, p) where n is dimension, p is number of generators
        This ensures consistent representation across all zonotope operations.
        """
        # Match MATLAB: aux_computeProperties simply sets G = zeros(size(c,1),0) if G is empty
        
        # Ensure c is a column vector if it has data
        # Unified representation: c is always (n, 1) column vector
        if c.size > 0:
            c = c.reshape(-1, 1)
        
        # If G is empty, set correct dimension based on c
        # MATLAB: if isempty(G), G = zeros(size(c,1),0); end
        if G.size == 0:
            # Determine dimension from c
            if c.size > 0:
                # c has data, use its dimension
                dim = c.shape[0]
            elif len(c.shape) > 0 and c.shape[0] > 0:
                # c is empty but has correct dimension (zeros(n,0))
                dim = c.shape[0]
            elif len(G.shape) > 0 and G.shape[0] > 0:
                # G has dimension info, use it (shouldn't happen in practice)
                dim = G.shape[0]
                c = np.zeros((dim, 0))
            else:
                # Both are completely empty, error
                raise CORAerror('CORA:wrongInputInConstructor', 'Cannot determine dimension for empty zonotope')
            
            # Set G to match dimension
            G = np.zeros((dim, 0))
        
        return c, G
    
    def __str__(self) -> str:
        """String representation of zonotope"""
        if hasattr(self, 'display') and callable(self.display):
            return self.display()
        return f"Zonotope with center {self.c.flatten()} and {self.G.shape[1]} generators"
    
    def __repr__(self) -> str:
        """Representation of zonotope"""
        return self.__str__()
    
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
