"""
capsule - object constructor for capsules

Description:
   This class represents capsule objects defined as
   C := L + S, L = {c + g*a | a ∈ [-1,1]}, S = {x | ||x||_2 <= r}.

Syntax:
   obj = capsule(c)
   obj = capsule(c,g)
   obj = capsule(c,g,r)

Inputs:
   c - center
   g - generator
   r - radius

Outputs:
   obj - capsule object

Example:
   c = [1;2];
   g = [2;1];
   r = 1;
   C = capsule(c,g,r);
   plot(C);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: interval, polytope

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 04-March-2019 (MATLAB)
Last update: 02-May-2020 (MW, add property validation)
             19-March-2021 (MW, error messages, remove capsule(r) case)
             14-December-2022 (TL, property check in inputArgsCheck)
Last revision: 16-June-2023 (MW, restructure using auxiliary functions)
Python translation: 2025
"""

import numpy as np
from typing import Union, Optional, Tuple, Any
from ..contSet import ContSet
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.contSet.emptySet.empty import empty as emptySet
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contSet.ellipsoid import Ellipsoid # Assuming ellipsoid is implemented
from cora_python.contSet.capsule.representsa_ import representsa_ # Import the new function
from cora_python.contSet.capsule.isemptyobject import isemptyobject # Import the new function
from cora_python.contSet.capsule.empty import empty # Import the empty method
from cora_python.contSet.capsule.display import display # Import the display method


class Capsule(ContSet):
    """
    Capsule class representing capsule objects
    
    A capsule is defined as C := L + S, where:
    - L = {c + g*a | a ∈ [-1,1]} (line segment)
    - S = {x | ||x||_2 <= r} (ball)
    
    Attributes:
        c: center vector
        g: generator vector  
        r: radius scalar
    """
    
    def __init__(self, *args, **kwargs):
        """
        Constructor for Capsule objects
        
        Args:
            *args: Variable arguments - can be (c), (c,g), or (c,g,r)
            **kwargs: Keyword arguments
        """
        # 0. avoid empty instantiation
        if len(args) == 0:
            raise ValueError("Capsule constructor requires at least one argument")
        
        # 1. copy constructor
        if len(args) == 1 and isinstance(args[0], Capsule):
            other = args[0]
            self.c = other.c.copy() if other.c is not None else None
            self.g = other.g.copy() if other.g is not None else None
            self.r = other.r
            super().__init__()
            return
        
        # 2. parse input arguments: args -> vars
        c, g, r = self._parse_input_args(*args)
        
        # 3. check correctness of input arguments
        self._check_input_args(c, g, r)
        
        # 4. compute properties
        c, g, r = self._compute_properties(c, g, r)
        
        # 5. assign properties
        self.c = c
        self.g = g
        self.r = r
        
        # Initialize parent class
        super().__init__()
        
        # 6. set precedence (fixed)
        self.precedence = 60
    
    def _parse_input_args(self, *args) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Parse input arguments from user and assign to variables
        
        Args:
            *args: Variable arguments
            
        Returns:
            Tuple of (c, g, r)
        """
        # Default values
        c, g, r = None, None, None
        
        if len(args) >= 1:
            c = args[0]
        if len(args) >= 2:
            g = args[1]
        if len(args) >= 3:
            r_raw = args[2]
            # Ensure r is a scalar float
            if isinstance(r_raw, (np.ndarray, list, tuple)):
                if np.size(r_raw) == 1:
                    r = float(np.asarray(r_raw).item())
                elif np.size(r_raw) == 0:
                    r = 0.0  # Treat empty array as radius 0
                else:
                    raise ValueError("Radius must be a scalar.")
            else:
                r = float(r_raw)
        
        return c, g, r
    
    def _check_input_args(self, c, g, r) -> None:
        """
        Check correctness of input arguments
        
        Args:
            c: center
            g: generator
            r: radius
        """
        # Convert inputs to numpy arrays if needed
        if c is not None:
            c = np.asarray(c, dtype=float)
            if c.ndim == 1:
                c = c.reshape(-1, 1)
        
        if g is not None:
            g = np.asarray(g, dtype=float)
            if g.ndim == 1:
                g = g.reshape(-1, 1)
        
        if r is not None:
            r = float(r)
            if r < 0:
                raise ValueError("Radius must be non-negative")
        
        # Check dimension compatibility
        if c is not None and g is not None:
            if c.shape != g.shape:
                raise ValueError("Dimension of center and generator do not match")
    
    def _compute_properties(self, c, g, r) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute properties
        
        Args:
            c: center
            g: generator
            r: radius
            
        Returns:
            Tuple of (c, g, r) with proper defaults
        """
        # Convert to numpy arrays
        if c is not None:
            c = np.asarray(c, dtype=float)
            if c.ndim == 1:
                c = c.reshape(-1, 1)
        
        if g is None:
            if c is not None:
                g = np.zeros((c.shape[0], 1))
            else:
                g = np.zeros((0, 1))
        else:
            g = np.asarray(g, dtype=float)
            if g.ndim == 1:
                g = g.reshape(-1, 1)
        
        if r is None:
            r = 0.0
        else:
            r = float(r)
        
        return c, g, r
    
    def dim(self) -> int:
        """
        Get dimension of the capsule
        
        Returns:
            Dimension of the capsule
        """
        if self.c is not None:
            return self.c.shape[0]
        return 0
    
    def is_empty(self) -> bool:
        """Check if capsule is empty"""
        return isemptyobject(self)
    
    def __str__(self) -> str:
        """String representation of the capsule"""
        return f"capsule (dimension: {self.dim()})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"Capsule(c={self.c.flatten() if self.c is not None else None}, " \
               f"g={self.g.flatten() if self.g is not None else None}, r={self.r})"

    # Assign the function to the class
    representsa_ = representsa_
    isemptyobject = isemptyobject
    
    @staticmethod
    def empty(n: int = 0):
        """Create an empty capsule"""
        return empty(n)
    
    def display(self):
        """Display capsule properties"""
        return display(self) 