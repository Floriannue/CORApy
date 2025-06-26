"""
conZonotope - object constructor for constrained zonotopes [1]

Description:
    This class represents constrained zonotope objects defined as
    {c + G * beta | ||beta||_Inf <= 1, A * beta = b}.

Syntax:
    obj = conZonotope(c,G)
    obj = conZonotope(c,G,A,b)
    obj = conZonotope(Z)
    obj = conZonotope(Z,A,b)

Inputs:
    c - center vector of the zonotope
    G - generator matrix of the zonotope
    Z - matrix containing zonotope center and generators Z = [c,G]
    A - constraint matrix A*beta = b
    b - constraint vector A*beta = b

Outputs:
    obj - generated conZonotope object

Example: 
    c = [0;0]
    G = [3 0 1; 0 2 1]
    A = [1 0 1]; b = 1
    cZ = conZonotope(c,G,A,b)
    plot(cZ)

References:
    [1] Scott, Joseph K., et al. "Constrained zonotopes:
           A new tool for set-based estimation and fault detection."
           Automatica 69 (2016): 126-136.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: interval

Authors:       Dmitry Grebenyuk, Mark Wetzlinger, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       03-September-2017 (MATLAB)
Last update:   19-March-2021 (MW, error messages, MATLAB)
               14-December-2022 (TL, property check in inputArgsCheck, MATLAB)
               29-March-2023 (TL, optimized constructor, MATLAB)
               13-September-2023 (TL, replaced Z property with c and G, MATLAB)
               29-October-2024 (TL, A & b get converted to double, MATLAB)
Last revision: 02-May-2020 (MW, methods list, rewrite methods(hidden), add property validation, MATLAB)
               16-June-2023 (MW, restructure using auxiliary functions, MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple, Union, List, Optional

from cora_python.contSet.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.macros import CHECKS_ENABLED
from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning

if TYPE_CHECKING:
    pass


class ConZonotope(ContSet):
    """
    Class for representing constrained zonotopes
    
    Properties (SetAccess = {?contSet, ?matrixSet}, GetAccess = public):
        c: center vector - x = c + G*beta; |beta| <= 1
        G: generator matrix - x = c + G*beta; |beta| <= 1
        Z: legacy property [c,g_1,...,g_p] (deprecated)
        A: constraint matrix - A*beta = b; |beta| <= 1 (format: matrix)
        b: constraint vector - A*beta = b; |beta| <= 1 (format: column vector)
        ksi: the value of beta at vertices (format: column vector)
        R: [rho_l, rho_h] (A.3) (format: column vector)
    """
    
    def __init__(self, *varargin):
        """
        Class constructor for constrained zonotopes
        """
        # 0. avoid empty instantiation
        if len(varargin) == 0:
            raise CORAerror('CORA:noInputInSetConstructor')
        assertNarginConstructor(list(range(1, 5)), len(varargin))

        # 1. copy constructor
        if len(varargin) == 1 and isinstance(varargin[0], ConZonotope):
            # Direct assignment like MATLAB
            other = varargin[0]
            self.c = other.c
            self._G = other._G
            self.A = other.A
            self.b = other.b
            self.ksi = other.ksi if hasattr(other, 'ksi') else np.array([])
            self.R = other.R if hasattr(other, 'R') else np.array([])
            super().__init__()
            self.precedence = 90
            return

        # 2. parse input arguments: varargin -> vars
        c, G, A, b = _aux_parseInputArgs(*varargin)

        # 3. check correctness of input arguments
        _aux_checkInputArgs(c, G, A, b, len(varargin))

        # 4. compute properties
        c, G, A, b = _aux_computeProperties(c, G, A, b)

        # 5. assign properties
        self.c = c
        self._G = G  # Use private attribute for G with getter/setter
        self.A = A
        self.b = b
        
        # Initialize additional properties
        self.ksi = np.array([])  # the value of beta at vertices
        self.R = np.array([])    # [rho_l, rho_h] (A.3)

        # 6. set precedence (fixed) and initialize parent
        super().__init__()
        self.precedence = 90

    @property
    def G(self):
        """Generator matrix property with automatic dimension fixing"""
        return self._G

    @G.setter  
    def G(self, value):
        """Setter for G property - fix dimension if empty"""
        # MATLAB: if isempty(G), G = zeros(size(c,1),0); end
        if value is None or value.size == 0:
            if hasattr(self, 'c') and self.c.size > 0:
                value = np.zeros((self.c.shape[0], 0))
            else:
                value = np.array([])
        self._G = value

    @property
    def Z(self):
        """Legacy Z property getter - deprecated, use c and G instead"""
        CORAwarning('CORA:deprecated', 'property', 'conZonotope.Z', 'CORA v2024',
                   'Please use conZonotope.c and conZonotope.G instead.',
                   'This change was made to be consistent with the notation in papers.')
        if hasattr(self, 'c') and hasattr(self, '_G'):
            if self.c.size > 0 and self._G.size > 0:
                return np.column_stack([self.c, self._G])
            elif self.c.size > 0:
                return self.c.reshape(-1, 1)
        return np.array([])

    @Z.setter
    def Z(self, value):
        """Legacy Z property setter - deprecated, use c and G instead"""
        CORAwarning('CORA:deprecated', 'property', 'conZonotope.Z', 'CORA v2024',
                   'Please use conZonotope.c and conZonotope.G instead.',
                   'This change was made to be consistent with the notation in papers.')
        if value is not None and value.size > 0:
            # read out center and generators
            self.c = value[:, 0]
            if value.shape[1] > 1:
                self._G = value[:, 1:]
            else:
                self._G = np.zeros((value.shape[0], 0))
        else:
            self.c = np.array([])
            self._G = np.array([])

    def __repr__(self) -> str:
        """
        Official string representation of ConZonotope
        
        Returns:
            str: String representation showing center, generators, and constraints
        """
        if hasattr(self, 'c') and self.c.size > 0:
            c_str = np.array2string(self.c.flatten(), precision=4, suppress_small=True)
            if hasattr(self, 'G') and self.G.size > 0:
                G_str = np.array2string(self.G, precision=4, suppress_small=True)
                repr_str = f"ConZonotope(c={c_str}, G={G_str}"
            else:
                repr_str = f"ConZonotope(c={c_str}"
            
            if hasattr(self, 'A') and self.A.size > 0 and hasattr(self, 'b') and self.b.size > 0:
                A_str = np.array2string(self.A, precision=4, suppress_small=True)
                b_str = np.array2string(self.b.flatten(), precision=4, suppress_small=True)
                repr_str += f", A={A_str}, b={b_str}"
            
            repr_str += ")"
            return repr_str
        else:
            return "ConZonotope(empty)"


# Auxiliary functions -----------------------------------------------------

def _aux_parseInputArgs(*varargin) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse input arguments from user and assign to variables"""
    
    # set default values depending on nargin
    if len(varargin) == 1 or len(varargin) == 3:
        # only center given, or [c,G] with A and b
        c, A, b = setDefaultValues([[], [], []], list(varargin))
        if hasattr(varargin[0], 'shape') and len(varargin[0].shape) > 1 and varargin[0].shape[1] > 0:
            c_matrix = np.array(varargin[0])
            G = c_matrix[:, 1:]
            c = c_matrix[:, 0]
        else:
            G = np.array([])
    elif len(varargin) == 2 or len(varargin) == 4:
        # c,G or c,G,A,b given
        defaults = [[], [], [], []] if len(varargin) == 4 else [[], [], None, None]
        if len(varargin) == 2:
            c, G = setDefaultValues(defaults[:2], list(varargin))
            A, b = np.array([]), np.array([])
        else:
            c, G, A, b = setDefaultValues(defaults, list(varargin))
    else:
        c, G, A, b = np.array([]), np.array([]), np.array([]), np.array([])

    # Convert to numpy arrays
    c = np.array(c) if c is not None else np.array([])
    G = np.array(G) if G is not None else np.array([])
    A = np.array(A) if A is not None else np.array([])
    b = np.array(b) if b is not None else np.array([])

    return c, G, A, b


def _aux_checkInputArgs(c: np.ndarray, G: np.ndarray, A: np.ndarray, b: np.ndarray, n_in: int):
    """Check correctness of input arguments"""
    
    # only check if macro set to true
    if CHECKS_ENABLED and n_in > 0:

        if n_in == 1 or n_in == 3:
            inputChecks = [
                [c, 'att', 'numeric', ['finite']],
                [G, 'att', 'numeric', ['finite', 'matrix']]
            ]

        elif n_in == 2 or n_in == 4:
            # check whether c and G fit together to avoid bad message
            if c.size > 0 and not (c.ndim == 1 or (c.ndim == 2 and c.shape[1] == 1)):
                raise CORAerror('CORA:wrongInputInConstructor',
                              'The center has to be a column vector.')
            elif G.size > 0 and c.size > 0 and len(c) != G.shape[0]:
                raise CORAerror('CORA:wrongInputInConstructor',
                              'The dimensions of the center and the generator matrix do not match.')
            
            # Combine c and G for input checking following MATLAB format
            if c.size > 0 and G.size > 0:
                c_reshaped = c.reshape(-1, 1) if c.ndim == 1 else c
                combined = np.column_stack([c_reshaped, G])
                inputChecks = [
                    [combined, 'att', 'numeric', 'finite']
                ]
            elif c.size > 0:
                inputChecks = [
                    [c, 'att', 'numeric', 'finite']
                ]
            else:
                inputChecks = []

        # Add A and b checks (only if they have content)
        if A.size > 0:
            inputChecks.append([A, 'att', 'numeric', ['finite', 'matrix']])
        if b.size > 0:
            inputChecks.append([b, 'att', 'numeric', 'finite'])
        
        inputArgsCheck(inputChecks)

        # check correctness of A and b, also w.r.t G
        if A.size > 0 and b.size > 0:
            if not (b.ndim == 1 or (b.ndim == 2 and b.shape[1] == 1)):  # b is a vector?
                raise CORAerror('CORA:wrongInputInConstructor',
                              'The constraint offset has to be a vector.')
            elif G.size > 0 and A.shape[1] != G.shape[1]:  # A fits G?
                raise CORAerror('CORA:wrongInputInConstructor',
                              'The dimensions of the generator matrix and the constraint matrix do not match.')
            elif A.shape[0] != len(b):  # A fits b?
                raise CORAerror('CORA:wrongInputInConstructor',
                              'The dimensions of the constraint matrix and the constraint offset do not match.')


def _aux_computeProperties(c: np.ndarray, G: np.ndarray, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute properties"""
    
    # if G is empty, set correct dimension
    if G.size == 0 and c.size > 0:
        G = np.zeros((c.shape[0], 0))

    # if no constraints, set correct dimension
    if A.size == 0:
        A = np.zeros((0, G.shape[1] if G.size > 0 else 0))
        b = np.zeros((0, 1))

    # convert A,b to double for internal processing
    A = A.astype(float)
    b = b.astype(float)

    return c, G, A, b 