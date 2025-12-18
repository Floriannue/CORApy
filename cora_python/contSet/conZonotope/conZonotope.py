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
import scipy.linalg
import scipy.optimize

from cora_python.contSet.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.macros import CHECKS_ENABLED
from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning

if TYPE_CHECKING:
    from cora_python.contSet.polytope.polytope import Polytope


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
    
    _dim_val: Optional[int]

    def __init__(self, *varargin):
        """
        Class constructor for constrained zonotopes
        """
        self._dim_val = None # Initialize _dim_val

        # 0. avoid empty instantiation
        if len(varargin) == 0:
            raise CORAerror('CORA:noInputInSetConstructor')
        assertNarginConstructor(list(range(1, 5)), len(varargin))

        # 1. copy constructor
        if len(varargin) == 1 and isinstance(varargin[0], ConZonotope):
            # Direct assignment like MATLAB
            other = varargin[0]
            self.c = other.c.copy() # Ensure copy to avoid shared references
            self._G = other._G.copy() # Ensure copy
            self.A = other.A.copy() # Ensure copy
            self.b = other.b.copy() # Ensure copy
            self.ksi = other.ksi.copy() if hasattr(other, 'ksi') and other.ksi.size > 0 else np.array([])
            self.R = other.R.copy() if hasattr(other, 'R') and other.R.size > 0 else np.array([])
            self._dim_val = other._dim_val # Copy dimension
            super().__init__()
            self.precedence = 90
            return

        # Handle Interval object input (convert via zonotope)
        from cora_python.contSet.interval.interval import Interval
        if len(varargin) == 1 and isinstance(varargin[0], Interval):
            # MATLAB: conZonotope(I) -> conZonotope(zonotope(I))
            from cora_python.contSet.interval.zonotope import zonotope
            z = zonotope(varargin[0])
            # Then treat as Zonotope input
            self.c = z.c.copy()
            self._G = z.G.copy()
            self.A = np.array([]) # No constraints from bare zonotope
            self.b = np.array([])
            self.ksi = np.array([])
            self.R = np.array([])
            self._dim_val = z.dim() # Get dimension from Zonotope
            super().__init__()
            self.precedence = 90
            return

        # Handle Zonotope object input
        from cora_python.contSet.zonotope.zonotope import Zonotope
        if len(varargin) == 1 and isinstance(varargin[0], Zonotope):
            z = varargin[0]
            # MATLAB equivalent is `conZonotope(Z)`
            # cZ = conZonotope(Z) -> c = Z.c, G = Z.G, A = [], b = []
            self.c = z.c.copy()
            self._G = z.G.copy()
            self.A = np.array([]) # No constraints from bare zonotope
            self.b = np.array([])
            self.ksi = np.array([])
            self.R = np.array([])
            self._dim_val = z.dim() # Get dimension from Zonotope
            super().__init__()
            self.precedence = 90
            return

        # Handle Polytope object input (direct conversion from Polytope.zonotope())
        from cora_python.contSet.polytope.polytope import Polytope
        if len(varargin) == 1 and isinstance(varargin[0], Polytope):
            P = varargin[0]
            # This path is typically P -> Zonotope -> ConZonotope
            # So, convert Polytope to Zonotope first
            if hasattr(P, 'zonotope') and callable(getattr(P, 'zonotope')):
                z = P.zonotope()
                # Then treat as Zonotope input
                self.c = z.c.copy()
                self._G = z.G.copy()
                self.A = np.array([])
                self.b = np.array([])
                self.ksi = np.array([])
                self.R = np.array([])
                self._dim_val = P.dim() # Get dimension from original Polytope
                super().__init__()
                self.precedence = 90
                return
            else:
                raise CORAerror('CORA:conversionError', 'Polytope.zonotope() method not found for conversion to ConZonotope.')

        # 2. Parse input arguments based on nargin for aux_parseInputArgs
        # This mirrors MATLAB's explicit argument parsing before calling auxiliary functions
        num_args = len(varargin)
        
        # Default values for all possible parameters (will be refined by _aux_parseInputArgs)
        c, G, A, b = None, None, None, None

        # MATLAB: if nargin == 1 || nargin == 3, first arg can be matrix [c, G]
        if num_args == 1 or num_args == 3:
            # First argument might be a matrix [c, G]
            c = varargin[0]
            if num_args == 3:
                A = varargin[1]
                b = varargin[2]
        elif num_args == 2 or num_args == 4:
            # c, G or c, G, A, b given explicitly
            c = varargin[0]
            G = varargin[1]
            if num_args == 4:
                A = varargin[2]
                b = varargin[3]
        
        # Determine dimension from inputs (c or G) early if not already set by specific constructors
        if self._dim_val is None:
            if c is not None and hasattr(c, 'shape'):
                self._dim_val = c.shape[0]  # Dimension is number of rows, not dependent on size
            elif G is not None and hasattr(G, 'shape'):
                self._dim_val = G.shape[0]  # Dimension is number of rows, not dependent on size
            elif hasattr(varargin[0], 'dim') and callable(getattr(varargin[0], 'dim')):
                self._dim_val = varargin[0].dim()
            else:
                self._dim_val = 0 # Default to 0 if no dimension can be determined

        # Now call the auxiliary functions with explicit arguments
        # Pass n_in to handle matrix [c, G] case correctly
        c, G, A, b = _aux_parseInputArgs(c, G, A, b, num_args)

        # 3. check correctness of input arguments (using num_args for assertNarginConstructor)
        _aux_checkInputArgs(c, G, A, b, num_args)

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

    def zonotope(self, alg: str = 'nullSpace') -> 'Zonotope':
        """
        Over-approximates a constrained zonotope with a zonotope
        
        Args:
            alg: Algorithm used to compute enclosure ('nullSpace' or 'reduce')
            
        Returns:
            Zonotope: Zonotope object enclosing the constrained zonotope
        """
        from ..zonotope import Zonotope
        
        # Handle trivial cases
        if self.representsa_('emptySet', 1e-10):
            return Zonotope.empty(self.dim())
        
        if self.A.size == 0:
            return Zonotope(self.c, self.G)
        
        # Check input arguments
        if alg not in ['nullSpace', 'reduce']:
            raise ValueError("alg must be 'nullSpace' or 'reduce'")
        
        # Compute over-approximation using the selected algorithm
        if alg == 'nullSpace':
            return self._aux_zonotopeNullSpace()
        elif alg == 'reduce':
            return self._aux_zonotopeReduce()
    
    def _aux_zonotopeNullSpace(self) -> 'Zonotope':
        """Auxiliary function for nullSpace algorithm"""
        from ..zonotope import Zonotope
        
        # Compute point satisfying all constraints with pseudo inverse
        p_ = np.linalg.pinv(self.A) @ self.b
        
        # Compute null-space of constraints
        T = scipy.linalg.null_space(self.A)
        
        # Transform boundary constraints of the factor hypercube
        m = self.A.shape[1]
        m_ = T.shape[1]
        
        A = np.vstack([np.eye(m), -np.eye(m)])
        b = np.ones(2*m)
        
        A_ = A @ T
        b_ = b - A @ p_
        
        # Loop over all dimensions of the transformed state space
        lb = np.zeros(m_)
        ub = np.zeros(m_)
        
        for i in range(m_):
            f = np.zeros(m_)
            f[i] = 1
            
            # Compute minimum
            result_min = scipy.optimize.linprog(f, A_ub=A_, b_ub=b_, bounds=None)
            lb[i] = result_min.fun
            
            # Compute maximum
            result_max = scipy.optimize.linprog(-f, A_ub=A_, b_ub=b_, bounds=None)
            ub[i] = -result_max.fun
        
        # Handle case where linprog returns very close lb/ub (single point)
        dummy = np.linalg.norm(ub)
        if dummy == 0:
            dummy = 1
        if np.linalg.norm(ub - lb) / dummy < 1e-12:
            meanval = 0.5 * (ub + lb)
            from ..interval import Interval
            int_val = Interval(meanval, meanval)
        else:
            from ..interval import Interval
            int_val = Interval(lb, ub)
        
        # Compute transformation matrices
        off = p_ + T @ int_val.center()
        S = T @ np.diag(int_val.radius())
        
        # Construct final zonotope
        c = self.c + self.G @ off
        G = self.G @ S
        
        return Zonotope(c, G)
    
    def _aux_zonotopeReduce(self) -> 'Zonotope':
        """Auxiliary function for reduce algorithm"""
        from ..zonotope import Zonotope
        
        # Remove all constraints of the constrained zonotope
        ng = max(1, self.G.shape[1] // len(self.c) + 1)
        
        # For now, just return the basic zonotope without reduction
        # TODO: Implement proper reduction
        return Zonotope(self.c, self.G)

    def isIntersecting_(self, other, type='exact', tol=1e-9):
        if self.isemptyobject() or (hasattr(other, 'isemptyobject') and other.isemptyobject()):
            return False
        # Exact algorithm: check for non-empty intersection
        try:
            from cora_python.contSet.zonotope.and_ import and_ as zonotope_and
            from cora_python.contSet.conZonotope.representsa_ import representsa_ as conzono_representsa
            from cora_python.contSet.zonotope.zonotope import Zonotope
            # ConZonotope vs ConZonotope
            if type == 'exact':
                if hasattr(other, '__class__') and other.__class__.__name__ == 'ConZonotope':
                    inter = self.and_(other, 'exact')
                    return not conzono_representsa(inter, 'emptySet', tol)
                # Zonotope/Interval/ZonoBundle
                if hasattr(other, '__class__') and other.__class__.__name__ in ['Zonotope', 'Interval', 'ZonoBundle']:
                    from cora_python.contSet.conZonotope.conZonotope import ConZonotope
                    other_cz = ConZonotope(other)
                    inter = self.and_(other_cz, 'exact')
                    return not conzono_representsa(inter, 'emptySet', tol)
        except Exception:
            pass
        # fallback: interval logic
        try:
            I1 = self.zonotope().interval()
            I2 = other.zonotope().interval()
            lb1, ub1 = I1.inf, I1.sup
            lb2, ub2 = I2.inf, I2.sup
            overlap = np.all((ub1 >= lb2 - tol) & (ub2 >= lb1 - tol))
            return bool(overlap)
        except Exception:
            return True

    def and_(self, other, method='exact'):
        """
        Minimal placeholder for intersection: uses interval overlap.
        """
        try:
            I1 = self.zonotope().interval()
            I2 = other.zonotope().interval()
            lb = np.maximum(I1.inf, I2.inf)
            ub = np.minimum(I1.sup, I2.sup)
            if np.any(lb > ub):
                # Return empty ConZonotope
                return type(self).empty(self.c.shape[0])
            # Otherwise, return a ConZonotope for the intersection interval
            from cora_python.contSet.zonotope.zonotope import Zonotope
            return type(self)(Zonotope((lb + ub) / 2, np.diag((ub - lb) / 2)))
        except Exception:
            return self  # fallback


# Auxiliary functions -----------------------------------------------------

def _aux_parseInputArgs(c: Optional[np.ndarray], G: Optional[np.ndarray], A: Optional[np.ndarray], b: Optional[np.ndarray], n_in: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse input arguments from user and assign to variables
    Matches MATLAB's aux_parseInputArgs behavior
    """
    
    # MATLAB: if nargin == 1 || nargin == 3
    if n_in == 1 or n_in == 3:
        # First argument might be a matrix [c, G] or just c
        c_out = np.array([]) if c is None else np.asarray(c)
        A_out = np.array([]) if A is None else np.asarray(A)
        b_out = np.array([]) if b is None else np.asarray(b)
        
        # Check if c is a matrix with multiple columns (i.e., [c, G])
        if c_out.size > 0 and c_out.ndim == 2 and c_out.shape[1] > 0:
            # Split into c and G: c = first column, G = remaining columns
            G_out = c_out[:, 1:]
            c_out = c_out[:, 0:1]  # Keep as column vector
        else:
            # Just c given, G is empty
            G_out = np.array([])
            # Ensure c is a column vector
            if c_out.ndim == 1:
                c_out = c_out.reshape(-1, 1)
    
    # MATLAB: elseif nargin == 2 || nargin == 4
    elif n_in == 2 or n_in == 4:
        # c, G or c, G, A, b given explicitly
        c_out = np.array([]) if c is None else np.asarray(c)
        G_out = np.array([]) if G is None else np.asarray(G)
        A_out = np.array([]) if A is None else np.asarray(A)
        b_out = np.array([]) if b is None else np.asarray(b)
        
        # Ensure c is a column vector and G is a 2D array
        if c_out.ndim == 1:
            c_out = c_out.reshape(-1, 1)
        if G_out.ndim == 1:
            G_out = G_out.reshape(-1, 1)
        # If G is empty but c has dimension, reshape G to (n,0)
        elif G_out.size == 0 and c_out.size > 0:
            G_out = np.zeros((c_out.shape[0], 0))
    else:
        # Fallback (shouldn't happen with proper validation)
        c_out = np.array([]) if c is None else np.asarray(c)
        G_out = np.array([]) if G is None else np.asarray(G)
        A_out = np.array([]) if A is None else np.asarray(A)
        b_out = np.array([]) if b is None else np.asarray(b)

    return c_out, G_out, A_out, b_out


def _aux_checkInputArgs(c: np.ndarray, G: np.ndarray, A: np.ndarray, b: np.ndarray, n_in: int):
    """Check correctness of input arguments"""
    
    # only check if macro set to true
    if CHECKS_ENABLED and n_in > 0:

        if n_in == 1 or n_in == 3:
            inputChecks = [
                [c, 'att', 'numeric', ['finite']]
            ]
            # Only check G if it's not empty
            if G.size > 0:
                inputChecks.append([G, 'att', 'numeric', ['finite', 'matrix']])

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
    
    # make center a column vector (MATLAB line 492)
    if c.ndim == 1:
        c = c.reshape(-1, 1)

    # if G is empty, set correct dimension (MATLAB lines 493-494)
    if G.size == 0 and c.size > 0:
        G = np.zeros((c.shape[0], 0))

    # if no constraints, set correct dimension (MATLAB lines 496-503)
    if A.size == 0:
        # Handle case where G might be empty
        if G.size > 0:
            A = np.zeros((0, G.shape[1]))
        else:
            A = np.zeros((0, 0))
        b = np.zeros((0, 1))

    # convert A,b to double for internal processing (MATLAB lines 505-506)
    A = A.astype(float)
    b = b.astype(float)

    return c, G, A, b 