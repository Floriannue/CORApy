"""
spectraShadow - object constructor for spectraShadow objects

Description:
    This class represents spectrahedral shadow objects defined as
    SpS = {Gx+c|A0 + x1*A1 + ... + xm*Am >= 0}
      (projection representation)
    or, equivalently,
    SpS = {y|∃z, A0 + y1*B1 + ... + ym*Bm + z1*C1 + ... + zl*Cl>=0}
      (existential sum representation)
    where the Ai, Bi, Ci are all real, symmetric matrices, and the ">=0"
    means that the preceding matrix is positive semi-definite.

Syntax:
    SpS = spectraShadow(A)
    SpS = spectraShadow(A,c)
    SpS = spectraShadow(A,c,G)
    SpS = spectraShadow(ESumRep)

Inputs:
    A - coefficient matrices, i.e., A = [A0 A1 ... Am]
    c - center vector
    G - generator matrix
    ESumRep - Cell array with two elements, containing the existential sum
       representation of the spectrahedral shadow, i.e.,
       ESumRep = {[B0 B1 ... Bm] [C1 ... Cl]}

Outputs:
    obj - generated spectraShadow object

Example: 
    % 2 dimensional box with radius 3 around point [-1;2]:
    A0 = eye(3)
    A1 = [0 1 0;1 0 0;0 0 0]
    A2 = [0 0 1;0 0 0;1 0 0]
    SpS = spectraShadow([A0,A1,A2],[-1;2],3*eye(2))

References:
    [1] T. Netzer. "Spectrahedra and Their Shadows", 2011
 
Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Maximilian Perschl, Adrian Kulmburg (MATLAB)
               Python translation by AI Assistant
Written:       18-April-2023 (MATLAB)
Last update:   01-August-2023 (AK, restructured arguments, MATLAB)
Last revision: ---
Python translation: 2025
"""

import numpy as np
from scipy import sparse
from typing import TYPE_CHECKING, Tuple, Union, List, Optional

from cora_python.contSet.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.macros import CHECKS_ENABLED

if TYPE_CHECKING:
    pass


class SetProperty:
    """Helper class to mimic MATLAB's setproperty functionality"""
    def __init__(self):
        self.val = None


class SpectraShadow(ContSet):
    """
    Class for representing spectrahedral shadow objects
    
    Properties (SetAccess = private, GetAccess = public):
        A: spectrahedron coefficient matrices A = [A0,A1,...,Ad]
           m+1 matrices of dimension k x k => k x k*(m+1)
        G: generator matrix (n x m matrix)
        c: center vector (n x 1 vector)
        
    Properties (SetAccess = protected, GetAccess = public):
        emptySet: emptiness
        fullDim: full-dimensionality
        bounded: boundedness
        center: center of the 'base' spectrahedron
        ESumRep: Existential Sum representation
    """
    
    def __init__(self, *varargin):
        """
        Class constructor for spectrahedral shadow objects
        """
        # 0. avoid empty instantiation
        if len(varargin) == 0:
            raise CORAerror('CORA:noInputInSetConstructor')
        assertNarginConstructor(list(range(1, 5)), len(varargin))
        
        # 0. init hidden properties
        self.emptySet = SetProperty()
        self.fullDim = SetProperty()
        self.bounded = SetProperty()
        self.center = SetProperty()
        self.ESumRep = SetProperty()
        
        # 1. copy constructor
        if len(varargin) == 1 and isinstance(varargin[0], SpectraShadow):
            # MATLAB: obj = varargin{1}; return (direct assignment)
            # In Python, we copy arrays to avoid shared references
            SpS = varargin[0]
            # Copy sparse matrices/arrays
            self.A = SpS.A.copy() if hasattr(SpS.A, 'copy') else SpS.A
            self.c = SpS.c.copy() if hasattr(SpS.c, 'copy') else SpS.c
            self.G = SpS.G.copy() if hasattr(SpS.G, 'copy') else SpS.G
            
            # Copy SetProperty values (shallow copy is fine for these)
            self.ESumRep.val = SpS.ESumRep.val
            self.emptySet.val = SpS.emptySet.val
            self.fullDim.val = SpS.fullDim.val
            self.bounded.val = SpS.bounded.val
            self.center.val = SpS.center.val
            super().__init__()
            self.precedence = SpS.precedence
            return

        # Handle Interval object input (convert using interval.spectraShadow)
        if len(varargin) == 1:
            from cora_python.contSet.interval.interval import Interval
            if isinstance(varargin[0], Interval):
                # MATLAB: spectraShadow(I) converts directly using interval bounds
                # Use the interval.spectraShadow conversion function
                from cora_python.contSet.interval.spectraShadow import spectraShadow
                result = spectraShadow(varargin[0])
                # Copy all attributes from result
                self.A = result.A
                self.c = result.c
                self.G = result.G
                self.ESumRep = result.ESumRep
                self.emptySet = result.emptySet
                self.fullDim = result.fullDim
                self.bounded = result.bounded
                self.center = result.center
                super().__init__()
                self.precedence = result.precedence
                return

        # 2. parse input arguments: varargin -> vars
        A, c, G, ESumRep = _aux_parseInputArgs(*varargin)
        # Note that at this stage, A may be a cell containing ESumRep

        # 3. check correctness of input arguments
        _aux_checkInputArgs(A, c, G, ESumRep, len(varargin))
        
        # 4. compute properties and hidden properties
        A, c, G, ESumRep = _aux_computeProperties(A, c, G, ESumRep, len(varargin))

        # 5. assign properties
        # Ensure arrays are 2D before converting to sparse matrices
        # Ensure A is a numpy array (should be after _aux_parseInputArgs, but check to be safe)
        if isinstance(A, list):
            if all(isinstance(x, np.ndarray) for x in A):
                A = np.hstack(A) if len(A) > 0 else np.array([0])
            else:
                A = np.array(A) if len(A) > 0 else np.array([0])
        
        if not sparse.issparse(A):
            if A.ndim == 0:
                A = np.array([[A.item()]])
            elif A.ndim == 1:
                A = A.reshape(1, -1)
            elif A.ndim > 2:
                raise ValueError(f"Cannot convert {A.ndim}D array to sparse matrix. A must be 2D.")
            self.A = sparse.csr_matrix(A)
        else:
            self.A = A
            
        if not sparse.issparse(c):
            # Ensure c is a column vector (2D)
            if c.ndim == 0:
                c = np.array([[c.item()]])
            elif c.ndim == 1:
                c = c.reshape(-1, 1)
            elif c.ndim > 2:
                raise ValueError(f"Cannot convert {c.ndim}D array to sparse matrix. c must be 2D.")
            self.c = sparse.csr_matrix(c)
        else:
            self.c = c
            
        if not sparse.issparse(G):
            if G.ndim == 0:
                G = np.array([[G.item()]])
            elif G.ndim == 1:
                G = G.reshape(1, -1)
            elif G.ndim > 2:
                raise ValueError(f"Cannot convert {G.ndim}D array to sparse matrix. G must be 2D.")
            self.G = sparse.csr_matrix(G)
        else:
            self.G = G
            
        self.ESumRep.val = ESumRep

        # 6. set precedence (fixed) and initialize parent
        super().__init__()
        self.precedence = 40

    def __repr__(self) -> str:
        """String representation of the SpectraShadow object"""
        return self.display()


# Auxiliary functions -----------------------------------------------------

def _aux_parseInputArgs(*varargin) -> Tuple[Union[np.ndarray, List], np.ndarray, np.ndarray, Optional[List]]:
    """Parse input arguments from user and assign to variables"""
    
    # Check if first argument is a list BEFORE setDefaultValues to avoid flattening
    # In MATLAB: if nargin == 1 && iscell(A)
    # ESumRep is a cell array with exactly 2 elements (matrices)
    if len(varargin) == 1 and isinstance(varargin[0], list):
        A_raw = varargin[0]
        # Check if it's ESumRep: exactly 2 elements, both are 2D numpy arrays
        if (len(A_raw) == 2 and 
            isinstance(A_raw[0], np.ndarray) and isinstance(A_raw[1], np.ndarray) and
            A_raw[0].ndim == 2 and A_raw[1].ndim == 2):
            # This is ESumRep case
            ESumRep = A_raw
            A = 0
            c = []
            G = []
        elif len(A_raw) == 1:
            # List with 1 element - this is an incorrect ESumRep structure
            # In MATLAB, a cell array with 1 element is treated as incorrect ESumRep
            # We'll set ESumRep to trigger the error check in _aux_checkInputArgs
            ESumRep = A_raw
            A = 0
            c = []
            G = []
        else:
            # This is a list of matrices for A - concatenate them (matches MATLAB behavior)
            # In MATLAB, [A0, A1, A2] horizontally concatenates
            if all(isinstance(x, np.ndarray) for x in A_raw):
                A = np.hstack(A_raw) if len(A_raw) > 0 else np.array([0])
            else:
                # Convert list to numpy array (shouldn't happen in normal usage, but handle it)
                A = np.array(A_raw) if len(A_raw) > 0 else np.array([0])
            ESumRep = None
            c = []
            G = []
    else:
        # Set default values (normal case)
        defaults = setDefaultValues([0, [], []], list(varargin))
        A, c, G = defaults
        ESumRep = None

    # Convert to numpy arrays where appropriate
    # Ensure A is always a numpy array (not a list)
    if isinstance(A, list):
        # If A is still a list, try to concatenate if all elements are arrays
        if all(isinstance(x, np.ndarray) for x in A):
            A = np.hstack(A) if len(A) > 0 else np.array([0])
        else:
            A = np.array(A) if len(A) > 0 else np.array([0])
    elif not sparse.issparse(A):
        A = np.array(A) if A is not None else np.array([0])
    c = np.array(c) if c is not None else np.array([])
    G = np.array(G) if G is not None else np.array([])

    return A, c, G, ESumRep


def _aux_checkInputArgs(A: Union[np.ndarray, List], c: np.ndarray, G: np.ndarray, ESumRep: Optional[List], n_in: int):
    """Check correctness of input arguments"""

    if CHECKS_ENABLED:
        # First, we need to check whether the first argument is a cell; if it
        # is, we just set ESumRep and stop right there. Otherwise, we continue
        # with the 'usual' set up of the spectrahedron
        if n_in == 1 and ESumRep is not None:
            # To construct the corresponding A,c,G, we need to check that
            # ESumRep does indeed consist of matrices

            if CHECKS_ENABLED:

                if len(ESumRep) != 2:
                    # (That was a little hack to make sure we have a (2,1) or
                    # (1,2) cell array)
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'The cell array for ESumRep is not a (2,1) or (1,2) array.')

                # Check if elements are numpy arrays (matrices)
                # In MATLAB: ismatrix(ESumRep{1}) checks if it's a 2D array
                if not isinstance(ESumRep[0], np.ndarray) or ESumRep[0].ndim != 2:
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'One of the entries of ESumRep is not a matrix.')
                if not isinstance(ESumRep[1], np.ndarray) or ESumRep[1].ndim != 2:
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'One of the entries of ESumRep is not a matrix.')

                if (ESumRep[0].shape[0] != ESumRep[1].shape[0] and 
                    ESumRep[0].size > 0 and ESumRep[1].size > 0):
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'The dimensions of the two matrices in ESumRep do not match.')

        # For all other cases (n_in > 0 and not ESumRep case), check dimensions
        if n_in > 0 and not (n_in == 1 and ESumRep is not None):
            from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
            
            # Checking types (matches MATLAB lines 289-293)
            inputArgsCheck([
                (A, 'att', 'numeric', ['nonnan', 'nonempty']),
                (c, 'att', 'numeric', 'nonnan'),
                (G, 'att', 'numeric', 'nonnan')
            ])
            
            # Check that A is a matrix (dense or sparse)
            from scipy import sparse
            is_sparse_matrix = sparse.issparse(A) and A.ndim == 2
            is_dense_matrix = isinstance(A, np.ndarray) and A.ndim == 2
            if not (is_sparse_matrix or is_dense_matrix):
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Coefficient matrix A is not a matrix.')
            # Check that G is a matrix
            if G.size > 0 and (not isinstance(G, np.ndarray) or G.ndim != 2):
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Generator matrix G is not a matrix.')
            # Check that c is a vector
            if c.size > 0 and not (c.ndim == 1 or (c.ndim == 2 and (c.shape[0] == 1 or c.shape[1] == 1))):
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Center vector c is not a vector.')
            
            # Check that A is not empty
            if A.size == 0:
                raise CORAerror('CORA:wrongInputInConstructor',
                              'The matrix A is empty. If you are trying to create an empty or a fullspace spectrahedral shadow, please consider calling spectraShadow.empty or spectraShadow.Inf instead.')
            
            # Check special case: empty c and G with square A
            if c.size == 0 and G.size == 0 and A.shape[0] == A.shape[1]:
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Both the generator matrix G and the center vector c are empty, while the coefficient matrix A is only a k x k-matrix. It is thus impossible to deduce in which space the spectrahedron lives in (i.e., for which n there holds that S is in R^n).')
            
            # Check that c and G are consistent
            if c.size == 0 and G.size > 0:
                raise CORAerror('CORA:wrongInputInConstructor',
                              'c is empty, but G is not. This is not consistent.')
            
            # Deduce dimensions and check consistency
            # For n_in == 2, G will be computed in _aux_computeProperties, so we check dimensions there
            # For n_in == 3, we check dimensions here
            if n_in == 3 and c.size > 0 and G.size > 0:
                k = A.shape[0]
                m = A.shape[1] // k - 1
                n = c.shape[0] if c.ndim == 1 else (c.shape[0] if c.shape[1] == 1 else c.shape[1])
                
                # Check that m is an integer
                if (m % 1 != 0) and (m != 0):
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'The coefficient matrix A does not have the right dimension (should be k x k*(m+1)).')
                
                # Check dimensions
                if G.shape[0] != n:
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'Dimension mismatch between the center vector c and the generator matrix G.')
                if G.shape[1] != m:
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'Dimension mismatch between the coefficient matrix A and the generator matrix G.')
            elif n_in == 2 and c.size > 0:
                # For n_in == 2, compute G first (matches MATLAB line 283)
                # Then check dimensions
                n = c.shape[0] if c.ndim == 1 else (c.shape[0] if c.shape[1] == 1 else c.shape[1])
                G = np.eye(n)  # Compute G for dimension checking
                
                k = A.shape[0]
                m = A.shape[1] // k - 1
                
                # Check that m is an integer
                if (m % 1 != 0) and (m != 0):
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'The coefficient matrix A does not have the right dimension (should be k x k*(m+1)).')
                
                # Check dimensions (matches MATLAB lines 342-348)
                if G.shape[0] != n:
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'Dimension mismatch between the center vector c and the generator matrix G.')
                if G.shape[1] != m:
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'Dimension mismatch between the coefficient matrix A and the generator matrix G.')


def _aux_computeProperties(A: Union[np.ndarray, List], c: np.ndarray, G: np.ndarray, ESumRep: Optional[List], n_in: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List]]:
    """Compute properties and hidden properties"""
    
    if n_in == 1 and ESumRep is not None:
        # If only ESumRep is given, we automatically compute A, G, and c
        A = np.hstack([ESumRep[0], ESumRep[1]])
        
        k = A.shape[0]
        m1 = ESumRep[0].shape[1] // k - 1
        m2 = ESumRep[1].shape[1] // k
        
        # G = [speye(m1) sparse(m1,m2)]
        G = np.hstack([np.eye(m1), np.zeros((m1, m2))])
        c = np.zeros((m1, 1))
    elif n_in == 1 and ESumRep is None:
        # If A is given, we set G and c; we don't set ESumRep, as this
        # would require further computation
        k = A.shape[0]
        m = A.shape[1] // k - 1
        G = np.eye(m)
        c = np.zeros((m, 1))
        ESumRep = [[], []]  # Empty list representation
    elif n_in == 2:
        # If all but G are set, automatically set G to be the identity
        G = np.eye(c.shape[0])
        ESumRep = [[], []]  # Empty list representation
    elif n_in == 3:
        # If everything is set (except ESumRep), just initialize ESumRep
        ESumRep = [[], []]  # Empty list representation

    return A, c, G, ESumRep 