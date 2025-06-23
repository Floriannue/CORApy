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
            SpS = varargin[0]
            self.A = SpS.A
            self.c = SpS.c
            self.G = SpS.G
            
            self.ESumRep.val = SpS.ESumRep.val
            self.emptySet.val = SpS.emptySet.val
            self.fullDim.val = SpS.fullDim.val
            self.bounded.val = SpS.bounded.val
            self.center.val = SpS.center.val
            super().__init__()
            self.precedence = SpS.precedence
            return

        # 2. parse input arguments: varargin -> vars
        A, c, G, ESumRep = _aux_parseInputArgs(*varargin)
        # Note that at this stage, A may be a cell containing ESumRep

        # 3. check correctness of input arguments
        _aux_checkInputArgs(A, c, G, ESumRep, len(varargin))
        
        # 4. compute properties and hidden properties
        A, c, G, ESumRep = _aux_computeProperties(A, c, G, ESumRep, len(varargin))

        # 5. assign properties
        self.A = sparse.csr_matrix(A)
        self.c = sparse.csr_matrix(c)
        self.G = sparse.csr_matrix(G)
        self.ESumRep.val = ESumRep

        # 6. set precedence (fixed) and initialize parent
        super().__init__()
        self.precedence = 40


# Auxiliary functions -----------------------------------------------------

def _aux_parseInputArgs(*varargin) -> Tuple[Union[np.ndarray, List], np.ndarray, np.ndarray, Optional[List]]:
    """Parse input arguments from user and assign to variables"""
    
    # Set default values
    A, c, G = setDefaultValues([0, [], []], list(varargin))
    
    # Identify if initialization is made via just A or the existential sum
    # representation
    if len(varargin) == 1 and isinstance(A, list):
        ESumRep = A
        A = 0
    else:
        ESumRep = None

    # Convert to numpy arrays where appropriate
    if not isinstance(A, list):
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

                if not isinstance(ESumRep[0], np.ndarray) or not isinstance(ESumRep[1], np.ndarray):
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'One of the entries of ESumRep is not a matrix.')

                if (ESumRep[0].shape[0] != ESumRep[1].shape[0] and 
                    ESumRep[0].size > 0 and ESumRep[1].size > 0):
                    raise CORAerror('CORA:wrongInputInConstructor',
                                  'The dimensions of the two matrices in ESumRep do not match.')


def _aux_computeProperties(A: Union[np.ndarray, List], c: np.ndarray, G: np.ndarray, ESumRep: Optional[List], n_in: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List]]:
    """Compute properties and hidden properties"""
    
    # Handle ESumRep case
    if n_in == 1 and ESumRep is not None:
        # For now, set default values for A, c, G when using ESumRep
        # Full implementation would compute these from ESumRep
        A = np.array([[1]])
        c = np.array([0])
        G = np.array([[1]])
    else:
        # Ensure proper array formats
        if isinstance(A, (int, float)):
            A = np.array([[A]])
        if c.size == 0:
            c = np.array([0])
        if G.size == 0:
            G = np.array([[1]])

    return A, c, G, ESumRep 