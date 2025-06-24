"""
polyZonotope - object constructor for polynomial zonotopes

Definition: see CORA manual, Sec. 2.2.1.5.

Syntax:
    obj = polyZonotope(pZ)
    obj = polyZonotope(c)
    obj = polyZonotope(c,G)
    obj = polyZonotope(c,G,GI)
    obj = polyZonotope(c,[],GI)
    obj = polyZonotope(c,G,GI,E)
    obj = polyZonotope(c,G,[],E)
    obj = polyZonotope(c,G,GI,E,id)
    obj = polyZonotope(c,G,[],E,id)

Inputs:
    pZ - polyZonotope object
    c - center of the polynomial zonotope (dimension: [nx,1])
    G - generator matrix containing the dependent generators 
       (dimension: [nx,N])
    GI - generator matrix containing the independent generators
            (dimension: [nx,M])
    E - matrix containing the exponents for the dependent generators
             (dimension: [p,N])
    id - vector containing the integer identifiers for the dependent
         factors (dimension: [p,1])

Outputs:
    obj - polyZonotope object

Example: 
    c = [0;0]
    G = [2 0 1;0 2 1]
    GI = [0;0.5]
    E = [1 0 3;0 1 1]
 
    pZ = polyZonotope(c,G,GI,E)
 
    plot(pZ,[1,2],'FaceColor','r')

References:
    [1] Kochdumper, N., et al. (2020). Sparse polynomial zonotopes: A novel
        set representation for reachability analysis. IEEE Transactions on 
        Automatic Control.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope

Authors:       Niklas Kochdumper, Mark Wetzlinger, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       26-March-2018 (MATLAB)
Last update:   02-May-2020 (MW, add property validation, def constructor, MATLAB)
               21-March-2021 (MW, error messages, size checks, restructuring, MATLAB)
               14-December-2022 (TL, restructuring, MATLAB)
               29-March-2023 (TL, optimized constructor, MATLAB)
               13-September-2023 (TL, replaced Grest/expMat properties with GI/E, MATLAB)
Last revision: 16-June-2023 (MW, restructure using auxiliary functions, MATLAB)
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
from cora_python.g.functions.helper.sets.contSet.polyZonotope import removeRedundantExponents

if TYPE_CHECKING:
    pass


class PolyZonotope(ContSet):
    """
    Class for representing polynomial zonotopes
    
    Properties (SetAccess = {?contSet, ?matrixSet}, GetAccess = public):
        c: center
        G: dependent generator matrix
        GI: independent generator matrix
        E: exponent matrix
        id: identifier vector
        Grest: legacy property (deprecated)
        expMat: legacy property (deprecated)
    """
    
    def __init__(self, *varargin):
        """
        Class constructor for polynomial zonotopes
        """
        # 0. avoid empty instantiation
        if len(varargin) == 0:
            raise CORAerror('CORA:noInputInSetConstructor')
        assertNarginConstructor(list(range(1, 6)), len(varargin))

        # 1. copy constructor
        if len(varargin) == 1 and isinstance(varargin[0], PolyZonotope):
            # Direct assignment like MATLAB
            other = varargin[0]
            self.c = other.c
            self.G = other.G
            self.GI = other.GI
            self.E = other.E
            self.id = other.id
            super().__init__()
            self.precedence = 70
            return

        # 2. parse input arguments: varargin -> vars
        c, G, GI, E, id = _aux_parseInputArgs(*varargin)

        # 3. check correctness of input arguments
        _aux_checkInputArgs(c, G, GI, E, id, len(varargin))

        # 4. compute properties
        c, G, GI, E, id = _aux_computeProperties(c, G, GI, E, id)

        # 5. assign properties
        self.c = c
        self.G = G
        self.GI = GI
        self.E = E
        self.id = id

        # 6. set precedence (fixed) and initialize parent
        super().__init__()
        self.precedence = 70

    # Legacy Grest property getter/setter (deprecated)
    def _get_Grest(self):
        """Legacy Grest property getter - deprecated, use GI instead"""
        CORAwarning('CORA:deprecated', 'property', 'polyZonotope.Grest', 'CORA v2024',
                   'Please use polyZonotope.GI instead.',
                   'This change was made to be consistent with the notation in papers.')
        return self.GI

    def _set_Grest(self, Grest):
        """Legacy Grest property setter - deprecated, use GI instead"""
        CORAwarning('CORA:deprecated', 'property', 'polyZonotope.Grest', 'CORA v2024',
                   'Please use polyZonotope.GI instead.',
                   'This change was made to be consistent with the notation in papers.')
        self.GI = Grest

    # Legacy Grest property (deprecated)
    Grest = property(_get_Grest, _set_Grest)

    # Legacy expMat property getter/setter (deprecated)
    def _get_expMat(self):
        """Legacy expMat property getter - deprecated, use E instead"""
        CORAwarning('CORA:deprecated', 'property', 'polyZonotope.expMat', 'CORA v2024',
                   'Please use polyZonotope.E instead.',
                   'This change was made to be consistent with the notation in papers.')
        return self.E

    def _set_expMat(self, expMat):
        """Legacy expMat property setter - deprecated, use E instead"""
        CORAwarning('CORA:deprecated', 'property', 'polyZonotope.expMat', 'CORA v2024',
                   'Please use polyZonotope.E instead.',
                   'This change was made to be consistent with the notation in papers.')
        self.E = expMat

    # Legacy expMat property (deprecated)
    expMat = property(_get_expMat, _set_expMat)
    
    def __repr__(self) -> str:
        """Official string representation for programmers"""
        return f"PolyZonotope(c={self.c.shape}, G={self.G.shape}, GI={self.GI.shape}, E={self.E.shape}, id={self.id.shape})"


# Auxiliary functions -----------------------------------------------------

def _aux_parseInputArgs(*varargin) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse input arguments from user and assign to variables"""
    
    # no input arguments
    if len(varargin) == 0:
        c = np.array([])
        G = np.array([])
        GI = np.array([])
        E = np.array([])
        id = np.array([])
        return c, G, GI, E, id

    # set default values
    c, G, GI, E, id = setDefaultValues([[], [], [], [], []], list(varargin))

    # Convert to numpy arrays
    c = np.array(c) if c is not None else np.array([])
    G = np.array(G) if G is not None else np.array([])
    GI = np.array(GI) if GI is not None else np.array([])
    E = np.array(E) if E is not None else np.array([])
    id = np.array(id) if id is not None else np.array([])

    return c, G, GI, E, id


def _aux_checkInputArgs(c: np.ndarray, G: np.ndarray, GI: np.ndarray, E: np.ndarray, id: np.ndarray, n_in: int):
    """Check correctness of input arguments"""
    
    # only check if macro set to true
    if CHECKS_ENABLED and n_in > 0:

        inputChecks = [
            [c, 'att', 'numeric', 'finite'],
            [G, 'att', 'numeric', ['finite', 'matrix']],
            [GI, 'att', 'numeric', ['finite', 'matrix']],
            [E, 'att', 'numeric', ['integer', 'nonnegative', 'matrix']],
            [id, 'att', 'numeric', 'integer']
        ]
        
        inputArgsCheck(inputChecks)

        # check dimensions ---
        # c
        if c.size > 0 and c.ndim > 1 and c.shape[1] != 1:
            raise CORAerror('CORA:wrongInputInConstructor',
                          'Center should be a column vector.')
        
        # G
        if G.size > 0 and c.size > 0 and G.shape[0] != c.shape[0]:
            raise CORAerror('CORA:wrongInputInConstructor',
                          'Dimension mismatch between center and dependent generator matrix.')
        
        # GI
        if GI.size > 0 and c.size > 0 and GI.shape[0] != c.shape[0]:
            raise CORAerror('CORA:wrongInputInConstructor',
                          'Dimension mismatch between center and independent generator matrix.')
        
        # E
        if E.size > 0 and G.size > 0 and E.shape[1] != G.shape[1]:
            raise CORAerror('CORA:wrongInputInConstructor',
                          'Dimension mismatch between dependent generator matrix and exponent matrix.')
        
        # id
        if id.size > 0 and id.ndim > 1 and id.shape[1] != 1:
            raise CORAerror('CORA:wrongInputInConstructor',
                          'Identifier vector should be a column vector.')
        
        if E.size > 0 and id.size > 0 and id.shape[0] != E.shape[0]:
            raise CORAerror('CORA:wrongInputInConstructor',
                          'Dimension mismatch between exponent matrix and identifier vector.')


def _aux_computeProperties(c: np.ndarray, G: np.ndarray, GI: np.ndarray, E: np.ndarray, id: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute properties"""
    
    # remove redundancies
    if E.size > 0:
        E, G = removeRedundantExponents(E, G)

    # if G/GI is empty, set correct dimension
    if G.size == 0 and c.size > 0:
        G = np.zeros((c.shape[0], 0))
    if GI.size == 0 and c.size > 0:
        GI = np.zeros((c.shape[0], 0))

    # default value for exponent matrix
    if E.size == 0 and G.size > 0:
        E = np.eye(G.shape[1])

    # number of dependent factors
    if id.size == 0 and E.size > 0:
        p = E.shape[0]
        id = np.arange(1, p + 1).reshape(-1, 1)  # column vector

    return c, G, GI, E, id 