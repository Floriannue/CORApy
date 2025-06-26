"""
Constrained polynomial zonotope class

A constrained polynomial zonotope (conPolyZono) is defined as:
cPZ := {c + G*β + Σ(E_i*β^E[i,:]) + GI*γ | A_EC*β^EC = b, β ∈ [-1,1]^p, γ ∈ [-1,1]^q}

Properties:
    c: center vector (n × 1)
    G: generator matrix (n × p)  
    E: exponent matrix (h × p)
    A: constraint matrix (m × r)
    b: constraint vector (m × 1)
    EC: constraint exponent matrix (h × r)
    GI: independent generator matrix (n × q)
    id: identifier vector (h × 1)

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 03-March-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, Optional, Any, Tuple
from ..contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check import inputArgsCheck
from cora_python.g.macros import CHECKS_ENABLED

class ConPolyZono(ContSet):
    """
    Constrained polynomial zonotope class
    
    A constrained polynomial zonotope represents sets of the form:
    cPZ := {c + G*β + Σ(E_i*β^E[i,:]) + GI*γ | A_EC*β^EC = b, β ∈ [-1,1]^p, γ ∈ [-1,1]^q}
    """
    
    def __init__(self, *varargin):
        """
        Constructor for constrained polynomial zonotope objects
        
        Args:
            *varargin: Variable arguments
                     - conPolyZono(c, G, E, [A, b, EC, GI, id])
                     - conPolyZono(other_conPolyZono): copy constructor
        """
        # 0. avoid empty instantiation
        if len(varargin) == 0:
            raise CORAerror('CORA:noInputInSetConstructor')
        
        # Check number of input arguments
        if len(varargin) < 1 or len(varargin) > 8:
            raise CORAerror('CORA:wrongInputInConstructor', f'Expected 1-8 arguments, got {len(varargin)}')

        # 1. copy constructor
        if len(varargin) == 1 and isinstance(varargin[0], ConPolyZono):
            other = varargin[0]
            self.c = other.c.copy() if hasattr(other, 'c') else np.array([])
            self.G = other.G.copy() if hasattr(other, 'G') else np.array([])
            self.E = other.E.copy() if hasattr(other, 'E') else np.array([])
            self.A = other.A.copy() if hasattr(other, 'A') else np.array([])
            self.b = other.b.copy() if hasattr(other, 'b') else np.array([])
            self.EC = other.EC.copy() if hasattr(other, 'EC') else np.array([])
            self.GI = other.GI.copy() if hasattr(other, 'GI') else np.array([])
            self.id = other.id.copy() if hasattr(other, 'id') else np.array([])
            super().__init__()
            self.precedence = 30
            return

        # 2. parse input arguments: varargin -> vars
        c, G, E, A, b, EC, GI, id_ = _aux_parseInputArgs(*varargin)

        # 3. check correctness of input arguments
        _aux_checkInputArgs(c, G, E, A, b, EC, GI, id_, len(varargin))

        # 4. compute properties
        c, G, E, A, b, EC, GI, id_ = _aux_computeProperties(c, G, E, A, b, EC, GI, id_)

        # 5. assign properties
        self.c = c
        self.G = G
        self.E = E
        self.A = A
        self.b = b
        self.EC = EC
        self.GI = GI
        self.id = id_

        # 6. set precedence (fixed) and initialize parent
        super().__init__()
        self.precedence = 30

    def __repr__(self):
        """String representation"""
        if hasattr(self, 'c') and self.c.size > 0:
            return f"ConPolyZono(dim={len(self.c)}, generators={self.G.shape[1] if self.G.size > 0 else 0})"
        else:
            return "ConPolyZono(empty)"

    # Legacy property support for backward compatibility
    @property
    def Grest(self):
        """Legacy property for GI (deprecated)"""
        try:
            from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning
            CORAwarning('CORA:deprecated', 'property', 'conPolyZono.Grest', 'CORA v2024',
                       'Please use conPolyZono.GI instead.',
                       'This change was made to be consistent with the notation in papers.')
        except ImportError:
            pass  # Skip warning if CORAwarning not available
        return self.GI
    
    @Grest.setter
    def Grest(self, value):
        """Legacy property setter for GI (deprecated)"""
        try:
            from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning
            CORAwarning('CORA:deprecated', 'property', 'conPolyZono.Grest', 'CORA v2024',
                       'Please use conPolyZono.GI instead.',
                       'This change was made to be consistent with the notation in papers.')
        except ImportError:
            pass
        self.GI = value

    @property
    def expMat(self):
        """Legacy property for E (deprecated)"""
        try:
            from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning
            CORAwarning('CORA:deprecated', 'property', 'conPolyZono.expMat', 'CORA v2024',
                       'Please use conPolyZono.E instead.',
                       'This change was made to be consistent with the notation in papers.')
        except ImportError:
            pass
        return self.E
    
    @expMat.setter
    def expMat(self, value):
        """Legacy property setter for E (deprecated)"""
        try:
            from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning
            CORAwarning('CORA:deprecated', 'property', 'conPolyZono.expMat', 'CORA v2024',
                       'Please use conPolyZono.E instead.',
                       'This change was made to be consistent with the notation in papers.')
        except ImportError:
            pass
        self.E = value

    @property
    def expMat_(self):
        """Legacy property for EC (deprecated)"""
        try:
            from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning
            CORAwarning('CORA:deprecated', 'property', 'conPolyZono.expMat_', 'CORA v2024',
                       'Please use conPolyZono.EC instead.',
                       'This change was made to be consistent with the notation in papers.')
        except ImportError:
            pass
        return self.EC
    
    @expMat_.setter
    def expMat_(self, value):
        """Legacy property setter for EC (deprecated)"""
        try:
            from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning
            CORAwarning('CORA:deprecated', 'property', 'conPolyZono.expMat_', 'CORA v2024',
                       'Please use conPolyZono.EC instead.',
                       'This change was made to be consistent with the notation in papers.')
        except ImportError:
            pass
        self.EC = value


# Auxiliary functions -----------------------------------------------------

def _aux_parseInputArgs(*varargin) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse input arguments from user and assign to variables"""
    
    c, G, E, A, b, EC, GI, id_ = (np.array([]) for _ in range(8))
    
    if len(varargin) == 1 and isinstance(varargin[0], ConPolyZono):
        pass # Dealt with in __init__
    elif len(varargin) >= 3:
        c, G, E = varargin[0], varargin[1], varargin[2]
        if len(varargin) >= 6:
            A, b, EC = varargin[3], varargin[4], varargin[5]
        if len(varargin) >= 4:
            if len(varargin) == 4 or len(varargin) == 7:
                GI = varargin[-1]
            elif len(varargin) == 5 or len(varargin) == 8:
                GI, id_ = varargin[-2], varargin[-1]

    # Ensure all are numpy arrays and correct shape before validation
    c, G, E, A, b, EC, GI, id_ = [np.asarray(arg) for arg in [c, G, E, A, b, EC, GI, id_]]
    
    if c.ndim == 1: c = c.reshape(-1, 1)
    if b.ndim == 1: b = b.reshape(-1, 1)
    if id_.ndim == 1: id_ = id_.reshape(-1, 1)
    
    if E.size > 0 and id_.size == 0:
        id_ = np.arange(1, E.shape[0] + 1).reshape(-1, 1)
    
    return c, G, E, A, b, EC, GI, id_


def _aux_checkInputArgs(c: np.ndarray, G: np.ndarray, E: np.ndarray, A: np.ndarray, 
                       b: np.ndarray, EC: np.ndarray, GI: np.ndarray, id_: np.ndarray, n_in: int):
    """Check correctness of input arguments by mirroring MATLAB's validation"""
    
    if CHECKS_ENABLED and n_in > 1:
        # Individual argument checks
        inputChecks = [[c, 'att', 'numeric', ['finite']]]
        if G.size > 0: inputChecks.append([G, 'att', 'numeric', ['finite', 'matrix']])
        if E.size > 0: inputChecks.append([E, 'att', 'numeric', ['integer', 'matrix']])
        
        # In MATLAB, these checks are added based on nargin > 5.
        # This is a bit tricky to replicate perfectly with Python's optional args,
        # but we can check if they are non-empty.
        if A.size > 0 or b.size > 0 or EC.size > 0:
            inputChecks.append([A, 'att', 'numeric', ['finite', 'matrix']])
            inputChecks.append([b, 'att', 'numeric', ['finite', 'matrix']])
            inputChecks.append([EC, 'att', 'numeric', ['integer', 'matrix']])

        if GI.size > 0: inputChecks.append([GI, 'att', 'numeric', ['finite', 'matrix']])
        if id_.size > 0: inputChecks.append([id_, 'att', 'numeric', ['finite']])
        
        inputArgsCheck(inputChecks)
        
        # center must be a vector
        if c.size == 0:
            if any(arg.size > 0 for arg in [G, E, A, b, EC, GI, id_]):
                raise CORAerror('CORA:wrongInputInConstructor', 'Either all or none input arguments are empty.')
        elif c.ndim > 1 and c.shape[1] > 1:
            raise CORAerror('CORA:wrongInputInConstructor', 'Center must be a vector.')

        # check inter-argument dimensions
        if G.size > 0 and G.shape[0] != c.shape[0]:
            raise CORAerror('CORA:wrongInputInConstructor', 'Dimension mismatch between center and G.')

        if E.size > 0 and G.size > 0 and E.shape[1] != G.shape[1]:
            raise CORAerror('CORA:wrongInputInConstructor', 'E and G must have the same number of columns.')
        
        if EC.size > 0:
            if E.size > 0 and E.shape[0] != EC.shape[0]:
                raise CORAerror('CORA:wrongInputInConstructor', 'E and EC must have the same number of rows.')
            if A.size > 0:
                if A.shape[1] != EC.shape[1]:
                    raise CORAerror('CORA:wrongInputInConstructor', 'A and EC must have the same number of columns.')
                if b.size > 0 and (b.shape[0] != A.shape[0] or b.shape[1] != 1):
                    raise CORAerror('CORA:wrongInputInConstructor', 'b must be a column vector of the same height as A.')
            elif b.size > 0:
                 raise CORAerror('CORA:wrongInputInConstructor', 'If b is provided, A must be provided.')
        elif A.size > 0 or b.size > 0:
            raise CORAerror('CORA:wrongInputInConstructor', 'If A or b is provided, EC must be provided.')


def _aux_computeProperties(c: np.ndarray, G: np.ndarray, E: np.ndarray, A: np.ndarray,
                          b: np.ndarray, EC: np.ndarray, GI: np.ndarray, id_: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reshape vectors to column vectors for consistency"""
    if c.ndim == 1:
        c = c.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if id_.ndim == 1:
        id_ = id_.reshape(-1, 1)
    
    return c, G, E, A, b, EC, GI, id_ 