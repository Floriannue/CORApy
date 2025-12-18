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
from typing import Union, Optional, Any, Tuple, TYPE_CHECKING
from ..contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check import inputArgsCheck
from cora_python.g.macros import CHECKS_ENABLED
from cora_python.contSet.conZonotope.conZonotope import ConZonotope

if TYPE_CHECKING:
    from cora_python.contSet.conZonotope.conZonotope import ConZonotope

class ConPolyZono(ContSet):
    """
    Constrained polynomial zonotope class
    
    A constrained polynomial zonotope represents sets of the form:
    cPZ := {c + G*β + Σ(E_i*β^E[i,:]) + GI*γ | A_EC*β^EC = b, β ∈ [-1,1]^p, γ ∈ [-1,1]^q}
    """
    
    # Additional property for ambient dimension for empty sets
    _dim_val: Optional[int]

    def __init__(self, *varargin, **kwargs):
        """
        Constructor for constrained polynomial zonotope objects
        
        Args:
            *varargin: Variable arguments
                     - conPolyZono(c, G, E, [A, b, EC, GI, id])
                     - conPolyZono(other_conPolyZono): copy constructor
                     - conPolyZono(other_conZonotope): conversion from conZonotope
        """
        # Initialize _dim_val
        self._dim_val = None

        # 0. avoid empty instantiation
        if len(varargin) == 0:
            raise CORAerror('CORA:noInputInSetConstructor')
        
        # Check number of input arguments (MATLAB assertNarginConstructor handles this for regular calls)
        # This check is kept to ensure robust handling of various input types before more specific parsing.
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
            self._dim_val = other._dim_val # Copy dimension
            super().__init__()
            self.precedence = 30
            return
        
        # Handle ConZonotope conversion
        if len(varargin) == 1 and isinstance(varargin[0], ConZonotope):
            cz = varargin[0]
            self.c = cz.c
            self.G = cz.G
            self.A = cz.A
            self.b = cz.b
            self._dim_val = cz.dim() # Get dimension from ConZonotope
            # Initialize other properties as empty
            self.E = np.array([]).reshape(0,0)
            self.EC = np.array([]).reshape(0,0)
            self.GI = np.array([]).reshape(self.c.shape[0],0) if self.c.size > 0 else np.array([])
            self.id = np.array([]).reshape(0,1)
            super().__init__()
            self.precedence = 30
            return

        # Handle PolyZonotope conversion
        if len(varargin) == 1:
            from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope
            if isinstance(varargin[0], PolyZonotope):
                pz = varargin[0]
                # Copy properties from PolyZonotope
                self.c = pz.c.copy() if hasattr(pz, 'c') else np.array([])
                self.G = pz.G.copy() if hasattr(pz, 'G') else np.array([])
                self.E = pz.E.copy() if hasattr(pz, 'E') else np.array([])
                self.GI = pz.GI.copy() if hasattr(pz, 'GI') else np.array([])
                self.id = pz.id.copy() if hasattr(pz, 'id') else np.array([])
                # Initialize constraint properties as empty (no constraints)
                self.A = np.array([]).reshape(0, 0)
                self.b = np.array([]).reshape(0, 1)
                self.EC = np.array([]).reshape(0, 0)
                self._dim_val = pz.dim() if hasattr(pz, 'dim') else (pz.c.shape[0] if hasattr(pz, 'c') and pz.c.size > 0 else 0)
                super().__init__()
                self.precedence = 30
                return
        
        # Handle Interval conversion (via polyZonotope)
        if len(varargin) == 1:
            from cora_python.contSet.interval.interval import Interval
            if isinstance(varargin[0], Interval):
                # Use interval.conPolyZono() method which converts via polyZonotope
                from cora_python.contSet.interval.conPolyZono import conPolyZono
                result = conPolyZono(varargin[0])
                # Copy all attributes from result
                self.c = result.c
                self.G = result.G
                self.E = result.E
                self.A = result.A
                self.b = result.b
                self.EC = result.EC
                self.GI = result.GI
                self.id = result.id
                self._dim_val = result._dim_val
                super().__init__()
                self.precedence = 30
                return

        # 2. Parse input arguments directly within __init__ based on nargin
        num_args = len(varargin)
        
        # Determine dimension early if possible, primarily from c, otherwise G
        # Check if varargin[0] has a size attribute (property or method)
        # For numpy arrays, size is a property; for some ContSets, it might be a method
        size_val = None
        if num_args >= 1 and varargin[0] is not None:
            if hasattr(varargin[0], 'shape'):
                # Try to get size - could be property or method
                if hasattr(varargin[0], 'size'):
                    size_attr = getattr(varargin[0], 'size')
                    if callable(size_attr):
                        try:
                            size_val = size_attr()
                        except:
                            size_val = None
                    else:
                        size_val = size_attr
                elif hasattr(varargin[0], '__len__'):
                    try:
                        size_val = len(varargin[0])
                    except:
                        size_val = None
        
        if num_args >= 1 and varargin[0] is not None and hasattr(varargin[0], 'shape') and (size_val is not None and size_val > 0):
            self._dim_val = varargin[0].shape[0]
        elif num_args >= 2 and varargin[1] is not None and hasattr(varargin[1], 'shape'):
            # Check size for varargin[1] as well
            size_val_1 = None
            if hasattr(varargin[1], 'size'):
                size_attr_1 = getattr(varargin[1], 'size')
                if callable(size_attr_1):
                    try:
                        size_val_1 = size_attr_1()
                    except:
                        size_val_1 = None
                else:
                    size_val_1 = size_attr_1
            elif hasattr(varargin[1], '__len__'):
                try:
                    size_val_1 = len(varargin[1])
                except:
                    size_val_1 = None
            if size_val_1 is not None and size_val_1 > 0:
                self._dim_val = varargin[1].shape[0]
        elif 'dim' in kwargs and isinstance(kwargs['dim'], int):
            self._dim_val = kwargs['dim']

        # Default values for all possible parameters (will be refined by _aux_parseInputArgs)
        c, G, E, A, b, EC, GI, id_ = None, None, None, None, None, None, None, None

        if num_args >= 1:
            c = varargin[0]
        if num_args >= 2:
            G = varargin[1]
        if num_args >= 3:
            E = varargin[2]
        
        if num_args == 4 or num_args == 5: # Handles GI, id
            GI = varargin[3]
            if num_args == 5:
                id_ = varargin[4]
        elif num_args >= 6: # Handles A, b, EC, then optional GI, id
            A = varargin[3]
            b = varargin[4]
            EC = varargin[5]
            if num_args >= 7:
                GI = varargin[6]
            if num_args == 8:
                id_ = varargin[7]
        
        # Now call the auxiliary functions with explicit arguments
        c, G, E, A, b, EC, GI, id_ = _aux_parseInputArgs(c, G, E, A, b, EC, GI, id_)

        # If _dim_val is still None (e.g., empty inputs provided), and there's no implicit dim from c/G, set to 0.
        if self._dim_val is None:
            if c is not None and hasattr(c, 'shape') and len(c.shape) > 0 and c.size > 0:
                self._dim_val = c.shape[0]
            elif c is not None and hasattr(c, 'size') and c.size > 0:
                # Handle scalar or 1D array
                if hasattr(c, 'shape') and len(c.shape) > 0:
                    self._dim_val = c.shape[0]
                else:
                    self._dim_val = 1
            else:
                self._dim_val = 0

        # 3. check correctness of input arguments
        _aux_checkInputArgs(c, G, E, A, b, EC, GI, id_, num_args)

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

def _aux_parseInputArgs(c: Optional[np.ndarray], G: Optional[np.ndarray], E: Optional[np.ndarray],
                       A: Optional[np.ndarray], b: Optional[np.ndarray], EC: Optional[np.ndarray],
                       GI: Optional[np.ndarray], id_: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse input arguments from user and assign to variables (auxiliary function)"""

    # Default values from MATLAB's aux_parseInputArgs (lines 173-175)
    # This function now receives pre-parsed args, so it just needs to apply defaults
    
    # Note: `setDefaultValues` expects a list of defaults and a list of actual inputs
    # We need to construct these lists carefully based on whether the input was provided or is None.

    # For c, G, E - these are always expected to be present, but might be None if not passed.
    # MATLAB uses `setDefaultValues({[],[],[]}, varargin)` where varargin contains c,G,E
    # We will assume these are passed as non-None if they were present in the original varargin.

    # Explicitly set defaults for potentially None inputs to empty numpy arrays
    c = np.array([]) if c is None else np.asarray(c)
    G = np.array([]) if G is None else np.asarray(G)
    E = np.array([]) if E is None else np.asarray(E)
    A = np.array([]) if A is None else np.asarray(A)
    b = np.array([]) if b is None else np.asarray(b)
    EC = np.array([]) if EC is None else np.asarray(EC)
    GI = np.array([]) if GI is None else np.asarray(GI)
    id_ = np.array([]) if id_ is None else np.asarray(id_)

    # set identifiers (MATLAB lines 187-189)
    if E.size > 0 and id_.size == 0:
        # MATLAB: id = (1:size(E,1))';
        id_ = np.arange(1, E.shape[0] + 1).reshape(-1, 1)
    
    # Reshape vectors to column vectors for consistency (already in aux_computeProperties in MATLAB, but good to ensure early)
    if c.ndim == 1: c = c.reshape(-1, 1)
    if b.ndim == 1: b = b.reshape(-1, 1)
    if id_.ndim == 1: id_ = id_.reshape(-1, 1)

    return c, G, E, A, b, EC, GI, id_


def _aux_checkInputArgs(c: np.ndarray, G: np.ndarray, E: np.ndarray, A: np.ndarray, 
                       b: np.ndarray, EC: np.ndarray, GI: np.ndarray, id_: np.ndarray, n_in: int):
    """Check correctness of input arguments by mirroring MATLAB's validation"""
    
    if CHECKS_ENABLED and n_in > 0:

        # check correctness of user input 
        # Only check if they have content (empty arrays are valid for empty sets)
        inputChecks = []
        if c.size > 0:
            inputChecks.append([c, 'att', 'numeric', ['finite']])
        if G.size > 0:
            inputChecks.append([G, 'att', 'numeric', ['finite', 'matrix']])
        if E.size > 0:
            inputChecks.append([E, 'att', 'numeric', ['integer', 'matrix']])
        
        if n_in > 5:
            # only add constraints checks if they were in the input
            # to correctly indicate the position of the wrong input
            # Only check if they have content (empty arrays are valid for empty sets)
            if A.size > 0:
                inputChecks.append([A, 'att', 'numeric', ['finite', 'matrix']])
            if b.size > 0:
                inputChecks.append([b, 'att', 'numeric', ['finite', 'matrix']])
            if EC.size > 0:
                inputChecks.append([EC, 'att', 'numeric', ['finite', 'matrix']])
        
        # Add GI and id checks universally as MATLAB does at the end
        # (even if they were not explicitly passed, they are `np.array([])` from `_aux_parseInputArgs`)
        # Only check if they have content (empty arrays are valid for empty sets)
        if GI.size > 0:
            inputChecks.append([GI, 'att', 'numeric', ['finite', 'matrix']])
        if id_.size > 0:
            inputChecks.append([id_, 'att', 'numeric', ['finite']])
        
        inputArgsCheck(inputChecks)
        
        # center must be a vector
        if c.size == 0:
            # MATLAB: ~isempty(G) || ~isempty(E) || ~isempty(A) || ~isempty(b) || ~isempty(EC) || ~isempty(GI) || ~isempty(id)
            if G.size > 0 or E.size > 0 or A.size > 0 or b.size > 0 or EC.size > 0 or GI.size > 0 or id_.size > 0:
                raise CORAerror('CORA:wrongInputInConstructor', 'Either all or none input arguments are empty.')
        elif c.ndim > 1 and c.shape[1] > 1:
            raise CORAerror('CORA:wrongInputInConstructor', 'Center must be a vector.')

        # check inter-argument dimensions
        # MATLAB: size(E,2) ~= size(G,2)
        if E.size > 0 and G.size > 0 and E.shape[1] != G.shape[1]:
            raise CORAerror('CORA:wrongInputInConstructor', 'E and G must have the same number of columns.')
        
        if EC.size > 0:
            # MATLAB: size(E,1) ~= size(EC,1)
            if E.size > 0 and E.shape[0] != EC.shape[0]:
                raise CORAerror('CORA:wrongInputInConstructor', 'Input arguments "E" and "EC" are not compatible.')
            
            # MATLAB: ~all(all(floor(EC) == EC)) || ~all(all(EC >= 0))
            if not (np.all(np.floor(EC) == EC) and np.all(EC >= 0)):
                raise CORAerror('CORA:wrongInputInConstructor', 'Invalid constraint exponent matrix.')
            
            # check A, b
            # MATLAB: isempty(A) || size(A,2) ~= size(EC,2)
            if A.size == 0 or (A.size > 0 and A.shape[1] != EC.shape[1]):
                raise CORAerror('CORA:wrongInputInConstructor', 'Input arguments "A" and "EC" are not compatible.')
            
            # MATLAB: isempty(b) || size(b,2) > 1 || size(b,1) ~= size(A,1)
            if b.size == 0 or (b.size > 0 and (b.shape[1] > 1 or b.shape[0] != A.shape[0])):
                raise CORAerror('CORA:wrongInputInConstructor', 'Input arguments "A" and "b" are not compatible.')
        
        # MATLAB: elseif ~isempty(A) || ~isempty(b)
        # This handles cases where A or b are given, but EC is not (which is an error)
        elif (A.size > 0 or b.size > 0) and EC.size == 0:
            raise CORAerror('CORA:wrongInputInConstructor', 'Invalid constraint exponent matrix.')


def _aux_computeProperties(c: np.ndarray, G: np.ndarray, E: np.ndarray, A: np.ndarray,
                          b: np.ndarray, EC: np.ndarray, GI: np.ndarray, id_: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute properties"""

    # make center a column vector
    if c.ndim == 1:
        c = c.reshape(-1, 1)

    # Reshape other vectors to column vectors for consistency
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if id_.ndim == 1:
        id_ = id_.reshape(-1, 1)

    # set generator matrices to correct dimensions (MATLAB lines 273-279)
    n = c.shape[0] # Get dimension from center
    
    if G.size == 0 and n > 0:
        G = np.zeros((n, 0))
    
    if GI.size == 0 and n > 0:
        GI = np.zeros((n, 0))
    
    return c, G, E, A, b, EC, GI, id_ 