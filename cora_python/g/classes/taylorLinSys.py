"""
taylorLinSys - helper class for storing Taylor series computations for linear systems

This class stores precomputed matrices and values for Taylor series expansions
used in reachability analysis of linear systems.

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 07-May-2007 (MATLAB)
Last update: 25-July-2016 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Dict, Any, Optional, List
from scipy.linalg import expm, inv
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TaylorLinSys:
    """
    Helper class for storing Taylor series computations for linear systems
    
    This class precomputes and stores matrices needed for reachability analysis
    including exponential matrices, Taylor series powers, and error bounds.
    
    Properties:
        A: System matrix
        timeStep: Current time step size
        eAt: Matrix exponential e^(A*timeStep)
        eAdt: Alias for eAt
        Ainv: Inverse of A (if it exists)
        powers: List of matrix powers A^i
        error: Error bound matrix
        F: Correction matrix for state
        G: Correction matrix for input
        V: Input set
        RV: Particular solution due to time-varying inputs
        Rtrans: Particular solution due to constant inputs
        inputCorr: Input correction term
        eAtInt: Interval matrix exponential
    """
    
    def __init__(self, A: np.ndarray):
        """
        Constructor for taylorLinSys
        
        Args:
            A: System matrix
        """
        # Brief check
        if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1] or \
           not np.all(np.isfinite(A)) or not np.all(np.isreal(A)):
            raise CORAerror('CORA:wrongInputInConstructor',
                           'A must be numeric, finite, 2D, square, and real-valued')
        
        self.A = A
        self.A_abs = np.abs(A)
        # Don't compute inverse because A might be singular (only if explicitly requested)
        self.Ainv = None
        
        # Initialize eAt to None (will be computed when needed)
        self.eAt = None
        
        # Initialize time step arrays (cell arrays in MATLAB)
        self.timeStep = []  # List of time steps that have been used
        self.eAdt = []  # List of eAdt matrices for each time step
        self.E = []  # List of remainder matrices
        self.F = []  # List of correction matrices for state
        self.G = []  # List of correction matrices for input
        self.dtoverfac = []  # List of dt^i/i! factors
        
        # Fields depending on truncationOrder (directly as index usable)
        # MATLAB: obj.Apower{1} = A; (1-based indexing, stores A^1)
        # Python: self.Apower[0] = A (0-based indexing, stores A^1)
        # So MATLAB index i → Python index i-1
        self.Apower = [A]  # List of A^i: A (index 0 = A^1), A^2 (index 1), A^3 (index 2), ...
        self.Apower_abs = [np.abs(A)]  # List of |A|^i
        # Apos and Aneg
        Apos = A.copy()
        Apos[A < 0] = 0
        Aneg = A.copy()
        Aneg[A > 0] = 0
        self.Apos = [Apos]
        self.Aneg = [Aneg]
        
        # Fields depending on timeStep (used as first index) and truncationOrder (used as second index)
        self.Apower_dt_fact = []  # List of lists: (A*dt)^i/i!
        self.Apower_abs_dt_fact = []  # List of lists: (|A|*dt)^i/i!
    
    def computeField(self, name: str, **kwargs) -> Any:
        """
        Compute and return a field value, computing it if necessary
        
        Args:
            name: Name of the field to compute
            **kwargs: Additional parameters for computation
                - For 'Apower': needs 'ithpower' parameter
                - For 'eAdt' or 'eAt': needs 'timeStep' parameter
                - For 'dtoverfac': needs 'timeStep' and 'ithpower' parameters
                
        Returns:
            Computed field value
        """
        if name == 'eAt' or name == 'eAdt':
            return self._computeEAt(**kwargs)
        elif name == 'Ainv':
            # Compute Ainv if not already computed and A is invertible
            if self.Ainv is None:
                # Check if A is invertible (rank == nrOfDims)
                rank_A = np.linalg.matrix_rank(self.A)
                if rank_A == self.A.shape[0]:
                    self.Ainv = inv(self.A)
                else:
                    # A is singular, Ainv remains None
                    pass
            return self.Ainv
        elif name == 'Apower':
            ithpower = kwargs.get('ithpower', 1)
            return self._computeApower(ithpower)
        elif name == 'Apower_abs':
            ithpower = kwargs.get('ithpower', 1)
            return self._computeApower_abs(ithpower)
        elif name == 'dtoverfac':
            timeStep = kwargs.get('timeStep')
            ithpower = kwargs.get('ithpower', 1)
            if timeStep is None:
                raise CORAerror('CORA:specialError', 'Time step required for dtoverfac')
            return self._computeDtoverfac(timeStep, ithpower)
        else:
            raise CORAerror('CORA:specialError', f'Unknown field: {name}')
    
    def _computeEAt(self, timeStep: Optional[float] = None) -> np.ndarray:
        """
        Compute matrix exponential e^(A*timeStep)
        
        Args:
            timeStep: Time step size (required)
            
        Returns:
            Matrix exponential
        """
        if timeStep is None:
            raise CORAerror('CORA:specialError', 'Time step not specified')
        
        # Check if already computed for this time step
        idx = self.getIndexForTimeStep(timeStep)
        if idx != -1 and idx < len(self.eAdt) and self.eAdt[idx] is not None:
            # Update self.eAt for backward compatibility
            self.eAt = self.eAdt[idx]
            return self.eAdt[idx]
        
        # Compute matrix exponential
        eAt_computed = expm(self.A * timeStep)
        
        # Store in the list structure
        if idx == -1:
            self.makeNewTimeStep(timeStep)
            idx = len(self.timeStep) - 1
        self.eAdt[idx] = eAt_computed
        
        # Also update self.eAt for backward compatibility
        self.eAt = eAt_computed
        
        return eAt_computed
    
    def _computeApower(self, ithpower: int) -> np.ndarray:
        """
        Compute i-th power of matrix A (recursively, matching MATLAB exactly)
        
        MATLAB: compute_Apower(obj,ithpower) uses 1-based indexing
        Python: Convert to 0-based indexing: MATLAB ithpower=1,2,3... → Python index=0,1,2...
        
        Args:
            ithpower: Power to compute (1-based, matching MATLAB)
            
        Returns:
            A^ithpower
        """
        # MATLAB: if length(obj.Apower) >= ithpower
        # Convert to 0-based: check if list has enough elements
        if len(self.Apower) >= ithpower:
            # MATLAB: Apower_i = readField(obj,'Apower',ithpower);
            # Python: Access with 0-based index
            return self.Apower[ithpower - 1]
        
        # MATLAB: Apower_mm = readField(obj,'Apower',ithpower-1);
        # Get previous power (MATLAB uses 1-based, so ithpower-1 is previous)
        if len(self.Apower) >= ithpower - 1:
            Apower_mm = self.Apower[ithpower - 2]  # 0-based index for ithpower-1
        else:
            # MATLAB: if isempty(Apower_mm), recursive call
            Apower_mm = self._computeApower(ithpower - 1)
        
        # MATLAB: Apower_i = Apower_mm * obj.A;
        Apower_i = Apower_mm @ self.A
        
        # MATLAB: obj.Apower{ithpower} = Apower_i;
        # Extend list if needed, then store at 0-based index
        while len(self.Apower) < ithpower:
            self.Apower.append(None)
        self.Apower[ithpower - 1] = Apower_i
        
        return Apower_i
    
    def _computeApower_abs(self, ithpower: int) -> np.ndarray:
        """
        Compute i-th power of |A| (recursively, matching MATLAB exactly)
        
        MATLAB: compute_Apower_abs(obj,ithpower) uses 1-based indexing
        Python: Convert to 0-based indexing: MATLAB ithpower=1,2,3... → Python index=0,1,2...
        
        Args:
            ithpower: Power to compute (1-based, matching MATLAB)
            
        Returns:
            |A|^ithpower
        """
        # MATLAB: if length(obj.Apower_abs) >= ithpower
        if len(self.Apower_abs) >= ithpower:
            # MATLAB: Apower_abs_i = readField(obj,'Apower_abs',ithpower);
            return self.Apower_abs[ithpower - 1]
        
        # MATLAB: Apower_abs_mm = readField(obj,'Apower_abs',ithpower-1);
        if len(self.Apower_abs) >= ithpower - 1:
            Apower_abs_mm = self.Apower_abs[ithpower - 2]
        else:
            # MATLAB: if isempty(Apower_abs_mm), recursive call
            Apower_abs_mm = self._computeApower_abs(ithpower - 1)
        
        # MATLAB: Apower_abs_i = Apower_abs_mm * obj.A_abs;
        # Need to compute A_abs if not stored
        if not hasattr(self, 'A_abs'):
            self.A_abs = np.abs(self.A)
        Apower_abs_i = Apower_abs_mm @ self.A_abs
        
        # MATLAB: obj.Apower_abs{ithpower} = Apower_abs_i;
        while len(self.Apower_abs) < ithpower:
            self.Apower_abs.append(None)
        self.Apower_abs[ithpower - 1] = Apower_abs_i
        
        return Apower_abs_i
    
    def _computeDtoverfac(self, timeStep: float, ithpower: int) -> float:
        """
        Compute dt^ithpower / ithpower! recursively (matching MATLAB exactly)
        
        MATLAB: compute_dtoverfac(obj,timeStep,ithpower) uses 1-based indexing
        Python: Convert to 0-based indexing: MATLAB ithpower=1,2,3... → Python index=0,1,2...
        
        Args:
            timeStep: Time step size
            ithpower: Power/factorial index (1-based, matching MATLAB)
            
        Returns:
            dt^ithpower / ithpower!
        """
        # MATLAB: idx = getIndexForTimeStep(obj,timeStep);
        idx = self.getIndexForTimeStep(timeStep)
        if idx == -1:
            # MATLAB: makeNewTimeStep(obj,timeStep); idx = length(obj.timeStep);
            self.makeNewTimeStep(timeStep)
            idx = len(self.timeStep) - 1
            dtoverfac_mm = []
        else:
            # MATLAB: dtoverfac_mm = obj.dtoverfac{idx};
            dtoverfac_mm = self.dtoverfac[idx]
        
        # MATLAB: if ~isempty(dtoverfac_mm) && length(dtoverfac_mm) >= ithpower
        if len(dtoverfac_mm) > 0 and len(dtoverfac_mm) >= ithpower:
            # MATLAB: dtoverfac = obj.dtoverfac{idx}(ithpower);
            # Python: Access with 0-based index
            return dtoverfac_mm[ithpower - 1]
        
        # MATLAB: if isempty(dtoverfac_mm) && ithpower == 1
        if len(dtoverfac_mm) == 0 and ithpower == 1:
            # MATLAB: dtoverfac = timeStep; obj.dtoverfac{idx}(ithpower) = dtoverfac;
            dtoverfac = timeStep
            # Extend list and store at 0-based index
            while len(self.dtoverfac[idx]) < ithpower:
                self.dtoverfac[idx].append(None)
            self.dtoverfac[idx][ithpower - 1] = dtoverfac
        # MATLAB: elseif length(dtoverfac_mm) < ithpower-1
        elif len(dtoverfac_mm) < ithpower - 1:
            # MATLAB: recursive call
            dtoverfac = self._computeDtoverfac(timeStep, ithpower - 1)
            # MATLAB: dtoverfac = dtoverfac * timeStep / ithpower;
            dtoverfac = dtoverfac * timeStep / ithpower
            # MATLAB: obj.dtoverfac{idx}(ithpower) = dtoverfac;
            # Extend list and store at 0-based index
            while len(self.dtoverfac[idx]) < ithpower:
                self.dtoverfac[idx].append(None)
            self.dtoverfac[idx][ithpower - 1] = dtoverfac
        else:
            # MATLAB: dtoverfac = dtoverfac_mm(end) * timeStep / ithpower;
            dtoverfac = dtoverfac_mm[-1] * timeStep / ithpower
            # MATLAB: obj.dtoverfac{idx}(ithpower) = dtoverfac;
            # Extend list and store at 0-based index
            while len(self.dtoverfac[idx]) < ithpower:
                self.dtoverfac[idx].append(None)
            self.dtoverfac[idx][ithpower - 1] = dtoverfac
        
        return dtoverfac
    
    def getTaylor(self, name: str, **kwargs) -> Any:
        """
        Get Taylor series related values
        
        Args:
            name: Name of the value to get
            **kwargs: Additional parameters
            
        Returns:
            Requested value
        """
        return self.computeField(name, **kwargs)
    
    def readFieldForTimeStep(self, field: str, timeStep: float) -> Optional[Any]:
        """
        Read a field value for a specific time step
        
        Args:
            field: Name of the field ('E', 'F', 'G', 'eAdt')
            timeStep: Time step size
            
        Returns:
            Field value or None if not found
        """
        idx = self.getIndexForTimeStep(timeStep)
        return self.readField(field, idx)
    
    def readField(self, field: str, idx1: int, idx2: Optional[int] = None) -> Optional[Any]:
        """
        Read a field value by index
        
        Args:
            field: Name of the field
            idx1: Index relating to timeStep
            idx2: Optional index relating to ithpower
            
        Returns:
            Field value or None if not found
        """
        val = None
        if idx1 != -1:
            try:
                if field == 'E':
                    if idx1 < len(self.E):
                        val = self.E[idx1]
                elif field == 'F':
                    if idx1 < len(self.F):
                        val = self.F[idx1]
                elif field == 'G':
                    if idx1 < len(self.G):
                        val = self.G[idx1]
                elif field == 'eAdt':
                    if idx1 < len(self.eAdt):
                        val = self.eAdt[idx1]
                elif field == 'Apower':
                    if idx1 < len(self.Apower):
                        val = self.Apower[idx1]
                elif field == 'Apower_abs':
                    if idx1 < len(self.Apower_abs):
                        val = self.Apower_abs[idx1]
                elif field == 'Apower_dt_fact':
                    if idx1 < len(self.Apower_dt_fact) and idx2 is not None:
                        if idx2 < len(self.Apower_dt_fact[idx1]):
                            val = self.Apower_dt_fact[idx1][idx2]
                elif field == 'Apower_abs_dt_fact':
                    if idx1 < len(self.Apower_abs_dt_fact) and idx2 is not None:
                        if idx2 < len(self.Apower_abs_dt_fact[idx1]):
                            val = self.Apower_abs_dt_fact[idx1][idx2]
                elif field == 'Apos':
                    if idx1 < len(self.Apos):
                        val = self.Apos[idx1]
                elif field == 'Aneg':
                    if idx1 < len(self.Aneg):
                        val = self.Aneg[idx1]
                elif field == 'dtoverfac':
                    if idx1 < len(self.dtoverfac):
                        val = self.dtoverfac[idx1]
                else:
                    raise CORAerror('CORA:wrongValue', 'third',
                                  ['has to be "E", "F", "G", "Apower", "Apower_abs", '
                                   '"Apos", "Aneg", or "dtoverfac"'])
            except (IndexError, KeyError):
                # Likely index out of range
                val = None
        return val
    
    def getIndexForTimeStep(self, timeStep: float) -> int:
        """
        Get index for a given time step size
        
        Args:
            timeStep: Time step size
            
        Returns:
            Index in timeStep list, or -1 if not found
        """
        from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
        idx = -1
        for i, ts in enumerate(self.timeStep):
            if withinTol(ts, timeStep, 1e-10):
                idx = i
                break
        return idx
    
    def makeNewTimeStep(self, timeStep: float):
        """
        Append a new time step to the list
        
        Args:
            timeStep: Time step size
        """
        self.timeStep.append(timeStep)
        self.eAdt.append(None)
        self.Apower_dt_fact.append([])
        self.Apower_abs_dt_fact.append([])
        self.E.append(None)
        self.F.append(None)
        self.G.append(None)
        self.dtoverfac.append([])
    
    def insertFieldTimeStep(self, field: str, val: Any, timeStep: float):
        """
        Insert a field value for a specific time step
        
        Args:
            field: Name of the field ('E', 'F', 'G', 'eAdt')
            val: Value to insert
            timeStep: Time step size
        """
        idx = self.getIndexForTimeStep(timeStep)
        if idx == -1:
            self.makeNewTimeStep(timeStep)
            idx = len(self.timeStep) - 1
        self.insertField(field, val, idx)
    
    def insertField(self, field: str, val: Any, idx: int):
        """
        Insert a field value at a specific index
        
        Args:
            field: Name of the field
            val: Value to insert
            idx: Index in the list
        """
        if field == 'E':
            if idx >= len(self.E):
                self.E.extend([None] * (idx - len(self.E) + 1))
            self.E[idx] = val
        elif field == 'F':
            if idx >= len(self.F):
                self.F.extend([None] * (idx - len(self.F) + 1))
            self.F[idx] = val
        elif field == 'G':
            if idx >= len(self.G):
                self.G.extend([None] * (idx - len(self.G) + 1))
            self.G[idx] = val
        elif field == 'eAdt':
            if idx >= len(self.eAdt):
                self.eAdt.extend([None] * (idx - len(self.eAdt) + 1))
            self.eAdt[idx] = val
        else:
            raise CORAerror('CORA:wrongValue', 'third',
                          'has to be "E", "F", "G", or "eAdt"') 