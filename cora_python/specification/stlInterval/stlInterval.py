"""
stlInterval - open/closed time intervals for signal temporal logic specifications

Time intervals are always one dimensional and are subsets of the non-negative reals.
The boundaries of the interval can be open or closed. The interval can be empty.

Syntax:
    int = StlInterval()
    int = StlInterval(x)
    int = StlInterval(I)
    int = StlInterval(SI)
    int = StlInterval(lb, ub)
    int = StlInterval(lb, ub, closed)
    int = StlInterval(lb, ub, lc, rc)

Inputs:
    x - numeric value
    I - interval object
    SI - stlInterval object
    lb - lower bound
    ub - upper bound
    closed - boolean for both left and right closed (default: True)
    lc - is left closed (default: True)
    rc - is right closed (default: True)

Outputs:
    int - generated StlInterval object

Example:
    int0 = StlInterval()                 # empty interval
    int1 = StlInterval(1)                # singular interval [1,1]
    int2 = StlInterval(0, 1)             # closed interval [0,1]
    int3 = StlInterval(0, 1, False)      # open interval (0,1)
    int4 = StlInterval(0, 1, True, False) # half open interval [0,1)

Authors:       Florian Lercher (MATLAB)
               Python translation by AI Assistant
Written:       06-February-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, Optional, Tuple, Any
from cora_python.contSet.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.macros.CHECKS_ENABLED import CHECKS_ENABLED


class StlInterval(ContSet):
    """
    StlInterval class for open/closed time intervals in signal temporal logic
    
    This class represents time intervals that can have open or closed boundaries,
    used in signal temporal logic specifications.
    """
    
    def __init__(self, *args):
        """
        Constructor for StlInterval
        
        Args:
            *args: Variable arguments for different constructor patterns
        """
        # Initialize properties
        self.lower: Optional[float] = None
        self.leftClosed: bool = True
        self.upper: Optional[float] = None
        self.rightClosed: bool = True
        
        # 0. check number of input arguments
        assertNarginConstructor(list(range(5)), len(args))
        
        # 1. copy constructor
        if len(args) == 1 and isinstance(args[0], StlInterval):
            other = args[0]
            self.lower = other.lower
            self.upper = other.upper
            self.leftClosed = other.leftClosed
            self.rightClosed = other.rightClosed
            super().__init__()
            self.precedence = 13
            return
        
        # 2. parse input arguments: args -> vars
        lb, ub, lc, rc = self._aux_parseInputArgs(*args)
        
        # 3. check correctness of input arguments
        self._aux_checkInputArgs(lb, ub, lc, rc, len(args))
        
        # 4. compute properties
        lb, ub, lc, rc = self._aux_computeProperties(lb, ub, lc, rc)
        
        # 5. assign properties
        self.lower = lb
        self.upper = ub
        self.leftClosed = lc
        self.rightClosed = rc
        
        # 6. call parent constructor and set precedence
        super().__init__()
        self.precedence = 13
    
    def _aux_parseInputArgs(self, *args) -> Tuple[Optional[float], Optional[float], bool, bool]:
        """Parse input arguments"""
        
        # Default values - empty arrays like in MATLAB
        lb = []  # Empty array like MATLAB
        ub = []  # Empty array like MATLAB
        lc = True
        rc = True
        
        # Use setDefaultValues to handle argument parsing
        defaults, _ = setDefaultValues([lb, ub, lc, rc], list(args))
        lb, ub, lc, rc = defaults
        
        # Handle special cases
        if len(args) == 1:
            ub = lb
        elif len(args) == 3:
            rc = lc
            
        # Convert empty arrays to None for Python
        if isinstance(lb, list) and len(lb) == 0:
            lb = None
        if isinstance(ub, list) and len(ub) == 0:
            ub = None
            
        return lb, ub, lc, rc
    
    def _aux_checkInputArgs(self, lb: Optional[float], ub: Optional[float], 
                           lc: bool, rc: bool, n_in: int):
        """Check correctness of input arguments"""
        
        if CHECKS_ENABLED and n_in > 0:
            if lb is not None and lb < 0:
                raise CORAerror('CORA:wrongInputInConstructor',
                              'Lower bound must be non-negative.')
    
    def _aux_computeProperties(self, lb: Optional[float], ub: Optional[float], 
                              lc: bool, rc: bool) -> Tuple[Optional[float], Optional[float], bool, bool]:
        """Compute properties"""
        
        if lb is not None and ub is not None:
            # Handle infinite bounds
            if abs(lb) == np.inf:
                lc = False
            if abs(ub) == np.inf:
                rc = False
                
            # Check for empty interval
            if lb > ub or (lb == ub and (not lc or not rc)):
                # Make empty interval
                lb, ub, lc, rc = self._aux_parseInputArgs()
                
        return lb, ub, lc, rc
    
    def toStr(self) -> str:
        """String representation"""
        if self.isemptyobject():
            return '∅'
        
        # Handle None bounds (shouldn't happen in normal cases, but be safe)
        if self.lower is None or self.upper is None:
            return '∅'
        
        lterm = '[' if self.leftClosed else '('
        rterm = ']' if self.rightClosed else ')'
        
        return f"{lterm}{self.lower}, {self.upper}{rterm}"
    
    def __str__(self) -> str:
        """String representation for print"""
        return self.toStr()
    
    def __repr__(self) -> str:
        """String representation for repr"""
        return f"StlInterval({self.toStr()})"

