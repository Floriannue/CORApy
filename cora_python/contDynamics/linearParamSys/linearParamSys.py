"""
linearParamSys - Linear parametric system class

This class represents linear systems with parametric uncertainty.

Syntax:
    sys = LinearParamSys(A)
    sys = LinearParamSys(A, B)
    sys = LinearParamSys(A, B, c)
    sys = LinearParamSys(A, type)
    sys = LinearParamSys(A, B, type)
    sys = LinearParamSys(A, B, c, type)
    sys = LinearParamSys(name, A, type)
    sys = LinearParamSys(name, A, B, type)
    sys = LinearParamSys(name, A, B, c, type)

Inputs:
    name - name of the system
    A - system matrix (can be numeric, intervalMatrix, or matZonotope)
    B - input matrix (can be numeric, intervalMatrix, or matZonotope)
    c - constant input (can be numeric, interval, or zonotope)
    type - constant/time-varying parameters
           - 'constParam' (constant parameters, default)
           - 'varParam' (time-varying parameters)

Authors: Matthias Althoff (MATLAB)
         Python translation: 2025
"""

from typing import Optional, Union, Any
import numpy as np
from ..contDynamics import ContDynamics

# Import matrix set types
try:
    from cora_python.matrixSet.intervalMatrix.intervalMatrix import IntervalMatrix
except ImportError:
    IntervalMatrix = None

try:
    from cora_python.matrixSet.matZonotope import matZonotope
except ImportError:
    matZonotope = None


class LinearParamSys(ContDynamics):
    """
    Linear parametric system class
    
    This class represents linear systems with parametric uncertainty in the system matrix.
    The system dynamics are:
        x'(t) = A(p) x(t) + B(p) u(t) + c
    
    where A and B can depend on parameters p.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Constructor for LinearParamSys
        
        Supports multiple calling patterns matching MATLAB.
        """
        # Parse input arguments
        name, A, B, c, type_str = self._parse_input_args(*args)
        
        # Check input arguments
        self._check_input_args(name, A, B, c, type_str, len(args))
        
        # Compute properties
        name, A, B, c, type_str, states, inputs = self._compute_properties(name, A, B, c, type_str)
        
        # Initialize parent class
        super().__init__(name, states, inputs, 0, 0, 0)
        
        # Assign properties
        self.A = A
        self.B = B
        self.c = c
        self.type = type_str
        self.constParam = (type_str == 'constParam')
        
        # Initialize computation properties (set during reachability analysis)
        self.stepSize = 1.0
        self.taylorTerms = None
        self.mappingMatrixSet = {}
        self.power = {'zono': [], 'int': []}
        self.E = None  # Note: not disturbance matrix
        self.F = None  # Note: not noise matrix
        self.inputF = None
        self.inputCorr = None
        self.Rinput = None
        self.Rtrans = None
        self.RV = None
        self.sampleMatrix = None
    
    def _parse_input_args(self, *args):
        """Parse input arguments from user"""
        # Default values
        def_name = 'linParamSys'
        type_str = 'constParam'
        A = None
        B = None
        c = None
        
        if len(args) == 0:
            return def_name, A, B, c, type_str
        
        # Check if first argument is a name (string)
        if len(args) > 0 and isinstance(args[0], str):
            # First arg is name
            name = args[0] if len(args) > 0 else def_name
            A = args[1] if len(args) > 1 else A
            B = args[2] if len(args) > 2 else B
            c = args[3] if len(args) > 3 else c
            type_str = args[4] if len(args) > 4 else type_str
        else:
            # No name provided
            name = def_name
            A = args[0] if len(args) > 0 else A
            B = args[1] if len(args) > 1 else B
            c = args[2] if len(args) > 2 else c
            type_str = args[3] if len(args) > 3 else type_str
        
        # Handle ambiguous cases (type vs c)
        if isinstance(c, str):
            type_str = c
            c = None
        
        return name, A, B, c, type_str
    
    def _check_input_args(self, name, A, B, c, type_str, n_in):
        """Check correctness of input arguments"""
        if n_in == 0:
            return
        
        # Check type
        if type_str not in ['constParam', 'varParam']:
            raise ValueError(f"type must be 'constParam' or 'varParam', got '{type_str}'")
        
        # Check A
        if A is not None:
            valid = isinstance(A, np.ndarray)
            if not valid and IntervalMatrix is not None:
                valid = isinstance(A, IntervalMatrix)
            if not valid and matZonotope is not None:
                valid = isinstance(A, matZonotope)
            if not valid:
                raise TypeError("A must be numeric, IntervalMatrix, or matZonotope")
        
        # Check B
        if B is not None:
            valid = isinstance(B, (np.ndarray, int, float, np.number))
            if not valid and IntervalMatrix is not None:
                valid = isinstance(B, IntervalMatrix)
            if not valid and matZonotope is not None:
                valid = isinstance(B, matZonotope)
            if not valid:
                raise TypeError("B must be numeric, IntervalMatrix, or matZonotope")
        
        # Check c
        if c is not None:
            from cora_python.contSet.interval import Interval
            from cora_python.contSet.zonotope import Zonotope
            if not isinstance(c, (np.ndarray, Interval, Zonotope)):
                raise TypeError("c must be numeric, interval, or zonotope")
            if isinstance(c, np.ndarray) and c.ndim > 1 and c.shape[1] != 1:
                raise ValueError("c must be a vector")
    
    def _compute_properties(self, name, A, B, c, type_str):
        """Compute number of states and inputs, set defaults"""
        # Number of states from A
        if A is None:
            raise ValueError("A (system matrix) must be provided")
        
        if isinstance(A, np.ndarray):
            states = A.shape[0]
        elif (IntervalMatrix is not None and isinstance(A, IntervalMatrix)) or (matZonotope is not None and isinstance(A, matZonotope)):
            states = A.dim()[0]
        else:
            raise TypeError(f"Unsupported type for A: {type(A)}")
        
        # Input matrix
        if B is None:
            B = np.zeros((states, 1))
        
        # Number of inputs
        if isinstance(B, (int, float)) and (B == 0 or B == 1):
            # Scalar B means identity or zero
            inputs = states
        elif isinstance(B, np.ndarray):
            inputs = B.shape[1] if B.ndim > 1 else 1
        elif (IntervalMatrix is not None and isinstance(B, IntervalMatrix)) or (matZonotope is not None and isinstance(B, matZonotope)):
            inputs = B.dim()[1]
        else:
            inputs = states  # Default
        
        # Constant offset
        if c is None:
            c = np.zeros((states, 1))
        elif isinstance(c, np.ndarray) and c.ndim == 1:
            c = c.reshape(-1, 1)
        
        return name, A, B, c, type_str, states, inputs
    
    @property
    def nrOfDims(self):
        """MATLAB compatibility: nrOfDims property"""
        return self.nr_of_dims
    
    def __repr__(self):
        """String representation"""
        return (f"LinearParamSys(name='{self.name}', "
                f"states={self.nr_of_dims}, inputs={self.nr_of_inputs}, "
                f"type='{self.type}')")
