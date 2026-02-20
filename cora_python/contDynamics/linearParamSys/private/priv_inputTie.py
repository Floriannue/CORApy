"""
priv_inputTie - tie: time interval error; computes the error done by the
   linear assumption of the constant input solution

Syntax:
    sys = priv_inputTie(sys, options)

Inputs:
    sys - LinearParamSys object
    options - options struct

Outputs:
    sys - LinearParamSys object (modified with inputF property)

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import math
import numpy as np
from typing import Any, Dict
from cora_python.matrixSet.intervalMatrix import IntervalMatrix
from cora_python.contSet.interval import Interval


def priv_inputTie(sys: Any, options: Dict[str, Any]) -> Any:
    """
    Computes the input time interval error (tie) for linear parametric systems
    
    Args:
        sys: LinearParamSys object (must have power, taylorTerms, stepSize, A, E properties)
        options: Options dict (not used in this function but kept for MATLAB compatibility)
        
    Returns:
        sys: Modified LinearParamSys object with inputF property set
    """
    # Obtain powers and convert them to interval matrices
    # MATLAB: Apower=cell(1,length(obj.power.int));
    Apower = [None] * len(sys.power['int'])
    
    # Matrix zonotope
    # MATLAB: for i=1:length(obj.power.zono)
    for i in range(len(sys.power['zono'])):
        # MATLAB: Apower{i}=intervalMatrix(obj.power.zono{i});
        Apower[i] = IntervalMatrix(sys.power['zono'][i])
    
    # Interval matrix
    # MATLAB: for i=(length(obj.power.zono)+1):length(obj.power.int)
    for i in range(len(sys.power['zono']), len(sys.power['int'])):
        # MATLAB: Apower{i}=obj.power.int{i};
        Apower[i] = sys.power['int'][i]
    
    r = sys.stepSize
    
    # Initialize Asum
    # MATLAB: infimum=-0.25*r;%*r^2;already included in Apower
    # MATLAB: supremum=0;
    # MATLAB: timeInterval=intervalMatrix(0.5*(supremum+infimum),0.5*(supremum-infimum));
    # MATLAB: Asum=timeInterval*Apower{1}*(1/factorial(2));
    infimum = -0.25 * r  # *r^2 already included in Apower
    supremum = 0.0
    timeInterval = IntervalMatrix(0.5 * (supremum + infimum), 0.5 * (supremum - infimum))
    Asum = timeInterval * Apower[0] * (1.0 / math.factorial(2))  # Apower[1] in MATLAB is index 0 in Python
    
    # MATLAB: for i=3:obj.taylorTerms
    for i in range(3, sys.taylorTerms + 1):
        # Compute factor
        # MATLAB: exp1=-i/(i-1); exp2=-1/(i-1);
        # MATLAB: infimum = (i^exp1-i^exp2)*r;%*r^i;already included in Apower
        # MATLAB: supremum = 0;
        exp1 = -i / (i - 1)
        exp2 = -1 / (i - 1)
        infimum = (i ** exp1 - i ** exp2) * r  # *r^i already included in Apower
        supremum = 0.0
        timeInterval = IntervalMatrix(0.5 * (supremum + infimum), 0.5 * (supremum - infimum))
        
        # Compute powers
        # MATLAB: Aadd=timeInterval*Apower{i-1};
        # MATLAB uses 1-based indexing: Apower{i-1} where i starts at 3, so i-1 = 2 (1-based)
        # Python uses 0-based indexing: Apower[i-2] where i starts at 3, so i-2 = 1 (0-based)
        power_idx = i - 2  # Convert MATLAB i-1 to Python 0-based index
        if power_idx < len(Apower) and Apower[power_idx] is not None:
            Aadd = timeInterval * Apower[power_idx]
        else:
            # Skip if power doesn't exist (shouldn't happen, but handle gracefully)
            continue
        
        # Compute sum
        # MATLAB: Asum=Asum+Aadd*(1/factorial(i));
        Asum = Asum + Aadd * (1.0 / math.factorial(i))
    
    # Compute error due to finite Taylor series according to "M. L. Liou. A novel 
    # method of evaluating transient response. In Proceedings of the IEEE, 
    # volume 54, pages 20-23, 1966".
    # Consider that the power of A is less than for t due to the constant input
    # solution instead of the initial state solution
    
    # MATLAB: norm_A = norm(obj.A, inf);
    if isinstance(sys.A, np.ndarray):
        norm_A = np.linalg.norm(sys.A, np.inf)
    elif hasattr(sys.A, 'norm'):
        norm_A = sys.A.norm(np.inf)
    else:
        # For intervalMatrix or matZonotope, use center
        if hasattr(sys.A, 'center'):
            A_center = sys.A.center()
            norm_A = np.linalg.norm(A_center, np.inf)
        else:
            raise NotImplementedError(f"Cannot compute norm for A type: {type(sys.A)}")
    
    # MATLAB: epsilon = norm_A*r/(obj.taylorTerms + 2);
    epsilon = norm_A * r / (sys.taylorTerms + 2)
    
    if epsilon < 1:
        # MATLAB: phi = norm_A^(obj.taylorTerms+1)*r^(obj.taylorTerms+1)/(factorial(obj.taylorTerms + 1)*(1-epsilon));
        phi = (norm_A ** (sys.taylorTerms + 1) * r ** (sys.taylorTerms + 1) / 
               (math.factorial(sys.taylorTerms + 1) * (1 - epsilon)))
        # MATLAB: Einput = ones(length(obj.A))*interval(-1,1)*phi/norm_A;
        n = sys.nr_of_dims
        # Create interval with scaled bounds
        interval_scaled = Interval(-1, 1) * phi / norm_A
        # Create matrix where each element is the scaled interval
        # Extract center and radius from the interval
        center_val = interval_scaled.center()
        rad_val = interval_scaled.rad()
        # Create IntervalMatrix with center and width matrices
        Einput = IntervalMatrix(np.ones((n, n)) * center_val, np.ones((n, n)) * rad_val)
    else:
        # MATLAB: Einput = interval(-Inf,Inf);
        n = sys.nr_of_dims
        Einput = IntervalMatrix(np.full((n, n), -np.inf), np.full((n, n), np.inf))
        print('Taylor order not high enough')
    
    # Write to object structure
    # MATLAB: obj.inputF=Asum+Einput;
    sys.inputF = Asum + Einput
    
    return sys
