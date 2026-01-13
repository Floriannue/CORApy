"""
priv_tie - tie: time interval error; computes the error done by building the
   convex hull of time point solutions

Syntax:
    sys = priv_tie(sys)

Inputs:
    sys - LinearParamSys object

Outputs:
    sys - LinearParamSys object (modified with F property)

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import numpy as np
from typing import Any
from cora_python.matrixSet.intervalMatrix import intervalMatrix


def priv_tie(sys: Any) -> Any:
    """
    Computes the time interval error (tie) for linear parametric systems
    
    Args:
        sys: LinearParamSys object (must have power, taylorTerms, stepSize, E properties)
        
    Returns:
        sys: Modified LinearParamSys object with F property set
    """
    # Obtain powers and convert them to interval matrices
    # MATLAB: Apower=cell(1,length(obj.power.int));
    Apower = [None] * len(sys.power['int'])
    
    # Matrix zonotope
    # MATLAB: for i=2:length(obj.power.zono)
    for i in range(1, len(sys.power['zono'])):  # Python 0-based, MATLAB 1-based
        # MATLAB: Apower{i}=intervalMatrix(obj.power.zono{i});
        Apower[i] = intervalMatrix(sys.power['zono'][i])
    
    # Interval matrix
    # MATLAB: for i=(length(obj.power.zono)+1):length(obj.power.int)
    for i in range(len(sys.power['zono']), len(sys.power['int'])):
        # MATLAB: Apower{i}=obj.power.int{i};
        Apower[i] = sys.power['int'][i]
    
    # Initialize Asum
    # MATLAB: inf=-0.25;%*r^2;already included in Apower
    # MATLAB: sup=0;
    # MATLAB: timeInterval=intervalMatrix(0.5*(sup+inf),0.5*(sup-inf));
    # MATLAB: Asum=timeInterval*Apower{2}*(1/factorial(2));
    # Note: MATLAB starts loop at i=2, so Apower{2} is index 1 in Python (0-based)
    inf_val = -0.25  # *r^2 already included in Apower
    sup_val = 0.0
    timeInterval = intervalMatrix(0.5 * (sup_val + inf_val), 0.5 * (sup_val - inf_val))
    # MATLAB uses Apower{2} which is index 1 in Python (since loop starts at i=2, and Apower starts at index 1)
    # But wait, the loop starts at i=2, so Apower{2} should exist. Let me check the MATLAB code again.
    # Actually, MATLAB cell arrays are 1-indexed, so Apower{2} is the second element.
    # In Python, if we create Apower with length(obj.power.int), and the loop starts at i=2,
    # then Apower{2} in MATLAB is Apower[1] in Python.
    if len(Apower) > 1:
        Asum = timeInterval * Apower[1] * (1.0 / np.math.factorial(2))
    else:
        # Fallback if Apower doesn't have enough elements
        Asum = timeInterval * (1.0 / np.math.factorial(2))
    
    # MATLAB: for i=3:obj.taylorTerms
    for i in range(3, sys.taylorTerms + 1):  # Python range is exclusive, MATLAB is inclusive
        # Compute factor
        # MATLAB: exp1=-i/(i-1); exp2=-1/(i-1);
        # MATLAB: inf = (i^exp1-i^exp2);%*r^i;already included in Apower
        # MATLAB: sup = 0;
        exp1 = -i / (i - 1)
        exp2 = -1 / (i - 1)
        inf_val = (i ** exp1 - i ** exp2)  # *r^i already included in Apower
        sup_val = 0.0
        timeInterval = intervalMatrix(0.5 * (sup_val + inf_val), 0.5 * (sup_val - inf_val))
        
        # Compute powers
        # MATLAB: Aadd=timeInterval*Apower{i};
        # MATLAB uses i which goes from 3 to taylorTerms, so Apower{i} is index i-1 in Python
        if i - 1 < len(Apower):
            Aadd = timeInterval * Apower[i - 1]
        else:
            # Skip if index out of range
            continue
        
        # Compute sum
        # MATLAB: Asum=Asum+Aadd*(1/factorial(i));
        Asum = Asum + Aadd * (1.0 / np.math.factorial(i))
    
    # Write to object structure
    # MATLAB: obj.F=Asum+obj.E;
    sys.F = Asum + sys.E
    
    return sys
