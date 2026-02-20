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

import math
import numpy as np
from typing import Any
from cora_python.matrixSet.intervalMatrix.intervalMatrix import IntervalMatrix


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
    # In MATLAB, cell arrays are 1-indexed
    # We need enough elements to store powers from index 2 to taylorTerms
    max_index = max(len(sys.power['zono']), len(sys.power['int']), sys.taylorTerms) if sys.power['int'] else max(len(sys.power['zono']), sys.taylorTerms)
    Apower = [None] * (max_index + 1)  # +1 because MATLAB is 1-indexed, we need indices 0..max_index
    
    # Matrix zonotope
    # MATLAB: for i=2:length(obj.power.zono)
    # MATLAB uses 1-based indexing, so Apower{2} is index 1 in Python
    for i in range(1, len(sys.power['zono'])):  # Python 0-based, MATLAB 1-based
        # MATLAB: Apower{i}=intervalMatrix(obj.power.zono{i});
        # MATLAB i=2 means Python index 1, MATLAB i=3 means Python index 2, etc.
        # So Python index = MATLAB index - 1
        Apower[i] = IntervalMatrix(sys.power['zono'][i])
    
    # Interval matrix
    # MATLAB: for i=(length(obj.power.zono)+1):length(obj.power.int)
    # MATLAB uses 1-based indexing
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
    timeInterval = IntervalMatrix(0.5 * (sup_val + inf_val), 0.5 * (sup_val - inf_val))
    # MATLAB uses Apower{2} which is index 1 in Python (since loop starts at i=2, and Apower starts at index 1)
    # But wait, the loop starts at i=2, so Apower{2} should exist. Let me check the MATLAB code again.
    # Actually, MATLAB cell arrays are 1-indexed, so Apower{2} is the second element.
    # In Python, if we create Apower with length(obj.power.int), and the loop starts at i=2,
    # then Apower{2} in MATLAB is Apower[1] in Python.
    if len(Apower) > 1 and Apower[1] is not None:
        # MATLAB: Asum=timeInterval*Apower{2}*(1/factorial(2));
        # Handle scalar multiplication manually (IntervalMatrix doesn't support scalar * yet)
        temp = timeInterval * Apower[1]
        scalar = 1.0 / math.factorial(2)
        # Multiply interval by scalar
        Asum = IntervalMatrix(temp.int.inf * scalar, temp.int.sup * scalar - temp.int.inf * scalar)
    else:
        # Fallback if Apower doesn't have enough elements
        scalar = 1.0 / math.factorial(2)
        Asum = IntervalMatrix(timeInterval.int.inf * scalar, timeInterval.int.sup * scalar - timeInterval.int.inf * scalar)
    
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
        timeInterval = IntervalMatrix(0.5 * (sup_val + inf_val), 0.5 * (sup_val - inf_val))
        
        # Compute powers
        # MATLAB: Aadd=timeInterval*Apower{i};
        # MATLAB uses i which goes from 3 to taylorTerms, so Apower{i} is index i-1 in Python
        # MATLAB i=3 means Apower{3} which is index 2 in Python
        if i - 1 < len(Apower) and Apower[i - 1] is not None:
            Aadd = timeInterval * Apower[i - 1]
        else:
            # Skip if index out of range or None
            continue
        
        # Compute sum
        # MATLAB: Asum=Asum+Aadd*(1/factorial(i));
        # Handle scalar multiplication and addition manually
        scalar = 1.0 / math.factorial(i)
        Aadd_scaled = IntervalMatrix(Aadd.int.inf * scalar, Aadd.int.sup * scalar - Aadd.int.inf * scalar)
        # Manual addition: add intervals
        Asum_new_int_inf = Asum.int.inf + Aadd_scaled.int.inf
        Asum_new_int_sup = Asum.int.sup + Aadd_scaled.int.sup
        Asum = IntervalMatrix((Asum_new_int_inf + Asum_new_int_sup) / 2, (Asum_new_int_sup - Asum_new_int_inf) / 2)
    
    # Write to object structure
    # MATLAB: obj.F=Asum+obj.E;
    # Handle addition manually if E is IntervalMatrix
    if isinstance(sys.E, IntervalMatrix):
        F_new_int_inf = Asum.int.inf + sys.E.int.inf
        F_new_int_sup = Asum.int.sup + sys.E.int.sup
        sys.F = IntervalMatrix((F_new_int_inf + F_new_int_sup) / 2, (F_new_int_sup - F_new_int_inf) / 2)
    else:
        sys.F = Asum + sys.E
    
    return sys
