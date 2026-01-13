"""
plus - Overloaded '+' operator for the Minkowski addition of two interval
   matrices or a interval matrix with a matrix

Syntax:
    intMat = plus(summand1, summand2)
    intMat = summand1 + summand2

Inputs:
    summand1 - interval matrix object or numerical matrix
    summand2 - interval matrix object or numerical matrix

Outputs:
    intMat - interval matrix after Minkowski addition

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import numpy as np
from typing import Union
from .intervalMatrix import IntervalMatrix


def plus(summand1: Union[IntervalMatrix, np.ndarray], 
         summand2: Union[IntervalMatrix, np.ndarray]) -> IntervalMatrix:
    """
    Overloaded '+' operator for the Minkowski addition of two interval
    matrices or an interval matrix with a matrix
    
    Args:
        summand1: IntervalMatrix or numeric matrix
        summand2: IntervalMatrix or numeric matrix
        
    Returns:
        intMat: Resulting interval matrix after Minkowski addition
    """
    # Find the interval matrix object
    # MATLAB: if isa(summand1,'intervalMatrix')
    if isinstance(summand1, IntervalMatrix):
        intMat = summand1
        summand = summand2
    # MATLAB: elseif isa(summand2,'intervalMatrix')
    elif isinstance(summand2, IntervalMatrix):
        intMat = summand2
        summand = summand1
    else:
        raise TypeError("At least one argument must be an IntervalMatrix")
    
    # MATLAB: if isa(summand,'intervalMatrix')
    if isinstance(summand, IntervalMatrix):
        # MATLAB: intMat.int = intMat.int + summand.int;
        # Add intervals
        new_int = type(intMat.int)(intMat.int.inf + summand.int.inf, 
                                    intMat.int.sup + summand.int.sup)
        return IntervalMatrix((new_int.inf + new_int.sup) / 2, 
                               (new_int.sup - new_int.inf) / 2)
    
    # MATLAB: elseif isnumeric(summand)
    elif isinstance(summand, (int, float, np.ndarray, np.number)):
        # MATLAB: intMat.int = intMat.int + summand;
        # Add numeric to interval (adds to both inf and sup)
        summand_array = np.asarray(summand)
        # Ensure summand_array has the right shape
        if summand_array.ndim == 0:
            # Scalar - add to all elements
            new_int = type(intMat.int)(intMat.int.inf + summand_array, 
                                        intMat.int.sup + summand_array)
        else:
            # Array - add element-wise
            new_int = type(intMat.int)(intMat.int.inf + summand_array, 
                                        intMat.int.sup + summand_array)
        return IntervalMatrix((new_int.inf + new_int.sup) / 2, 
                               (new_int.sup - new_int.inf) / 2)
    
    else:
        raise TypeError(f"Unsupported addition: {type(intMat)} + {type(summand)}")
