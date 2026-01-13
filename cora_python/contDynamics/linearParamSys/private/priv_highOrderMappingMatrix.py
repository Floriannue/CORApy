"""
priv_highOrderMappingMatrix - computes a mapping matrix set without the first
   two orders

Syntax:
    sys = priv_highOrderMappingMatrix(sys, intermediateTerms)

Inputs:
    sys - LinearParamSys object
    intermediateTerms - order until which the original matrix set representation is used

Outputs:
    sys - LinearParamSys object (modified with highOrder mapping matrices)

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import numpy as np
from typing import Any
# center is attached to Interval class, use object.center()


def priv_highOrderMappingMatrix(sys: Any, intermediateTerms: int) -> Any:
    """
    Computes high-order mapping matrices for linear parametric systems
    
    Args:
        sys: LinearParamSys object (must have power, taylorTerms, stepSize, E, mappingMatrixSet properties)
        intermediateTerms: Order until which original matrix set representation is used
        
    Returns:
        sys: Modified LinearParamSys object with highOrder mapping matrices
    """
    # Powers
    zPow = sys.power['zono']
    iPow = sys.power['int']
    
    # Remainder
    E = sys.E
    
    # Step size
    r = sys.stepSize
    
    # Zonotope computations
    # MATLAB: eZ = zeros(obj.nrOfDims);
    # MATLAB: eZ_input = zeros(obj.nrOfDims);
    eZ = np.zeros((sys.nr_of_dims, sys.nr_of_dims))
    eZ_input = np.zeros((sys.nr_of_dims, sys.nr_of_dims))
    
    # MATLAB: for i=3:intermediateTerms
    for i in range(3, intermediateTerms + 1):
        # MATLAB: eZ = eZ + zPow{i}*(1/factorial(i));
        if i - 1 < len(zPow):  # Python 0-based indexing
            eZ = eZ + zPow[i - 1] * (1.0 / np.math.factorial(i))
            # MATLAB: eZ_input = eZ_input + zPow{i}*(r/factorial(i+1));
            eZ_input = eZ_input + zPow[i - 1] * (r / np.math.factorial(i + 1))
    
    # Interval computations
    # MATLAB: eI = zeros(obj.nrOfDims);
    # MATLAB: eI_input = zeros(obj.nrOfDims);
    eI = np.zeros((sys.nr_of_dims, sys.nr_of_dims))
    eI_input = np.zeros((sys.nr_of_dims, sys.nr_of_dims))
    
    # MATLAB: for i=(intermediateTerms+1):obj.taylorTerms
    for i in range(intermediateTerms + 1, sys.taylorTerms + 1):
        # MATLAB: eI = eI + iPow{i}*(1/factorial(i));
        if i - 1 < len(iPow):  # Python 0-based indexing
            eI = eI + iPow[i - 1] * (1.0 / np.math.factorial(i))
            # MATLAB: eI_input = eI_input + iPow{i}*(r/factorial(i+1));
            eI_input = eI_input + iPow[i - 1] * (r / np.math.factorial(i + 1))
    
    # MATLAB: eI = eI + E;
    # MATLAB: eI_input = eI_input + E*r;
    eI = eI + E
    eI_input = eI_input + E * r
    
    # Center of interval computations
    # MATLAB: eImid = center(eI.int);
    # MATLAB: eImid_input = center(eI_input.int);
    # Note: eI and eI_input are numpy arrays, not interval matrices at this point
    # We need to convert them to interval matrices first if E is an interval matrix
    if hasattr(E, 'int'):
        # E is an interval matrix, so eI should be converted
        from cora_python.matrixSet.intervalMatrix import intervalMatrix
        if not isinstance(eI, intervalMatrix):
            eI = intervalMatrix(eI, np.zeros_like(eI))
        if not isinstance(eI_input, intervalMatrix):
            eI_input = intervalMatrix(eI_input, np.zeros_like(eI_input))
        eImid = eI.int.center()
        eImid_input = eI_input.int.center()
    else:
        # E is a numpy array, so eI is also a numpy array
        eImid = eI
        eImid_input = eI_input
    
    # Save results
    # MATLAB: obj.mappingMatrixSet.highOrderZono = eZ + eImid;
    # MATLAB: obj.mappingMatrixSet.highOrderInt = eI + (-eImid);
    sys.mappingMatrixSet['highOrderZono'] = eZ + eImid
    sys.mappingMatrixSet['highOrderInt'] = eI + (-eImid)
    
    # MATLAB: obj.mappingMatrixSet.highOrderZonoInput = eZ_input + eImid_input;
    # MATLAB: obj.mappingMatrixSet.highOrderIntInput = eI_input + (-eImid_input);
    sys.mappingMatrixSet['highOrderZonoInput'] = eZ_input + eImid_input
    sys.mappingMatrixSet['highOrderIntInput'] = eI_input + (-eImid_input)
    
    return sys
