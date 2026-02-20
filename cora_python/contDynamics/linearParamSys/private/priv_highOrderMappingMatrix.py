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

import math
import numpy as np
from typing import Any
# center is attached to Interval class, use object.center()
from cora_python.matrixSet.matZonotope import matZonotope
from cora_python.matrixSet.intervalMatrix import IntervalMatrix


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
            eZ = eZ + zPow[i - 1] * (1.0 / math.factorial(i))
            # MATLAB: eZ_input = eZ_input + zPow{i}*(r/factorial(i+1));
            eZ_input = eZ_input + zPow[i - 1] * (r / math.factorial(i + 1))
    
    # Interval computations
    # MATLAB: eI = zeros(obj.nrOfDims);
    # MATLAB: eI_input = zeros(obj.nrOfDims);
    eI = np.zeros((sys.nr_of_dims, sys.nr_of_dims))
    eI_input = np.zeros((sys.nr_of_dims, sys.nr_of_dims))
    
    # MATLAB: for i=(intermediateTerms+1):obj.taylorTerms
    # Check if iPow is a list and not empty (expmOneParam sets iPow = [])
    if isinstance(iPow, list) and len(iPow) > 0:
        for i in range(intermediateTerms + 1, sys.taylorTerms + 1):
            # MATLAB: eI = eI + iPow{i}*(1/factorial(i));
            if i - 1 < len(iPow) and iPow[i - 1] is not None:  # Python 0-based indexing
                eI = eI + iPow[i - 1] * (1.0 / math.factorial(i))
                # MATLAB: eI_input = eI_input + iPow{i}*(r/factorial(i+1));
                eI_input = eI_input + iPow[i - 1] * (r / math.factorial(i + 1))
    
    # MATLAB: eI = eI + E;
    # MATLAB: eI_input = eI_input + E*r;
    # Convert eI and eI_input to IntervalMatrix before adding E
    from cora_python.matrixSet.intervalMatrix import IntervalMatrix
    if hasattr(E, 'int'):
        # E is an IntervalMatrix, convert eI and eI_input to IntervalMatrix
        if not isinstance(eI, IntervalMatrix):
            # eI is numpy array, convert to IntervalMatrix with zero width
            eI = IntervalMatrix(eI, np.zeros_like(eI))
        if not isinstance(eI_input, IntervalMatrix):
            eI_input = IntervalMatrix(eI_input, np.zeros_like(eI_input))
        # Add E (both are IntervalMatrix now)
        # MATLAB: intMat.int = intMat.int + summand.int (for intervalMatrix + intervalMatrix)
        # For now, manually add the intervals
        eI_int = eI.int
        E_int = E.int
        eI_new_int = type(eI_int)(eI_int.inf + E_int.inf, eI_int.sup + E_int.sup)
        eI = IntervalMatrix((eI_new_int.inf + eI_new_int.sup) / 2, (eI_new_int.sup - eI_new_int.inf) / 2)
        
        eI_input_int = eI_input.int
        eI_input_new_int = type(eI_input_int)(eI_input_int.inf + E_int.inf * r, eI_input_int.sup + E_int.sup * r)
        eI_input = IntervalMatrix((eI_input_new_int.inf + eI_input_new_int.sup) / 2, (eI_input_new_int.sup - eI_input_new_int.inf) / 2)
    else:
        # E is a numpy array
        eI = eI + E
        eI_input = eI_input + E * r
    
    # Center of interval computations
    # MATLAB: eImid = center(eI.int);
    # MATLAB: eImid_input = center(eI_input.int);
    if isinstance(eI, IntervalMatrix):
        eImid = eI.int.center()
        eImid_input = eI_input.int.center()
    else:
        # eI is a numpy array
        eImid = eI
        eImid_input = eI_input
    
    # Save results
    # MATLAB: obj.mappingMatrixSet.highOrderZono = eZ + eImid;
    # MATLAB: obj.mappingMatrixSet.highOrderInt = eI + (-eImid);
    # eZ is matZonotope, eImid is numpy array (center)
    if isinstance(eZ, matZonotope):
        sys.mappingMatrixSet['highOrderZono'] = matZonotope(eZ.C + eImid, eZ.G)
    else:
        sys.mappingMatrixSet['highOrderZono'] = eZ + eImid
    
    # eI is IntervalMatrix, eImid is numpy array (center)
    # MATLAB: intMat.int = intMat.int + summand (adds to both inf and sup)
    if isinstance(eI, IntervalMatrix):
        eI_int = eI.int
        new_inf = eI_int.inf - eImid
        new_sup = eI_int.sup - eImid
        new_center = (new_inf + new_sup) / 2
        new_width = (new_sup - new_inf) / 2
        sys.mappingMatrixSet['highOrderInt'] = IntervalMatrix(new_center, new_width)
    else:
        sys.mappingMatrixSet['highOrderInt'] = eI - eImid
    
    # MATLAB: obj.mappingMatrixSet.highOrderZonoInput = eZ_input + eImid_input;
    # MATLAB: obj.mappingMatrixSet.highOrderIntInput = eI_input + (-eImid_input);
    if isinstance(eZ_input, matZonotope):
        sys.mappingMatrixSet['highOrderZonoInput'] = matZonotope(eZ_input.C + eImid_input, eZ_input.G)
    else:
        sys.mappingMatrixSet['highOrderZonoInput'] = eZ_input + eImid_input
    
    if isinstance(eI_input, IntervalMatrix):
        eI_input_int = eI_input.int
        new_inf = eI_input_int.inf - eImid_input
        new_sup = eI_input_int.sup - eImid_input
        new_center = (new_inf + new_sup) / 2
        new_width = (new_sup - new_inf) / 2
        sys.mappingMatrixSet['highOrderIntInput'] = IntervalMatrix(new_center, new_width)
    else:
        sys.mappingMatrixSet['highOrderIntInput'] = eI_input - eImid_input
    
    return sys
