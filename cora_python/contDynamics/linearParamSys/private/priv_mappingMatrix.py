"""
priv_mappingMatrix - computes the set of matrices which map the states for the
   next point in time.

Syntax:
    sys = priv_mappingMatrix(sys, params, options)

Inputs:
    sys - LinearParamSys object
    params - model parameters
    options - options struct (must contain 'intermediateTerms')

Outputs:
    sys - LinearParamSys object (modified with mappingMatrixSet, power, E properties)

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import math
import numpy as np
from typing import Any, Dict
from cora_python.matrixSet.matZonotope import matZonotope
from cora_python.matrixSet.intervalMatrix import intervalMatrix
# center is attached to Interval class, use object.center()


def priv_mappingMatrix(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Computes mapping matrices for linear parametric systems
    
    Args:
        sys: LinearParamSys object (must have A, stepSize, taylorTerms, constParam)
        params: Model parameters dict
        options: Options dict (must contain 'intermediateTerms')
        
    Returns:
        sys: Modified LinearParamSys object
    """
    # MATLAB: if isa(obj.A,'matZonotope') && (obj.A.numgens() == 1) && obj.constParam
    if isinstance(sys.A, matZonotope) and sys.A.numgens() == 1 and sys.constParam:
        # MATLAB: [eZ,eI,zPow,iPow,E,RconstInput] = expmOneParam(obj.A,obj.stepSize,obj.taylorTerms,params);
        # NOTE: expmOneParam needs to be translated
        from cora_python.matrixSet.matZonotope.expmOneParam import expmOneParam
        eZ, eI, zPow, iPow, E, RconstInput = expmOneParam(sys.A, sys.stepSize, sys.taylorTerms, params)
        # Constant input solution
        sys.Rtrans = RconstInput
    else:
        # Multiply system matrix with stepSize
        # MATLAB: A = obj.A * obj.stepSize;
        A = sys.A * sys.stepSize
        
        # Convert IntervalMatrix to matZonotope if needed (expmMixed expects matZonotope)
        from cora_python.matrixSet.intervalMatrix import IntervalMatrix
        if isinstance(A, IntervalMatrix):
            # Convert IntervalMatrix to matZonotope: center is center, generators are zero
            A_center = A.center()
            A_rad = A.rad()
            # Create matZonotope with center and zero generators (non-parametric)
            A = matZonotope(A_center, np.zeros((*A_center.shape, 0)))
            # Note: The radius information is lost, but this matches MATLAB behavior
            # where IntervalMatrix is converted to matZonotope for expmMixed
        
        # Obtain mapping matrix
        # Mixed computation: first terms are matrix zonotopes, further terms are interval matrices
        # MATLAB: if obj.constParam
        if sys.constParam:
            # MATLAB: [eZ,eI,zPow,iPow,E]= expmMixed(A,obj.stepSize,options.intermediateTerms,obj.taylorTerms);
            from cora_python.matrixSet.matZonotope.expmMixed import expmMixed
            eZ, eI, zPow, iPow, E = expmMixed(A, sys.stepSize, options['intermediateTerms'], sys.taylorTerms)
        else:
            # MATLAB: [eZ,eI,zPow,iPow,E]= expmIndMixed(A,options.intermediateTerms,obj.taylorTerms);
            from cora_python.matrixSet.matZonotope.expmIndMixed import expmIndMixed
            eZ, eI, zPow, iPow, E = expmIndMixed(A, options['intermediateTerms'], sys.taylorTerms)
    
    # Save results
    # MATLAB: eImid=center(eI.int);
    if hasattr(eI, 'int'):
        eImid = eI.int.center()
    else:
        # eI might already be a numpy array
        eImid = eI if isinstance(eI, np.ndarray) else np.zeros((sys.nr_of_dims, sys.nr_of_dims))
    
    # Mapping matrices
    # MATLAB: obj.mappingMatrixSet.zono = eZ + eImid;
    # MATLAB: obj.mappingMatrixSet.int = eI + (-eImid);
    # Convert eImid to matZonotope for addition with eZ (if eZ is matZonotope)
    if isinstance(eZ, matZonotope):
        # eImid is numeric, add to center of matZonotope
        sys.mappingMatrixSet['zono'] = matZonotope(eZ.C + eImid, eZ.G)
    else:
        sys.mappingMatrixSet['zono'] = eZ + eImid
    
    # For interval part, eI is intervalMatrix, eImid is numeric
    # MATLAB: intMat.int = intMat.int + summand (adds to both inf and sup)
    if hasattr(eI, 'int'):
        # eI.int is an Interval, add -eImid to both inf and sup
        from cora_python.matrixSet.intervalMatrix import IntervalMatrix
        eI_int = eI.int
        # Create new interval with updated bounds
        new_inf = eI_int.inf - eImid
        new_sup = eI_int.sup - eImid
        # IntervalMatrix constructor expects center and width
        new_center = (new_inf + new_sup) / 2
        new_width = (new_sup - new_inf) / 2
        sys.mappingMatrixSet['int'] = IntervalMatrix(new_center, new_width)
    else:
        sys.mappingMatrixSet['int'] = eI - eImid
    
    # Powers
    # MATLAB: obj.power.zono = zPow;
    # MATLAB: obj.power.int = iPow;
    sys.power['zono'] = zPow
    sys.power['int'] = iPow
    
    # Powers for input computation
    # MATLAB: for i=1:length(obj.power.zono)
    sys.power['zono_input'] = []
    for i in range(len(sys.power['zono'])):
        # MATLAB: obj.power.zono_input{i} = obj.power.zono{i}*(obj.stepSize/factorial(i+1));
        sys.power['zono_input'].append(sys.power['zono'][i] * (sys.stepSize / math.factorial(i + 1)))
    
    # MATLAB: for i=1:length(obj.power.int)
    sys.power['int_input'] = []
    for i in range(len(sys.power['int'])):
        # MATLAB: obj.power.int_input{i} = obj.power.int{i}*(obj.stepSize/factorial(i+1));
        sys.power['int_input'].append(sys.power['int'][i] * (sys.stepSize / math.factorial(i + 1)))
    
    # Remainder
    # MATLAB: obj.E = E;
    sys.E = E
    
    return sys
