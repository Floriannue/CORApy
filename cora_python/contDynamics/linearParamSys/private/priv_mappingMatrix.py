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
    sys.mappingMatrixSet['zono'] = eZ + eImid
    sys.mappingMatrixSet['int'] = eI + (-eImid)
    
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
        sys.power['zono_input'].append(sys.power['zono'][i] * (sys.stepSize / np.math.factorial(i + 1)))
    
    # MATLAB: for i=1:length(obj.power.int)
    sys.power['int_input'] = []
    for i in range(len(sys.power['int'])):
        # MATLAB: obj.power.int_input{i} = obj.power.int{i}*(obj.stepSize/factorial(i+1));
        sys.power['int_input'].append(sys.power['int'][i] * (sys.stepSize / np.math.factorial(i + 1)))
    
    # Remainder
    # MATLAB: obj.E = E;
    sys.E = E
    
    return sys
