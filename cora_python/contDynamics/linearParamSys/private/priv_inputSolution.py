"""
priv_inputSolution - computes the bloating due to the input

Syntax:
    sys = priv_inputSolution(sys, params, options)

Inputs:
    sys - LinearParamSys object
    params - model parameters (must contain 'U' key, optionally 'uTrans')
    options - options struct (must contain 'reductionTechnique', 'zonotopeOrder', 'originContained')

Outputs:
    sys - LinearParamSys object (modified with Rinput, Rtrans, RV, inputCorr properties)

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import math
import numpy as np
from typing import Any, Dict
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.matrixSet.intervalMatrix import intervalMatrix
from .priv_inputTie import priv_inputTie


def priv_inputSolution(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Computes the reachable set due to input for linear parametric systems
    
    Args:
        sys: LinearParamSys object
        params: Model parameters dict (must contain 'U', optionally 'uTrans')
        options: Options dict (must contain 'reductionTechnique', 'zonotopeOrder', 'originContained')
        
    Returns:
        sys: Modified LinearParamSys object
    """
    # Set of possible inputs
    # MATLAB: V=obj.B*params.U+obj.c;
    V = sys.B * params['U'] + sys.c
    
    # Compute vTrans if possible
    # MATLAB: try; vTrans=obj.B*params.uTrans; catch; vTrans=[]; end
    try:
        vTrans = sys.B * params['uTrans']
    except (KeyError, TypeError):
        vTrans = None
    
    # Load data from object/options structure
    r = sys.stepSize
    
    # Initialize the reachable set due to input
    # MATLAB: inputSet = V*r;
    # MATLAB: intM = eye(obj.nrOfDims)*r; %integral of the mapping matrix
    inputSet = V * r
    intM = np.eye(sys.nr_of_dims) * r  # Integral of the mapping matrix
    
    # Matrix zonotope
    # MATLAB: for i=1:length(obj.power.zono)
    for i in range(len(sys.power['zono'])):
        # MATLAB: taylorTerm = obj.power.zono{i}*(r/factorial(i+1));
        taylorTerm = sys.power['zono'][i] * (r / math.factorial(i + 1))
        # MATLAB: inputSet = inputSet + taylorTerm*V;
        inputSet = inputSet + taylorTerm * V
        # MATLAB: intM = intM + intervalMatrix(taylorTerm);
        from cora_python.matrixSet.intervalMatrix import IntervalMatrix
        intM = intM + IntervalMatrix(taylorTerm)
    
    # Interval matrix
    # MATLAB: for i=(length(obj.power.zono)+1):length(obj.power.int)
    for i in range(len(sys.power['zono']), len(sys.power['int'])):
        # MATLAB: taylorTerm = obj.power.int{i}*(r/factorial(i+1));
        taylorTerm = sys.power['int'][i] * (r / math.factorial(i + 1))
        # MATLAB: inputSet = inputSet + taylorTerm*V;
        inputSet = inputSet + taylorTerm * V
        # MATLAB: intM = intM + taylorTerm;
        intM = intM + taylorTerm
    
    # Remainder term
    # MATLAB: Vabs=zonotope(interval(-1,1)*supremum(abs(interval(V))));
    from cora_python.contSet.zonotope.interval import interval as zonotope_interval
    from cora_python.contSet.interval import Interval
    # supremum is attached to Interval class, use object.supremum()
    V_interval = zonotope_interval(V) if hasattr(V, '__class__') and 'Zonotope' in V.__class__.__name__ else Interval(V, V)
    Vabs_sup = V_interval.abs().supremum()
    Vabs = Zonotope(np.zeros((sys.nr_of_dims, 1)), 
                    np.abs(Vabs_sup).reshape(-1, 1) if Vabs_sup.ndim == 1 else np.abs(Vabs_sup))
    # MATLAB: inputSet = inputSet + obj.E*r*Vabs;
    inputSet = inputSet + sys.E * r * Vabs
    # MATLAB: intM = intM + obj.E*r;
    intM = intM + sys.E * r
    
    # Input solution due certain input
    # MATLAB: inputSetTrans = intM*zonotope(interval(vTrans));
    if vTrans is not None:
        vTrans_zonotope = Zonotope(vTrans, np.zeros((sys.nr_of_dims, 0)))
        inputSetTrans = intM * vTrans_zonotope
    else:
        inputSetTrans = Zonotope(np.zeros((sys.nr_of_dims, 1)), np.zeros((sys.nr_of_dims, 0)))
    
    # Delete zero generators in zonotope representation
    # MATLAB: inputSet=compact_(inputSet,'zeros',eps);
    # MATLAB: inputSetTrans=compact_(inputSetTrans,'zeros',eps);
    # Note: compact_ method should exist on Zonotope, but if not, we skip for now
    if hasattr(inputSet, 'compact_'):
        inputSet = inputSet.compact_('zeros', np.finfo(float).eps)
    if hasattr(inputSetTrans, 'compact_'):
        inputSetTrans = inputSetTrans.compact_('zeros', np.finfo(float).eps)
    
    # Compute additional uncertainty if origin is not contained in input set
    # MATLAB: if options.originContained
    if options.get('originContained', False):
        # MATLAB: inputCorr = zeros(obj.nrOfDims,1);
        inputCorr = np.zeros((sys.nr_of_dims, 1))
    else:
        # Compute inputF
        # MATLAB: obj = priv_inputTie(obj,options);
        sys = priv_inputTie(sys, options)
        # MATLAB: inputCorr = obj.inputF*zonotope(vTrans);
        if vTrans is not None:
            vTrans_zonotope = Zonotope(vTrans, np.zeros((sys.nr_of_dims, 0)))
            inputCorr = sys.inputF * vTrans_zonotope
        else:
            inputCorr = np.zeros((sys.nr_of_dims, 1))
    
    # Write to object structure
    # MATLAB: obj.Rinput = reduce(inputSet + inputSetTrans,options.reductionTechnique,options.zonotopeOrder);
    # MATLAB: obj.Rtrans = inputSetTrans;
    # MATLAB: obj.RV = reduce(inputSet,options.reductionTechnique,options.zonotopeOrder);
    # MATLAB: obj.inputCorr = inputCorr;
    from cora_python.contSet.zonotope.reduce import reduce
    sys.Rinput = reduce(inputSet + inputSetTrans, options['reductionTechnique'], options['zonotopeOrder'])
    sys.Rtrans = inputSetTrans
    sys.RV = reduce(inputSet, options['reductionTechnique'], options['zonotopeOrder'])
    sys.inputCorr = inputCorr
    
    return sys
