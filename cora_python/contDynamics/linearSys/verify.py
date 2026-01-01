"""
verify - verifies a hybrid system against the given specifications

Syntax:
    R = verify(linsys,params,options,spec)

Inputs:
    linsys - contDynamics object
    params - parameter defining the reachability problem
    options - options for the computation of the reachable set
       .verifyAlg: 'reachavoid:zonotope','reachavoid:supportFunc',
                   'stl:kochdumper','stl:seidl'
    spec - object of class specification (reach-avoid) or stl

Outputs:
    res - true/false whether specifications are satisfied
    varargout - depending on selected verification algorithm

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contDynamics/verify

Authors:       Tobias Ladner
Written:       19-October-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Dict, Tuple, Optional
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.macros.CHECKS_ENABLED import CHECKS_ENABLED
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def verify(linsys: Any, params: Dict[str, Any], options: Dict[str, Any], spec: Any) -> Tuple[bool, ...]:
    """
    Verifies a linear system against the given specifications
    
    Args:
        linsys: linearSys object
        params: parameter defining the reachability problem
        options: options for the computation of the reachable set
            verifyAlg: 'reachavoid:zonotope','reachavoid:supportFunc',
                      'stl:kochdumper','stl:seidl'
        spec: object of class specification (reach-avoid) or stl
        
    Returns:
        res: true/false whether specifications are satisfied
        Additional return values depending on selected verification algorithm
    """
    
    # 1. check number of inputs
    # narginchk(4,4); # Python handles this differently
    
    # 2. validate inputs
    if CHECKS_ENABLED:
        inputArgsCheck([
            [linsys, 'att', 'linearSys'],
            [params, 'att', 'dict'],
            [options, 'att', 'dict'],
            [spec, 'att', ['specification', 'stl']]
        ])
    
    # 3. select verification algorithm
    verifyAlg, options = _aux_selectVerifyAlgorithm(linsys, params, options, spec)
    
    # 5. call verify function depending on given specification
    if verifyAlg == 'reachavoid:zonotope':
        from .private.priv_verifyRA_zonotope import priv_verifyRA_zonotope
        res, *varargout = priv_verifyRA_zonotope(linsys, params, options, spec)
        return (res, *varargout)
    elif verifyAlg == 'reachavoid:supportFunc':
        from .private.priv_verifyRA_supportFunc import priv_verifyRA_supportFunc
        res, *varargout = priv_verifyRA_supportFunc(linsys, params, options, spec)
        return (res, *varargout)
    elif verifyAlg == 'stl:kochdumper':
        from .private.priv_verifySTL_kochdumper import priv_verifySTL_kochdumper
        res, *varargout = priv_verifySTL_kochdumper(linsys, params, options, spec)
        return (res, *varargout)
    elif verifyAlg == 'stl:seidl':
        from .verifySTL_seidl import verifySTL_seidl
        res = verifySTL_seidl(linsys, params, options, spec)
        return (res,)
    else:
        raise CORAerror('CORA:noops', linsys, params, options, spec)


def _aux_selectVerifyAlgorithm(linsys: Any, params: Dict[str, Any], 
                               options: Dict[str, Any], spec: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Select verification algorithm based on options and specification type
    """
    # check if algorithm is given
    if 'verifyAlg' in options:
        verifyAlg = options['verifyAlg']
        options = options.copy()  # Don't modify original
        del options['verifyAlg']
    else:
        # Check if spec is stl or specification
        spec_type = type(spec).__name__ if hasattr(spec, '__class__') else str(type(spec))
        if 'stl' in spec_type.lower():
            verifyAlg = 'stl:seidl'
        elif 'specification' in spec_type.lower():
            # the supportFunc algorithm is definitely faster for specifications
            # given as halfspaces. For others, one would need check which algorithm
            # is faster. Default is 'reachavoid:supportFunc' for now ...
            verifyAlg = 'reachavoid:supportFunc'
        else:
            raise CORAerror('CORA:wrongValue', 'options.verifyAlg',
                          'Verification algorithm not specified and unable to determine one automatically.')
    
    if CHECKS_ENABLED:
        # validate chosen verification algorithm
        possibleValues = [
            'reachavoid:zonotope', 'reachavoid:supportFunc',
            'stl:kochdumper', 'stl:seidl'
        ]
        if verifyAlg not in possibleValues:
            validrange = "'" + "', '".join(possibleValues) + "'"
            raise CORAerror('CORA:wrongValue', 'options.verifyAlg', validrange)
        
        spec_type = type(spec).__name__ if hasattr(spec, '__class__') else str(type(spec))
        if 'stl' in spec_type.lower() and not verifyAlg.startswith('stl:'):
            raise CORAerror('CORA:wrongValue', 'options.verifyAlg',
                          'Given stl specifications need an algorithm designed for stl.')
        
        # check specification
        if 'specification' in spec_type.lower():
            # Check if it's a list/array of specifications
            if isinstance(spec, (list, tuple)):
                spec_list = spec
            else:
                spec_list = [spec]
            
            for i, s in enumerate(spec_list):
                # check each specification
                if hasattr(s, 'type') and s.type == 'logic':
                    if not verifyAlg.startswith('stl:'):
                        raise CORAerror('CORA:wrongValue', 'options.verifyAlg',
                                      'Given logic specifications need an algorithm designed for stl.')
                else:
                    # only reachavoid is implemented
                    if not verifyAlg.startswith('reachavoid:'):
                        raise CORAerror('CORA:wrongValue', 'options.verifyAlg',
                                      'Given reach-avoid specifications need an algorithm designed for reachavoid.')
    
    return verifyAlg, options

