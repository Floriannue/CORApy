"""
initReach_adaptive - computes the reachable continuous set
   for the first time step

Syntax:
    [Rnext,options] = initReach_adaptive(nlnsys,options)

Inputs:
    nlnsys - nonlinearSys object
    options - struct containing the algorithm settings

Outputs:
    Rnext - reachable set
    options - struct containing the algorithm settings

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       14-January-2020
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Dict, Tuple
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def initReach_adaptive(nlnsys: Any, options: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Computes the reachable continuous set for the first time step using adaptive algorithm
    
    Args:
        nlnsys: nonlinearSys object
        options: struct containing the algorithm settings (must contain 'R')
        
    Returns:
        Rnext: reachable set struct with fields:
            - tp: time-point reachable set
            - ti: time-interval reachable set
            - R0: initial set
        options: updated options struct
    """
    
    # Check that options.R exists
    if 'R' not in options:
        raise CORAerror('CORA:wrongInputInConstructor',
                        'options must contain field R (initial reachable set)')
    
    # Call linReach_adaptive
    # NOTE: linReach_adaptive for nonlinearSys needs to be translated first
    # For now, we'll import it and call it
    # MATLAB: [Rti,Rtp,~,options] = linReach_adaptive(nlnsys,options,options.R);
    # In Python, use object.method pattern and pass params explicitly
    params = options.copy() if 'params' not in options else options.get('params', {})
    Rti, Rtp, options = nlnsys.linReach_adaptive(options['R'], params, options)
    
    # store the results
    Rnext = {'tp': Rtp, 'ti': Rti, 'R0': options['R']}
    
    return Rnext, options

