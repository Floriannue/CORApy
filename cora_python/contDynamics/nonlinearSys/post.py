"""
post - computes the reachable continuous set for one time step of a
   nonlinear system by overapproximative linearization

Syntax:
    [Rnext,options] = post(nlnsys,R,params,options)

Inputs:
    nlnsys - nonlinearSys object
    R - reachable set of the previous time step
    params - model parameters
    options - options for the computation of the reachable set

Outputs:
    Rnext - reachable set of the next time step
    options - options for the computation of the reachable set

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff
Written:       03-January-2008
Last update:   29-June-2009
               10-August-2016
               19-November-2017
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Tuple, Any, Dict
from cora_python.contSet.polyZonotope import PolyZonotope


def post(nlnsys: Any, R: Dict[str, Any], params: Dict[str, Any],
        options: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Computes the reachable continuous set for one time step of a nonlinear system
    by overapproximative linearization
    
    Args:
        nlnsys: nonlinearSys object
        R: reachable set of the previous time step (dict with 'tp' and 'ti' keys)
        params: model parameters
        options: options for the computation of the reachable set (must contain 'alg',
                'reductionTechnique', 'zonotopeOrder', and optionally 'polyZono')
        
    Returns:
        Rnext: reachable set of the next time step (dict with 'tp', 'ti', 'R0' keys)
        options: options for the computation of the reachable set (may be modified)
    """
    
    # potentially restructure the polynomial zonotope
    # MATLAB: if strcmp(options.alg,'poly') && isa(R.tp{1}.set,'polyZonotope') && ...
    #    isfield(options,'polyZono') && ~isinf(options.polyZono.maxPolyZonoRatio)
    should_restructure = (options['alg'] == 'poly' and 
                         len(R['tp']) > 0 and
                         isinstance(R['tp'][0]['set'], PolyZonotope) and
                         'polyZono' in options and
                         'maxPolyZonoRatio' in options['polyZono'] and
                         not np.isinf(options['polyZono']['maxPolyZonoRatio']))
    
    if should_restructure:
        # MATLAB: for i=1:length(R.tp)
        for i in range(len(R['tp'])):
            
            # compute ratio of dependent to independent part 
            # MATLAB: ratio = approxVolumeRatio(R.tp{i}.set,options.polyZono.volApproxMethod);
            from cora_python.contSet.polyZonotope.approxVolumeRatio import approxVolumeRatio
            ratio = approxVolumeRatio(R['tp'][i]['set'], options['polyZono']['volApproxMethod'])
            
            # restructure the polynomial zonotope
            # MATLAB: if ratio > options.polyZono.maxPolyZonoRatio
            if ratio > options['polyZono']['maxPolyZonoRatio']:
                # MATLAB: R.tp{i}.set = restructure(R.tp{i}.set, ...
                #                                options.polyZono.restructureTechnique, ...
                #                                options.polyZono.maxDepGenOrder);
                from cora_python.contSet.polyZonotope.restructure import restructure
                R['tp'][i]['set'] = restructure(
                    R['tp'][i]['set'],
                    options['polyZono']['restructureTechnique'],
                    options['polyZono']['maxDepGenOrder']
                )
    
    # In contrast to the linear system: the nonlinear system has to be constantly
    # initialized due to the linearization procedure
    # MATLAB: [Rnext,options] = initReach(nlnsys,R.tp,params,options);
    from cora_python.contDynamics.nonlinearSys.initReach import initReach
    Rnext, options = initReach(nlnsys, R['tp'], params, options)
    
    # reduce zonotopes
    # MATLAB: for i=1:length(Rnext.tp)
    for i in range(len(Rnext['tp'])):
        # MATLAB: if ~representsa_(Rnext.tp{i}.set,'emptySet',eps)
        eps = np.finfo(float).eps
        set_obj = Rnext['tp'][i]['set']
        is_empty = set_obj.representsa_('emptySet', eps)

        if not is_empty:
            # MATLAB: Rnext.tp{i}.set=reduce(Rnext.tp{i}.set,options.reductionTechnique,options.zonotopeOrder);
            Rnext['tp'][i]['set'] = Rnext['tp'][i]['set'].reduce(
                options['reductionTechnique'], options['zonotopeOrder']
            )
            
            # MATLAB: Rnext.ti{i}=reduce(Rnext.ti{i},options.reductionTechnique,options.zonotopeOrder);
            Rnext['ti'][i] = Rnext['ti'][i].reduce(
                options['reductionTechnique'], options['zonotopeOrder']
            )
    
    # delete redundant reachable sets
    # MATLAB: Rnext = deleteRedundantSets(Rnext,R,options);
    from cora_python.contDynamics.contDynamics.private.deleteRedundantSets import deleteRedundantSets
    Rnext = deleteRedundantSets(Rnext, R, options)
    
    return Rnext, options

