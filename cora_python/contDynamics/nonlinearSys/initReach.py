"""
initReach - computes the reachable continuous set for the first time step

Syntax:
    [Rnext,options] = initReach(nlnsys,Rinit,params,options)

Inputs:
    nlnsys - nonlinearSys object
    Rinit - initial reachable set
    params - model parameters
    options - struct containing the algorithm settings

Outputs:
    Rfirst - first reachable set
    options - struct containing the algorithm settings

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff, Niklas Kochdumper
Written:       29-October-2007 
Last update:   04-January-2008
               27-April-2009
               16-August-2016
               17-May-2019
               02-January-2020 (NK, cleaned up and simplified code)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Tuple, Any, Dict, List, Optional, Union



def initReach(nlnsys: Any, Rinit: Union[Any, List[Dict[str, Any]]], 
              params: Dict[str, Any], options: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Computes the reachable continuous set for the first time step
    
    Args:
        nlnsys: nonlinearSys object
        Rinit: initial reachable set (can be a single set or a list of dicts with 'set' and 'error' keys)
        params: model parameters
        options: struct containing the algorithm settings (must contain 'alg', 'maxError')
        
    Returns:
        Rnext: dict with keys 'tp', 'ti', 'R0' containing reachable sets
        options: struct containing the algorithm settings (may be modified)
    """
    
    # initialization for the case that it is the first time step
    # MATLAB: if ~iscell(Rinit)
    if not isinstance(Rinit, list):
        # MATLAB: R{1}.set = Rinit;
        # MATLAB: R{1}.error = zeros(size(options.maxError));
        # MATLAB: Rinit = R;
        # Set default maxError if not present (default: Inf for each dimension)
        if 'maxError' not in options:
            # MATLAB default: maxError = Inf(nrOfDims, 1)
            nr_of_dims = getattr(nlnsys, 'nr_of_dims', None)
            if nr_of_dims is None:
                nr_of_dims = getattr(nlnsys, 'nrOfDims', 6)  # fallback to 6 if not found
            options['maxError'] = np.full((nr_of_dims, 1), np.inf)
        R = [{'set': Rinit, 'error': np.zeros_like(options['maxError'])}]
        Rinit = R
    
    # compute reachable set using the options.alg = 'linRem' algorithm
    # MATLAB: if strcmp(options.alg,'linRem')
    if options['alg'] == 'linRem':
        # MATLAB: [Rnext,options] = aux_initReach_linRem(nlnsys,Rinit,params,options);
        Rnext, options = aux_initReach_linRem(nlnsys, Rinit, params, options)
        return Rnext, options
    
    # loop over all parallel sets
    # MATLAB: setCounter = 1; Rtp = {}; Rti = {}; R0 = {};
    setCounter = 1
    Rtp = []
    Rti = []
    R0 = []
    
    # MATLAB: for i = 1:length(Rinit)
    for i in range(len(Rinit)):
        
        # compute reachable set of abstraction
        # MATLAB: [Rtemp_ti,Rtemp_tp,dimForSplit,opts] = ...
        #            linReach(nlnsys,Rinit{i},params,options);
        # linReach is implemented in cora_python.contDynamics.contDynamics.linReach
        from cora_python.contDynamics.contDynamics.linReach import linReach
        Rtemp_ti, Rtemp_tp, dimForSplit, opts = linReach(nlnsys, Rinit[i], params, options)
        
        # save POpt (has to be returned by reach)
        # MATLAB: if isfield(opts,'POpt')
        if 'POpt' in opts:
            # MATLAB: options.POpt = opts.POpt;
            options['POpt'] = opts['POpt']
        
        # check if initial set has to be split
        # MATLAB: if isempty(dimForSplit)
        if dimForSplit is None or (isinstance(dimForSplit, (list, np.ndarray)) and len(dimForSplit) == 0):
            # no splitting
            # MATLAB: Rtp{setCounter} = Rtemp_tp;
            # MATLAB: Rtp{setCounter}.prev = i;
            Rtemp_tp_dict = Rtemp_tp if isinstance(Rtemp_tp, dict) else {'set': Rtemp_tp}
            Rtemp_tp_dict['prev'] = i
            Rtp.append(Rtemp_tp_dict)
            
            # MATLAB: Rti{setCounter} = Rtemp_ti;
            Rti.append(Rtemp_ti)
            
            # MATLAB: R0{setCounter} = Rinit{i};
            R0.append(Rinit[i])
            
            setCounter += 1
        else:
            # splitting
            # MATLAB: Rtmp = split(Rinit{i}.set,dimForSplit);
            set_obj = Rinit[i]['set']
            Rtmp = set_obj.split(dimForSplit)

            
            # MATLAB: Rsplit{1}.set = Rtmp{1};
            # MATLAB: Rsplit{2}.set = Rtmp{2};
            Rsplit = [
                {'set': Rtmp[0]},
                {'set': Rtmp[1]}
            ]
            
            # reset the linearization error
            # MATLAB: Rsplit{1}.error = zeros(size(options.maxError));
            # MATLAB: Rsplit{2}.error = zeros(size(options.maxError));
            Rsplit[0]['error'] = np.zeros_like(options['maxError'])
            Rsplit[1]['error'] = np.zeros_like(options['maxError'])
            
            # recursively compute the reachable set for the split sets
            # MATLAB: Rres = initReach(nlnsys,Rsplit,params,options);
            Rres, _ = initReach(nlnsys, Rsplit, params, options)
            
            # MATLAB: for j = 1:length(Rres.tp)
            for j in range(len(Rres['tp'])):
                # MATLAB: Rtp{setCounter} = Rres.tp{j};
                # MATLAB: Rtp{setCounter}.parent = i;
                Rtp_j = Rres['tp'][j] if isinstance(Rres['tp'][j], dict) else {'set': Rres['tp'][j]}
                Rtp_j['parent'] = i
                Rtp.append(Rtp_j)
                
                # MATLAB: Rti{setCounter} = Rres.ti{j};
                Rti.append(Rres['ti'][j])
                
                # MATLAB: R0{setCounter} = Rres.R0{j};
                R0.append(Rres['R0'][j])
                
                setCounter += 1
    
    # store the results
    # MATLAB: Rnext.tp = Rtp;
    # MATLAB: Rnext.ti = Rti;
    # MATLAB: Rnext.R0 = R0;
    Rnext = {
        'tp': Rtp,
        'ti': Rti,
        'R0': R0
    }
    
    return Rnext, options


# Auxiliary functions -----------------------------------------------------

def aux_initReach_linRem(nlnsys: Any, Rinit: List[Dict[str, Any]], 
                         params: Dict[str, Any], options: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute the reachable set using the 'linRem' algorithm
    
    Args:
        nlnsys: nonlinearSys object
        Rinit: list of initial reachable sets (dicts with 'set' and 'error' keys)
        params: model parameters
        options: struct containing the algorithm settings
        
    Returns:
        Rnext: dict with keys 'tp', 'ti' containing reachable sets
        options: struct containing the algorithm settings
    """
    
    # compute the reachable set using the options.alg = 'lin' algorithm to
    # obtain a first rough over-approximation of the reachable set
    # MATLAB: options_ = options;
    options_ = options.copy()
    
    # MATLAB: options_.alg = 'lin';
    options_['alg'] = 'lin'
    
    # compute the one-step reachable sets
    # MATLAB: R_onestep = initReach(nlnsys,Rinit,params,options_);
    R_onestep, _ = initReach(nlnsys, Rinit, params, options_)
    
    # loop over all parallel sets 
    # MATLAB: Rtp = cell(length(R_onestep.ti),1); 
    # MATLAB: Rti = cell(length(R_onestep.ti),1);
    Rtp = [None] * len(R_onestep['ti'])
    Rti = [None] * len(R_onestep['ti'])
    
    # MATLAB: for i = 1:length(R_onestep.ti)
    for i in range(len(R_onestep['ti'])):
        
        # compute reachable set with the "linear remainder" algorithm to
        # obtain a refined reachable set
        # MATLAB: R0 = R_onestep.R0{i};
        R0 = R_onestep['R0'][i]
        
        # MATLAB: options.Ronestep = R_onestep.ti{i};
        options['Ronestep'] = R_onestep['ti'][i]
        
        # MATLAB: [Rti{i},Rtp{i}] = linReach(nlnsys,R0,params,options);
        from cora_python.contDynamics.contDynamics.linReach import linReach
        Rti_i, Rtp_i, _, _ = linReach(nlnsys, R0, params, options)
        Rti[i] = Rti_i
        Rtp[i] = Rtp_i
        
        # copy information about previous reachble set and parent
        # MATLAB: Rtp{i}.prev = R_onestep.tp{i}.prev;
        if isinstance(Rtp[i], dict) and 'prev' in R_onestep['tp'][i]:
            Rtp[i]['prev'] = R_onestep['tp'][i]['prev']
        elif not isinstance(Rtp[i], dict):
            Rtp_dict = {'set': Rtp[i]}
            if isinstance(R_onestep['tp'][i], dict) and 'prev' in R_onestep['tp'][i]:
                Rtp_dict['prev'] = R_onestep['tp'][i]['prev']
            Rtp[i] = Rtp_dict
        
        # MATLAB: if isfield(R_onestep.tp{i},'parent')
        if isinstance(R_onestep['tp'][i], dict) and 'parent' in R_onestep['tp'][i]:
            # MATLAB: Rtp{i}.parent = R_onestep.tp{i}.parent;
            if not isinstance(Rtp[i], dict):
                Rtp[i] = {'set': Rtp[i]}
            Rtp[i]['parent'] = R_onestep['tp'][i]['parent']
    
    # store the results
    # MATLAB: Rnext.tp = Rtp;
    # MATLAB: Rnext.ti = Rti;
    Rnext = {
        'tp': Rtp,
        'ti': Rti
    }
    
    return Rnext, options

