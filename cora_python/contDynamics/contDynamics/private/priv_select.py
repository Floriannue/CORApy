"""
priv_select - selects the split strategy of the reachable set causing the
   least linearization error

Syntax:
    dimForSplit = priv_select(sys,Rinit,params,options)

Inputs:
    sys - nonlinearSys or nonlinParamSys object
    Rinit - initial reachable set
    params - model parameters
    options - struct containing the algorithm settings

Outputs:
    dimForSplit - dimension that is split to reduce the linearization
                  error

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: linReach

Authors:       Matthias Althoff, Niklas Kochdumper
Written:       04-January-2008 
Last update:   29-January-2008
               29-June-2009
               12-September-2017
               02-January-2019 (NK, cleaned up the code)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, Optional
from cora_python.contSet.zonotope.split import split
from cora_python.contDynamics.contDynamics.linReach import linReach


def priv_select(sys: Any, Rinit: Dict[str, Any], params: Dict[str, Any],
                options: Dict[str, Any]) -> Optional[int]:
    """
    Selects the split strategy of the reachable set causing the least linearization error
    
    Args:
        sys: nonlinearSys or nonlinParamSys object
        Rinit: initial reachable set (dict with 'set' and 'error' keys)
        params: model parameters
        options: struct containing the algorithm settings (must contain 'maxError', 'alg')
        
    Returns:
        dimForSplit: dimension that is split to reduce the linearization error (0-based index)
    """
    
    # compute all possible splits of the maximum reachable set
    # MATLAB: Rtmp = split(Rinit.set);
    Rtmp = split(Rinit['set'])
    
    # MATLAB: R = cell(length(Rtmp),1);
    R = [None] * len(Rtmp)
    
    # MATLAB: for i = 1:length(Rtmp)
    for i in range(len(Rtmp)):
        # MATLAB: R{i}.set = Rtmp{i}{1};        % only test one of the two split sets
        # MATLAB: R{i}.error = zeros(size(options.maxError));
        R[i] = {
            'set': Rtmp[i][0],  # only test one of the two split sets
            'error': np.zeros_like(options['maxError'])
        }
    
    # adapt the options for reachability analysis
    # MATLAB: maxError = options.maxError;
    maxError = options['maxError'].copy()
    
    # MATLAB: options.maxError = inf * maxError;
    options = options.copy()  # Don't modify original
    options['maxError'] = np.inf * maxError
    
    # MATLAB: if strcmp(options.alg,'linRem')
    if options['alg'] == 'linRem':
        # MATLAB: options.alg = 'lin';
        options['alg'] = 'lin'
    
    # loop over all split sets
    # MATLAB: perfInd = zeros(length(R),1);
    perfInd = np.zeros(len(R))
    
    # MATLAB: for i=1:length(R)
    for i in range(len(R)):
        
        # compute the reachable set for the splitted set
        # MATLAB: [~,Rtp] = linReach(sys,R{i},params,options);
        _, Rtp, _, _ = linReach(sys, R[i], params, options)
        
        # compute performance index (max lin. error) for the split 
        # MATLAB: perfInd(i) = max(Rtp.error./maxError);
        perfInd[i] = np.max(Rtp['error'] / maxError)
    
    # find best performance index
    # MATLAB: [~,dimForSplit] = min(perfInd);
    # Note: MATLAB uses 1-based indexing, Python uses 0-based
    # But dimForSplit is used as an index for split, which should be 0-based in Python
    dimForSplit = int(np.argmin(perfInd))
    
    return dimForSplit

