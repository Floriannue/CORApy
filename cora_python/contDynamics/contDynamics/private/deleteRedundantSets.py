"""
deleteRedundantSets - delete reachable sets that are already covered by
   other sets

Syntax:
    R = deleteRedundantSets(R,Rold,options)

Inputs:
    R - reachable sets
    Rold - reachable sets of previous time steps
    options - options for the computation of the reachable set

Outputs:
    R - reachable sets

Example: 
    ---

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff
Written:       29-June-2009
Last update:   03-February-2011
               29-June-2018
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.mldivide import mldivide


def deleteRedundantSets(R: Dict[str, Any], Rold: Dict[str, Any], 
                       options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Delete reachable sets that are already covered by other sets
    
    Args:
        R: reachable sets (dict with 'tp' key containing list of dicts with 'set' key)
        Rold: reachable sets of previous time steps (dict with 'P' key containing list of polytopes)
        options: options for the computation of the reachable set (must contain 'reductionInterval' and 'maxError')
        
    Returns:
        R: reachable sets (modified dict)
    """
    
    # set reduction method
    # MATLAB: redMethod='pca';
    redMethod = 'pca'
    
    # increase internal count
    # MATLAB: try
    # MATLAB:     R.internalCount=Rold.internalCount+1;
    # MATLAB: catch
    # MATLAB:     %for first run
    # MATLAB:     R.internalCount=3;
    # MATLAB: end
    try:
        R['internalCount'] = Rold['internalCount'] + 1
    except (KeyError, TypeError):
        # for first run
        R['internalCount'] = 3
    
    # if internal count='some constant'
    # MATLAB: if R.internalCount==options.reductionInterval
    if R['internalCount'] == options['reductionInterval']:
        # reset internal count
        # MATLAB: R.internalCount=1;
        R['internalCount'] = 1
        
        # overapproximate reachable set of time point(!) by parallelpipeds and
        # save them as polytopes
        # MATLAB: R.P=[];
        R['P'] = []
        
        # MATLAB: for i=1:length(R.tp)
        for i in range(len(R['tp'])):
            # MATLAB: R.tp{i}.set=reduce(R.tp{i}.set,redMethod,1);
            R['tp'][i]['set'] = R['tp'][i]['set'].reduce(redMethod, 1)
            
            # generate polytope
            # MATLAB: R.P{i}=polytope(R.tp{i}.set);
            R['P'].append(R['tp'][i]['set'].polytope())
    
    # MATLAB: elseif R.internalCount==2
    elif R['internalCount'] == 2:
        # intersect each reachable set with each previous reachable set
        # MATLAB: for iNewSet=1:length(R.tp)
        Pnew = []
        for iNewSet in range(len(R['tp'])):
            # approximate new set of time points by parallelpiped 
            # MATLAB: R.tp{iNewSet}.set=reduce(R.tp{iNewSet}.set,redMethod,1);
            R['tp'][iNewSet]['set'] = R['tp'][iNewSet]['set'].reduce(redMethod, 1)
            
            # generate mpt polytope
            # MATLAB: Pnew{iNewSet}=polytope(R.tp{iNewSet}.set);
            Pnew.append(R['tp'][iNewSet]['set'].polytope())
        
        # initialize Pcut
        # MATLAB: Pcut=Pnew;
        Pcut = Pnew.copy()
        
        # intersection with previous time step
        # MATLAB: for iNewSet=1:length(Pcut)
        for iNewSet in range(len(Pcut)):
            # intersect with previous sets
            # MATLAB: for iOldSet=1:length(Rold.P)
            if 'P' in Rold and len(Rold['P']) > 0:
                for iOldSet in range(len(Rold['P'])):
                    # MATLAB: Pcut{iNewSet}=Pcut{iNewSet}\Rold.P{iOldSet};
                    Pcut[iNewSet] = mldivide(Pcut[iNewSet], Rold['P'][iOldSet])
        
        # reset iChecked counter
        # MATLAB: iChecked=1;
        iChecked = 1
        
        # intersection with actual time step
        # MATLAB: for iNewSet=1:length(Pcut)
        Rnew = []
        for iNewSet in range(len(Pcut)):
            # intersect with actual sets
            # MATLAB: for iOtherSet=1:length(Pnew)
            for iOtherSet in range(len(Pnew)):
                # MATLAB: if iOtherSet~=iNewSet
                if iOtherSet != iNewSet:
                    # MATLAB: Pcut{iNewSet}=Pcut{iNewSet}\Pnew{iOtherSet};
                    Pcut[iNewSet] = mldivide(Pcut[iNewSet], Pnew[iOtherSet])
            
            # is polytope empty?
            # note: new polytope toolbox does not support mldivide, so we skip
            # this check (unclear if formally correct, too)
            # MATLAB: if true % ~representsa_(Pcut{iNewSet},'emptySet',eps)
            if True:  # ~representsa_(Pcut[iNewSet],'emptySet',eps)
                # MATLAB: Rnew{iChecked}.set = R.tp{iNewSet}.set;
                # MATLAB: Rnew{iChecked}.error = 0*options.maxError;
                # MATLAB: Rnew{iChecked}.prev = iNewSet;
                Rnew_dict = {
                    'set': R['tp'][iNewSet]['set'],
                    'error': np.zeros_like(options['maxError']),
                    'prev': iNewSet
                }
                
                # MATLAB: if isfield(R.tp{iNewSet},'parent')
                if 'parent' in R['tp'][iNewSet]:
                    # MATLAB: Rnew{iChecked}.parent = R.tp{iNewSet}.parent;
                    Rnew_dict['parent'] = R['tp'][iNewSet]['parent']
                
                Rnew.append(Rnew_dict)
                iChecked += 1
            else:
                # MATLAB: disp('canceled!!');
                print('canceled!!')
        
        # copy only checked reachable sets
        # MATLAB: R.tp=[];
        # MATLAB: for i=1:length(Rnew)
        # MATLAB:     R.tp{i} = Rnew{i};
        # MATLAB: end
        R['tp'] = Rnew
    
    return R

