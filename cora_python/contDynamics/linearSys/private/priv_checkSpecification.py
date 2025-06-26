"""
priv_checkSpecification - check safety properties for current time-interval
   reachable set; if a violation occurs, return truncated structs

Syntax:
   [res, timeInt, timePoint] = priv_checkSpecification(spec, XtimeInt, YtimeInt, YtimePoint, idx)

Inputs:
   spec - object of class specification
   XtimeInt - reachable set for current time interval 
   YtimeInt - struct about time-interval output set
   YtimePoint - struct about time-point output set
   idx - index of current time interval 

Outputs:
   res - true if specifications are satisfied, otherwise false
   YtimeInt - (truncated) struct about time-interval output set
              (only required if a violation was detected)
   YtimePoint - (truncated) struct about time-point output set
              (only required if a violation was detected)

Example: 
   -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 19-November-2022 (MATLAB)
Last update: 07-December-2022 (MW, bug fix for spec.type = 'invariant') (MATLAB)
Python translation: 2025
"""

from typing import Dict, Any, Tuple, Union, List
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def priv_checkSpecification(spec: Union[List, Any], XtimeInt: Any, YtimeInt: Dict[str, List], 
                          YtimePoint: Dict[str, List], idx: int) -> Tuple[bool, Dict[str, List], Dict[str, List]]:
    """
    Check safety properties for current time-interval reachable set
    
    Args:
        spec: Specification object or list of specification objects
        XtimeInt: Time-interval reachable set
        YtimeInt: Struct about time-interval output set
        YtimePoint: Struct about time-point output set
        idx: Index of current time interval (1-based)
        
    Returns:
        Tuple of (res, YtimeInt, YtimePoint)
    """
    # Init satisfaction
    res = True
    
    # Handle single specification or list of specifications
    if not isinstance(spec, list):
        spec = [spec]
    
    for i in range(len(spec)):
        if hasattr(spec[i], 'type') and spec[i].type == 'invariant':
            # Specification is an invariant -> called in hybrid system analysis to
            # check if the reachable set has left the invariant, therefore we use
            # the reachable set instead of the output set
            
            # The linearSys algorithms 'decomp', 'krylov', and 'adaptive' cannot
            # be used in conjunction with hybrid system analysis; thus, the call to
            # this function has XtimeInt = []
            if XtimeInt is None or (hasattr(XtimeInt, '__len__') and len(XtimeInt) == 0):
                raise CORAerror('CORA:notSupported',
                    'The chosen algorithm for linear systems (options.linAlg)\n'
                    'cannot be used for the analysis of hybrid systems.')
            
            if not spec[i].check(XtimeInt, YtimeInt['time'][idx-1]):  # idx-1 for 0-indexing
                # Violation
                res = False
                # Truncate reachable set until current time interval
                YtimeInt['set'] = YtimeInt['set'][:idx]
                YtimeInt['time'] = YtimeInt['time'][:idx]
                # Index for time-point shifted by one as initial set at index 1
                YtimePoint['set'] = YtimePoint['set'][:idx+1]
                YtimePoint['time'] = YtimePoint['time'][:idx+1]
                return res, YtimeInt, YtimePoint
        else:
            # Specification on the output set
            if not spec[i].check(YtimeInt['set'][idx-1], YtimeInt['time'][idx-1]):  # idx-1 for 0-indexing
                # Violation
                res = False
                # Truncate output set until current time interval
                YtimeInt['set'] = YtimeInt['set'][:idx]
                YtimeInt['time'] = YtimeInt['time'][:idx]
                # Index for time-point shifted by one as initial set at index 1
                YtimePoint['set'] = YtimePoint['set'][:idx+1]
                YtimePoint['time'] = YtimePoint['time'][:idx+1]
                return res, YtimeInt, YtimePoint
    
    return res, YtimeInt, YtimePoint 