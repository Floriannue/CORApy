"""
derivatives - compute derivatives for nonlinear reset functions of all or
   a subset of all transitions within a given location

TRANSLATED FROM: cora_matlab/hybridDynamics/@location/derivatives.m

Syntax:
    loc = derivatives(loc)
    loc = derivatives(loc,transIdx)
    loc = derivatives(loc,transIdx,fpath)
    loc = derivatives(loc,transIdx,fpath,fname)

Inputs:
    loc - location object
    transIdx - index of transition in location object (0-based in Python)
    fpath - path to generated file
    fname - file name

Outputs:
    loc - updated location object

Example:
    -

Authors:       Mark Wetzlinger (MATLAB)
Written:       15-October-2024 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Union, List, Optional
import os
import numpy as np

if TYPE_CHECKING:
    from .location import Location


def derivatives(loc: 'Location', 
                transIdx: Optional[Union[np.ndarray, List[int], int]] = None,
                fpath: Optional[str] = None, 
                fname: Optional[str] = None) -> 'Location':
    """
    Compute derivatives for nonlinear reset functions of all or a subset of
    all transitions within a given location.
    
    Args:
        loc: location object
        transIdx: index/indices of transition(s) in location object (0-based in Python)
                  If None, all transitions are processed
        fpath: path to generated file (default: cora_python/models/auxiliary/location)
        fname: file name (default: 'reset')
    
    Returns:
        Location: updated location object
    """
    from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
    
    # Set default values
    if transIdx is None:
        # MATLAB: 1:numel(loc.transition) (1-based)
        # Python: 0:len(loc.transition) (0-based)
        transIdx = list(range(len(loc.transition)))
    
    if fpath is None:
        # MATLAB: [CORAROOT filesep 'models' filesep 'auxiliary' filesep 'location']
        fpath = os.path.join('cora_python', 'models', 'auxiliary', 'location')
    
    if fname is None:
        fname = 'reset'
    
    # Convert transIdx to list if single int
    if isinstance(transIdx, (int, np.integer)):
        transIdx = [int(transIdx)]
    elif isinstance(transIdx, np.ndarray):
        transIdx = transIdx.tolist()
    
    # MATLAB: for i=1:numel(loc.transition)
    # But MATLAB only processes transIdx, so we iterate over transIdx
    # Note: transIdx in MATLAB is 1-based, but we use 0-based in Python
    for i in transIdx:
        # Ensure i is 0-based
        i_py = int(i)
        # MATLAB uses 1-based, so if transIdx was provided as 1-based, convert
        # But since we're using 0-based everywhere, assume transIdx is already 0-based
        if 0 <= i_py < len(loc.transition):
            # MATLAB: loc.transition(i) = derivatives(loc.transition(i),...)
            # MATLAB:     fpath,sprintf('transition_%i_%s',i,fname));
            fname_i = f'transition_{i_py+1}_{fname}'  # MATLAB uses 1-based for filename
            loc.transition[i_py] = loc.transition[i_py].derivatives(fpath, fname_i)
    
    return loc

