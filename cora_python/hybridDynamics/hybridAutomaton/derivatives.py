"""
derivatives - compute derivatives for nonlinear reset functions of all
   transitions of all or a subset of all locations within a hybrid
   automaton

TRANSLATED FROM: cora_matlab/hybridDynamics/@hybridAutomaton/derivatives.m

Syntax:
    HA = derivatives(HA)
    HA = derivatives(HA,locIdx)
    HA = derivatives(HA,locIdx,fpath)
    HA = derivatives(HA,locIdx,fpath,fname)

Inputs:
    HA - hybridAutomaton object
    locIdx - indices of locations (0-based in Python)
    fpath - path to generated file
    fname - file name

Outputs:
    HA - updated hybridAutomaton object

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
    from .hybridAutomaton import HybridAutomaton


def derivatives(HA: 'HybridAutomaton',
                locIdx: Optional[Union[np.ndarray, List[int], int]] = None,
                fpath: Optional[str] = None,
                fname: Optional[str] = None) -> 'HybridAutomaton':
    """
    Compute derivatives for nonlinear reset functions of all transitions of
    all or a subset of all locations within a hybrid automaton.
    
    Args:
        HA: hybridAutomaton object
        locIdx: indices of locations (0-based in Python). If None, all locations are processed
        fpath: path to generated file (default: cora_python/models/auxiliary/hybridAutomaton/{HA.name})
        fname: file name (default: 'nonlinear_reset_function')
    
    Returns:
        HybridAutomaton: updated hybridAutomaton object
    """
    # Set default values
    if locIdx is None:
        # MATLAB: 1:numel(HA.location) (1-based)
        # Python: 0:len(HA.location) (0-based)
        locIdx = list(range(len(HA.location)))
    
    if fpath is None:
        # MATLAB: [CORAROOT filesep 'models' filesep 'auxiliary' filesep ...
        #          'hybridAutomaton' filesep HA.name]
        fpath = os.path.join('cora_python', 'models', 'auxiliary', 'hybridAutomaton', HA.name)
    
    if fname is None:
        fname = 'nonlinear_reset_function'
    
    # Convert locIdx to list if single int
    if isinstance(locIdx, (int, np.integer)):
        locIdx = [int(locIdx)]
    elif isinstance(locIdx, np.ndarray):
        locIdx = locIdx.tolist()
    
    # MATLAB: for i=1:numel(HA.location)
    # But MATLAB only processes locIdx, so we iterate over locIdx
    # Note: locIdx in MATLAB is 1-based, but we use 0-based in Python
    for i in locIdx:
        # Ensure i is 0-based
        i_py = int(i)
        if 0 <= i_py < len(HA.location):
            # MATLAB: HA.location(i) = derivatives(HA.location(i),...
            # MATLAB:     numel(HA.location(i).transition),...
            # MATLAB:     sprintf('%s%slocation_%i',fpath,filesep,i),fname);
            # Make a new folder for each location (note: we do not use the name of
            # the location since there may be duplicates)
            fpath_i = os.path.join(fpath, f'location_{i_py+1}')  # MATLAB uses 1-based for folder name
            num_transitions = len(HA.location[i_py].transition)
            # Process all transitions in this location
            transIdx = list(range(num_transitions))  # 0-based
            HA.location[i_py] = HA.location[i_py].derivatives(transIdx, fpath_i, fname)
    
    return HA

