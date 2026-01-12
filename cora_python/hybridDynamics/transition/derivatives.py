"""
derivatives - compute derivatives for nonlinear reset functions

TRANSLATED FROM: cora_matlab/hybridDynamics/@transition/derivatives.m

Syntax:
    trans = derivatives(trans)
    trans = derivatives(trans,fpath)
    trans = derivatives(trans,fpath,fname)

Inputs:
    trans - transition object or class array
    fpath - path to generated file
    fname - file name

Outputs:
    trans - updated transition object

Example:
    -

Authors:       Mark Wetzlinger (MATLAB)
Written:       15-October-2024 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Union, List, Optional
import os

if TYPE_CHECKING:
    from .transition import Transition


def derivatives(trans: Union['Transition', List['Transition']], 
                fpath: Optional[str] = None, 
                fname: Optional[str] = None) -> Union['Transition', List['Transition']]:
    """
    Compute derivatives for nonlinear reset functions.
    
    Args:
        trans: transition object or list of transition objects
        fpath: path to generated file (default: CORA models/auxiliary/transition)
        fname: file name (default: 'nonlinear_reset_function')
    
    Returns:
        Transition or list of Transition objects with updated reset functions
    """
    from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
    
    # Set default values
    if fpath is None:
        # MATLAB: [CORAROOT filesep 'models' filesep 'auxiliary' filesep 'transition']
        # Use relative path from project root
        fpath = os.path.join('cora_python', 'models', 'auxiliary', 'transition')
    
    if fname is None:
        fname = 'nonlinear_reset_function'
    
    # Handle single transition or list of transitions
    is_single = not isinstance(trans, list)
    if is_single:
        trans_list = [trans]
    else:
        trans_list = trans
    
    # MATLAB: for i=1:numel(trans)
    for i in range(len(trans_list)):
        trans_i = trans_list[i]
        # MATLAB: if isa(trans(i).reset,'nonlinearReset')
        if isinstance(trans_i.reset, NonlinearReset):
            # MATLAB: trans(i).reset = derivatives(trans(i).reset,...)
            # MATLAB:     fpath,sprintf('transition_%i_%s',i,fname));
            fname_i = f'transition_{i+1}_{fname}'  # MATLAB uses 1-based indexing
            trans_i.reset = trans_i.reset.derivatives(fpath, fname_i)
    
    # Return single object or list
    if is_single:
        return trans_list[0]
    else:
        return trans_list

