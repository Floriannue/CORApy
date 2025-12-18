"""
empty - instantiates an empty spectrahedral shadow

Syntax:
    sS = empty(n)

Inputs:
    n - dimension

Outputs:
    sS - empty spectraShadow object
"""

import numpy as np
from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck


def empty(n: int = 0) -> SpectraShadow:
    """
    Instantiates an empty spectrahedral shadow
    
    Args:
        n: dimension (default: 0)
        
    Returns:
        sS: empty spectraShadow object
    """
    # Parse input - match MATLAB behavior
    inputArgsCheck([[n, 'att', 'numeric', ['scalar', 'nonnegative']]])
    
    # MATLAB: the spectraShadow -1 + 0*x >= 0 is empty
    if n == 0:
        # MATLAB: SpS_out = spectraShadow([-1 0], zeros([0 1]), zeros([0 1]));
        A = np.array([[-1, 0]])
        c = np.zeros((0, 1))
        G = np.zeros((0, 1))
        SpS_out = SpectraShadow(A, c, G)
        # MATLAB: SpS_out.ESumRep.val = {[-1 zeros([1 n])], []};
        SpS_out.ESumRep.val = [np.array([[-1]]), np.array([]).reshape(1, 0)]
    else:
        # MATLAB: SpS_out = spectraShadow([-1 zeros([1 n])]);
        A = np.hstack([np.array([[-1]]), np.zeros((1, n))])
        SpS_out = SpectraShadow(A)
        # MATLAB: SpS_out.ESumRep.val = {[-1 zeros([1 n])], []};
        SpS_out.ESumRep.val = [np.hstack([np.array([[-1]]), np.zeros((1, n))]), np.array([]).reshape(1, 0)]
    
    # MATLAB: assign properties
    SpS_out.emptySet.val = True
    SpS_out.bounded.val = True
    SpS_out.fullDim.val = False
    SpS_out.center.val = np.array([]).reshape(0, 1)
    
    return SpS_out 