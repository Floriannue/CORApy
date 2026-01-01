"""
taylorMatrices - computes the remainder matrix of the exponential matrix
   and the correction matrices for the state and input (all private)

Syntax:
    [E,F,G] = taylorMatrices(linsys,timeStep,truncationOrder)

Inputs:
    linsys - linearSys object
    timeStep - time step size
    truncationOrder - maximum order for Taylor expansion

Outputs:
    E - remainder matrix of exponential matrix
    F - correction matrix for the state
    G - correction matrix for the input

Example: 
    linsys = linearSys([-1 -4; 4 -1]);
    timeStep = 0.05;
    truncationOrder = 6;
    [E,F,G] = taylorMatrices(linsys,timeStep,truncationOrder);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       05-April-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Tuple
from cora_python.g.classes.taylorLinSys import TaylorLinSys
from cora_python.contDynamics.linearSys.private.priv_expmRemainder import priv_expmRemainder
from cora_python.contDynamics.linearSys.private.priv_correctionMatrixState import priv_correctionMatrixState
from cora_python.contDynamics.linearSys.private.priv_correctionMatrixInput import priv_correctionMatrixInput


def taylorMatrices(linsys: Any, timeStep: float, truncationOrder: int) -> Tuple[Any, Any, Any]:
    """
    Computes the remainder matrix of the exponential matrix and the correction matrices
    
    Args:
        linsys: linearSys object
        timeStep: time step size
        truncationOrder: maximum order for Taylor expansion
        
    Returns:
        E: remainder matrix of exponential matrix
        F: correction matrix for the state
        G: correction matrix for the input
    """
    
    # since this function is public, we cannot assume that taylorLinSys has
    # already been instantiated
    if not hasattr(linsys, 'taylor') or linsys.taylor is None:
        linsys.taylor = TaylorLinSys(linsys.A)
    
    # compute remainder of exponential matrix
    E = priv_expmRemainder(linsys, timeStep, truncationOrder)
    
    # compute correction matrices
    F = priv_correctionMatrixState(linsys, timeStep, truncationOrder)
    G = priv_correctionMatrixInput(linsys, timeStep, truncationOrder)
    
    return E, F, G

