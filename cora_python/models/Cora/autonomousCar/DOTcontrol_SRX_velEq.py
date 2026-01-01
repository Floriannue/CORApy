"""
DOTcontrol_SRX_velEq - provides the steering angle speed of the Cadillac 
                       SRX (see [1])

Syntax:
    [carInput, k] = DOTcontrol_SRX_velEq(x, uComb)

Inputs:
    x - state
    uComb - combination of reference trajectory u_ref and noise y

Outputs:
    carInput - steering angle for the car
    k - control parameter

References:
    [1] M. Althoff and J. M. Dolan. Online verification of automated
        road vehicles using reachability analysis.
        IEEE Transactions on Robotics, 30(4):903-918, 2014.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: DOTBicycleDynamics_controlled_SRX_velEq

Authors:       Matthias Althoff
Written:       01-March-2012
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np


def DOTcontrol_SRX_velEq(x, uComb):
    """
    DOTcontrol_SRX_velEq - provides the steering angle speed of the Cadillac SRX
    
    Args:
        x: state vector
        uComb: combination of reference trajectory u_ref and noise y (9-dimensional)
        
    Returns:
        carInput: steering angle for the car (2-dimensional)
        k: control parameter (dict)
    """
    # control parameters
    k = {}
    k[1] = 0.5
    k[2] = 3
    k[3] = 1
    k[6] = 2  # gain for steering wheel
    
    # separate in reference u and noise y
    u = np.zeros((5, 1))
    u[0:5, 0] = uComb[0:5, 0]
    y = np.zeros((4, 1))
    y[0:4, 0] = uComb[5:9, 0]
    
    # compute steering angle and acceleration input from reference trajectory
    # steering angle delta
    delta = k[1] * (-np.sin(u[2, 0]) * (u[0, 0] - (x[3, 0] + y[0, 0])) + np.cos(u[2, 0]) * (u[1, 0] - (x[4, 0] + y[1, 0]))) \
        + k[2] * (u[2, 0] - (x[1, 0] + y[2, 0])) + k[3] * (u[3, 0] - (x[2, 0] + y[3, 0]))
    # steering wheel speed
    delta_dot = k[6] * (delta - x[5, 0])
    
    # write control to car input
    carInput = np.zeros((2, 1))
    carInput[0, 0] = delta_dot
    carInput[1, 0] = u[4, 0]
    
    return carInput, k

