"""
DOTBicycleDynamics_controlled_SRX_velEq - enhances bicycle model (see [1])
                                         with control for trajectory 
                                         tracking

Syntax:
    f = DOTBicycleDynamics_controlled_SRX_velEq(x, u)

Inputs:
    x - state vector
    u - input vector (here: reference trajectory)

Outputs:
    f - time-derivative of the state vector

References:
    [1] M. Althoff and J. M. Dolan. Online verification of automated
        road vehicles using reachability analysis.
        IEEE Transactions on Robotics, 30(4):903-918, 2014.

Other m-files required: DOTcontrol_SRX_velEq, DOTBicycleDynamics_SRX_velEq
Subfunctions: none
MAT-files required: none

See also: DOTcontrol_SRX_velEq, DOTBicycleDynamics_SRX_velEq

Authors:       Matthias Althoff
Written:       01-March-2012
Last update:   15-August-2016
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from cora_python.models.Cora.autonomousCar.DOTcontrol_SRX_velEq import DOTcontrol_SRX_velEq
from cora_python.models.Cora.autonomousCar.DOTBicycleDynamics_SRX_velEq import DOTBicycleDynamics_SRX_velEq


def DOTBicycleDynamics_controlled_SRX_velEq(x, u):
    """
    DOTBicycleDynamics_controlled_SRX_velEq - enhances bicycle model with control
    
    Args:
        x: state vector
        u: input vector (here: reference trajectory)
        
    Returns:
        f: time-derivative of the state vector
    """
    # obtain control inputs
    carInput, _ = DOTcontrol_SRX_velEq(x, u)
    
    # simulate vehicle dynamics
    f = DOTBicycleDynamics_SRX_velEq(x, carInput)
    
    return f

