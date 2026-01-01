"""
SRXparameters - parameter for the automomous vehicle (see [1])

Syntax:
    p = SRXparameters()

Inputs:
    ---

Outputs:
    p - struct storing the parameters

References:
    [1] M. Althoff and J. M. Dolan. Online verification of automated
        road vehicles using reachability analysis.
        IEEE Transactions on Robotics, 30(4):903-918, 2014.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: DOTBicycleDynamics_controlled_SRX_velEq

Authors:       Matthias Althoff
Written:       23-August-2011
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""


def SRXparameters():
    """
    SRXparameters - parameter for the automomous vehicle
    
    Returns:
        p: dict storing the parameters
    """
    # Create parameter structure
    class P:
        pass
    
    p = P()
    
    # masses
    p.m = 2273.4049  # vehicle mass [kg]
    
    # axes distances
    p.a = 1.2916  # distance from spring mass center of gravity to front axle [m]
    p.b = 1.5151  # distance from spring mass center of gravity to rear axle [m]
    
    # moments of inertia of sprung mass
    p.I_z = 4423.0301  # moment of inertia for sprung mass in yaw [kg m^2]
    
    # tire parameters from ADAMS handbook
    # lateral coefficients
    p.tire = type('obj', (object,), {})()  # Create nested object for tire
    p.tire.p_dy1 = 1  # Lateral friction Muy
    p.tire.p_ky1 = -107555.5915 / (p.m * 9.81 / 4)  # Maximum value of stiffness Kfy/Fznom
    
    return p

