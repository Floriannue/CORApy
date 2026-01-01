"""
vmodel_A_bicycle_linear_controlled - enhances bicycle model (see [1])
                                     with linear feedback control for 
                                     trajectory tracking

Syntax:
    dx = vmodel_A_bicycle_linear_controlled(x, u)

Inputs:
    x - state vector
    u - input vector (here: reference trajectory)

Outputs:
    f - time-derivative of the state vector

References:
    [1] M. Althoff and J. M. Dolan. Online verification of automated
        road vehicles using reachability analysis.
        IEEE Transactions on Robotics, 30(4):903-918, 2014.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: DOTcontrol_SRX_velEq, DOTBicycleDynamics_SRX_velEq

Authors:       Matthias Althoff
Written:       01-March-2012
Last update:   15-August-2016
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np


def vmodel_A_bicycle_linear_controlled(x, u):
    """
    vmodel_A_bicycle_linear_controlled - enhances bicycle model with linear feedback control
    
    Bicycle Model with 
    - normal force equilibrium for pitching-moments
    - linear tyre model
    state x=[X,Y,psi,vx,vy,omega]
    input u=[delta,omega_f,omega_r]
    
    Args:
        x: state vector (8-dimensional)
        u: input vector (26-dimensional: R matrix, Xn, W)
        
    Returns:
        dx: time-derivative of the state vector
    """
    # parameters: get parameters from p vector
    # body
    m = 1750
    J = 2500
    L = 2.7
    l_F = 1.43
    l_R = L - l_F
    h = 0.5
    
    # street
    mu0 = 1
    g = 9.81
    
    # tires
    # B=p(7),C=p(8) - paceijca parameter in ff_abs = sin(C*atan(B*sf_abs/mu0))
    C_F = 10.4 * 1.3
    C_R = 21.4 * 1.1
    
    # state
    # position
    X = x[0, 0]  # unused but kept for reference
    Y = x[1, 0]  # unused but kept for reference
    psi = x[2, 0]
    
    # velocity
    vx = x[3, 0]
    vy = x[4, 0]
    omega = x[5, 0]
    
    # acceleration
    Fb = x[6, 0] * m
    delta = x[7, 0]
    
    # control action
    # inputs are values of the state feedback matrix R, the reference state Xn,
    # and the feedforward value W
    R = np.array([[u[0, 0], u[1, 0], u[2, 0], u[3, 0], u[4, 0], u[5, 0], u[6, 0], u[7, 0]],
                  [u[8, 0], u[9, 0], u[10, 0], u[11, 0], u[12, 0], u[13, 0], u[14, 0], u[15, 0]]])
    Xn = np.array([[u[16, 0]], [u[17, 0]], [u[18, 0]], [u[19, 0]], 
                   [u[20, 0]], [u[21, 0]], [u[22, 0]], [u[23, 0]]])
    W = np.array([[u[24, 0]], [u[25, 0]]])
    v = -R @ (x - Xn) + W
    
    # calculate normal forces
    Fzf = (l_R * m * g - h * Fb) / (l_R + l_F)
    Fzr = m * g - Fzf
    
    # side-slip
    sf = (vy + l_F * omega) / vx - delta
    sr = (vy - l_R * omega) / vx
    
    # forces
    Fyf = -C_F * Fzf * sf
    Fyr = -C_R * Fzr * sr
    
    # ACCELERATIONS
    dvx = Fb / m + vy * omega
    dvy = (Fyf + Fyr) / m - vx * omega
    domega = (l_F * Fyf - l_R * Fyr) / J
    
    # position
    cp = np.cos(psi)
    sp = np.sin(psi)
    dx = np.zeros((8, 1))
    dx[0, 0] = cp * vx - sp * vy
    dx[1, 0] = sp * vx + cp * vy
    dx[2, 0] = omega
    # velocity
    dx[3, 0] = dvx
    dx[4, 0] = dvy
    dx[5, 0] = domega
    # acceleration
    dx[6, 0] = v[0, 0]
    dx[7, 0] = v[1, 0]
    
    return dx

