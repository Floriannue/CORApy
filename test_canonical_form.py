import numpy as np
from cora_python.contDynamics.linearSys import LinearSys
from cora_python.contSet.zonotope import Zonotope

# Test system
A = np.array([[-0.3780, 0.2839, 0.5403, -0.2962],
              [0.1362, 0.2742, 0.5195, 0.8266],
              [0.0502, -0.1051, -0.6572, 0.3874],
              [1.0227, -0.4877, 0.8342, -0.2372]])
B = 0.25 * np.array([[-2, 0, 3],
                     [2, 1, 0],
                     [0, 0, 1],
                     [0, -2, 1]])

sys = LinearSys(A, B)
print('System created successfully')

# Parameters
params = {
    'tFinal': 0.2,
    'R0': Zonotope(10 * np.ones((4, 1)), 0.5 * np.eye(4)),
    'U': Zonotope(np.zeros((3, 1)), 0.25 * np.eye(3))
}

# Options
options = {
    'timeStep': 0.05,
    'taylorTerms': 6,
    'zonotopeOrder': 50,
    'linAlg': 'standard'
}

print('Parameters and options set')

# Test canonicalForm directly
from cora_python.contDynamics.linearSys.canonicalForm import canonicalForm
from cora_python.contDynamics.linearSys.reach import _validateOptions

params_val, options_val = _validateOptions(sys, params, options)
print('Options validated')

linsys_can, U_can, u_can, V_can, v_can = canonicalForm(
    sys, params_val['U'], params_val['uTrans'],
    params_val['W'], params_val['V'], np.zeros((sys.nr_of_outputs, 1))
)
print('canonicalForm completed successfully!')
print('Canonical system shape:', linsys_can.A.shape)
print('U_can:', U_can)
print('u_can shape:', u_can.shape)
print('V_can:', V_can)
print('v_can shape:', v_can.shape) 