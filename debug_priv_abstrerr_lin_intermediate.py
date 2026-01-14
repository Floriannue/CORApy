"""
Debug script to compare intermediate values in priv_abstrerr_lin
"""
import numpy as np
import sys
sys.path.insert(0, 'cora_python')

from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.contDynamics.derivatives import derivatives
from cora_python.contDynamics.contDynamics.linReach import linReach
from cora_python.contDynamics.contDynamics.private.priv_abstrerr_lin import priv_abstrerr_lin

# Setup same as test
dim_x = 6
params = {
    'R0': Zonotope(np.array([[2], [4], [4], [2], [10], [4]]), 0.2 * np.eye(dim_x)),
    'U': Zonotope(np.zeros((1, 1)), 0.005 * np.eye(1)),
    'tFinal': 4,
    'uTrans': np.zeros((1, 1))
}

options = {
    'timeStep': 4,
    'taylorTerms': 4,
    'zonotopeOrder': 50,
    'alg': 'lin',
    'tensorOrder': 2,
    'reductionTechnique': 'girard',
    'errorOrder': 10,
    'intermediateOrder': 10,
    'maxError': np.full((dim_x, 1), np.inf)
}

tank = NonlinearSys(tank6Eq, states=6, inputs=1)
derivatives(tank, options)

for i in range(1, options['taylorTerms'] + 2):
    options['factor'] = options.get('factor', [])
    options['factor'].append((options['timeStep'] ** i) / np.math.factorial(i))

Rstart = {'set': params['R0'], 'error': np.zeros((dim_x, 1))}

# Call linReach
Rti, Rtp, dimForSplit, options_out = linReach(tank, Rstart, params, options)

print(f"Rti center: {Rti.center().flatten()}")
print(f"Rti generators shape: {Rti.generators().shape}")
print(f"Rti interval: {Rti.interval()}")

# Now manually call priv_abstrerr_lin with Rti to see intermediate values
# But we need Rmax = Rti + RallError. Let's check what Rmax was in linReach
# Actually, let's modify priv_abstrerr_lin temporarily to print debug info

print(f"\nFinal error from linReach: {Rtp['error'].flatten()}")
print(f"Expected error: [0.000206863579523074, 0.000314066666873806, 0.000161658311464827, 0.00035325543180986, 0.000358487021465299, 0.000209190642349808]")
