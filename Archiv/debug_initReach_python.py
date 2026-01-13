"""
Debug script to trace intermediate values in initReach computation
"""
import numpy as np
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contDynamics.nonlinearSys.initReach import initReach

# Setup (same as test)
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
    'tensorOrder': 2
}

# System
tank = NonlinearSys(tank6Eq, states=6, inputs=1)

# Compute derivatives
from cora_python.contDynamics.contDynamics.derivatives import derivatives
derivatives(tank, options)

# Get factors
options['factor'] = []
for i in range(dim_x):
    options['factor'].append(1)

# Compute initReach
print("=== Python Debug Output ===")
print(f"R0 center: {params['R0'].c}")
print(f"R0 generators shape: {params['R0'].G.shape}")
print(f"U center: {params['U'].c}")
print(f"U generators shape: {params['U'].G.shape}")
print(f"timeStep: {options['timeStep']}")
print(f"taylorTerms: {options['taylorTerms']}")

Rfirst, _ = initReach(tank, params['R0'], params, options)

print(f"\nRfirst keys: {Rfirst.keys()}")
print(f"Rfirst['tp'] length: {len(Rfirst['tp'])}")
print(f"Rfirst['ti'] length: {len(Rfirst['ti'])}")

if len(Rfirst['tp']) > 0:
    Rtp0 = Rfirst['tp'][0]
    print(f"\nRtp[0] keys: {Rtp0.keys()}")
    print(f"Rtp[0]['set'] type: {type(Rtp0['set'])}")
    if hasattr(Rtp0['set'], 'c'):
        print(f"Rtp[0]['set'] center: {Rtp0['set'].c}")
    if hasattr(Rtp0['set'], 'G'):
        print(f"Rtp[0]['set'] generators shape: {Rtp0['set'].G.shape}")
    
    # Get interval hull
    IH_tp = Interval(Rtp0['set'])
    print(f"\nIH_tp inf: {IH_tp.inf.flatten()}")
    print(f"IH_tp sup: {IH_tp.sup.flatten()}")

if len(Rfirst['ti']) > 0:
    Rti0 = Rfirst['ti'][0]
    print(f"\nRti[0] type: {type(Rti0)}")
    if hasattr(Rti0, 'c'):
        print(f"Rti[0] center: {Rti0.c}")
    IH_ti = Interval(Rti0)
    print(f"IH_ti inf: {IH_ti.inf.flatten()}")
    print(f"IH_ti sup: {IH_ti.sup.flatten()}")
