import numpy as np
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.nonlinearSys.linearize import linearize

sys = NonlinearSys(tank6Eq, states=6, inputs=1)
params = {
    'tFinal': 1,
    'R0': PolyZonotope(
        np.array([[2],[4],[4],[2],[10],[4]]),
        0.2*np.eye(6),
        np.array([]).reshape(6,0),
        np.array([]).reshape(0,0)
    ),
    'U': Zonotope(np.zeros((1,1)), 0.005*np.eye(1))
}
options = {
    'zonotopeOrder': 1000,
    'taylorTerms': 5,
    'tensorOrder': 3,
    'intermediateOrder': 10,
    'errorOrder': 10,
    'reductionTechnique': 'girard',
    'alg': 'poly',
    'timeStep': params['tFinal']
}

sys.derivatives(options)
if 'uTrans' not in params:
    params['uTrans'] = params['U'].center()
params['tStart'] = 0.0

sys, linsys, linParams, linOptions = linearize(sys, params['R0'], params, options)
print('p.x=', sys.linError.p.x.reshape(-1))
print('A finite?', np.isfinite(linsys.A).all())
print('A=', linsys.A)
