import numpy as np
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.contSet.zonotope import Zonotope
from cora_python.models.auxiliary.tank6Eq.jacobian_tank6Eq import jacobian_tank6Eq

sys = NonlinearSys(tank6Eq, states=6, inputs=1)
# disable derivatives in reach
sys.derivatives = lambda options: None
sys.jacobian = jacobian_tank6Eq

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

orig_jac = sys.jacobian

def wrap_jac(x,u):
    print('jacobian input x=', np.asarray(x).reshape(-1))
    return orig_jac(x,u)

sys.jacobian = wrap_jac

from cora_python.contDynamics.contDynamics.reach import reach
try:
    reach(sys, params, options)
except Exception as e:
    print('reach exception:', e)
    import traceback; traceback.print_exc()
