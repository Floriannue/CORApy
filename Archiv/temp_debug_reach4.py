import numpy as np
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.contSet.zonotope import Zonotope
import importlib

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

sys = NonlinearSys(tank6Eq, states=6, inputs=1)

lin_mod = importlib.import_module('cora_python.contDynamics.nonlinearSys.linearize')
orig_linearize = lin_mod.linearize

def wrapped_linearize(nlnsys, R, params, options):
    print('linearize R type:', type(R))
    try:
        c = R.center()
        print('R.center=', np.asarray(c).reshape(-1))
    except Exception as e:
        print('R.center error', e)
    res = orig_linearize(nlnsys, R, params, options)
    print('p.x=', nlnsys.linError.p.x.reshape(-1))
    print('A finite?', np.isfinite(res[1].A).all())
    return res

lin_mod.linearize = wrapped_linearize

from cora_python.contDynamics.contDynamics.reach import reach
try:
    reach(sys, params, options)
except Exception as e:
    print('reach exception:', e)
    import traceback; traceback.print_exc()
