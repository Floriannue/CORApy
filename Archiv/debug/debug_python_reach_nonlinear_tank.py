import numpy as np
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.models.Cora.tank.tank6Eq import tank6Eq


def main():
    dim_x = 6
    params = {
        'tFinal': 400,
        'R0': Zonotope(np.array([[2], [4], [4], [2], [10], [4]]), 0.2 * np.eye(dim_x)),
        'U': Zonotope(np.zeros((1, 1)), 0.005 * np.eye(1))
    }
    options = {
        'timeStep': 4,
        'taylorTerms': 4,
        'zonotopeOrder': 50,
        'alg': 'lin',
        'tensorOrder': 2,
        'compOutputSet': False
    }

    tank = NonlinearSys(tank6Eq, states=6, inputs=1)

    # Monkeypatch priv_abstrerr_lin to log Rmax interval near failure time
    import importlib
    abstrerr_mod = importlib.import_module("cora_python.contDynamics.contDynamics.private.priv_abstrerr_lin")
    original_priv_abstrerr_lin = abstrerr_mod.priv_abstrerr_lin
    def priv_abstrerr_lin_debug(sys_obj, R, params_in, options_in):
        try:
            IHx = R.interval()
            ihx_inf = IHx.infimum()
            if np.any(ihx_inf < 0):
                print("DEBUG priv_abstrerr_lin t =", options_in.get('t'))
                print("DEBUG Rmax.interval.inf =", ihx_inf.reshape(-1))
                print("DEBUG Rmax.interval.sup =", IHx.supremum().reshape(-1))
        except Exception:
            pass
        return original_priv_abstrerr_lin(sys_obj, R, params_in, options_in)
    abstrerr_mod.priv_abstrerr_lin = priv_abstrerr_lin_debug
    # Also patch linReach's local reference (imported directly in module)
    linreach_mod = importlib.import_module("cora_python.contDynamics.contDynamics.linReach")
    linreach_mod.priv_abstrerr_lin = priv_abstrerr_lin_debug
    # Wrap setHessian to log interval bounds passed to hessian (debug)
    original_set_hessian = tank.setHessian
    def setHessian_debug(version):
        result = original_set_hessian(version)
        if version == 'int':
            original_hessian = tank.hessian
            def hessian_wrapper(x, u, *args):
                try:
                    x_inf = x.infimum()
                    x_sup = x.supremum()
                    print("DEBUG totalInt_x.inf =", x_inf.reshape(-1))
                    print("DEBUG totalInt_x.sup =", x_sup.reshape(-1))
                except Exception:
                    pass
                return original_hessian(x, u, *args)
            tank.hessian = hessian_wrapper
        return result
    tank.setHessian = setHessian_debug
    try:
        R = tank.reach(params, options)
    except Exception as exc:
        print("DEBUG reach failed at t =", options.get('t'))
        print("DEBUG exception:", repr(exc))
        raise
    np.set_printoptions(precision=15, suppress=False)
    if not hasattr(R.timeInterval, 'set') or len(R.timeInterval.set) == 0:
        print("timeInterval is empty")
        print("timePoint length:", len(R.timePoint.set) if hasattr(R.timePoint, 'set') else "missing")
        return

    last_set = R.timeInterval.set[-1]
    print("last_set type:", type(last_set))
    if isinstance(last_set, dict) and 'set' in last_set:
        last_set = last_set['set']
    elif isinstance(last_set, list) and len(last_set) > 0:
        last_set = last_set[-1]
    # Prefer set method to avoid constructor ambiguity
    if hasattr(last_set, 'interval'):
        IH = last_set.interval()
    else:
        IH = Interval(last_set)
    print("IH_inf =", IH.infimum().reshape(-1))
    print("IH_sup =", IH.supremum().reshape(-1))


if __name__ == "__main__":
    main()
