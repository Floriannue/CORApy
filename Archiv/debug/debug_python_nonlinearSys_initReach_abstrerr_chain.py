import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.contDynamics.linReach import linReach


def _print_interval(label, interval_obj):
    inf = interval_obj.infimum()
    sup = interval_obj.supremum()
    np.set_printoptions(precision=15, suppress=False)
    print(f"{label} inf:\n{inf.reshape(-1, 1)}")
    print(f"{label} sup:\n{sup.reshape(-1, 1)}")


def main():
    print("=== Python Debug: nonlinearSys initReach -> abstrerr chain ===")
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
    tank.derivatives(options)

    options['factor'] = []
    for i in range(1, options['taylorTerms'] + 2):
        options['factor'].append((options['timeStep'] ** i) / np.math.factorial(i))

    Rstart = {'set': params['R0'], 'error': np.zeros((dim_x, 1))}

    print("\n--- Test 1: linReach (alg='lin') ---")
    Rti, Rtp, dimForSplit, _ = linReach(tank, Rstart, params, options)
    _print_interval("IH_ti (lin)", Rti.interval())
    _print_interval("IH_tp (lin)", Rtp['set'].interval())
    print(f"Rtp.error (lin):\n{Rtp['error']}")

    print("\n--- Test 2: linReach (alg='poly') ---")
    options_poly = options.copy()
    options_poly['alg'] = 'poly'
    options_poly['tensorOrder'] = 3
    tank.derivatives(options_poly)
    Rti_p, Rtp_p, dimForSplit, _ = linReach(tank, Rstart, params, options_poly)
    _print_interval("IH_ti (poly)", Rti_p.interval())
    _print_interval("IH_tp (poly)", Rtp_p['set'].interval())
    print(f"Rtp.error (poly):\n{Rtp_p['error']}")


if __name__ == "__main__":
    main()
