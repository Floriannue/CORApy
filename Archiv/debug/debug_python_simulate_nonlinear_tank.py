import numpy as np
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.models.Cora.tank.tank6Eq import tank6Eq


def main():
    tank = NonlinearSys(tank6Eq, states=6, inputs=1)
    params = {
        'tStart': 0.0,
        'tFinal': 1.0,
        'timeStep': 0.1,
        'x0': np.array([[2], [4], [4], [2], [10], [4]]),
        'u': np.zeros((1, 1))
    }

    t, x, _, _ = tank.simulate(params)
    np.set_printoptions(precision=15, suppress=False)
    print("t =", t)
    print("x =", x.flatten())


if __name__ == "__main__":
    main()
