"""
supportFunc_ - Calculate the upper or lower bound of a zonotope bundle
   along a certain direction

Syntax:
    val = supportFunc_(zB,dir)
    val, x = supportFunc_(zB,dir,type)

Inputs:
    zB - zonoBundle object
    dir - direction for which the bounds are calculated (vector of size (n,1))
    type - upper bound, lower bound, or both ('upper','lower','range')

Outputs:
    val - bound of the zonotope bundle in the specified direction
    x - support vector

Example: 
    Z1 = zonotope([0 1 2 0;0 1 0 2]);
    Z2 = zonotope([3 -0.5 3 0;-1 0.5 0 3]);
    zB = zonoBundle({Z1,Z2});
    val = supportFunc(zB,[1;1]);
   
    figure; hold on;
    plot(Z1); plot(Z2); plot(zB);
    plot(polytope([],[],[1,1],val),[1,2],'g');

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/supportFunc, conZonotope/supportFunc_

Authors:       Niklas Kochdumper, Mark Wetzlinger
Written:       19-November-2019
Last update:   23-April-2023 (MW, fix empty case)
Last revision: 27-March-2023 (MW, rename supportFunc_)
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Tuple, Union
import numpy as np
from scipy.linalg import block_diag

from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def supportFunc_(zB: 'ZonoBundle',
                 direction: np.ndarray,
                 type_: str = 'upper',
                 *args, **kwargs) -> Tuple[Union[float, Interval], np.ndarray]:
    """
    Calculate upper/lower support function value and optional support vector.
    """
    # Ensure direction is a column vector
    direction = np.asarray(direction)
    if direction.ndim == 1:
        direction = direction.reshape(-1, 1)
    elif direction.shape[0] == 1 and direction.shape[1] > 1:
        direction = direction.T

    n = zB.dim()

    # initialization
    Aeq_blocks = []
    beq_list = []
    lb_list = []
    ub_list = []

    # loop over all parallel sets
    for i in range(zB.parallelSets):
        Z = zB.Z[i]
        c = Z.center()
        G = Z.generators()
        nr_gens = G.shape[1]

        # construct equality constraint matrices
        Aeq_blocks.append(-G)
        beq_list.append(c.reshape(-1, 1))
        lb_list.append(-np.ones((nr_gens, 1)))
        ub_list.append(np.ones((nr_gens, 1)))

    # block diagonal for generators
    if len(Aeq_blocks) > 0:
        Aeq = block_diag(*Aeq_blocks)
    else:
        Aeq = np.zeros((0, 0))

    beq = np.vstack(beq_list) if len(beq_list) > 0 else np.zeros((0, 1))
    lb = np.vstack(lb_list) if len(lb_list) > 0 else np.zeros((0, 1))
    ub = np.vstack(ub_list) if len(ub_list) > 0 else np.zeros((0, 1))

    # add optimal point as an additional variable
    if Aeq.shape[1] > 0:
        A = np.vstack([np.eye(Aeq.shape[1]), -np.eye(Aeq.shape[1])])
        Aineq = np.hstack([np.zeros((A.shape[0], n)), A])
        bineq = np.vstack([ub, -lb])
    else:
        Aineq = np.zeros((0, n))
        bineq = np.zeros((0, 1))

    if zB.parallelSets > 0:
        Aeq_full = np.hstack([np.tile(np.eye(n), (zB.parallelSets, 1)), Aeq])
    else:
        Aeq_full = np.zeros((0, n + Aeq.shape[1]))

    f = np.vstack([direction, np.zeros((Aeq.shape[1], 1))])

    problem = {
        'Aineq': Aineq,
        'bineq': bineq,
        'Aeq': Aeq_full,
        'beq': beq,
        'lb': [],
        'ub': []
    }

    # upper/lower bound or range
    if type_ == 'lower':
        problem['f'] = f.flatten()
        x, val, exitflag, _, _ = CORAlinprog(problem)
        if exitflag == -2:
            val = float('inf')
            x = np.array([])
        elif exitflag <= -3:
            raise CORAerror('CORA:solverIssue')

    elif type_ == 'upper':
        problem['f'] = (-f).flatten()
        x, val, exitflag, _, _ = CORAlinprog(problem)
        if exitflag == -2:
            val = float('-inf')
            x = np.array([])
        elif exitflag <= -3:
            raise CORAerror('CORA:solverIssue')
        val = -val

    elif type_ == 'range':
        # solve linear program for upper bound
        problem['f'] = (-f).flatten()
        x_upper, val_upper, exitflag, _, _ = CORAlinprog(problem)
        if exitflag == -2:
            return Interval(float('-inf'), float('inf')), np.array([])
        elif exitflag <= -3:
            raise CORAerror('CORA:solverIssue')
        val_upper = -val_upper

        # solve linear program for lower bound
        problem['f'] = f.flatten()
        x_lower, val_lower, _, _, _ = CORAlinprog(problem)

        val = Interval(val_lower, val_upper)
        x = np.vstack([x_lower[:n], x_upper[:n]]) if x_lower is not None and x_upper is not None else np.array([])

    else:
        raise ValueError(f"Invalid type '{type_}'. Use 'lower', 'upper', or 'range'.")

    if type_ != 'range':
        x = x[:n] if isinstance(x, np.ndarray) and x.size > 0 else np.array([])

    return val, x

