"""
distance - computes the shortest distance from a polytope to another set
   or point cloud; the cases are:
   - one operand is the empty set: Inf
   - the operands intersect: 0
   - the operands do not intersect: > 0
   - unbounded + bounded: some value < Inf
   - unbounded + unbounded: some value < Inf  

Syntax:
   val = distance(P,S)

Inputs:
   P - polytope object
   S - contSet object or point (cloud)

Outputs:
   val - shortest distance

Example: 
   P = polytope([-1 -1; 1 0;-1 0; 0 1; 0 -1],[2;3;2;3;2]);
   S = polytope([ 1 0;-1 0; 0 1; 0 -1],[1;1;1;1]);
   val = distance(P,S);

Reference: MPT-Toolbox https://www.mpt3.org/

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Victor Gassmann, Viktor Kotsev (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       26-July-2021 (MATLAB)
Last update:   07-November-2022 (added polytope-polytope case, MATLAB)
               18-December-2023 (MW, add intersection check, MATLAB)
Last revision: ---
"""

import numpy as np
import scipy.optimize as opt
from typing import Union, List, Tuple

from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

# Assuming these are available as methods or globally imported functions
# P.dim()
# P.representsa_()
# P.isIntersecting_()
# P.constraints() - ensures A, b, Aeq, beq are set
# Polytope.A_.val, Polytope.b_.val, Polytope.Ae_.val, Polytope.be_.val
# findClassArg - helper for reordering
from cora_python.g.functions.matlab.validate.preprocessing.find_class_arg import find_class_arg

def distance(P_in, S_in):
    # check input
    inputArgsCheck([
        [P_in, 'att', 'polytope'], # Removed 'scalar' attribute
        [S_in, 'att', ['ellipsoid', 'numeric', 'polytope']]
    ])

    # set tolerance
    tol = 1e-12

    # check dimensions
    equal_dim_check(P_in, S_in)

    # Assign P and S based on type. find_class_arg is for finding the *one* instance of a classname
    # when mixed types are possible. For polytope to polytope (or empty polytope), we know types.
    if isinstance(P_in, Polytope) and isinstance(S_in, Polytope):
        P = P_in
        S = S_in
    else:
        # For cases like Polytope to numeric, or Polytope to Ellipsoid
        P, S = find_class_arg(P_in, S_in, 'polytope')

    # empty set case: distance defined as infinity
    S_is_empty_set = False
    if isinstance(S, np.ndarray):
        if S.size == 0:
            S_is_empty_set = True
    elif hasattr(S, 'representsa_'): # Only call representsa_ if the object has it
        if S.representsa_('emptySet'):
            S_is_empty_set = True

    if P.representsa_('emptySet') or S_is_empty_set:
        return np.inf

    # sets intersect (LP): shortest distance is 0
    if P.isIntersecting_(S, 'exact', tol):
        return 0.0

    # we require the halfspace representation
    P.constraints() # This ensures P.A_ etc are set

    # select case
    if isinstance(S, np.ndarray):
        # distance to a point or point cloud
        if S.shape[1] == 1 and P.representsa_('hyperplane', 1e-12):
            # Analytical solution for hyperplane-point distance
            # MATLAB: val = abs(P.be-P.Ae*S) / (P.Ae*P.Ae');
            # P.Ae is P.Aeq, P.be is P.beq for a hyperplane
            # For a hyperplane A*x=b, distance from point S is |A*S - b| / ||A||_2
            # Aeq is (1,n), beq is (1,1)
            val = np.abs(P.Aeq @ S - P.beq) / np.linalg.norm(P.Aeq) # Assuming Aeq is vector
            return val[0,0] # Return scalar value
        else:
            return aux_distancePointCloud(P, S)

    elif isinstance(S, Ellipsoid):
        # call ellipsoid function
        # This is where ellipsoid.distance(S, P) is called
        # Ensure that ellipsoid.distance is robust for Ellipsoid to Polytope distance
        return S.distance(P)

    elif isinstance(S, Polytope):
        return aux_distancePoly(P, S)

    else:
        # throw error
        raise CORAerror('CORA:noops', P, S)

# Auxiliary functions (will be defined below)
def aux_distancePointCloud(P: Polytope, Y: np.ndarray) -> Union[float, np.ndarray]:
    """
    aux_distancePointCloud - computes the shortest distance from a polytope to a point or point cloud.

    Inputs:
        P - polytope object
        Y - numpy array (point or point cloud, (n, N) where N is number of points)

    Outputs:
        val - shortest distance(s)
    """
    n, N = Y.shape

    # Ensure halfspace representation of polytope is available
    P.constraints()

    # Get constraint matrices
    A = P.A
    b = P.b
    Aeq = P.Ae
    beq = P.be

    # Define the objective function for minimization: norm(x - y_i)
    # The objective is ||x - y_i||_2, so we minimize its square for differentiability: ||x - y_i||_2^2
    # f(x) = (x - y_i).T @ (x - y_i)
    # grad_f(x) = 2 * (x - y_i)
    # hess_f(x) = 2 * I (identity matrix)

    def objective(x, y_target):
        return np.sum((x - y_target.flatten())**2)

    def jacobian(x, y_target):
        return 2 * (x - y_target.flatten())

    def hessian(x, y_target):
        return 2 * np.eye(n)

    # Define the constraints for scipy.optimize.minimize
    # Inequality constraints: A_ineq @ x - b_ineq <= 0
    # Equality constraints: A_eq @ x - b_eq = 0

    constraints = []
    if A.shape[0] > 0: # If there are inequality constraints
        constraints.append(opt.LinearConstraint(A, -np.inf, b.flatten())) # A*x <= b => A*x - b <= 0, so upper bound is b
    if Aeq.shape[0] > 0: # If there are equality constraints
        constraints.append(opt.LinearConstraint(Aeq, beq.flatten(), beq.flatten())) # Aeq*x = beq

    # Initial guess for x (e.g., origin or average of point cloud)
    x0 = np.zeros(n)

    val = np.zeros(N)
    for i in range(N):
        y_target_i = Y[:, i:i+1] # Get current column as (n, 1) vector

        # Solve the optimization problem
        res = opt.minimize(
            objective,
            x0,
            args=(y_target_i,),
            method='trust-constr', # Suitable for non-linear objective with constraints
            jac=jacobian,
            hess=hessian,
            constraints=constraints,
            options={'verbose': 0, 'gtol': 1e-12, 'xtol': 1e-12}
        )

        if not res.success:
            # Handle solver failure. This might mean the polytope is empty or unbounded in a problematic way.
            # For distance, if it can't find a solution, it's often due to numerical issues
            # or the problem being infeasible for the solver.
            raise CORAerror('CORA:polytopeDistance:solverIssue',
                            f'scipy.optimize.minimize failed to compute distance to point cloud: {res.message}')

        val[i] = np.sqrt(res.fun) # res.fun is the squared norm (distance^2)

    if N == 1:
        return val[0]
    return val

def aux_distancePoly(P: Polytope, S_poly: Polytope) -> float:
    """
    aux_distancePoly - computes the shortest distance between two polytopes by solving a quadratic program.

    Inputs:
        P      - first polytope object
        S_poly - second polytope object

    Outputs:
        val    - shortest distance
    """
    # Ensure halfspace representation of both polytopes is available
    P.constraints()
    S_poly.constraints()

    n = P.dim() # Dimension of the ambient space

    # Constraints for the combined variable z = [x_P; x_S]
    # x_P is a point in P, x_S is a point in S_poly
    # Minimize ||x_P - x_S||^2
    # Objective: x_P.T @ x_P - 2 * x_P.T @ x_S + x_S.T @ x_S

    # The MATLAB code builds a problem structure for quadprog.
    # problem.H is the Hessian, problem.f is the linear term of the objective.
    # In scipy.optimize.minimize, we define the objective, Jacobian, and Hessian functions.

    # Combined variable dimension: 2 * n
    # z = [x_P; x_S]
    dim_z = 2 * n

    # Objective function components (minimize 0.5 * z.T @ H @ z + f.T @ z)
    # The objective is ||x_P - x_S||^2
    # f(x_P, x_S) = (x_P - x_S).T @ (x_P - x_S)
    #               = x_P.T @ x_P - 2 * x_P.T @ x_S + x_S.T @ x_S
    # Hessian matrix H for quadratic objective 0.5 * z.T @ H @ z
    # H = 2 * [[I, -I], [-I, I]] for minimizing (x_P - x_S)^2
    H = 2 * np.block([[np.eye(n), -np.eye(n)],
                      [-np.eye(n), np.eye(n)]])
    f = np.zeros(dim_z) # No linear term

    def objective(z):
        # z is a 1D array from the solver, need to reshape to get x_P and x_S
        x_P = z[:n].reshape(-1, 1)
        x_S = z[n:].reshape(-1, 1)
        return np.sum((x_P - x_S)**2)

    def jacobian(z):
        x_P = z[:n]
        x_S = z[n:]
        grad_x_P = 2 * (x_P - x_S)
        grad_x_S = 2 * (x_S - x_P)
        return np.concatenate((grad_x_P, grad_x_S))

    def hessian(z):
        # Hessian is constant for this quadratic objective
        return H

    # Linear inequality constraints: A_ineq @ z <= b_ineq
    # Combine P's and S_poly's constraints
    # P.A @ x_P <= P.b
    # S_poly.A @ x_S <= S_poly.b
    A_ineq_P = np.hstack([P.A, np.zeros((P.A.shape[0], n))])
    b_ineq_P = P.b.flatten()

    A_ineq_S = np.hstack([np.zeros((S_poly.A.shape[0], n)), S_poly.A])
    b_ineq_S = S_poly.b.flatten()

    A_ineq = np.vstack([A_ineq_P, A_ineq_S]) if A_ineq_P.shape[0] > 0 or A_ineq_S.shape[0] > 0 else np.array([[]]).reshape(0, dim_z)
    b_ineq = np.concatenate([b_ineq_P, b_ineq_S]) if b_ineq_P.shape[0] > 0 or b_ineq_S.shape[0] > 0 else np.array([])


    # Linear equality constraints: A_eq @ z == b_eq
    # P.Ae @ x_P == P.be
    # S_poly.Ae @ x_S == S_poly.be
    A_eq_P = np.hstack([P.Ae, np.zeros((P.Ae.shape[0], n))])
    b_eq_P = P.be.flatten()

    A_eq_S = np.hstack([np.zeros((S_poly.Ae.shape[0], n)), S_poly.Ae])
    b_eq_S = S_poly.be.flatten()


    A_eq = np.vstack([A_eq_P, A_eq_S]) if A_eq_P.shape[0] > 0 or A_eq_S.shape[0] > 0 else np.array([[]]).reshape(0, dim_z)
    b_eq = np.concatenate([b_eq_P, b_eq_S]) if b_eq_P.shape[0] > 0 or b_eq_S.shape[0] > 0 else np.array([])


    constraints = []
    if A_ineq.shape[0] > 0:
        constraints.append(opt.LinearConstraint(A_ineq, -np.inf, b_ineq))
    if A_eq.shape[0] > 0:
        constraints.append(opt.LinearConstraint(A_eq, b_eq, b_eq))

    # Initial guess
    x0 = np.zeros(dim_z)

    res = opt.minimize(
        objective,
        x0,
        method='trust-constr', # Suitable for quadratic objective with linear constraints
        jac=jacobian,
        hess=hessian,
        constraints=constraints,
        options={'verbose': 0, 'gtol': 1e-12, 'xtol': 1e-12}
    )

    if not res.success:
        raise CORAerror('CORA:polytopeDistance:solverIssue',
                        f'scipy.optimize.minimize failed to compute distance between polytopes: {res.message}')

    val_squared = res.fun
    val = np.sqrt(val_squared)

    return val 