"""
testLong_nonlinearSys_reach_time - unit_test_function of nonlinear
   reachability analysis for following a reference trajectory

Checks the solution of an autonomous car following a reference trajectory;
It is checked whether the final reachable set encloses the end points of
the simulated trajectories

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/testLong_nonlinearSys_reach_time.py

Inputs:
    -

Outputs:
    res - true/false

Authors:       Niklas Kochdumper
Written:       23-March-2018
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
import scipy.io
import os
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.contDynamics.reach import reach
from cora_python.contDynamics.contDynamics.simulateRandom import simulateRandom
from cora_python.contSet.polytope import Polytope
from cora_python.models.Cora.autonomousCar.vmodel_A_bicycle_linear_controlled import vmodel_A_bicycle_linear_controlled




def uTRansVec4CASreach():
    """
    uTRansVec4CASreach - Returns reference trajectory input vector
    
    Loads data from the .mat file:
    'linearized_controller_09_double_lane_change_jy(0)=0,jy(1)=0.mat'
    
    This file contains R, Xn, W cell arrays that are converted to uTransVec.
    R is a cell array of (2, 8) matrices (feedback matrix)
    Xn is a cell array of (8, 1) vectors (reference state)
    W is a cell array of (2, 1) vectors (feedforward input)
    
    Returns:
        uTransVec: Matrix of size (26, N) where N is the number of time steps
                   Columns: [R(1,1:8), R(2,1:8), Xn(1:8), W(1:2)]
    """
    # Load the .mat file
    # MATLAB location: cora_matlab/models/Cora/contDynamics/nonlinearSys/models/
    from cora_python.g.macros.CORAROOT import CORAROOT
    mat_file_path = os.path.join(
        CORAROOT(), 'models', 'Cora', 'contDynamics', 'nonlinearSys', 'models',
        'linearized_controller_09_double_lane_change_jy(0)=0,jy(1)=0.mat'
    )
    
    data = scipy.io.loadmat(mat_file_path)
    R = data['R']  # Shape: (N, 1) cell array, each element is (2, 8)
    Xn = data['Xn']  # Shape: (N, 1) cell array, each element is (8, 1)
    W = data['W']  # Shape: (N, 1) cell array, each element is (2, 1)
    
    num_time_steps = len(R)
    uTransVec = np.zeros((26, num_time_steps))
    
    for i in range(num_time_steps):
        R_i = R[i, 0]  # Extract (2, 8) matrix
        Xn_i = Xn[i, 0]  # Extract (8, 1) vector
        W_i = W[i, 0]  # Extract (2, 1) vector
        
        # Fill uTransVec: [R(1,1:8), R(2,1:8), Xn(1:8), W(1:2)]
        uTransVec[0:8, i] = R_i[0, :].flatten()
        uTransVec[8:16, i] = R_i[1, :].flatten()
        uTransVec[16:24, i] = Xn_i.flatten()
        uTransVec[24:26, i] = W_i.flatten()
    
    return uTransVec


class TestLongNonlinearSysReachTime:
    """Test class for nonlinearSys reach functionality with reference trajectory"""
    
    def test_long_nonlinearSys_reach_time(self):
        """Test reach for autonomous car following reference trajectory"""
        # assume true
        # MATLAB: res = true;
        res = True
        
        # Parameters --------------------------------------------------------------
        # MATLAB: dim_x = 8;
        # MATLAB: params.tFinal=4;
        # MATLAB: params.R0 = zonotope([[0; 0; 0; 22; 0 ; 0; -2.1854; 0],0.05*diag(ones(dim_x,1))]);
        # MATLAB: params.u = uTRansVec4CASreach();
        # MATLAB: params.u = params.u(:,1:400);
        # MATLAB: params.U = zonotope([0*params.u(:,1), 0.05*diag([ones(5,1);zeros(21,1)])]);
        dim_x = 8
        u_trans = uTRansVec4CASreach()
        u_trans = u_trans[:, :400]  # Take first 400 columns
        
        params = {
            'tFinal': 4,
            'R0': Zonotope(np.array([[0], [0], [0], [22], [0], [0], [-2.1854], [0]]), 
                          0.05 * np.diag(np.ones(dim_x))),
            'u': u_trans,
            'U': Zonotope(np.zeros((26, 1)), 0.05 * np.diag(np.concatenate([np.ones(5), np.zeros(21)])))
        }
        
        # Reachability settings ---------------------------------------------------
        # MATLAB: options.timeStep=0.01;
        # MATLAB: options.taylorTerms=5;
        # MATLAB: options.zonotopeOrder=200;
        # MATLAB: options.maxError = ones(dim_x,1);
        # MATLAB: options.alg = 'lin';
        # MATLAB: options.tensorOrder = 2;
        # MATLAB: options.reductionInterval = Inf;
        options = {
            'timeStep': 0.01,
            'taylorTerms': 5,
            'zonotopeOrder': 200,
            'maxError': np.ones((dim_x, 1)),
            'alg': 'lin',
            'tensorOrder': 2,
            'reductionInterval': np.inf
        }
        
        # System Dynamics ---------------------------------------------------------
        # MATLAB: vehicle = nonlinearSys(@vmodel_A_bicycle_linear_controlled,8,26);
        vehicle = NonlinearSys(vmodel_A_bicycle_linear_controlled, states=8, inputs=26)
        
        # Reachability Analysis --------------------------------------------------- 
        # MATLAB: Rset = reach(vehicle, params, options);
        Rset = reach(vehicle, params, options)
        
        # Simulation --------------------------------------------------------------
        # MATLAB: simOpt.points = 20;
        # MATLAB: simOpt.fracVert = 0.5;
        # MATLAB: simOpt.fracInpVert = 1;
        # MATLAB: simRes = simulateRandom(vehicle,params,simOpt);
        simOpt = {
            'points': 20,
            'fracVert': 0.5,
            'fracInpVert': 1
        }
        simRes = simulateRandom(vehicle, params, simOpt)
        
        # Numerical Evaluation ----------------------------------------------------
        # check if end points are inside the final reachable set
        # MATLAB: R = Rset.timeInterval.set{end};
        # MATLAB: R = reduce(R,'girard',1);
        # MATLAB: P = polytope(R);
        if isinstance(Rset, dict):
            timeInterval_set = Rset.get('timeInterval', {}).get('set', [])
        else:
            timeInterval_set = Rset.timeInterval.set if hasattr(Rset, 'timeInterval') else []
        
        if len(timeInterval_set) > 0:
            R = timeInterval_set[-1]
            R = R.reduce('girard', 1)  # Using object method reduce()
            P = Polytope(R)
            
            # MATLAB: simEndPoints = reshape(cell2mat(arrayfun(@(s) s.x{1}(end,:)', simRes, 'UniformOutput', false)),8,[]);
            # Extract end points from simulation results
            simEndPoints = []
            for s in simRes:
                if isinstance(s, dict):
                    x_data = s.get('x', [])
                    if len(x_data) > 0:
                        simEndPoints.append(x_data[0][-1, :].reshape(-1, 1))
                else:
                    if hasattr(s, 'x') and len(s.x) > 0:
                        simEndPoints.append(s.x[0][-1, :].reshape(-1, 1))
            
            if len(simEndPoints) > 0:
                simEndPoints = np.hstack(simEndPoints)
                
                # MATLAB: res = all(contains(P,simEndPoints));
                from cora_python.contSet.polytope.contains_ import contains_
                res_contains, _, _ = contains_(P, simEndPoints)
                res = np.all(res_contains) if isinstance(res_contains, np.ndarray) else res_contains
            else:
                res = False
        else:
            res = False
        
        assert res, "All simulated end points should be contained in the final reachable set"
        
        return res


def testLong_nonlinearSys_reach_time():
    """Test function for nonlinearSys reach method with reference trajectory.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestLongNonlinearSysReachTime()
    result = test.test_long_nonlinearSys_reach_time()
    
    print("testLong_nonlinearSys_reach_time: all tests passed")
    return result


if __name__ == "__main__":
    testLong_nonlinearSys_reach_time()

