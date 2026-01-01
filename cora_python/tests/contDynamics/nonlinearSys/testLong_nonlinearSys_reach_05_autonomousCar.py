"""
testLong_nonlinearSys_reach_05_autonomousCar - unit_test_function of 
   nonlinear reachability analysis for following a reference trajectory
   Checks the solution of an autonomous car following a reference
   trajectory; It is checked whether the reachable set is enclosed
   in the initial set after a certain amount of time.

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/testLong_nonlinearSys_reach_05_autonomousCar.py

Inputs:
    -

Outputs:
    res - true/false

Authors:       Matthias Althoff
Written:       10-September-2015
Last update:   12-August-2016
               23-April-2020 (restructure params/options)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
import scipy.io
import os
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contDynamics.contDynamics.reach import reach
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
    
    # MATLAB: for i = 1:length(R)
    for i in range(num_time_steps):
        # MATLAB: R{i} is (2, 8) matrix
        R_i = R[i, 0]  # Extract (2, 8) matrix
        # MATLAB: uTransVec(1:16, i) = [R{i}(1,1:8), R{i}(2,1:8)]
        uTransVec[0, i] = R_i[0, 0]   # R{i}(1,1)
        uTransVec[1, i] = R_i[0, 1]   # R{i}(1,2)
        uTransVec[2, i] = R_i[0, 2]   # R{i}(1,3)
        uTransVec[3, i] = R_i[0, 3]   # R{i}(1,4)
        uTransVec[4, i] = R_i[0, 4]   # R{i}(1,5)
        uTransVec[5, i] = R_i[0, 5]   # R{i}(1,6)
        uTransVec[6, i] = R_i[0, 6]   # R{i}(1,7)
        uTransVec[7, i] = R_i[0, 7]   # R{i}(1,8)
        uTransVec[8, i] = R_i[1, 0]   # R{i}(2,1)
        uTransVec[9, i] = R_i[1, 1]   # R{i}(2,2)
        uTransVec[10, i] = R_i[1, 2]  # R{i}(2,3)
        uTransVec[11, i] = R_i[1, 3]  # R{i}(2,4)
        uTransVec[12, i] = R_i[1, 4]  # R{i}(2,5)
        uTransVec[13, i] = R_i[1, 5]  # R{i}(2,6)
        uTransVec[14, i] = R_i[1, 6]  # R{i}(2,7)
        uTransVec[15, i] = R_i[1, 7]  # R{i}(2,8)
        
        # MATLAB: Xn{i} is (8, 1) vector
        Xn_i = Xn[i, 0]  # Extract (8, 1) vector
        # MATLAB: uTransVec(17:24, i) = Xn{i}(1:8)
        uTransVec[16, i] = Xn_i[0, 0]  # Xn{i}(1)
        uTransVec[17, i] = Xn_i[1, 0]  # Xn{i}(2)
        uTransVec[18, i] = Xn_i[2, 0]  # Xn{i}(3)
        uTransVec[19, i] = Xn_i[3, 0]  # Xn{i}(4)
        uTransVec[20, i] = Xn_i[4, 0]  # Xn{i}(5)
        uTransVec[21, i] = Xn_i[5, 0]  # Xn{i}(6)
        uTransVec[22, i] = Xn_i[6, 0]  # Xn{i}(7)
        uTransVec[23, i] = Xn_i[7, 0]  # Xn{i}(8)
        
        # MATLAB: W{i} is (2, 1) vector
        W_i = W[i, 0]  # Extract (2, 1) vector
        # MATLAB: uTransVec(25:26, i) = W{i}(1:2)
        uTransVec[24, i] = W_i[0, 0]  # W{i}(1)
        uTransVec[25, i] = W_i[1, 0]  # W{i}(2)
    
    return uTransVec


class TestLongNonlinearSysReach05AutonomousCar:
    """Test class for nonlinearSys reach functionality (autonomousCar long test)"""
    
    def test_long_nonlinearSys_reach_05_autonomousCar(self):
        """Test reach for autonomous car example"""
        # Parameters --------------------------------------------------------------
        # MATLAB: dim_x = 8;
        dim_x = 8
        # MATLAB: params.tFinal=0.1; %final time
        # MATLAB: params.R0 = zonotope([[0; 0; 0; 22; 0 ; 0; -2.1854; 0],...
        # MATLAB:     0.05*diag([1, 1, 1, 1, 1, 1, 1, 1])]); %initial state for reachability analysis
        # MATLAB: params.u = uTRansVec4CASreach();
        # MATLAB: params.u = params.u(:,1:10);
        # MATLAB: params.U = zonotope([0*params.u(:,1), 0.05*diag([ones(5,1);zeros(21,1)])]);
        u_trans = uTRansVec4CASreach()
        params = {
            'tFinal': 0.1,  # final time
            'R0': Zonotope(
                np.array([[0], [0], [0], [22], [0], [0], [-2.1854], [0]]),
                0.05 * np.diag([1, 1, 1, 1, 1, 1, 1, 1])
            ),
            'u': u_trans[:, :10],  # Take first 10 columns
            'U': Zonotope(
                np.zeros((26, 1)),
                0.05 * np.diag(np.concatenate([np.ones(5), np.zeros(21)]))
            )
        }
        
        # Reachability Settings ---------------------------------------------------
        # MATLAB: options.timeStep=0.01;
        # MATLAB: options.taylorTerms=5;
        # MATLAB: options.zonotopeOrder=200;
        # MATLAB: options.maxError = ones(dim_x,1); % for comparison reasons
        # MATLAB: options.alg = 'lin';
        # MATLAB: options.tensorOrder = 2;
        options = {
            'timeStep': 0.01,
            'taylorTerms': 5,
            'zonotopeOrder': 200,
            'maxError': np.ones((dim_x, 1)),  # for comparison reasons
            'alg': 'lin',
            'tensorOrder': 2
        }
        
        # System Dynamics ---------------------------------------------------------
        # MATLAB: vehicle = nonlinearSys(@vmodel_A_bicycle_linear_controlled);
        vehicle = NonlinearSys(vmodel_A_bicycle_linear_controlled, states=8, inputs=26)
        
        # Reachability Analysis --------------------------------------------------- 
        # MATLAB: R = reach(vehicle, params, options);
        R = reach(vehicle, params, options)
        
        # Numerical Evaluation ----------------------------------------------------
        # enclose result by interval
        # MATLAB: IH = interval(R.timeInterval.set{end});
        IH = Interval(R['timeInterval']['set'][-1])
        
        # saved result
        # MATLAB: IH_saved = interval( ...
        IH_saved = Interval(
            np.array([[1.9113165972834527], [-0.1763919418894342], [-0.0628054820847352], 
                     [21.7144766641770168], [-0.1081823785867376], [-0.2077292813633349], 
                     [-2.5375737787154540], [-0.0418129308050226]]),
            np.array([[2.2486917066149412], [0.1775907076510275], [0.0638901147646950], 
                     [21.8628568496963389], [0.1329959142044251], [0.2499135613777886], 
                     [-1.9819546869205054], [0.0646318759557788]])
        )
        
        # final result
        # MATLAB: assert(isequal(IH,IH_saved,1e-8));
        assert IH.isequal(IH_saved, 1e-8), \
            f"Interval hull {IH} does not match saved result {IH_saved}"
        
        # test completed
        # MATLAB: res = true;
        res = True
        
        return res


def testLong_nonlinearSys_reach_05_autonomousCar():
    """Test function for nonlinearSys reach method (autonomousCar long test).
    
    Runs all test methods to verify correct implementation.
    
    NOTE: This test requires the actual .mat file data from uTRansVec4CASreach
    to fully replicate MATLAB results. The placeholder implementation may not
    produce identical results.
    """
    test = TestLongNonlinearSysReach05AutonomousCar()
    result = test.test_long_nonlinearSys_reach_05_autonomousCar()
    
    print("testLong_nonlinearSys_reach_05_autonomousCar: all tests passed")
    return result


if __name__ == "__main__":
    testLong_nonlinearSys_reach_05_autonomousCar()

