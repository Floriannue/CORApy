"""
Test file for linearSys reach function

Tests the reachability analysis functionality for linear time-invariant systems,
including standard, wrapping-free, and fromStart algorithms.
"""

import numpy as np
import pytest
from cora_python.contDynamics.linearSys import LinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

from cora_python.g.classes.reachSet import ReachSet


class TestLinearSysReach:
    """Test class for linearSys reach functionality"""

    def test_reach_basic_standard(self):
        """Test basic reachability analysis with standard algorithm"""
        # System dynamics
        A = np.array([[-0.3780, 0.2839, 0.5403, -0.2962],
                      [0.1362, 0.2742, 0.5195, 0.8266],
                      [0.0502, -0.1051, -0.6572, 0.3874],
                      [1.0227, -0.4877, 0.8342, -0.2372]])
        B = 0.25 * np.array([[-2, 0, 3],
                             [2, 1, 0],
                             [0, 0, 1],
                             [0, -2, 1]])
        
        sys = LinearSys(A, B)
        
        # Parameters
        params = {
            'tFinal': 0.2,
            'R0': Zonotope(10 * np.ones((4, 1)), 0.5 * np.eye(4)),
            'U': Zonotope(np.zeros((3, 1)), 0.25 * np.eye(3))
        }
        
        # Options
        options = {
            'timeStep': 0.05,
            'taylorTerms': 6,
            'zonotopeOrder': 50,
            'linAlg': 'standard'
        }
        
        # Compute reachable set
        R = sys.reach(params, options)
        
        # Verify result structure
        assert isinstance(R, ReachSet)
        assert hasattr(R, 'timePoint')
        assert hasattr(R, 'timeInterval')
        assert len(R.timePoint.set) > 0
        assert len(R.timeInterval.set) > 0
        
        # Check time consistency
        expected_steps = int(params['tFinal'] / options['timeStep']) + 1
        assert len(R.timePoint.time) == expected_steps

    def test_reach_wrapping_free(self):
        """Test reachability analysis with wrapping-free algorithm"""
        # 5-dimensional system
        A = np.array([[-1, -4, 0, 0, 0],
                      [4, -1, 0, 0, 0],
                      [0, 0, -3, 1, 0],
                      [0, 0, -1, -3, 0],
                      [0, 0, 0, 0, -2]])
        B = np.array([[1], [0], [0], [0], [0]])
        
        sys = LinearSys(A, B)
        
        # Parameters
        params = {
            'tFinal': 0.2,
            'R0': Zonotope(np.ones((5, 1)), 0.1 * np.eye(5)),
            'U': Zonotope(np.array([[1]]), 
                         0.5 * np.array([[0.2]]))
        }
        
        # Options
        options = {
            'timeStep': 0.04,
            'taylorTerms': 4,
            'zonotopeOrder': 10,
            'linAlg': 'wrapping-free'
        }
        
        # Compute reachable set
        R = sys.reach(params, options)
        
        # Verify result structure
        assert isinstance(R, ReachSet)
        assert len(R.timePoint.set) > 0
        assert len(R.timeInterval.set) > 0

    def test_reach_zero_dynamics(self):
        """Test reachability analysis for zero dynamics (A = 0)"""
        # Zero dynamics system
        dim_x = 5
        A = np.zeros((dim_x, dim_x))
        B = np.zeros((dim_x, 1))
        
        sys = LinearSys(A, B)
        
        # Parameters
        params = {
            'tFinal': 3.0,
            'R0': Zonotope(np.ones((dim_x, 1)), 0.1 * np.eye(dim_x)),
            'U': Zonotope(np.zeros((1, 1)))
        }
        
        # Options
        options = {
            'timeStep': 0.04,
            'taylorTerms': 3,
            'zonotopeOrder': 100,
            'linAlg': 'standard'
        }
        
        # Compute reachable set
        R = sys.reach(params, options)
        
        # For zero dynamics, final set should be same as initial set
        final_set = R.timePoint.set[-1]
        initial_set = params['R0']
        
        # Check if sets are approximately equal (within tolerance)
        assert final_set.isequal(initial_set, 1e-8)

    def test_reach_constant_input(self):
        """Test reachability analysis with constant input offset"""
        # System with constant input (gravity example)
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [0]])
        c = np.array([[0], [-9.81]])
        
        sys = LinearSys(A, B, c)
        
        # Parameters
        params = {
            'tFinal': 0.2,
            'R0': Zonotope(np.array([[1], [0]]), np.diag([0.05, 0.05])),
            'U': Zonotope(np.zeros((1, 1)))
        }
        
        # Options
        options = {
            'timeStep': 0.1,
            'taylorTerms': 10,
            'zonotopeOrder': 20,
            'linAlg': 'standard'
        }
        
        # Compute reachable set
        R = sys.reach(params, options)
        
        # Verify result structure
        assert isinstance(R, ReachSet)
        assert len(R.timePoint.set) > 0

    def test_reach_double_integrator(self):
        """Test reachability analysis for double integrator (A = 0)"""
        # Double integrator system
        A = np.zeros((2, 2))
        B = np.array([[1], [1]])
        
        sys = LinearSys(A, B)
        
        # Parameters
        params = {
            'tFinal': 1.0,
            'R0': Zonotope(np.ones((2, 1)), 0.1 * np.eye(2)),
            'U': Zonotope(np.array([[1]]), np.array([[0.1]]))
        }
        
        # Options
        options = {
            'timeStep': 0.04,
            'taylorTerms': 4,
            'zonotopeOrder': 10,
            'linAlg': 'wrapping-free'
        }
        
        # Compute reachable set
        R = sys.reach(params, options)
        
        # Check final interval hull
        final_set = R.timeInterval.set[-1]
        IH = final_set.interval()
        
        # Expected result (from MATLAB) - column vectors, not matrices
        # MATLAB returns shape [2 1], not [2 2]
        expected_lower = np.array([[1.76], [1.76]])
        expected_upper = np.array([[2.20], [2.20]])

        # Ensure both are column vectors for comparison
        IH_inf = IH.infimum()
        IH_sup = IH.supremum()
        if IH_inf.ndim == 1:
            IH_inf = IH_inf.reshape(-1, 1)
        if IH_sup.ndim == 1:
            IH_sup = IH_sup.reshape(-1, 1)

        assert np.allclose(IH_inf, expected_lower, atol=1e-2)
        assert np.allclose(IH_sup, expected_upper, atol=1e-2)

    def test_reach_with_output_matrix(self):
        """Test reachability analysis with output matrix"""
        # System with output matrix
        A = np.array([[-0.1, -2], [2, -0.1]])
        B = np.array([[1], [0]])
        C = np.array([[2, -1]])
        
        sys = LinearSys(A, B, output_matrix=C)
        
        # Parameters
        params = {
            'tFinal': 1.0,
            'R0': Zonotope(10 * np.ones((2, 1)), 0.5 * np.eye(2)),
            'U': Zonotope(np.ones((1, 1)), 0.25 * np.eye(1))
        }
        
        # Options
        options = {
            'timeStep': 0.1,
            'taylorTerms': 4,
            'zonotopeOrder': 20,
            'linAlg': 'standard'
        }
        
        # Compute reachable set
        R = sys.reach(params, options)
        
        # Verify result structure
        assert isinstance(R, ReachSet)
        assert len(R.timePoint.set) > 0
        
        # Output dimension should match C matrix
        output_set = R.timePoint.set[-1]
        assert output_set.dim() == C.shape[0]

    def test_reach_input_trajectory(self):
        """Test reachability analysis with input trajectory"""
        # 5-dimensional system
        A = np.array([[-1, -4, 0, 0, 0],
                      [4, -1, 0, 0, 0],
                      [0, 0, -3, 1, 0],
                      [0, 0, -1, -3, 0],
                      [0, 0, 0, 0, -2]])
        B = np.array([[1], [0], [0], [0], [0]])
        
        sys = LinearSys(A, B)
        
        # Parameters with input trajectory
        u_traj = 100 * np.array([[1, 0, 0]])
        
        params = {
            'tFinal': 0.12,
            'R0': Zonotope(np.ones((5, 1))),
            'u': u_traj,
            'U': Zonotope(np.zeros((1, 1)))  # Input dimension should match B (1 input)
        }
        
        # Options
        options = {
            'timeStep': 0.04,
            'taylorTerms': 4,
            'zonotopeOrder': 200,
            'linAlg': 'standard'
        }
        
        # Compute reachable set
        R = sys.reach(params, options)
        
        # Verify result structure
        assert isinstance(R, ReachSet)
        assert len(R.timePoint.set) > 0

    def test_reach_different_algorithms(self):
        """Test that different algorithms produce valid results"""
        # Simple 2D system
        A = np.array([[-1, -4], [4, -1]])
        B = np.array([[1], [0]])
        
        sys = LinearSys(A, B)
        
        # Parameters
        params = {
            'tFinal': 0.2,
            'R0': Zonotope(np.array([[10], [5]]), 0.5 * np.eye(2)),
            'U': Zonotope(np.zeros((1, 1)), 0.25 * np.eye(1))
        }
        
        # Options for different algorithms
        base_options = {
            'timeStep': 0.05,
            'taylorTerms': 4,
            'zonotopeOrder': 20
        }
        
        algorithms = ['standard', 'wrapping-free']  # 'fromStart' not yet implemented
        results = {}
        
        for alg in algorithms:
            options = base_options.copy()
            options['linAlg'] = alg
            
            R = sys.reach(params, options)
            results[alg] = R
            
            # Verify result structure
            assert isinstance(R, ReachSet)
            assert len(R.timePoint.set) > 0
            assert len(R.timeInterval.set) > 0

    def test_reach_time_shifted(self):
        """Test reachability analysis with shifted start time"""
        # System dynamics
        A = np.array([[-0.3780, 0.2839, 0.5403, -0.2962],
                      [0.1362, 0.2742, 0.5195, 0.8266],
                      [0.0502, -0.1051, -0.6572, 0.3874],
                      [1.0227, -0.4877, 0.8342, -0.2372]])
        B = 0.25 * np.array([[-2, 0, 3],
                             [2, 1, 0],
                             [0, 0, 1],
                             [0, -2, 1]])
        C = np.array([[1, 1, 0, 0],
                      [0, -0.5, 0.5, 0]])
        
        sys = LinearSys(A, B, output_matrix=C)
        
        # Parameters with shifted start time
        params = {
            'tStart': 0.5,
            'tFinal': 1.0,
            'R0': Zonotope(10 * np.ones((4, 1)), 0.5 * np.eye(4)),
            'U': Zonotope(np.zeros((3, 1)), 0.05 * np.eye(3))
        }
        
        # Options
        options = {
            'timeStep': 0.1,
            'taylorTerms': 4,
            'zonotopeOrder': 20,
            'linAlg': 'standard'
        }
        
        # Compute reachable set
        R = sys.reach(params, options)
        
        # Check if times are correct (allow for floating point precision and time step rounding)
        assert np.isclose(R.timePoint.time[0], params['tStart'], atol=1e-10)  # Precise floating-point tolerance
        assert np.isclose(R.timePoint.time[-1], params['tFinal'], atol=1e-10)  # Precise floating-point tolerance

    def test_reach_error_handling(self):
        """Test error handling for invalid inputs"""
        # Simple system
        A = np.array([[-1, 0], [0, -1]])
        B = np.array([[1], [0]])
        sys = LinearSys(A, B)
        
        # Test missing required parameters
        with pytest.raises((KeyError, ValueError, CORAerror)):
            sys.reach({}, {})
        
        # Test invalid algorithm
        params = {
            'tFinal': 1.0,
            'R0': Zonotope(np.ones((2, 1)), np.eye(2))
        }
        options = {
            'linAlg': 'invalid_algorithm'
        }
        
        with pytest.raises((ValueError, CORAerror)):
            sys.reach(params, options)

    def test_reach_dimension_consistency(self):
        """Test dimension consistency checks"""
        # 2D system
        A = np.array([[-1, 0], [0, -1]])
        B = np.array([[1], [0]])
        sys = LinearSys(A, B)
        
        # Test with wrong dimension initial set
        params = {
            'tFinal': 1.0,
            'R0': Zonotope(np.ones((3, 1)), np.eye(3))  # Wrong dimension
        }
        options = {
            'linAlg': 'standard'
        }
        
        with pytest.raises((ValueError, AssertionError, CORAerror)):
            sys.reach(params, options)