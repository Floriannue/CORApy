"""
test_linearSys_simulate - unit test for simulate

This module contains unit tests for the simulate function of the LinearSys class.

Authors: Florian Nüssel (Python implementation)
Date: 8.6.2025
"""

import pytest
import numpy as np
from cora_python.contDynamics.linearSys import LinearSys


class TestLinearSysSimulate:
    """Test class for LinearSys.simulate method"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        # Define linearSys test systems
        
        # Stable system matrix: n x n
        self.A = np.array([
            [-0.3780, 0.2839, 0.5403, -0.2962],
            [0.1362, 0.2742, 0.5195, 0.8266],
            [0.0502, -0.1051, -0.6572, 0.3874],
            [1.0227, -0.4877, 0.8342, -0.2372]
        ])
        self.states = len(self.A)
        
        # Input matrix: n x m
        self.B = 0.25 * np.array([
            [-2, 0, 3],
            [2, 1, 0],
            [0, 0, 1],
            [0, -2, 1]
        ])
        self.inputs = self.B.shape[1]
        
        # Constant offset: n x 1
        self.c = 0.05 * np.array([-4, 2, 3, 1])
        
        # Output matrix: q x n
        self.C = np.array([
            [1, 1, 0, 0],
            [0, -0.5, 0.5, 0]
        ])
        
        # Feedthrough matrix: q x m
        self.D = np.array([
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        # Constant input: q x 1
        self.k = np.array([0, 0.02])
        
        # Disturbance matrix: n x r
        self.E = np.array([
            [1, 0.5],
            [0, -0.5],
            [1, -1],
            [0, 1]
        ])
        self.dists = self.E.shape[1]
        
        # Noise matrix: q x s
        self.F = np.array([[1], [0.5]])
        self.noises = self.F.shape[1]
        
        # Initialize different linearSys objects
        # sys_A has scalar input (like MATLAB linearSys(A,1))
        self.sys_A = LinearSys(self.A, np.ones((self.states, 1)))
        self.sys_AB = LinearSys(self.A, self.B)
        self.sys_ABC = LinearSys(self.A, self.B, None, self.C)
        self.sys_ABCD = LinearSys(self.A, self.B, None, self.C, self.D)
        self.sys_ABcCDk = LinearSys(self.A, self.B, self.c, self.C, self.D, self.k)
        self.sys_ABcCDkE = LinearSys(self.A, self.B, self.c, self.C, self.D, self.k, self.E)
        self.sys_ABcCDkEF = LinearSys(self.A, self.B, self.c, self.C, self.D, self.k, self.E, self.F)
        
        self.systems = [
            self.sys_A, self.sys_AB, self.sys_ABC, self.sys_ABCD,
            self.sys_ABcCDk, self.sys_ABcCDkE, self.sys_ABcCDkEF
        ]
        
        # Model parameters
        self.params = {}
        self.params['tFinal'] = 1.0
        self.dt_steps = 10
        
        # Initial state (simplified without zonotope for now)
        self.params['x0'] = 10 * np.ones(self.states)
        
        # Input vectors
        np.random.seed(42)  # For reproducible tests
        self.u_n = np.random.randn(self.states, self.dt_steps + 1)
        self.u_n_noD = np.random.randn(self.states, self.dt_steps)
        self.u_m = np.random.randn(self.inputs, self.dt_steps + 1)
        self.u_m_noD = np.random.randn(self.inputs, self.dt_steps)
        
        # Disturbance and noise (simplified)
        self.W_center = 0.02 + np.zeros(self.dists)
        self.V_center = -0.01 + np.zeros(self.noises)
    
    def _get_u(self, sys, states, inputs, u_n, u_m, u_n_noD, u_m_noD):
        """Return corresponding u based on system"""
        if sys.nr_of_inputs == states:
            if np.any(sys.D != 0):
                return u_n
            else:
                return u_n_noD
        elif sys.nr_of_inputs == inputs:
            if np.any(sys.D != 0):
                return u_m
            else:
                return u_m_noD
        elif sys.nr_of_inputs == 1:
            # For sys_A which has 1 input
            if np.any(sys.D != 0) if sys.D is not None else False:
                return np.random.randn(1, self.dt_steps + 1)
            else:
                return np.random.randn(1, self.dt_steps)
        else:
            return np.zeros((sys.nr_of_inputs, self.dt_steps + 1))
    
    def _get_w(self, sys, W_center, dt_steps):
        """Return corresponding w based on system"""
        if sys.nr_of_disturbances == 1:
            return np.random.randn(1, dt_steps)
        else:
            # Simplified random point generation
            return 0.02 * np.random.randn(sys.nr_of_disturbances, dt_steps)
    
    def _get_v(self, sys, V_center, dt_steps):
        """Return corresponding v based on system"""
        if sys.nr_of_noises == 1:
            return np.random.randn(1, dt_steps + 1)
        else:
            # Simplified random point generation
            return -0.01 + 0.01 * np.random.randn(sys.nr_of_noises, dt_steps + 1)
    
    def test_simulate_basic(self):
        """Test basic simulation functionality"""
        for sys in self.systems:
            # Basic simulation without inputs
            t, x, ind, y = sys.simulate(self.params)
            
            # Check outputs
            assert isinstance(t, np.ndarray), "Time should be numpy array"
            assert isinstance(x, np.ndarray), "State should be numpy array"
            assert ind is None, "Event index should be None for basic simulation"
            
            # Check dimensions
            assert len(t) == len(x), "Time and state should have same length"
            assert x.shape[1] == sys.nr_of_dims, "State should have correct dimension"
            
            # Check initial condition
            assert np.allclose(x[0, :], self.params['x0']), "Initial condition should match"
    
    def test_simulate_with_inputs(self):
        """Test simulation with various input combinations"""
        for sys in self.systems:
            params = self.params.copy()
            
            # Get appropriate input vectors
            u_sys = self._get_u(sys, self.states, self.inputs, 
                               self.u_n, self.u_m, self.u_n_noD, self.u_m_noD)
            w_sys = self._get_w(sys, self.W_center, self.dt_steps)
            v_sys = self._get_v(sys, self.V_center, self.dt_steps)
            
            # Test with input u only
            params['u'] = u_sys
            t, x, ind, y = sys.simulate(params)
            assert len(t) == len(x), "Time and state should have same length"
            del params['u']
            
            # Test with disturbance w only
            params['w'] = w_sys
            t, x, ind, y = sys.simulate(params)
            assert len(t) == len(x), "Time and state should have same length"
            del params['w']
            
            # Test with noise v only
            params['v'] = v_sys
            t, x, ind, y = sys.simulate(params)
            assert len(t) == len(x), "Time and state should have same length"
            del params['v']
            
            # Test with u and w
            params['u'] = u_sys
            params['w'] = w_sys
            t, x, ind, y = sys.simulate(params)
            assert len(t) == len(x), "Time and state should have same length"
            del params['u']
            del params['w']
            
            # Test with u and v
            params['u'] = u_sys
            params['v'] = v_sys
            t, x, ind, y = sys.simulate(params)
            assert len(t) == len(x), "Time and state should have same length"
            del params['u']
            del params['v']
            
            # Test with w and v
            params['w'] = w_sys
            params['v'] = v_sys
            t, x, ind, y = sys.simulate(params)
            assert len(t) == len(x), "Time and state should have same length"
            del params['w']
            del params['v']
            
            # Test with u, w, and v
            params['u'] = u_sys
            params['w'] = w_sys
            params['v'] = v_sys
            t, x, ind, y = sys.simulate(params)
            assert len(t) == len(x), "Time and state should have same length"
            del params['u']
            del params['w']
            del params['v']
    
    def test_simulate_with_timestep(self):
        """Test simulation with specified time step"""
        sys = self.sys_AB
        params = self.params.copy()
        params['timeStep'] = 0.1
        
        t, x, ind, y = sys.simulate(params)
        
        # Check time step
        expected_steps = int(params['tFinal'] / params['timeStep']) + 1
        assert len(t) == expected_steps, f"Expected {expected_steps} time points, got {len(t)}"
        
        # Check time spacing
        dt = np.diff(t)
        assert np.allclose(dt, params['timeStep'], atol=1e-10), "Time step should be consistent"
    
    def test_simulate_with_start_time(self):
        """Test simulation with non-zero start time"""
        sys = self.sys_AB
        params = self.params.copy()
        params['tStart'] = 0.5
        
        t, x, ind, y = sys.simulate(params)
        
        # Check start time
        assert np.isclose(t[0], params['tStart']), "Start time should match"
        assert np.isclose(t[-1], params['tFinal']), "End time should match"
    
    def test_simulate_output_computation(self):
        """Test output computation for systems with output matrices"""
        # Test system with output matrix
        sys = self.sys_ABC
        params = self.params.copy()
        
        t, x, ind, y = sys.simulate(params)
        
        if sys.C is not None and sys.C.size > 0:
            assert y is not None, "Output should be computed when C matrix exists"
            assert y.shape[0] == len(t), "Output should have same length as time"
            assert y.shape[1] == sys.nr_of_outputs, "Output should have correct dimension"
        else:
            assert y is None, "Output should be None when no C matrix"
    
    def test_simulate_error_handling(self):
        """Test error handling for invalid inputs"""
        sys = self.sys_AB
        
        # Test missing required parameters
        with pytest.raises(KeyError):
            sys.simulate({})  # Missing x0 and tFinal
        
        # Test invalid parameter sizes
        params = self.params.copy()
        params['timeStep'] = 0.1
        params['u'] = np.random.randn(sys.nr_of_inputs, 5)  # Wrong size
        
        with pytest.raises(ValueError):
            sys.simulate(params)
    
    def test_simulate_time_consistency(self):
        """Test time consistency for different scenarios"""
        sys = self.sys_AB
        
        # Test only time horizon
        params = {'x0': self.params['x0'], 'tFinal': 1.0}
        t, x, ind, y = sys.simulate(params)
        
        assert t[0] == 0, "Start time should be 0 by default"
        assert np.isclose(t[-1], params['tFinal']), "End time should match tFinal"
        
        # Test with start time
        params['tStart'] = 0.5
        t, x, ind, y = sys.simulate(params)
        
        assert np.isclose(t[0], params['tStart']), "Start time should match"
        assert np.isclose(t[-1], params['tFinal']), "End time should match"
        
        # Test with time step
        params['timeStep'] = 0.1
        steps = round((params['tFinal'] - params['tStart']) / params['timeStep'])
        t, x, ind, y = sys.simulate(params)
        
        assert np.isclose(t[0], params['tStart']), "Start time should match"
        assert len(t) == steps + 1, f"Should have {steps + 1} time points"
        assert len(x) == steps + 1, f"Should have {steps + 1} state points"
        assert np.isclose(t[-1], params['tFinal']), "End time should match"
    
    def test_simulate_example_from_docstring(self):
        """Test the example from the function docstring"""
        A = np.array([[1, 0], [0, 2]])
        B = np.array([[1], [2]])
        linsys = LinearSys(A, B)
        
        params = {'x0': np.array([1, 2]), 'tFinal': 2}
        
        t, x, ind, y = linsys.simulate(params)
        
        # Basic checks
        assert isinstance(t, np.ndarray), "Time should be numpy array"
        assert isinstance(x, np.ndarray), "State should be numpy array"
        assert len(t) == len(x), "Time and state should have same length"
        assert np.allclose(x[0, :], params['x0']), "Initial condition should match"
    
    def test_simulate_with_options(self):
        """Test simulation with solver options"""
        sys = self.sys_AB
        params = self.params.copy()
        
        # Test with solver options
        options = {'rtol': 1e-8, 'atol': 1e-10}
        t, x, ind, y = sys.simulate(params, options)
        
        # Should complete without error
        assert len(t) == len(x), "Time and state should have same length"
    
    def test_simulate_matrix_dimensions(self):
        """Test that matrix operations work correctly with different dimensions"""
        # Test 1D system
        A1 = np.array([[-1]])
        B1 = np.array([[1]])
        sys1 = LinearSys(A1, B1)
        
        params1 = {'x0': np.array([1]), 'tFinal': 1}
        t, x, ind, y = sys1.simulate(params1)
        
        assert x.shape[1] == 1, "1D system should have 1 state dimension"
        
        # Test higher dimensional system
        A3 = np.random.randn(5, 5)
        A3 = A3 - 2 * np.eye(5)  # Make stable
        B3 = np.random.randn(5, 2)
        sys3 = LinearSys(A3, B3)
        
        params3 = {'x0': np.random.randn(5), 'tFinal': 0.5}
        t, x, ind, y = sys3.simulate(params3)
        
        assert x.shape[1] == 5, "5D system should have 5 state dimensions"


def test_simulate_function_directly():
    """Test calling simulate function directly (not as method)"""
    from cora_python.contDynamics.linearSys.simulate import simulate
    
    A = np.array([[-1, 0], [0, -2]])
    B = np.array([[1], [1]])
    sys = LinearSys(A, B)
    
    params = {'x0': np.array([1, 1]), 'tFinal': 1}
    
    t, x, ind, y = simulate(sys, params)
    
    assert len(t) == len(x), "Time and state should have same length"
    assert np.allclose(x[0, :], params['x0']), "Initial condition should match"


if __name__ == '__main__':
    pytest.main([__file__]) 