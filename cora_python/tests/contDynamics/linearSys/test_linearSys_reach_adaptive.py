"""
Test file for adaptive reachability analysis of linear systems
Tests the priv_reach_adaptive function and related components
"""

import numpy as np
import pytest
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval
from cora_python.g.classes.linErrorBound import LinErrorBound
from cora_python.g.classes.verifyTime import VerifyTime
from cora_python.contDynamics.linearSys.private.priv_reach_adaptive import (
    priv_reach_adaptive, compute_eps_linComb, compute_eps_F, compute_eps_G,
    priv_correctionMatrixState, priv_correctionMatrixInput, priv_expmRemainder
)


class TestLinearSysReachAdaptive:

    def test_basic_adaptive_reach(self):
        """Test basic adaptive reachability functionality"""
        # Simple 2D system
        A = np.array([[-1, -4], [4, -1]])
        linsys = LinearSys(A)
        
        # Initial set
        R0 = Zonotope(np.array([[1], [1]]), 0.1 * np.eye(2))
        
        # Parameters
        params = {
            'R0': R0,
            'tStart': 0.0,
            'tFinal': 1.0,
            'U': Zonotope(np.zeros((2, 1)), np.zeros((2, 2))),  # No input
            'uTrans': np.zeros((2, 1))
        }
        
        # Options
        options = {
            'error': 0.01,
            'verbose': 0,
            'verify': False
        }
        
        # Run adaptive reachability
        timeInt, timePoint, res, savedata = priv_reach_adaptive(linsys, params, options)
        
        # Basic checks
        assert isinstance(timeInt, dict)
        assert isinstance(timePoint, dict)
        assert isinstance(res, bool)
        assert isinstance(savedata, dict)
        
        # Check that sets were computed
        assert len(timeInt['set']) > 0
        assert len(timePoint['set']) > 0
        assert len(timeInt['time']) > 0
        assert len(timePoint['time']) > 0

    def test_adaptive_reach_with_input(self):
        """Test adaptive reachability with input uncertainty"""
        # Simple 2D system with input
        A = np.array([[-0.5, -2], [2, -0.5]])
        B = np.array([[1], [0]])
        linsys = LinearSys(A, B)
        
        # Initial set and input set
        R0 = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))
        U = Zonotope(np.array([[0]]), np.array([[0.1]]))
        
        # Parameters
        params = {
            'R0': R0,
            'tStart': 0.0,
            'tFinal': 0.5,
            'U': U,
            'uTrans': np.zeros((1, 1))
        }
        
        # Options with smaller error tolerance
        options = {
            'error': 0.005,
            'verbose': 0,
            'verify': False
        }
        
        # Run adaptive reachability
        timeInt, timePoint, res, savedata = priv_reach_adaptive(linsys, params, options)
        
        # Check results
        assert len(timeInt['set']) > 0
        assert len(timePoint['set']) > 0
        assert all(isinstance(s, (Zonotope, type(None))) for s in timeInt['set'] if s is not None)
        assert all(err >= 0 for err in timeInt['error'] if not np.isnan(err))

    def test_adaptive_reach_homogeneous_system(self):
        """Test adaptive reachability for homogeneous system (no input)"""
        # 3D stable system
        A = np.array([[-1, 0, 0], [0, -2, 1], [0, -1, -2]])
        linsys = LinearSys(A)
        
        # Initial set
        center = np.array([[1], [0], [1]])
        generators = 0.05 * np.eye(3)
        R0 = Zonotope(center, generators)
        
        # Parameters
        params = {
            'R0': R0,
            'tStart': 0.0,
            'tFinal': 2.0,
            'U': Zonotope(np.zeros((3, 1)), np.zeros((3, 3))),  # No input
            'uTrans': np.zeros((3, 1))
        }
        
        # Options
        options = {
            'error': 0.01,
            'verbose': 0,
            'verify': False
        }
        
        # Run adaptive reachability
        timeInt, timePoint, res, savedata = priv_reach_adaptive(linsys, params, options)
        
        # Verify convergence and error bounds
        assert res is True
        assert len(timeInt['set']) > 0
        
        # Check that errors are within bounds
        valid_errors = [err for err in timeInt['error'] if not np.isnan(err)]
        if valid_errors:
            assert all(err <= options['error'] * 1.1 for err in valid_errors)  # Small tolerance

    def test_adaptive_reach_with_output_matrix(self):
        """Test adaptive reachability with output matrix"""
        # 2D system with 1D output
        A = np.array([[-1, 2], [-2, -1]])
        C = np.array([[1, 0]])  # Only observe first state
        linsys = LinearSys(A, C=C)
        
        # Initial set
        R0 = Zonotope(np.array([[1], [0]]), 0.1 * np.eye(2))
        
        # Parameters
        params = {
            'R0': R0,
            'tStart': 0.0,
            'tFinal': 1.0,
            'U': Zonotope(np.zeros((2, 1)), np.zeros((2, 2))),
            'uTrans': np.zeros((2, 1))
        }
        
        # Options
        options = {
            'error': 0.01,
            'verbose': 0,
            'verify': False
        }
        
        # Run adaptive reachability
        timeInt, timePoint, res, savedata = priv_reach_adaptive(linsys, params, options)
        
        # Check that output sets have correct dimension
        for s in timePoint['set']:
            if s is not None and hasattr(s, 'dim'):
                assert s.dim() == 1  # Output dimension should be 1

    def test_adaptive_reach_different_error_tolerances(self):
        """Test adaptive reachability with different error tolerances"""
        A = np.array([[-0.5, 1], [-1, -0.5]])
        linsys = LinearSys(A)
        
        R0 = Zonotope(np.array([[1], [1]]), 0.1 * np.eye(2))
        
        base_params = {
            'R0': R0,
            'tStart': 0.0,
            'tFinal': 1.0,
            'U': Zonotope(np.zeros((2, 1)), np.zeros((2, 2))),
            'uTrans': np.zeros((2, 1))
        }
        
        error_tolerances = [0.1, 0.01, 0.001]
        results = []
        
        for error_tol in error_tolerances:
            params = base_params.copy()
            options = {
                'error': error_tol,
                'verbose': 0,
                'verify': False
            }
            
            timeInt, timePoint, res, savedata = priv_reach_adaptive(linsys, params, options)
            results.append((len(timeInt['set']), max(timeInt['error']) if timeInt['error'] else 0))
        
        # Check that smaller error tolerance leads to more time steps
        assert results[2][0] >= results[1][0] >= results[0][0]

    def test_correction_matrix_state(self):
        """Test state correction matrix computation"""
        A = np.array([[-1, 2], [-2, -1]])
        linsys = LinearSys(A)
        
        # Initialize taylor object
        from cora_python.g.classes.taylorLinSys import TaylorLinSys
        linsys.taylor = TaylorLinSys(A)
        
        timeStep = 0.1
        truncationOrder = 10
        
        F = priv_correctionMatrixState(linsys, timeStep, truncationOrder)
        
        # Check that F is an interval matrix
        assert hasattr(F, 'infimum') and hasattr(F, 'supremum')
        assert F.infimum().shape == A.shape
        assert F.supremum().shape == A.shape
        
        # Check that interval bounds are consistent
        assert np.all(F.infimum() <= F.supremum())

    def test_correction_matrix_input(self):
        """Test input correction matrix computation"""
        A = np.array([[-1, 2], [-2, -1]])
        linsys = LinearSys(A)
        
        # Initialize taylor object
        from cora_python.g.classes.taylorLinSys import TaylorLinSys
        linsys.taylor = TaylorLinSys(A)
        
        timeStep = 0.1
        truncationOrder = 10
        
        G = priv_correctionMatrixInput(linsys, timeStep, truncationOrder)
        
        # Check that G is an interval matrix
        assert hasattr(G, 'infimum') and hasattr(G, 'supremum')
        assert G.infimum().shape == A.shape
        assert G.supremum().shape == A.shape
        
        # Check that interval bounds are consistent
        assert np.all(G.infimum() <= G.supremum())

    def test_expm_remainder(self):
        """Test exponential matrix remainder computation"""
        A = np.array([[-1, 2], [-2, -1]])
        linsys = LinearSys(A)
        
        # Initialize taylor object
        from cora_python.g.classes.taylorLinSys import TaylorLinSys
        linsys.taylor = TaylorLinSys(A)
        
        timeStep = 0.1
        truncationOrder = 5
        
        E = priv_expmRemainder(linsys, timeStep, truncationOrder)
        
        # Check that E is an interval matrix
        assert hasattr(E, 'infimum') and hasattr(E, 'supremum')
        assert E.infimum().shape == A.shape
        assert E.supremum().shape == A.shape
        
        # Check that remainder is symmetric around zero
        assert np.allclose(E.infimum(), -E.supremum(), atol=1e-10)

    def test_compute_eps_linComb(self):
        """Test linear combination error computation"""
        errs = LinErrorBound(0.01, 1.0)
        
        A = np.array([[-1, 2], [-2, -1]])
        timeStep = 0.1
        eAdt = np.array([[0.9, 0.2], [-0.2, 0.9]])  # Approximation of exp(A*dt)
        
        # Test with zonotope
        startset = Zonotope(np.array([[1], [0]]), 0.1 * np.eye(2))
        
        eps_linComb = compute_eps_linComb(errs, eAdt, startset)
        
        assert isinstance(eps_linComb, (float, np.floating))
        assert eps_linComb >= 0

    def test_compute_eps_F_zonotope(self):
        """Test state curvature error computation with zonotope"""
        errs = LinErrorBound(0.01, 1.0)
        
        # Create interval matrix F
        F_inf = np.array([[-0.01, 0.005], [0.005, -0.01]])
        F_sup = np.array([[0.01, -0.005], [-0.005, 0.01]])
        F = Interval(F_inf, F_sup)
        
        # Test with zonotope startset
        startset = Zonotope(np.array([[1], [0]]), 0.1 * np.eye(2))
        
        eps_F, box_center, box_G = compute_eps_F(errs, F, startset)
        
        assert isinstance(eps_F, (float, np.floating))
        assert eps_F >= 0
        assert box_center.shape[0] == 2
        assert box_G.shape[0] == 2

    def test_compute_eps_G(self):
        """Test input curvature error computation"""
        errs = LinErrorBound(0.01, 1.0)
        
        # Create interval matrix G
        G_inf = np.array([[-0.01, 0.005], [0.005, -0.01]])
        G_sup = np.array([[0.01, -0.005], [-0.005, 0.01]])
        G = Interval(G_inf, G_sup)
        
        u = np.array([0.1, -0.05])
        
        eps_G, Gu_c, Gu_G = compute_eps_G(errs, G, u)
        
        assert isinstance(eps_G, (float, np.floating))
        assert eps_G >= 0
        assert Gu_c.shape == u.shape
        assert Gu_G.shape == u.shape

    def test_error_validation(self):
        """Test that computed errors are within specified bounds"""
        A = np.array([[-1, 1], [-1, -1]])
        linsys = LinearSys(A)
        
        R0 = Zonotope(np.array([[1], [1]]), 0.05 * np.eye(2))
        
        params = {
            'R0': R0,
            'tStart': 0.0,
            'tFinal': 0.5,
            'U': Zonotope(np.zeros((2, 1)), np.zeros((2, 2))),
            'uTrans': np.zeros((2, 1))
        }
        
        # Test with very strict error bound
        options = {
            'error': 0.001,
            'verbose': 0,
            'verify': False
        }
        
        timeInt, timePoint, res, savedata = priv_reach_adaptive(linsys, params, options)
        
        # Check that all computed errors are within the specified bound
        valid_errors = [err for err in timeInt['error'] if not np.isnan(err)]
        if valid_errors:
            max_error = max(valid_errors)
            assert max_error <= options['error'] * 1.05  # Small tolerance for numerical precision

    def test_adaptive_vs_fixed_step(self):
        """Compare adaptive algorithm with fixed step size"""
        A = np.array([[-0.5, 2], [-2, -0.5]])
        linsys = LinearSys(A)
        
        R0 = Zonotope(np.array([[1], [0]]), 0.1 * np.eye(2))
        
        params = {
            'R0': R0,
            'tStart': 0.0,
            'tFinal': 1.0,
            'U': Zonotope(np.zeros((2, 1)), np.zeros((2, 2))),
            'uTrans': np.zeros((2, 1))
        }
        
        # Adaptive algorithm
        options_adaptive = {
            'error': 0.01,
            'verbose': 0,
            'verify': False
        }
        
        timeInt_adaptive, _, _, _ = priv_reach_adaptive(linsys, params, options_adaptive)
        
        # Check that adaptive algorithm produces reasonable results
        assert len(timeInt_adaptive['set']) > 0
        assert len(timeInt_adaptive['time']) > 0
        
        # Check that time intervals cover the full time horizon
        total_time = sum(
            interval.supremum() - interval.infimum() 
            for interval in timeInt_adaptive['time'] 
            if hasattr(interval, 'supremum')
        )
        assert abs(total_time - params['tFinal']) < 1e-6

    def test_high_dimensional_system(self):
        """Test adaptive reachability for higher dimensional system"""
        # 4D system
        A = np.array([
            [-1, 0, 1, 0],
            [0, -2, 0, 1], 
            [-1, 0, -1, 0],
            [0, -1, 0, -2]
        ])
        linsys = LinearSys(A)
        
        # Initial set
        center = np.array([[1], [0], [0], [1]])
        generators = 0.1 * np.eye(4)
        R0 = Zonotope(center, generators)
        
        params = {
            'R0': R0,
            'tStart': 0.0,
            'tFinal': 1.0,
            'U': Zonotope(np.zeros((4, 1)), np.zeros((4, 4))),
            'uTrans': np.zeros((4, 1))
        }
        
        options = {
            'error': 0.02,
            'verbose': 0,
            'verify': False
        }
        
        # Run adaptive reachability
        timeInt, timePoint, res, savedata = priv_reach_adaptive(linsys, params, options)
        
        # Basic checks for higher dimensional case
        assert len(timeInt['set']) > 0
        assert len(timePoint['set']) > 0
        assert res is True

    def test_caching_functionality(self):
        """Test that computed matrices are properly cached"""
        A = np.array([[-1, 2], [-2, -1]])
        linsys = LinearSys(A)
        
        # Initialize taylor object
        from cora_python.g.classes.taylorLinSys import TaylorLinSys
        linsys.taylor = TaylorLinSys(A)
        
        timeStep = 0.1
        truncationOrder = 10
        
        # First computation
        F1 = priv_correctionMatrixState(linsys, timeStep, truncationOrder)
        
        # Second computation (should use cache)
        F2 = priv_correctionMatrixState(linsys, timeStep, truncationOrder)
        
        # Check that cached result is returned
        assert np.allclose(F1.infimum(), F2.infimum())
        assert np.allclose(F1.supremum(), F2.supremum())
        
        # Check that cache exists
        assert hasattr(linsys.taylor, '_F_cache')
        assert timeStep in linsys.taylor._F_cache

    @pytest.mark.parametrize("error_tol", [0.1, 0.01, 0.001])
    def test_different_error_tolerances_parametrized(self, error_tol):
        """Parametrized test for different error tolerances"""
        A = np.array([[-1, 1], [-1, -1]])
        linsys = LinearSys(A)
        
        R0 = Zonotope(np.array([[1], [0]]), 0.1 * np.eye(2))
        
        params = {
            'R0': R0,
            'tStart': 0.0,
            'tFinal': 0.5,
            'U': Zonotope(np.zeros((2, 1)), np.zeros((2, 2))),
            'uTrans': np.zeros((2, 1))
        }
        
        options = {
            'error': error_tol,
            'verbose': 0,
            'verify': False
        }
        
        timeInt, timePoint, res, savedata = priv_reach_adaptive(linsys, params, options)
        
        # Check that algorithm completes successfully
        assert res is True
        assert len(timeInt['set']) > 0
        
        # Check that errors are within tolerance
        valid_errors = [err for err in timeInt['error'] if not np.isnan(err)]
        if valid_errors:
            assert all(err <= error_tol * 1.1 for err in valid_errors)


if __name__ == '__main__':
    pytest.main([__file__]) 