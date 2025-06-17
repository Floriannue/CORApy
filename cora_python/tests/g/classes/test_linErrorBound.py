"""
test_linErrorBound - unit tests for LinErrorBound class

This module contains comprehensive unit tests for the LinErrorBound class
used in adaptive reachability analysis.

Authors: Python translation by AI Assistant
Written: 2025
"""

import pytest
import numpy as np
import scipy.linalg
from cora_python.g.classes.linErrorBound import LinErrorBound
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class TestLinErrorBound:
    """Test class for LinErrorBound"""
    
    def test_init_basic(self):
        """Test basic initialization"""
        emax = 0.1
        tFinal = 5.0
        errs = LinErrorBound(emax, tFinal)
        
        assert errs.emax == emax
        assert errs.tFinal == tFinal
        assert errs.useApproxFun == False
        assert isinstance(errs.timeSteps, list)
        assert isinstance(errs.seq_nonacc, list)
        assert isinstance(errs.step_acc, list)
    
    def test_init_invalid_params(self):
        """Test initialization with invalid parameters"""
        # Test negative error - should work but might be caught by validation elsewhere
        try:
            errs = LinErrorBound(-0.1, 5.0)
            # If no exception, check that it was created
            assert errs.emax == -0.1
        except (ValueError, AssertionError):
            pass  # Expected behavior
        
        # Test zero error - should work but might be caught by validation elsewhere  
        try:
            errs = LinErrorBound(0.0, 5.0)
            assert errs.emax == 0.0
        except (ValueError, AssertionError):
            pass  # Expected behavior
    
    def test_checkErrors_basic(self):
        """Test basic error checking functionality"""
        errs = LinErrorBound(0.1, 5.0)
        
        # Initialize some basic data
        errs.timeSteps = [0.1, 0.2]
        errs.seq_nonacc = [0.01, 0.02]
        errs.step_acc = [0.005, 0.01]
        errs.bound_rem = [0.05, 0.04]
        errs.bound_acc = [0.02, 0.03]
        errs.bound_red = [0.01, 0.01]
        errs.cum_acc = [0.005, 0.015]
        errs.cum_red = [0.01, 0.02]
        errs.step_red = [0.01, 0.01]
        
        # Test with valid errors
        Rout_error = np.array([0.05, 0.06])
        Rout_tp_error = np.array([0.03, 0.04])
        
        result = errs.checkErrors(Rout_error, Rout_tp_error)
        assert isinstance(result, bool)
    
    def test_nextBounds(self):
        """Test next bounds computation"""
        errs = LinErrorBound(0.1, 5.0)
        
        timeStep = 0.1
        t = 1.0
        k = 0
        k_iter = 0
        
        errs.nextBounds(timeStep, t, k, k_iter)
        
        # Check that bounds were created
        assert len(errs.bound_acc) > k
        assert len(errs.bound_rem) > k
        assert len(errs.bound_red) > k
        assert len(errs.bound_acc[k]) > k_iter
        assert len(errs.bound_rem[k]) > k_iter
        assert len(errs.bound_red[k]) > k_iter
    
    def test_accumulateErrors(self):
        """Test error accumulation"""
        errs = LinErrorBound(0.1, 5.0)
        
        # Initialize step errors
        errs.step_acc = [[0.01], [0.02]]
        errs.step_red = [0.005, 0.01]
        
        # Test first step (k=0)
        errs.accumulateErrors(0, 0)
        assert len(errs.cum_acc) == 1
        assert len(errs.cum_red) == 1
        assert errs.cum_acc[0] == 0.01
        assert errs.cum_red[0] == 0.005
        
        # Test second step (k=1)
        errs.accumulateErrors(1, 0)
        assert len(errs.cum_acc) == 2
        assert len(errs.cum_red) == 2
        assert errs.cum_acc[1] == 0.01 + 0.02
        assert errs.cum_red[1] == 0.005 + 0.01
    
    def test_fullErrors(self):
        """Test full error computation"""
        errs = LinErrorBound(0.1, 5.0)
        
        # Initialize data
        errs.seq_nonacc = [0.02, 0.03]
        errs.cum_acc = [0.01, 0.025]
        errs.cum_red = [0.005, 0.01]
        
        # Test first step (k=0)
        Rcont_error, Rcont_tp_error = errs.fullErrors(0)
        expected_error = 0.02 + 0.005  # seq_nonacc[0] + cum_red[0]
        expected_tp_error = 0.01 + 0.005  # cum_acc[0] + cum_red[0]
        
        assert Rcont_error == expected_error
        assert Rcont_tp_error == expected_tp_error
        
        # Test second step (k=1)
        Rcont_error, Rcont_tp_error = errs.fullErrors(1)
        expected_error = 0.03 + 0.01 + 0.01  # seq_nonacc[1] + cum_acc[0] + cum_red[1]
        expected_tp_error = 0.025 + 0.01  # cum_acc[1] + cum_red[1]
        
        assert Rcont_error == expected_error
        assert Rcont_tp_error == expected_tp_error
    
    def test_computeErrorBoundReduction(self):
        """Test error bound reduction computation"""
        errs = LinErrorBound(0.1, 5.0)
        
        # Test with no generators
        A = np.array([[-1, 0], [0, -2]])
        G_U = np.zeros((2, 2))
        
        errs.computeErrorBoundReduction(A, G_U)
        assert errs.bound_red_max == 0.0
        
        # Test with generators
        G_U = np.array([[1, 0], [0, 1]])
        errs.computeErrorBoundReduction(A, G_U)
        assert hasattr(errs, 'bound_red_max')
        assert errs.bound_red_max >= 0.0
        assert errs.bound_red_max <= errs.emax
        
    def test_init_invalid_params(self):
        """Test initialization with invalid parameters"""
        from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError
        
        # Test negative error
        with pytest.raises(CORAError):
            LinErrorBound(-0.1, 5.0)
        
        # Test zero error
        with pytest.raises(CORAError):
            LinErrorBound(0.0, 5.0)
        
        # Test negative time
        with pytest.raises(CORAError):
            LinErrorBound(0.1, -5.0)
    
    def test_checkErrors_basic(self):
        """Test basic error checking functionality"""
        errs = LinErrorBound(0.1, 5.0)
        
        # Initialize some basic data
        errs.timeSteps = [0.1, 0.2]
        errs.seq_nonacc = [0.01, 0.02]
        errs.step_acc = [0.005, 0.01]
        errs.bound_rem = [0.05, 0.04]
        errs.bound_acc = [0.02, 0.03]
        errs.bound_red = [0.01, 0.01]
        errs.cum_acc = [0.005, 0.015]
        errs.cum_red = [0.01, 0.02]
        errs.step_red = [0.01, 0.01]
        
        # Test with valid errors
        Rout_error = np.array([0.05, 0.06])
        Rout_tp_error = np.array([0.03, 0.04])
        
        result = errs.checkErrors(Rout_error, Rout_tp_error)
        assert isinstance(result, bool)
    
    def test_checkErrors_exceeds_bounds(self):
        """Test error checking when errors exceed bounds"""
        errs = LinErrorBound(0.1, 5.0)
        
        # Initialize data
        errs.timeSteps = [0.1]
        errs.seq_nonacc = [0.05]
        errs.step_acc = [0.02]
        errs.bound_rem = [0.04]
        errs.bound_acc = [0.03]
        errs.bound_red = [0.01]
        errs.cum_acc = [0.02]
        errs.cum_red = [0.01]
        errs.step_red = [0.01]
        
        # Test with error exceeding maximum
        Rout_error = np.array([0.15])  # Exceeds emax
        result = errs.checkErrors(Rout_error)
        assert result == False
    
    def test_nextBounds(self):
        """Test next bounds computation"""
        errs = LinErrorBound(0.1, 5.0)
        
        timeStep = 0.1
        t = 1.0
        k = 0
        k_iter = 0
        
        errs.nextBounds(timeStep, t, k, k_iter)
        
        # Check that bounds were created
        assert len(errs.bound_acc) > k
        assert len(errs.bound_rem) > k
        assert len(errs.bound_red) > k
        assert len(errs.bound_acc[k]) > k_iter
        assert len(errs.bound_rem[k]) > k_iter
        assert len(errs.bound_red[k]) > k_iter
    
    def test_accumulateErrors(self):
        """Test error accumulation"""
        errs = LinErrorBound(0.1, 5.0)
        
        # Initialize step errors
        errs.step_acc = [[0.01], [0.02]]
        errs.step_red = [0.005, 0.01]
        
        # Test first step (k=0)
        errs.accumulateErrors(0, 0)
        assert len(errs.cum_acc) == 1
        assert len(errs.cum_red) == 1
        assert errs.cum_acc[0] == 0.01
        assert errs.cum_red[0] == 0.005
        
        # Test second step (k=1)
        errs.accumulateErrors(1, 0)
        assert len(errs.cum_acc) == 2
        assert len(errs.cum_red) == 2
        assert errs.cum_acc[1] == 0.01 + 0.02
        assert errs.cum_red[1] == 0.005 + 0.01
    
    def test_removeRedundantValues(self):
        """Test removal of redundant values"""
        errs = LinErrorBound(0.1, 5.0)
        
        # Initialize with multiple iterations
        errs.seq_nonacc = [[0.01, 0.015, 0.02]]
        errs.step_acc = [[0.005, 0.008, 0.01]]
        errs.timeSteps = [[0.1, 0.15, 0.2]]
        errs.bound_rem = [[0.05, 0.04, 0.03]]
        errs.bound_acc = [[0.02, 0.025, 0.03]]
        errs.bound_red = [[0.01, 0.012, 0.015]]
        
        # Remove redundant values, keeping only iteration 1
        errs.removeRedundantValues(0, 1)
        
        # Check that only the selected iteration remains
        assert errs.seq_nonacc[0] == 0.015
        assert errs.step_acc[0] == 0.008
        assert errs.timeSteps[0] == 0.15
        assert errs.bound_rem[0] == 0.04
        assert errs.bound_acc[0] == 0.025
        assert errs.bound_red[0] == 0.012
    
    def test_fullErrors(self):
        """Test full error computation"""
        errs = LinErrorBound(0.1, 5.0)
        
        # Initialize data
        errs.seq_nonacc = [0.02, 0.03]
        errs.cum_acc = [0.01, 0.025]
        errs.cum_red = [0.005, 0.01]
        
        # Test first step (k=0)
        Rcont_error, Rcont_tp_error = errs.fullErrors(0)
        expected_error = 0.02 + 0.005  # seq_nonacc[0] + cum_red[0]
        expected_tp_error = 0.01 + 0.005  # cum_acc[0] + cum_red[0]
        
        assert Rcont_error == expected_error
        assert Rcont_tp_error == expected_tp_error
        
        # Test second step (k=1)
        Rcont_error, Rcont_tp_error = errs.fullErrors(1)
        expected_error = 0.03 + 0.01 + 0.01  # seq_nonacc[1] + cum_acc[0] + cum_red[1]
        expected_tp_error = 0.025 + 0.01  # cum_acc[1] + cum_red[1]
        
        assert Rcont_error == expected_error
        assert Rcont_tp_error == expected_tp_error
    
    def test_computeErrorBoundReduction(self):
        """Test error bound reduction computation"""
        errs = LinErrorBound(0.1, 5.0)
        
        # Test with no generators
        A = np.array([[-1, 0], [0, -2]])
        G_U = np.zeros((2, 2))
        
        errs.computeErrorBoundReduction(A, G_U)
        assert errs.bound_red_max == 0.0
        
        # Test with generators
        G_U = np.array([[1, 0], [0, 1]])
        errs.computeErrorBoundReduction(A, G_U)
        assert hasattr(errs, 'bound_red_max')
        assert errs.bound_red_max >= 0.0
        assert errs.bound_red_max <= errs.emax
    
    def test_updateCoefficientsApproxFun_first_iteration(self):
        """Test approximation function coefficient updates for first iteration"""
        errs = LinErrorBound(0.1, 5.0)
        errs.useApproxFun = True
        
        # Initialize data for first iteration
        errs.idv_PUtkplus1 = [[0.01]]
        errs.timeSteps = [[0.1]]
        errs.idv_linComb = [[0.005]]
        errs.idv_PUtauk = [[0.003]]
        errs.idv_F = [[0.008]]
        errs.idv_G = [[0.006]]
        
        errs.updateCoefficientsApproxFun(0, 0, True)
        
        # Check that coefficients were initialized
        assert hasattr(errs, 'coeff_PUtkplus1_a')
        assert hasattr(errs, 'coeff_PUtkplus1_b')
        assert hasattr(errs, 'coeff_linComb_b')
        assert hasattr(errs, 'coeff_PUtauk_b')
        assert hasattr(errs, 'coeff_F_a')
        assert hasattr(errs, 'coeff_G_a')
    
    def test_updateCoefficientsApproxFun_second_iteration(self):
        """Test approximation function coefficient updates for second iteration"""
        errs = LinErrorBound(0.1, 5.0)
        errs.useApproxFun = True
        
        # Initialize data for two iterations
        errs.idv_PUtkplus1 = [[0.01, 0.015]]
        errs.timeSteps = [[0.1, 0.12]]
        errs.idv_F = [[0.008, 0.012]]
        errs.idv_G = [[0.006, 0.009]]
        
        errs.updateCoefficientsApproxFun(0, 1, True)
        
        # Check that coefficients were updated using two points
        assert hasattr(errs, 'coeff_PUtkplus1_a')
        assert hasattr(errs, 'coeff_PUtkplus1_b')
        assert hasattr(errs, 'coeff_F_a')
        assert hasattr(errs, 'coeff_F_b')
    
    def test_updateBisection_first_iteration(self):
        """Test bisection update for first iteration"""
        errs = LinErrorBound(0.1, 5.0)
        
        # Initialize required data structures
        errs.bound_acc_ok = [[]]
        errs.bound_nonacc_ok = [[]]
        errs.step_acc = [[]]
        errs.seq_nonacc = [[]]
        errs.bound_acc = [[]]
        errs.bound_rem = [[]]
        errs.idv_PUtkplus1 = [[0.01]]
        
        timeStep = 0.1
        errs.updateBisection(0, 0, True, timeStep)
        
        # Check that bisection bounds were initialized
        assert hasattr(errs, 'bisect_lb_timeStep_acc')
        assert hasattr(errs, 'bisect_ub_timeStep_acc')
        assert hasattr(errs, 'bisect_lb_timeStep_nonacc')
        assert hasattr(errs, 'bisect_ub_timeStep_nonacc')
        
        assert errs.bisect_lb_timeStep_acc == 0.0
        assert errs.bisect_ub_timeStep_acc == timeStep
    
    def test_estimateTimeStepSize_approximation(self):
        """Test time step size estimation using approximation functions"""
        errs = LinErrorBound(0.1, 5.0)
        errs.useApproxFun = True
        
        # Initialize coefficients
        errs.coeff_PUtkplus1_a = 0.1
        errs.coeff_PUtkplus1_b = 0.01
        errs.coeff_F_a = 0.05
        errs.coeff_F_b = 0.005
        errs.coeff_G_a = 0.03
        errs.coeff_G_b = 0.003
        errs.coeff_linComb_b = 0.002
        errs.coeff_PUtauk_b = 0.001
        errs.bound_remacc = [0.05]
        errs.bound_rem = [[0.04]]
        
        # Initialize bisection bounds
        errs.bisect_lb_timeStep_acc = 0.0
        errs.bisect_lb_timeStep_nonacc = 0.0
        errs.bisect_ub_timeStep_acc = 1.0
        errs.bisect_ub_timeStep_nonacc = 1.0
        errs.bisect_ub_accok = False
        errs.bisect_ub_nonaccok = False
        
        t = 1.0
        timeStep = 0.2
        maxTimeStep = 0.5
        
        new_timeStep = errs.estimateTimeStepSize(t, 0, 0, True, timeStep, maxTimeStep, True)
        
        assert isinstance(new_timeStep, float)
        assert new_timeStep > 0
        assert new_timeStep <= maxTimeStep
    
    def test_estimateTimeStepSize_bisection(self):
        """Test time step size estimation using bisection"""
        errs = LinErrorBound(0.1, 5.0)
        errs.useApproxFun = False
        
        # Initialize bisection bounds
        errs.bisect_lb_timeStep_acc = 0.05
        errs.bisect_ub_timeStep_acc = 0.2
        errs.bisect_lb_timeStep_nonacc = 0.03
        errs.bisect_ub_timeStep_nonacc = 0.15
        errs.bisect_lb_acc = 0.01
        errs.bisect_ub_acc = 0.04
        errs.bisect_lb_nonacc = 0.005
        errs.bisect_ub_nonacc = 0.02
        errs.bisect_ub_accok = False
        errs.bisect_ub_nonaccok = False
        errs.bisect_lb_acc_perc = 0.2
        errs.bisect_ub_acc_perc = 0.8
        errs.bisect_lb_nonacc_perc = 0.1
        errs.bisect_ub_nonacc_perc = 0.6
        errs.cum_acc = [0.02]
        
        t = 1.0
        timeStep = 0.1
        maxTimeStep = 0.3
        
        new_timeStep = errs.estimateTimeStepSize(t, 1, 0, True, timeStep, maxTimeStep, True)
        
        assert isinstance(new_timeStep, float)
        assert new_timeStep > 0
        assert new_timeStep <= maxTimeStep
    
    def test_priv_errOp(self):
        """Test private error operation method"""
        errs = LinErrorBound(0.1, 5.0)
        
        # Test with array-like object
        S = np.array([1.0, 2.0, 3.0])
        result = errs._priv_errOp(S)
        assert isinstance(result, float)
        assert result > 0
        
        # Test with mock zonotope-like object
        class MockZonotope:
            def generators(self):
                return np.array([[1, 0], [0, 1]])
            def center(self):
                return np.array([0, 0])
        
        mock_zono = MockZonotope()
        result = errs._priv_errOp(mock_zono)
        assert isinstance(result, float)
        assert result > 0
        
        # Test with mock interval-like object
        class MockInterval:
            def infimum(self):
                return np.array([-1, -2])
            def supremum(self):
                return np.array([1, 2])
        
        mock_int = MockInterval()
        result = errs._priv_errOp(mock_int)
        assert isinstance(result, float)
        assert result > 0


def test_linErrorBound_integration():
    """Integration test for LinErrorBound with typical adaptive algorithm usage"""
    emax = 0.05
    tFinal = 2.0
    errs = LinErrorBound(emax, tFinal)
    
    # Simulate typical usage pattern
    A = np.array([[-1, 0], [0, -2]])
    G_U = np.array([[1, 0], [0, 1]])
    
    # Compute error bound reduction
    errs.computeErrorBoundReduction(A, G_U)
    
    # Simulate first step
    timeStep = 0.1
    t = 0.0
    k = 0
    k_iter = 0
    
    errs.nextBounds(timeStep, t, k, k_iter)
    
    # Initialize some error data
    errs.idv_PUtkplus1 = [[0.01]]
    errs.timeSteps = [[timeStep]]
    errs.step_acc = [[0.005]]
    errs.step_red = [0.002]
    errs.seq_nonacc = [[0.008]]
    
    # Update coefficients
    errs.updateCoefficientsApproxFun(k, k_iter, True)
    
    # Update bisection
    errs.updateBisection(k, k_iter, True, timeStep)
    
    # Accumulate errors
    errs.accumulateErrors(k, k_iter)
    
    # Remove redundant values
    errs.removeRedundantValues(k, k_iter)
    
    # Compute full errors
    Rcont_error, Rcont_tp_error = errs.fullErrors(k)
    
    # Check errors
    Rout_error = np.array([Rcont_error])
    Rout_tp_error = np.array([Rcont_tp_error])
    result = errs.checkErrors(Rout_error, Rout_tp_error)
    
    assert isinstance(result, bool)
    assert Rcont_error >= 0
    assert Rcont_tp_error >= 0


if __name__ == '__main__':
    test = TestLinErrorBound()
    test.test_init_basic()
    test.test_init_invalid_params()
    test.test_checkErrors_basic()
    test.test_nextBounds()
    test.test_accumulateErrors()
    test.test_fullErrors()
    test.test_computeErrorBoundReduction()
    print("All LinErrorBound tests passed!") 