"""
Test cases for interval acosh method

Authors: AI Assistant
Written: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestIntervalAcosh:
    """Test class for interval acosh method"""
    
    def test_acosh_valid_interval(self):
        """Test acosh with valid intervals within [1, inf)"""
        # Test case 1: Normal interval within domain
        I = Interval(2, 4)
        res = I.acosh()
        
        # acosh is increasing, so acosh(inf) becomes inf and acosh(sup) becomes sup
        expected_inf = np.arccosh(2)
        expected_sup = np.arccosh(4)
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
        
    def test_acosh_boundary_values(self):
        """Test acosh with boundary value 1"""
        I = Interval(1, 3)
        res = I.acosh()
        
        expected_inf = np.arccosh(1)  # = 0
        expected_sup = np.arccosh(3)
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
        
    def test_acosh_single_point(self):
        """Test acosh with single point intervals"""
        # Test with 1
        I = Interval(1)
        res = I.acosh()
        expected = np.arccosh(1)  # 0
        
        assert np.allclose(res.inf, expected)
        assert np.allclose(res.sup, expected)
        
        # Test with e
        I = Interval(np.e)
        res = I.acosh()
        expected = np.arccosh(np.e)
        
        assert np.allclose(res.inf, expected)
        assert np.allclose(res.sup, expected)
        
    def test_acosh_matrix_intervals(self):
        """Test acosh with matrix intervals"""
        # 2x2 matrix interval
        inf_mat = np.array([[1.5, 2.2], [1.1, 2.8]])
        sup_mat = np.array([[3.3, 4.7], [5.6, 3.1]])
        
        I = Interval(inf_mat, sup_mat)
        res = I.acosh()
        
        # Check that result has correct shape
        assert res.inf.shape == (2, 2)
        assert res.sup.shape == (2, 2)
        
        # Check individual elements (acosh is increasing)
        expected_inf = np.arccosh(inf_mat)
        expected_sup = np.arccosh(sup_mat)
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
        
    def test_acosh_domain_violations(self):
        """Test acosh behavior when interval extends below 1"""
        # This should raise an error according to MATLAB behavior
        # when NaN values would be produced
        
        # Case: interval extends below 1
        I = Interval(0.5, 2)
        with pytest.raises(CORAerror):
            I.acosh()
            
        # Case: interval completely below 1
        I = Interval(0.2, 0.8)
        with pytest.raises(CORAerror):
            I.acosh()
            
        # Case: interval with negative values
        I = Interval(-1, 2)
        with pytest.raises(CORAerror):
            I.acosh()
            
    def test_acosh_monotonicity(self):
        """Test that acosh respects monotonicity (increasing)"""
        I1 = Interval(1.2, 2.0)
        I2 = Interval(3.0, 5.0)
        
        res1 = I1.acosh()
        res2 = I2.acosh()
        
        # Since acosh is increasing, larger input intervals should give larger output intervals
        assert np.all(res1.inf < res2.inf)  # acosh(1.2) < acosh(3.0)
        assert np.all(res1.sup < res2.sup)  # acosh(2.0) < acosh(5.0)
        
    def test_acosh_vector_intervals(self):
        """Test acosh with vector intervals"""
        # Vector interval
        inf_vec = np.array([1.5, 2.1, 1.0])
        sup_vec = np.array([2.2, 3.9, 4.5])
        
        I = Interval(inf_vec, sup_vec)
        res = I.acosh()
        
        # Check shape
        assert res.inf.shape == (3,)
        assert res.sup.shape == (3,)
        
        # Check values (acosh is increasing)
        expected_inf = np.arccosh(inf_vec)
        expected_sup = np.arccosh(sup_vec)
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
        
    def test_acosh_large_values(self):
        """Test acosh with large values"""
        # Very large interval
        I = Interval(100, 1000)
        res = I.acosh()
        
        # Should give finite results
        assert np.all(np.isfinite(res.inf))
        assert np.all(np.isfinite(res.sup))
        assert np.all(res.inf > 0)
        assert np.all(res.inf < res.sup)
        
    def test_acosh_close_to_boundary(self):
        """Test acosh with values very close to 1"""
        I = Interval(1 + 1e-10, 1 + 1e-5)
        res = I.acosh()
        
        # Should not raise error and give reasonable results
        assert np.all(np.isfinite(res.inf))
        assert np.all(np.isfinite(res.sup))
        assert np.all(res.inf >= 0)
        assert np.all(res.inf <= res.sup)
        
    def test_acosh_type_consistency(self):
        """Test that acosh returns proper Interval type"""
        I = Interval(1.3, 2.7)
        res = I.acosh()
        
        assert isinstance(res, Interval)
        assert hasattr(res, 'inf')
        assert hasattr(res, 'sup')
        
    def test_acosh_numerical_stability(self):
        """Test numerical stability of acosh implementation"""
        # Test with values very close to domain boundary
        test_cases = [
            [1 + 1e-15, 1 + 1e-10],  # Very close to 1
            [1, 2],                   # Starting exactly at 1
            [1.0001, 1.001],         # Small interval above 1
        ]
        
        for inf_val, sup_val in test_cases:
            I = Interval(inf_val, sup_val)
            res = I.acosh()
            
            # Results should be finite and within expected range
            assert np.all(np.isfinite(res.inf))
            assert np.all(np.isfinite(res.sup))
            assert np.all(res.inf >= 0)
            assert np.all(res.inf <= res.sup)  # Interval ordering should be preserved
            
    def test_acosh_infinite_inputs(self):
        """Test acosh with infinite inputs"""
        # Interval extending to infinity
        I = Interval(2, np.inf)
        res = I.acosh()
        
        assert np.allclose(res.inf, np.arccosh(2))
        assert res.sup == np.inf
        
    def test_acosh_growth_behavior(self):
        """Test that acosh grows logarithmically for large inputs"""
        I1 = Interval(10, 20)
        I2 = Interval(100, 200)
        
        res1 = I1.acosh()
        res2 = I2.acosh()
        
        # acosh grows slower than linearly for large inputs
        # The ratio should be less than the input ratio
        input_ratio = 100 / 10  # 10
        output_ratio = res2.inf / res1.inf
        
        assert output_ratio < input_ratio  # acosh grows sublinearly 