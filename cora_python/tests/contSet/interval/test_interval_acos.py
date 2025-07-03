"""
Test cases for interval acos method

Authors: AI Assistant
Written: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestIntervalAcos:
    """Test class for interval acos method"""
    
    def test_acos_valid_interval(self):
        """Test acos with valid intervals within [-1, 1]"""
        # Test case 1: Normal interval within domain
        I = Interval(-0.5, 0.5)
        res = I.acos()
        
        # acos is decreasing, so acos(sup) becomes inf and acos(inf) becomes sup
        expected_inf = np.arccos(0.5)  # acos(sup)
        expected_sup = np.arccos(-0.5)  # acos(inf)
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
        
    def test_acos_boundary_values(self):
        """Test acos with boundary values [-1, 1]"""
        I = Interval(-1, 1)
        res = I.acos()
        
        expected_inf = np.arccos(1)  # = 0
        expected_sup = np.arccos(-1)  # = pi
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
        
    def test_acos_single_point(self):
        """Test acos with single point intervals"""
        # Test with 0
        I = Interval(0)
        res = I.acos()
        expected = np.arccos(0)  # pi/2
        
        assert np.allclose(res.inf, expected)
        assert np.allclose(res.sup, expected)
        
        # Test with 1
        I = Interval(1)
        res = I.acos()
        expected = np.arccos(1)  # 0
        
        assert np.allclose(res.inf, expected)
        assert np.allclose(res.sup, expected)
        
    def test_acos_matrix_intervals(self):
        """Test acos with matrix intervals"""
        # 2x2 matrix interval
        inf_mat = np.array([[-0.5, 0.2], [0.1, -0.8]])
        sup_mat = np.array([[0.3, 0.7], [0.6, -0.1]])
        
        I = Interval(inf_mat, sup_mat)
        res = I.acos()
        
        # Check that result has correct shape
        assert res.inf.shape == (2, 2)
        assert res.sup.shape == (2, 2)
        
        # Check individual elements (acos is decreasing)
        expected_inf = np.arccos(sup_mat)
        expected_sup = np.arccos(inf_mat)
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
        
    def test_acos_partial_domain_violations(self):
        """Test acos behavior when interval partially extends beyond domain"""
        # This should raise an error according to MATLAB behavior
        # when NaN values would be produced
        
        # Case: interval extends below -1
        I = Interval(-1.5, 0)
        with pytest.raises(CORAerror):
            I.acos()
            
        # Case: interval extends above 1  
        I = Interval(0, 1.5)
        with pytest.raises(CORAerror):
            I.acos()
            
        # Case: interval extends both ways
        I = Interval(-1.5, 1.5)
        with pytest.raises(CORAerror):
            I.acos()
            
    def test_acos_completely_outside_domain(self):
        """Test acos with intervals completely outside domain"""
        # Interval completely above 1
        I = Interval(1.2, 1.8)
        with pytest.raises(CORAerror):
            I.acos()
            
        # Interval completely below -1
        I = Interval(-1.8, -1.2)
        with pytest.raises(CORAerror):
            I.acos()
            
    def test_acos_monotonicity(self):
        """Test that acos respects monotonicity (decreasing)"""
        I1 = Interval(-0.8, -0.2)
        I2 = Interval(0.2, 0.8)
        
        res1 = I1.acos()
        res2 = I2.acos()
        
        # Since acos is decreasing, larger input intervals should give smaller output intervals
        # For I1 (negative values), acos values should be larger
        # For I2 (positive values), acos values should be smaller
        assert np.all(res1.inf > res2.inf)  # acos(-0.2) > acos(0.8)
        assert np.all(res1.sup > res2.sup)  # acos(-0.8) > acos(0.2)
        
    def test_acos_vector_intervals(self):
        """Test acos with vector intervals"""
        # Vector interval
        inf_vec = np.array([-0.5, 0.1, -1.0])
        sup_vec = np.array([0.2, 0.9, 0.5])
        
        I = Interval(inf_vec, sup_vec)
        res = I.acos()
        
        # Check shape
        assert res.inf.shape == (3,)
        assert res.sup.shape == (3,)
        
        # Check values (acos is decreasing)
        expected_inf = np.arccos(sup_vec)
        expected_sup = np.arccos(inf_vec)
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
        
    def test_acos_edge_cases(self):
        """Test acos with edge cases"""
        # Very small interval around 0
        I = Interval(-1e-10, 1e-10)
        res = I.acos()
        
        # Should be close to pi/2
        assert np.allclose(res.inf, np.pi/2, atol=1e-9)
        assert np.allclose(res.sup, np.pi/2, atol=1e-9)
        
        # Interval very close to boundary
        I = Interval(-0.9999, 0.9999)
        res = I.acos()
        
        # Should not raise error and give reasonable results
        assert np.all(np.isfinite(res.inf))
        assert np.all(np.isfinite(res.sup))
        assert np.all(res.inf >= 0)
        assert np.all(res.sup <= np.pi)
        
    def test_acos_type_consistency(self):
        """Test that acos returns proper Interval type"""
        I = Interval(-0.3, 0.7)
        res = I.acos()
        
        assert isinstance(res, Interval)
        assert hasattr(res, 'inf')
        assert hasattr(res, 'sup')
        
    def test_acos_numerical_stability(self):
        """Test numerical stability of acos implementation"""
        # Test with values very close to domain boundaries
        test_cases = [
            [-1 + 1e-15, -1 + 1e-10],  # Very close to -1
            [1 - 1e-10, 1 - 1e-15],    # Very close to 1
            [-1e-15, 1e-15],           # Very close to 0
        ]
        
        for inf_val, sup_val in test_cases:
            I = Interval(inf_val, sup_val)
            res = I.acos()
            
            # Results should be finite and within expected range
            assert np.all(np.isfinite(res.inf))
            assert np.all(np.isfinite(res.sup))
            assert np.all(res.inf >= 0)
            assert np.all(res.sup <= np.pi)
            assert np.all(res.inf <= res.sup)  # Interval ordering should be preserved 