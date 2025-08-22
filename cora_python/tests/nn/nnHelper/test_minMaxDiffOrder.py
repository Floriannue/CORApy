"""
Test for nnHelper.minMaxDiffOrder function

This test verifies that the minMaxDiffOrder function works correctly for computing min/max difference order.
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.minMaxDiffOrder import minMaxDiffOrder


class MockLayer:
    """Mock layer class for testing"""
    
    def __init__(self, f):
        self.f = f
    
    def computeApproxPoly(self, l, u, order, method):
        """Mock polynomial approximation"""
        # Simple polynomial approximation for testing
        if method == 'regression':
            # Linear approximation: f(x) ≈ f(l) + (f(u)-f(l))/(u-l) * (x-l)
            coeffs = np.array([self.f(l), (self.f(u) - self.f(l)) / (u - l)])
        else:  # 'singh' method
            # Quadratic approximation
            coeffs = np.array([self.f(l), (self.f(u) - self.f(l)) / (u - l), 0.1])
        return coeffs, None
    
    def getDerBounds(self, l, u):
        """Mock derivative bounds"""
        # Simple derivative bounds for testing
        der1l = (self.f(u) - self.f(l)) / (u - l) - 0.1
        der1u = (self.f(u) - self.f(l)) / (u - l) + 0.1
        return der1l, der1u


class TestMinMaxDiffOrder:
    """Test class for minMaxDiffOrder function"""
    
    def test_minMaxDiffOrder_basic(self):
        """Test basic minMaxDiffOrder functionality"""
        # Test with sigmoid-like function
        def sigmoid_f(x):
            return 1 / (1 + np.exp(-x))
        
        layer = MockLayer(sigmoid_f)
        l, u = 1, 2
        order = 1
        
        # Get polynomial approximation
        coeffs, _ = layer.computeApproxPoly(l, u, order, 'regression')
        der1l, der1u = layer.getDerBounds(l, u)
        
        # Compute min/max difference order
        diff2l, diff2u = minMaxDiffOrder(coeffs, l, u, layer.f, der1l, der1u)
        
        # Check that results are real numbers
        assert np.isreal(diff2l)
        assert np.isreal(diff2u)
        
        # Check that lower bound <= upper bound
        assert diff2l <= diff2u
    
    def test_minMaxDiffOrder_different_methods(self):
        """Test minMaxDiffOrder with different approximation methods"""
        def sigmoid_f(x):
            return 1 / (1 + np.exp(-x))
        
        layer = MockLayer(sigmoid_f)
        l, u = 1, 2
        order = 1
        
        # Test regression method
        coeffs_reg, _ = layer.computeApproxPoly(l, u, order, 'regression')
        der1l, der1u = layer.getDerBounds(l, u)
        diff2l_reg, diff2u_reg = minMaxDiffOrder(coeffs_reg, l, u, layer.f, der1l, der1u)
        
        # Test singh method
        coeffs_singh, _ = layer.computeApproxPoly(l, u, order, 'singh')
        diff2l_singh, diff2u_singh = minMaxDiffOrder(coeffs_singh, l, u, layer.f, der1l, der1u)
        
        # Both should return valid bounds
        assert diff2l_reg <= diff2u_reg
        assert diff2l_singh <= diff2u_singh
    
    def test_minMaxDiffOrder_different_functions(self):
        """Test minMaxDiffOrder with different functions"""
        l, u = 1, 2
        order = 1
        
        # Test with sigmoid function
        def sigmoid_f(x):
            return 1 / (1 + np.exp(-x))
        
        layer_sigmoid = MockLayer(sigmoid_f)
        coeffs_sigmoid, _ = layer_sigmoid.computeApproxPoly(l, u, order, 'regression')
        der1l_sigmoid, der1u_sigmoid = layer_sigmoid.getDerBounds(l, u)
        diff2l_sigmoid, diff2u_sigmoid = minMaxDiffOrder(
            coeffs_sigmoid, l, u, layer_sigmoid.f, der1l_sigmoid, der1u_sigmoid
        )
        
        # Test with tanh function
        def tanh_f(x):
            return np.tanh(x)
        
        layer_tanh = MockLayer(tanh_f)
        coeffs_tanh, _ = layer_tanh.computeApproxPoly(l, u, order, 'regression')
        der1l_tanh, der1u_tanh = layer_tanh.getDerBounds(l, u)
        diff2l_tanh, diff2u_tanh = minMaxDiffOrder(
            coeffs_tanh, l, u, layer_tanh.f, der1l_tanh, der1u_tanh
        )
        
        # Both should return valid bounds
        assert diff2l_sigmoid <= diff2u_sigmoid
        assert diff2l_tanh <= diff2u_tanh
    
    def test_minMaxDiffOrder_different_intervals(self):
        """Test minMaxDiffOrder with different intervals"""
        def sigmoid_f(x):
            return 1 / (1 + np.exp(-x))
        
        layer = MockLayer(sigmoid_f)
        order = 1
        
        # Test different intervals
        test_intervals = [
            (0, 1),
            (1, 2),
            (-1, 1),
            (0, 5)
        ]
        
        for l, u in test_intervals:
            coeffs, _ = layer.computeApproxPoly(l, u, order, 'regression')
            der1l, der1u = layer.getDerBounds(l, u)
            
            diff2l, diff2u = minMaxDiffOrder(coeffs, l, u, layer.f, der1l, der1u)
            
            # Check that results are valid
            assert np.isreal(diff2l)
            assert np.isreal(diff2u)
            assert diff2l <= diff2u
    
    def test_minMaxDiffOrder_different_orders(self):
        """Test minMaxDiffOrder with different polynomial orders"""
        def sigmoid_f(x):
            return 1 / (1 + np.exp(-x))
        
        layer = MockLayer(sigmoid_f)
        l, u = 1, 2
        
        # Test different orders
        for order in [1, 2, 3]:
            coeffs, _ = layer.computeApproxPoly(l, u, order, 'regression')
            der1l, der1u = layer.getDerBounds(l, u)
            
            diff2l, diff2u = minMaxDiffOrder(coeffs, l, u, layer.f, der1l, der1u)
            
            # Check that results are valid
            assert np.isreal(diff2l)
            assert np.isreal(diff2u)
            assert diff2l <= diff2u
    
    def test_minMaxDiffOrder_accuracy(self):
        """Test accuracy of minMaxDiffOrder computation"""
        def sigmoid_f(x):
            return 1 / (1 + np.exp(-x))
        
        layer = MockLayer(sigmoid_f)
        l, u = 1, 2
        order = 1
        
        # Get polynomial approximation
        coeffs, _ = layer.computeApproxPoly(l, u, order, 'regression')
        der1l, der1u = layer.getDerBounds(l, u)
        
        # Compute bounds
        diff2l, diff2u = minMaxDiffOrder(coeffs, l, u, layer.f, der1l, der1u)
        
        # Test containment at several points
        x_test = np.linspace(l, u, 10)
        
        for x in x_test:
            # Evaluate polynomial
            y_poly = np.polyval(coeffs, x)
            
            # Evaluate original function
            y_func = layer.f(x)
            
            # Difference should be within bounds
            diff = y_func - y_poly
            assert diff2l <= diff <= diff2u
    
    def test_minMaxDiffOrder_edge_cases(self):
        """Test minMaxDiffOrder edge cases"""
        def sigmoid_f(x):
            return 1 / (1 + np.exp(-x))
        
        layer = MockLayer(sigmoid_f)
        order = 1
        
        # Test with very small interval
        l, u = 1, 1.001
        coeffs, _ = layer.computeApproxPoly(l, u, order, 'regression')
        der1l, der1u = layer.getDerBounds(l, u)
        
        diff2l, diff2u = minMaxDiffOrder(coeffs, l, u, layer.f, der1l, der1u)
        assert diff2l <= diff2u
        
        # Test with very large interval
        l, u = 0, 100
        coeffs, _ = layer.computeApproxPoly(l, u, order, 'regression')
        der1l, der1u = layer.getDerBounds(l, u)
        
        diff2l, diff2u = minMaxDiffOrder(coeffs, l, u, layer.f, der1l, der1u)
        assert diff2l <= diff2u
    
    def test_minMaxDiffOrder_consistency(self):
        """Test that minMaxDiffOrder produces consistent results"""
        def sigmoid_f(x):
            return 1 / (1 + np.exp(-x))
        
        layer = MockLayer(sigmoid_f)
        l, u = 1, 2
        order = 1
        
        coeffs, _ = layer.computeApproxPoly(l, u, order, 'regression')
        der1l, der1u = layer.getDerBounds(l, u)
        
        # Call multiple times
        diff2l1, diff2u1 = minMaxDiffOrder(coeffs, l, u, layer.f, der1l, der1u)
        diff2l2, diff2u2 = minMaxDiffOrder(coeffs, l, u, layer.f, der1l, der1u)
        
        # Should be consistent
        assert np.isclose(diff2l1, diff2l2, atol=1e-10)
        assert np.isclose(diff2u1, diff2u2, atol=1e-10)
    
    def test_minMaxDiffOrder_error_handling(self):
        """Test minMaxDiffOrder error handling"""
        def sigmoid_f(x):
            return 1 / (1 + np.exp(-x))
        
        layer = MockLayer(sigmoid_f)
        l, u = 1, 2
        order = 1
        
        coeffs, _ = layer.computeApproxPoly(l, u, order, 'regression')
        der1l, der1u = layer.getDerBounds(l, u)
        
        # Test with invalid interval (l >= u)
        with pytest.raises(ValueError):
            minMaxDiffOrder(coeffs, 2, 1, layer.f, der1l, der1u)
        
        with pytest.raises(ValueError):
            minMaxDiffOrder(coeffs, 1, 1, layer.f, der1l, der1u)
    
    def test_minMaxDiffOrder_numerical_stability(self):
        """Test numerical stability"""
        def sigmoid_f(x):
            return 1 / (1 + np.exp(-x))
        
        layer = MockLayer(sigmoid_f)
        l, u = 1, 2
        order = 1
        
        coeffs, _ = layer.computeApproxPoly(l, u, order, 'regression')
        der1l, der1u = layer.getDerBounds(l, u)
        
        # Test with extreme derivative bounds
        extreme_der1l = -1e6
        extreme_der1u = 1e6
        
        diff2l, diff2u = minMaxDiffOrder(coeffs, l, u, layer.f, extreme_der1l, extreme_der1u)
        
        # Should still be finite
        assert np.isfinite(diff2l)
        assert np.isfinite(diff2u)
        assert diff2l <= diff2u
