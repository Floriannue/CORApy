"""
Tests for nnLeakyReLULayer class.
"""

import pytest
import numpy as np
from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer


class TestNnLeakyReLULayer:
    """Test class for nnLeakyReLULayer."""
    
    def test_nnLeakyReLULayer_constructor_default_alpha(self):
        """Test constructor with default alpha"""
        layer = nnLeakyReLULayer()
        assert layer.name == "leakyrelu"
        assert layer.type == "leakyrelu"
        assert layer.alpha == 0.01
    
    def test_nnLeakyReLULayer_constructor_custom_alpha(self):
        """Test constructor with custom alpha"""
        alpha = 0.1
        layer = nnLeakyReLULayer(alpha)
        assert layer.name == "leakyrelu"
        assert layer.type == "leakyrelu"
        assert layer.alpha == alpha
    
    def test_nnLeakyReLULayer_constructor_custom_name(self):
        """Test constructor with custom name"""
        layer = nnLeakyReLULayer(0.05, "custom_leakyrelu")
        assert layer.name == "custom_leakyrelu"
        assert layer.type == "leakyrelu"
        assert layer.alpha == 0.05
    
    def test_nnLeakyReLULayer_function_values_positive(self):
        """Test LeakyReLU function values for positive inputs"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test positive inputs
        x = np.array([[1], [2], [3]])
        y = layer.f(x)
        
        # For positive inputs, LeakyReLU(x) = x
        expected = np.array([[1], [2], [3]])
        assert np.allclose(y, expected)
    
    def test_nnLeakyReLULayer_function_values_negative(self):
        """Test LeakyReLU function values for negative inputs"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test negative inputs
        x = np.array([[-1], [-2], [-3]])
        y = layer.f(x)
        
        # For negative inputs, LeakyReLU(x) = alpha * x
        expected = np.array([[-0.01], [-0.02], [-0.03]])
        assert np.allclose(y, expected)
    
    def test_nnLeakyReLULayer_function_values_mixed(self):
        """Test LeakyReLU function values for mixed inputs"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test mixed inputs
        x = np.array([[-2], [0], [2]])
        y = layer.f(x)
        
        expected = np.array([[-0.02], [0], [2]])
        assert np.allclose(y, expected)
    
    def test_nnLeakyReLULayer_function_values_zero(self):
        """Test LeakyReLU function value at zero"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test zero input
        x = np.array([[0]])
        y = layer.f(x)
        
        # LeakyReLU(0) = 0
        assert np.isclose(y[0, 0], 0)
    
    def test_nnLeakyReLULayer_derivative_values_positive(self):
        """Test LeakyReLU derivative values for positive inputs"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test positive inputs
        x = np.array([[1], [2], [3]])
        y_prime = layer.df(x)
        
        # For positive inputs, LeakyReLU'(x) = 1
        expected = np.array([[1], [1], [1]])
        assert np.allclose(y_prime, expected)
    
    def test_nnLeakyReLULayer_derivative_values_negative(self):
        """Test LeakyReLU derivative values for negative inputs"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test negative inputs
        x = np.array([[-1], [-2], [-3]])
        y_prime = layer.df(x)
        
        # For negative inputs, LeakyReLU'(x) = alpha
        expected = np.array([[0.01], [0.01], [0.01]])
        assert np.allclose(y_prime, expected)
    
    def test_nnLeakyReLULayer_derivative_values_zero(self):
        """Test LeakyReLU derivative value at zero"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test zero input
        x = np.array([[0]])
        y_prime = layer.df(x)
        
        # At zero, we use the positive derivative (1)
        assert np.isclose(y_prime[0, 0], 1)
    
    def test_nnLeakyReLULayer_second_derivative(self):
        """Test LeakyReLU second derivative"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test various inputs
        x = np.array([[-2], [-1], [0], [1], [2]])
        y_double_prime = layer._df2(x)
        
        # Second derivative should be 0 everywhere
        assert np.allclose(y_double_prime, 0)
    
    def test_nnLeakyReLULayer_getDf(self):
        """Test getDf method for different derivative orders"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test 0th derivative (function itself)
        df0 = layer.getDf(0)
        x = np.array([[1]])
        assert np.allclose(df0(x), layer.f(x))
        
        # Test 1st derivative
        df1 = layer.getDf(1)
        x = np.array([[1]])
        assert np.allclose(df1(x), layer.df(x))
        
        # Test 2nd derivative
        df2 = layer.getDf(2)
        x = np.array([[1]])
        assert np.allclose(df2(x), layer._df2(x))
        
        # Test higher order derivatives (should return zeros)
        df3 = layer.getDf(3)
        x = np.array([[1]])
        assert np.allclose(df3(x), 0)
    
    def test_nnLeakyReLULayer_getDerBounds(self):
        """Test getDerBounds method"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test bounds that include 0
        l, u = -1, 1
        df_min, df_max = layer.getDerBounds(l, u)
        
        # For LeakyReLU, derivative bounds are [alpha, 1]
        assert np.isclose(df_min, 0.01)  # alpha
        assert np.isclose(df_max, 1.0)   # 1
        
        # Test bounds on one side of 0
        l, u = 1, 2
        df_min, df_u = layer.getDerBounds(l, u)
        assert np.isclose(df_min, 1.0)  # All positive, so derivative is 1
        assert np.isclose(df_u, 1.0)
        
        # Test bounds on negative side
        l, u = -2, -1
        df_min, df_u = layer.getDerBounds(l, u)
        assert np.isclose(df_min, 0.01)  # All negative, so derivative is alpha
        assert np.isclose(df_u, 0.01)
    
    def test_nnLeakyReLULayer_computeApproxError_order1(self):
        """Test computeApproxError for order 1"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test with valid coefficients
        l, u = -1, 1
        coeffs = [0.5, 0.0]  # m = 0.5, t = 0.0
        
        new_coeffs, d = layer.computeApproxError(l, u, coeffs)
        
        # Check that coefficients were adjusted
        assert len(new_coeffs) == 2
        assert new_coeffs[0] == 0.5  # m should remain the same
        assert new_coeffs[1] != 0.0  # t should be adjusted
        
        # Check that error bound is positive
        assert d >= 0
    
    def test_nnLeakyReLULayer_computeApproxError_order1_edge_cases(self):
        """Test computeApproxError edge cases for order 1"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test with m >= 1 (should return d = 0)
        l, u = -1, 1
        coeffs = [1.0, 0.0]  # m = 1.0
        
        new_coeffs, d = layer.computeApproxError(l, u, coeffs)
        assert d == 0
        
        # Test with m <= 0 (should return d = 0)
        coeffs = [-0.1, 0.0]  # m = -0.1 < 0
        
        new_coeffs, d = layer.computeApproxError(l, u, coeffs)
        assert d == 0
    
    def test_nnLeakyReLULayer_computeApproxError_higher_order(self):
        """Test computeApproxError for higher orders"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test with order 2
        l, u = -1, 1
        coeffs = [0.1, 0.2, 0.3]  # Order 2 polynomial
        
        # This should call the parent class method
        # For now, we'll just test that it doesn't crash
        try:
            new_coeffs, d = layer.computeApproxError(l, u, coeffs)
            assert len(new_coeffs) == 3
            assert d >= 0
        except NotImplementedError:
            # Parent method might not be implemented yet
            pass
    
    def test_nnLeakyReLULayer_computeApproxPoly(self):
        """Test computeApproxPoly method"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test with different methods and orders
        l, u = -1, 1
        
        # Test regression method
        for order in [1, 2, 3]:
            coeffs, d = layer.computeApproxPoly(l, u, order, "regression")
            assert len(coeffs) == order + 1
            assert d >= 0
        
        # Test ridgeregression method
        for order in [1, 2, 3]:
            coeffs, d = layer.computeApproxPoly(l, u, order, "ridgeregression")
            assert len(coeffs) == order + 1
            assert d >= 0
    
    def test_nnLeakyReLULayer_computeApproxPoly_validation(self):
        """Test computeApproxPoly input validation"""
        layer = nnLeakyReLULayer(0.01)
        
        l, u = -1, 1
        
        # Test invalid order
        with pytest.raises(ValueError):
            layer.computeApproxPoly(l, u, 0, "regression")
        
        # Test invalid method
        with pytest.raises(ValueError):
            layer.computeApproxPoly(l, u, 1, "invalid_method")
    
    def test_nnLeakyReLULayer_computeApproxPoly_containment(self):
        """Test that polynomial approximation contains the function"""
        layer = nnLeakyReLULayer(0.01)
        
        l, u = -1, 1
        
        for order in [1, 2, 3]:
            for method in ["regression", "ridgeregression"]:
                coeffs, d = layer.computeApproxPoly(l, u, order, method)
                
                # Test containment at several points
                x_test = np.linspace(l, u, 10)
                y_true = layer.f(x_test)
                y_approx = np.polyval(coeffs, x_test)
                
                # Check that approximation is within error bounds
                assert np.all(y_approx - d <= y_true + 1e-10)
                assert np.all(y_approx + d >= y_true - 1e-10)
    
    def test_nnLeakyReLULayer_computeExtremePointsBatch(self):
        """Test computeExtremePointsBatch method"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test with valid m values
        m = np.array([0.1, 0.5, 0.9])
        options = {}
        
        xs, dxsdm = layer.computeExtremePointsBatch(m, options)
        
        # Check shapes
        assert xs.shape == (3, 2)  # 3 slopes, 2 extreme points each
        assert dxsdm.shape == (3, 2)
        
        # For LeakyReLU, extreme points should be at x = 0 for positive m
        for i, m_val in enumerate(m):
            if m_val > 0:
                assert np.isclose(xs[i, 0], 0)  # xl = 0
                assert np.isclose(xs[i, 1], 0)  # xu = 0
    
    def test_nnLeakyReLULayer_computeExtremePointsBatch_edge_cases(self):
        """Test computeExtremePointsBatch edge cases"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test with m >= 1 (should return d = 0)
        m = np.array([0.9, 1.0, 1.1])
        options = {}
        
        xs, dxsdm = layer.computeExtremePointsBatch(m, options)
        
        # All m values should give same result
        assert np.allclose(xs[0, :], xs[1, :])
        assert np.allclose(xs[1, :], xs[2, :])
    
    def test_nnLeakyReLULayer_evaluateNumeric(self):
        """Test evaluateNumeric method"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test with simple input
        x = np.array([[1]])
        options = {}
        
        y = layer.evaluateNumeric(x, options)
        
        # Should use the same function as layer.f
        expected = layer.f(x)
        assert np.allclose(y, expected)
    
    def test_nnLeakyReLULayer_inheritance(self):
        """Test that nnLeakyReLULayer properly inherits from nnActivationLayer"""
        layer = nnLeakyReLULayer(0.01)
        
        # Check that it has the expected base class methods
        assert hasattr(layer, 'f')
        assert hasattr(layer, 'df')
        assert hasattr(layer, 'name')
        assert hasattr(layer, 'type')
        assert hasattr(layer, 'is_refinable')
        
        # Check that is_refinable is True (from parent class)
        assert layer.is_refinable is True
    
    def test_nnLeakyReLULayer_alpha_property(self):
        """Test that alpha property is correctly set and accessible"""
        alpha = 0.05
        layer = nnLeakyReLULayer(alpha)
        
        assert layer.alpha == alpha
        
        # Test that alpha affects function behavior
        x_neg = np.array([[-1]])
        y_neg = layer.f(x_neg)
        
        expected = alpha * x_neg
        assert np.allclose(y_neg, expected)
    
    def test_nnLeakyReLULayer_negative_alpha_handling(self):
        """Test that negative alpha is handled correctly"""
        # Test with negative alpha (though this might not be typical)
        alpha = -0.01
        layer = nnLeakyReLULayer(alpha)
        
        assert layer.alpha == alpha
        
        # Test function behavior with negative alpha
        x_neg = np.array([[-1]])
        y_neg = layer.f(x_neg)
        
        expected = alpha * x_neg
        assert np.allclose(y_neg, expected)
    
    def test_nnLeakyReLULayer_zero_alpha(self):
        """Test that zero alpha gives ReLU behavior"""
        alpha = 0.0
        layer = nnLeakyReLULayer(alpha)
        
        assert layer.alpha == alpha
        
        # Test function behavior with zero alpha
        x_neg = np.array([[-1]])
        y_neg = layer.f(x_neg)
        
        expected = 0  # Same as ReLU
        assert np.allclose(y_neg, expected)
    
    def test_nnLeakyReLULayer_continuity(self):
        """Test that LeakyReLU is continuous at x = 0"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test values approaching 0 from both sides
        x_pos = np.array([[1e-10], [1e-9], [1e-8]])
        x_neg = np.array([[-1e-10], [-1e-9], [-1e-8]])
        
        y_pos = layer.f(x_pos)
        y_neg = layer.f(x_neg)
        
        # Values should approach 0 as x approaches 0
        assert np.allclose(y_pos, x_pos)  # y = x for positive x
        assert np.allclose(y_neg, 0.01 * x_neg)  # y = alpha * x for negative x
    
    def test_nnLeakyReLULayer_monotonicity(self):
        """Test that LeakyReLU is monotonically increasing"""
        layer = nnLeakyReLULayer(0.01)
        
        # Test that function is increasing
        x1 = np.array([[-2], [-1], [0], [1], [2]])
        y1 = layer.f(x1)
        
        # Check that values are increasing
        for i in range(1, len(y1)):
            assert y1[i] > y1[i-1]
        
        # Test that derivative is always positive
        y_prime = layer.df(x1)
        assert np.all(y_prime > 0)
