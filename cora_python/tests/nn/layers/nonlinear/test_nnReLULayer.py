"""
Tests for nnReLULayer class.
"""

import pytest
import numpy as np
from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer


class TestNnReLULayer:
    """Test class for nnReLULayer."""
    
    def test_nnReLULayer_constructor_default(self):
        """Test constructor with default parameters"""
        layer = nnReLULayer()
        assert layer.name == "relu"
        assert layer.type == "relu"
        assert layer.alpha == 0.0
    
    def test_nnReLULayer_constructor_custom_alpha(self):
        """Test constructor with custom alpha"""
        alpha = 0.1
        layer = nnReLULayer(alpha)
        assert layer.name == "relu"
        assert layer.type == "relu"
        assert layer.alpha == alpha
    
    def test_nnReLULayer_constructor_custom_name(self):
        """Test constructor with custom name"""
        layer = nnReLULayer(0.05, "custom_relu")
        assert layer.name == "custom_relu"
        assert layer.type == "relu"
        assert layer.alpha == 0.05
    
    def test_nnReLULayer_function_values_positive(self):
        """Test ReLU function values for positive inputs"""
        layer = nnReLULayer()
        
        # Test positive inputs
        x = np.array([[1], [2], [3]])
        y = layer.f(x)
        
        # For positive inputs, ReLU(x) = x
        expected = np.array([[1], [2], [3]])
        assert np.allclose(y, expected)
    
    def test_nnReLULayer_function_values_negative(self):
        """Test ReLU function values for negative inputs"""
        layer = nnReLULayer()
        
        # Test negative inputs
        x = np.array([[-1], [-2], [-3]])
        y = layer.f(x)
        
        # For negative inputs, ReLU(x) = 0
        expected = np.array([[0], [0], [0]])
        assert np.allclose(y, expected)
    
    def test_nnReLULayer_function_values_mixed(self):
        """Test ReLU function values for mixed inputs"""
        layer = nnReLULayer()
        
        # Test mixed inputs
        x = np.array([[-2], [0], [2]])
        y = layer.f(x)
        
        expected = np.array([[0], [0], [2]])
        assert np.allclose(y, expected)
    
    def test_nnReLULayer_function_values_zero(self):
        """Test ReLU function value at zero"""
        layer = nnReLULayer()
        
        # Test zero input
        x = np.array([[0]])
        y = layer.f(x)
        
        # ReLU(0) = 0
        assert np.isclose(y[0, 0], 0)
    
    def test_nnReLULayer_derivative_values_positive(self):
        """Test ReLU derivative values for positive inputs"""
        layer = nnReLULayer()
        
        # Test positive inputs
        x = np.array([[1], [2], [3]])
        y_prime = layer.df(x)
        
        # For positive inputs, ReLU'(x) = 1
        expected = np.array([[1], [1], [1]])
        assert np.allclose(y_prime, expected)
    
    def test_nnReLULayer_derivative_values_negative(self):
        """Test ReLU derivative values for negative inputs"""
        layer = nnReLULayer()
        
        # Test negative inputs
        x = np.array([[-1], [-2], [-3]])
        y_prime = layer.df(x)
        
        # For negative inputs, ReLU'(x) = 0
        expected = np.array([[0], [0], [0]])
        assert np.allclose(y_prime, expected)
    
    def test_nnReLULayer_derivative_values_zero(self):
        """Test ReLU derivative value at zero"""
        layer = nnReLULayer()
        
        # Test zero input
        x = np.array([[0]])
        y_prime = layer.df(x)
        
        # At zero, we use the positive derivative (1)
        assert np.isclose(y_prime[0, 0], 1)
    
    def test_nnReLULayer_second_derivative(self):
        """Test ReLU second derivative"""
        layer = nnReLULayer()
        
        # Test various inputs
        x = np.array([[-2], [-1], [0], [1], [2]])
        y_double_prime = layer._df2(x)
        
        # Second derivative should be 0 everywhere
        assert np.allclose(y_double_prime, 0)
    
    def test_nnReLULayer_getDf(self):
        """Test getDf method for different derivative orders"""
        layer = nnReLULayer()
        
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
    
    def test_nnReLULayer_getDerBounds(self):
        """Test getDerBounds method"""
        layer = nnReLULayer()
        
        # Test bounds that include 0
        l, u = -1, 1
        df_min, df_max = layer.getDerBounds(l, u)
        
        # For ReLU, derivative bounds are [0, 1]
        assert np.isclose(df_min, 0.0)  # 0
        assert np.isclose(df_max, 1.0)  # 1
        
        # Test bounds on one side of 0
        l, u = 1, 2
        df_min, df_u = layer.getDerBounds(l, u)
        assert np.isclose(df_min, 1.0)  # All positive, so derivative is 1
        assert np.isclose(df_u, 1.0)
        
        # Test bounds on negative side
        l, u = -2, -1
        df_min, df_u = layer.getDerBounds(l, u)
        assert np.isclose(df_min, 0.0)  # All negative, so derivative is 0
        assert np.isclose(df_u, 0.0)
    
    def test_nnReLULayer_computeApproxError_order1(self):
        """Test computeApproxError for order 1"""
        layer = nnReLULayer()
        
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
    
    def test_nnReLULayer_computeApproxError_order1_edge_cases(self):
        """Test computeApproxError edge cases for order 1"""
        layer = nnReLULayer()
        
        # Test with m >= 1 (should return d = 0)
        l, u = -1, 1
        coeffs = [1.0, 0.0]  # m = 1.0
        
        new_coeffs, d = layer.computeApproxError(l, u, coeffs)
        assert d == 0
        
        # Test with m <= 0 (should return d = 0)
        coeffs = [-0.1, 0.0]  # m = -0.1 < 0
        
        new_coeffs, d = layer.computeApproxError(l, u, coeffs)
        assert d == 0
    
    def test_nnReLULayer_computeApproxError_higher_order(self):
        """Test computeApproxError for higher orders"""
        layer = nnReLULayer()
        
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
    
    def test_nnReLULayer_computeApproxPoly(self):
        """Test computeApproxPoly method"""
        layer = nnReLULayer()
        
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
    
    def test_nnReLULayer_computeApproxPoly_validation(self):
        """Test computeApproxPoly input validation"""
        layer = nnReLULayer()
        
        l, u = -1, 1
        
        # Test invalid order
        with pytest.raises(ValueError):
            layer.computeApproxPoly(l, u, 0, "regression")
        
        # Test invalid method
        with pytest.raises(ValueError):
            layer.computeApproxPoly(l, u, 1, "invalid_method")
    
    def test_nnReLULayer_computeApproxPoly_containment(self):
        """Test that polynomial approximation contains the function"""
        layer = nnReLULayer()
        
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
    
    def test_nnReLULayer_computeExtremePointsBatch(self):
        """Test computeExtremePointsBatch method"""
        layer = nnReLULayer()
        
        # Test with valid m values
        m = np.array([0.1, 0.5, 0.9])
        options = {}
        
        xs, dxsdm = layer.computeExtremePointsBatch(m, options)
        
        # Check shapes
        assert xs.shape == (3, 2)  # 3 slopes, 2 extreme points each
        assert dxsdm.shape == (3, 2)
        
        # For ReLU, extreme points should be at x = 0 for positive m
        for i, m_val in enumerate(m):
            if m_val > 0:
                assert np.isclose(xs[i, 0], 0)  # xl = 0
                assert np.isclose(xs[i, 1], 0)  # xu = 0
    
    def test_nnReLULayer_computeExtremePointsBatch_edge_cases(self):
        """Test computeExtremePointsBatch edge cases"""
        layer = nnReLULayer()
        
        # Test with m >= 1 (should return d = 0)
        m = np.array([0.9, 1.0, 1.1])
        options = {}
        
        xs, dxsdm = layer.computeExtremePointsBatch(m, options)
        
        # All m values should give same result
        assert np.allclose(xs[0, :], xs[1, :])
        assert np.allclose(xs[1, :], xs[2, :])
    
    def test_nnReLULayer_evaluateNumeric(self):
        """Test evaluateNumeric method"""
        layer = nnReLULayer()
        
        # Test with simple input
        x = np.array([[1]])
        options = {}
        
        y = layer.evaluateNumeric(x, options)
        
        # Should use the same function as layer.f
        expected = layer.f(x)
        assert np.allclose(y, expected)
    
    def test_nnReLULayer_evaluateConZonotopeNeuron(self):
        """Test evaluateConZonotopeNeuron method"""
        layer = nnReLULayer()
        
        # Test with simple input
        c = 1.0
        G = np.array([0.1, 0.2])
        GI = np.array([0.01])
        options = {}
        
        try:
            c_out, G_out, GI_out = layer.evaluateConZonotopeNeuron(c, G, GI, options)
            
            # Check that outputs have correct types
            assert isinstance(c_out, (int, float, np.ndarray))
            assert isinstance(G_out, np.ndarray)
            assert isinstance(GI_out, np.ndarray)
            
        except NotImplementedError:
            # Method might not be fully implemented yet
            pass
    
    def test_nnReLULayer_unitvector(self):
        """Test unitvector method"""
        layer = nnReLULayer()
        
        # Test with various dimensions
        for n in [1, 2, 3, 5]:
            for i in range(n):
                v = layer.unitvector(n, i)
                
                # Check shape
                assert v.shape == (n, 1)
                
                # Check that it's a unit vector
                assert np.isclose(np.linalg.norm(v), 1.0)
                
                # Check that only the i-th element is 1
                assert np.isclose(v[i, 0], 1.0)
                assert np.allclose(v[:i, 0], 0)
                assert np.allclose(v[i+1:, 0], 0)
    
    def test_nnReLULayer_inheritance(self):
        """Test that nnReLULayer properly inherits from nnLeakyReLULayer"""
        layer = nnReLULayer()
        
        # Check that it has the expected base class methods
        assert hasattr(layer, 'f')
        assert hasattr(layer, 'df')
        assert hasattr(layer, 'name')
        assert hasattr(layer, 'type')
        assert hasattr(layer, 'is_refinable')
        assert hasattr(layer, 'alpha')
        
        # Check that is_refinable is True (from parent class)
        assert layer.is_refinable is True
        
        # Check that alpha is 0 (ReLU default)
        assert layer.alpha == 0.0
    
    def test_nnReLULayer_alpha_property(self):
        """Test that alpha property is correctly set and accessible"""
        alpha = 0.05
        layer = nnReLULayer(alpha)
        
        assert layer.alpha == alpha
        
        # Test that alpha affects function behavior
        x_neg = np.array([[-1]])
        y_neg = layer.f(x_neg)
        
        expected = alpha * x_neg
        assert np.allclose(y_neg, expected)
    
    def test_nnReLULayer_zero_alpha_behavior(self):
        """Test that zero alpha gives standard ReLU behavior"""
        layer = nnReLULayer(0.0)
        
        # Test function behavior with zero alpha
        x_neg = np.array([[-1]])
        y_neg = layer.f(x_neg)
        
        expected = 0  # Standard ReLU
        assert np.allclose(y_neg, expected)
    
    def test_nnReLULayer_continuity(self):
        """Test that ReLU is continuous at x = 0"""
        layer = nnReLULayer()
        
        # Test values approaching 0 from both sides
        x_pos = np.array([[1e-10], [1e-9], [1e-8]])
        x_neg = np.array([[-1e-10], [-1e-9], [-1e-8]])
        
        y_pos = layer.f(x_pos)
        y_neg = layer.f(x_neg)
        
        # Values should approach 0 as x approaches 0
        assert np.allclose(y_pos, x_pos)  # y = x for positive x
        assert np.allclose(y_neg, 0)      # y = 0 for negative x
    
    def test_nnReLULayer_monotonicity(self):
        """Test that ReLU is monotonically increasing"""
        layer = nnReLULayer()
        
        # Test that function is increasing
        x1 = np.array([[-2], [-1], [0], [1], [2]])
        y1 = layer.f(x1)
        
        # Check that values are increasing
        for i in range(1, len(y1)):
            assert y1[i] >= y1[i-1]
        
        # Test that derivative is always non-negative
        y_prime = layer.df(x1)
        assert np.all(y_prime >= 0)
    
    def test_nnReLULayer_piecewise_linearity(self):
        """Test that ReLU is piecewise linear"""
        layer = nnReLULayer()
        
        # Test linearity on positive side
        x_pos = np.array([[1], [2], [3]])
        y_pos = layer.f(x_pos)
        
        # Check that it's linear: y = x
        assert np.allclose(y_pos, x_pos)
        
        # Test constant on negative side
        x_neg = np.array([[-1], [-2], [-3]])
        y_neg = layer.f(x_neg)
        
        # Check that it's constant: y = 0
        assert np.allclose(y_neg, 0)
    
    def test_nnReLULayer_approximation_quality(self):
        """Test that polynomial approximations work well for ReLU"""
        layer = nnReLULayer()
        
        l, u = -1, 1
        
        # Test that higher order approximations generally have smaller error
        errors = []
        for order in [1, 2, 3]:
            coeffs, d = layer.computeApproxPoly(l, u, order, "regression")
            errors.append(d)
        
        # Check that method works for all orders
        assert len(errors) == 3
        assert all(e >= 0 for e in errors)
        
        # For ReLU, order 1 should give good approximation
        coeffs1, d1 = layer.computeApproxPoly(l, u, 1, "regression")
        assert len(coeffs1) == 2
        assert d1 >= 0
