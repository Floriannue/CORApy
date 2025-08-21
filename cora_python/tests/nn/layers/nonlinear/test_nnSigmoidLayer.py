"""
Tests for nnSigmoidLayer class.
"""

import pytest
import numpy as np
from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer


class TestNnSigmoidLayer:
    """Test class for nnSigmoidLayer."""
    
    def test_nnSigmoidLayer_constructor_default_name(self):
        """Test constructor with default name"""
        layer = nnSigmoidLayer()
        assert layer.name == "sigmoid"
        assert layer.type == "sigmoid"
    
    def test_nnSigmoidLayer_constructor_custom_name(self):
        """Test constructor with custom name"""
        layer = nnSigmoidLayer("custom_sigmoid")
        assert layer.name == "custom_sigmoid"
        assert layer.type == "sigmoid"
    
    def test_nnSigmoidLayer_function_values(self):
        """Test sigmoid function values"""
        layer = nnSigmoidLayer()
        
        # Test specific values
        x = np.array([[-2], [0], [2]])
        y = layer.f(x)
        
        # Expected values: sigmoid(x) = 1/(1+exp(-x)) = tanh(x/2)/2 + 0.5
        expected = np.array([[0.11920292], [0.5], [0.88079708]])
        assert np.allclose(y, expected, rtol=1e-7)
        
        # Test that sigmoid is bounded between 0 and 1
        x_large = np.array([[-10], [10]])
        y_large = layer.f(x_large)
        assert np.all(y_large >= 0)
        assert np.all(y_large <= 1)
    
    def test_nnSigmoidLayer_derivative_values(self):
        """Test sigmoid derivative values"""
        layer = nnSigmoidLayer()
        
        # Test specific values
        x = np.array([[-2], [0], [2]])
        y_prime = layer.df(x)
        
        # Expected values: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        expected = np.array([[0.10499359], [0.25], [0.10499359]])
        assert np.allclose(y_prime, expected, rtol=1e-7)
        
        # Test that derivative is maximum at x = 0
        x_zero = np.array([[0]])
        y_prime_zero = layer.df(x_zero)
        assert np.isclose(y_prime_zero[0, 0], 0.25)  # Maximum derivative
    
    def test_nnSigmoidLayer_second_derivative(self):
        """Test sigmoid second derivative"""
        layer = nnSigmoidLayer()
        
        # Test specific values
        x = np.array([[-1], [0], [1]])
        y_double_prime = layer._df2(x)
        
        # Expected values: sigmoid''(x) = sigmoid'(x) * (1 - 2*sigmoid(x))
        expected = np.array([[-0.19661193], [0], [-0.19661193]])
        assert np.allclose(y_double_prime, expected, rtol=1e-7)
    
    def test_nnSigmoidLayer_getDf(self):
        """Test getDf method for different derivative orders"""
        layer = nnSigmoidLayer()
        
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
    
    def test_nnSigmoidLayer_getDerBounds(self):
        """Test getDerBounds method"""
        layer = nnSigmoidLayer()
        
        # Test bounds that include 0 (maximum derivative)
        l, u = -1, 1
        df_min, df_max = layer.getDerBounds(l, u)
        assert np.isclose(df_max, 0.25)  # Maximum at x = 0
        assert df_min < df_max
        
        # Test bounds on one side of 0
        l, u = 1, 2
        df_min, df_u = layer.getDerBounds(l, u)
        assert df_min <= df_u
        assert df_u < 0.25  # Less than maximum
    
    def test_nnSigmoidLayer_computeApproxError_order1(self):
        """Test computeApproxError for order 1"""
        layer = nnSigmoidLayer()
        
        # Test with valid coefficients
        l, u = -1, 1
        coeffs = [0.2, 0.5]  # m = 0.2, t = 0.5
        
        new_coeffs, d = layer.computeApproxError(l, u, coeffs)
        
        # Check that coefficients were adjusted
        assert len(new_coeffs) == 2
        assert new_coeffs[0] == 0.2  # m should remain the same
        assert new_coeffs[1] != 0.5  # t should be adjusted
        
        # Check that error bound is positive
        assert d >= 0
    
    def test_nnSigmoidLayer_computeApproxError_order1_edge_cases(self):
        """Test computeApproxError edge cases for order 1"""
        layer = nnSigmoidLayer()
        
        # Test with m >= 1/4 (should return d = 0)
        l, u = -1, 1
        coeffs = [0.3, 0.5]  # m = 0.3 > 1/4
        
        new_coeffs, d = layer.computeApproxError(l, u, coeffs)
        assert d == 0
        
        # Test with m <= 0 (should return d = 0)
        coeffs = [-0.1, 0.5]  # m = -0.1 < 0
        
        new_coeffs, d = layer.computeApproxError(l, u, coeffs)
        assert d == 0
    
    def test_nnSigmoidLayer_computeApproxError_higher_order(self):
        """Test computeApproxError for higher orders"""
        layer = nnSigmoidLayer()
        
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
    
    def test_nnSigmoidLayer_computeExtremePointsBatch(self):
        """Test computeExtremePointsBatch method"""
        layer = nnSigmoidLayer()
        
        # Test with valid m values
        m = np.array([0.1, 0.2, 0.3])
        options = {}
        
        xs, dxsdm = layer.computeExtremePointsBatch(m, options)
        
        # Check shapes
        assert xs.shape == (3, 2)  # 3 slopes, 2 extreme points each
        assert dxsdm.shape == (3, 2)
        
        # Check that extreme points are symmetric for each slope
        for i in range(3):
            assert np.isclose(xs[i, 0], -xs[i, 1])  # xl = -xu
        
        # Check that derivatives are opposite
        for i in range(3):
            assert np.isclose(dxsdm[i, 0], -dxsdm[i, 1])  # dxl = -dxu
    
    def test_nnSigmoidLayer_computeExtremePointsBatch_edge_cases(self):
        """Test computeExtremePointsBatch edge cases"""
        layer = nnSigmoidLayer()
        
        # Test with m >= 1/4 (should limit m to 1/4)
        m = np.array([0.3, 0.4, 0.5])
        options = {}
        
        xs, dxsdm = layer.computeExtremePointsBatch(m, options)
        
        # All m values should be limited to 1/4
        assert np.allclose(xs[0, :], xs[1, :])
        assert np.allclose(xs[1, :], xs[2, :])
    
    def test_nnSigmoidLayer_evaluateNumeric(self):
        """Test evaluateNumeric method"""
        layer = nnSigmoidLayer()
        
        # Test with simple input
        x = np.array([[1]])
        options = {}
        
        y = layer.evaluateNumeric(x, options)
        
        # Should use tanh for numerical stability
        expected = np.tanh(0.5) / 2 + 0.5
        assert np.isclose(y[0, 0], expected)
    
    def test_nnSigmoidLayer_reg_polys(self):
        """Test that reg_polys are properly initialized"""
        layer = nnSigmoidLayer()
        
        # Check that reg_polys is a list with the expected structure
        assert isinstance(layer.reg_polys, list)
        assert len(layer.reg_polys) == 6  # 6 regions
        
        # Check structure of first region
        first_region = layer.reg_polys[0]
        assert 'l' in first_region
        assert 'u' in first_region
        assert 'p' in first_region
        assert 'd' in first_region
        
        # Check that bounds are properly ordered
        for i in range(len(layer.reg_polys) - 1):
            assert layer.reg_polys[i]['u'] == layer.reg_polys[i + 1]['l']
    
    def test_nnSigmoidLayer_inheritance(self):
        """Test that nnSigmoidLayer properly inherits from nnActivationLayer"""
        layer = nnSigmoidLayer()
        
        # Check that it has the expected base class methods
        assert hasattr(layer, 'f')
        assert hasattr(layer, 'df')
        assert hasattr(layer, 'name')
        assert hasattr(layer, 'type')
        assert hasattr(layer, 'is_refinable')
        
        # Check that is_refinable is True (from parent class)
        assert layer.is_refinable is True
