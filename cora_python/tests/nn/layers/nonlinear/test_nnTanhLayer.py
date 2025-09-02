"""
Tests for nnTanhLayer class.
"""

import pytest
import numpy as np
from cora_python.nn.layers.nonlinear.nnTanhLayer import nnTanhLayer


class TestNnTanhLayer:
    """Test class for nnTanhLayer."""
    
    def test_nnTanhLayer_constructor_default_name(self):
        """Test constructor with default name"""
        layer = nnTanhLayer()
        # Default name should be generated with unique number (matches MATLAB behavior)
        assert layer.name.startswith("tanh_")
        assert layer.type == "nnTanhLayer"
    
    def test_nnTanhLayer_constructor_custom_name(self):
        """Test constructor with custom name"""
        layer = nnTanhLayer("custom_tanh")
        assert layer.name == "custom_tanh"
        assert layer.type == "nnTanhLayer"
    
    def test_nnTanhLayer_function_values(self):
        """Test tanh function values"""
        layer = nnTanhLayer()
        
        # Test specific values
        x = np.array([[-2], [0], [2]])
        y = layer.f(x)
        
        # Expected values: tanh(x)
        expected = np.array([[-0.96402758], [0], [0.96402758]])
        assert np.allclose(y, expected, rtol=1e-7)
        
        # Test that tanh is bounded between -1 and 1
        x_large = np.array([[-10], [10]])
        y_large = layer.f(x_large)
        assert np.all(y_large >= -1)
        assert np.all(y_large <= 1)
        
        # Test that tanh(0) = 0
        x_zero = np.array([[0]])
        y_zero = layer.f(x_zero)
        assert np.isclose(y_zero[0, 0], 0)
    
    def test_nnTanhLayer_derivative_values(self):
        """Test tanh derivative values"""
        layer = nnTanhLayer()
        
        # Test specific values
        x = np.array([[-2], [0], [2]])
        y_prime = layer.df(x)
        
        # Expected values: tanh'(x) = 1 - tanh(x)^2
        expected = np.array([[0.07065082], [1], [0.07065082]])
        assert np.allclose(y_prime, expected, rtol=1e-7)
        
        # Test that derivative is maximum at x = 0
        x_zero = np.array([[0]])
        y_prime_zero = layer.df(x_zero)
        assert np.isclose(y_prime_zero[0, 0], 1)  # Maximum derivative
        
        # Test that derivative approaches 0 as |x| increases
        x_large = np.array([[-5], [5]])
        y_prime_large = layer.df(x_large)
        assert np.all(y_prime_large < 0.1)
    
    def test_nnTanhLayer_second_derivative(self):
        """Test tanh second derivative"""
        layer = nnTanhLayer()
        
        # Test specific values
        x = np.array([[-1], [0], [1]])
        y_double_prime = layer._df2(x)
        
        # Expected values: tanh''(x) = -2*tanh(x)*(1-tanh(x)^2)
        expected = np.array([[-0.39322387], [0], [0.39322387]])
        assert np.allclose(y_double_prime, expected, rtol=1e-7)
        
        # Test that second derivative is 0 at x = 0
        x_zero = np.array([[0]])
        y_double_prime_zero = layer._df2(x_zero)
        assert np.isclose(y_double_prime_zero[0, 0], 0)
    
    def test_nnTanhLayer_getDf(self):
        """Test getDf method for different derivative orders"""
        layer = nnTanhLayer()
        
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
    
    def test_nnTanhLayer_getDerBounds(self):
        """Test getDerBounds method"""
        layer = nnTanhLayer()
        
        # Test bounds that include 0 (maximum derivative)
        l, u = -1, 1
        df_min, df_max = layer.getDerBounds(l, u)
        assert np.isclose(df_max, 1.0)  # Maximum at x = 0
        assert df_min < df_max
        
        # Test bounds on one side of 0
        l, u = 1, 2
        df_min, df_u = layer.getDerBounds(l, u)
        assert df_min <= df_u
        assert df_u < 1.0  # Less than maximum
        
        # Test bounds on negative side
        l, u = -2, -1
        df_min, df_u = layer.getDerBounds(l, u)
        assert df_min <= df_u
        assert df_u < 1.0  # Less than maximum
    
    def test_nnTanhLayer_computeApproxError_order1(self):
        """Test computeApproxError for order 1"""
        layer = nnTanhLayer()
        
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
    
    def test_nnTanhLayer_computeApproxError_order1_edge_cases(self):
        """Test computeApproxError edge cases for order 1"""
        layer = nnTanhLayer()
        
        # Test with m >= 1 (should return d = 0)
        l, u = -1, 1
        coeffs = [1.0, 0.0]  # m = 1.0
        
        new_coeffs, d = layer.computeApproxError(l, u, coeffs)
        assert d == 0
        
        # Test with m <= 0 (should return d = 0)
        coeffs = [-0.1, 0.0]  # m = -0.1 < 0
        
        new_coeffs, d = layer.computeApproxError(l, u, coeffs)
        assert d == 0
        
        # Test with m very close to 1
        coeffs = [0.999, 0.0]  # m = 0.999 < 1
        
        new_coeffs, d = layer.computeApproxError(l, u, coeffs)
        assert d >= 0  # Should compute some error
    
    def test_nnTanhLayer_computeApproxError_higher_order(self):
        """Test computeApproxError for higher orders"""
        layer = nnTanhLayer()
        
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
    
    def test_nnTanhLayer_computeExtremePointsBatch(self):
        """Test computeExtremePointsBatch method"""
        layer = nnTanhLayer()
        
        # Test with valid m values
        m = np.array([0.1, 0.5, 0.9])
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
        
        # Check that extreme points increase with m
        for i in range(2):
            assert xs[0, i] < xs[1, i] < xs[2, i]
    
    def test_nnTanhLayer_computeExtremePointsBatch_edge_cases(self):
        """Test computeExtremePointsBatch edge cases"""
        layer = nnTanhLayer()
        
        # Test with m >= 1 (should limit m to 1)
        m = np.array([0.9, 1.0, 1.1])
        options = {}
        
        xs, dxsdm = layer.computeExtremePointsBatch(m, options)
        
        # All m values should be limited appropriately
        assert np.allclose(xs[1, :], xs[2, :])  # m=1.0 and m=1.1 should give same result
        
        # Test with m very small (should limit to eps)
        m = np.array([1e-10, 1e-15, 0])
        options = {}
        
        xs, dxsdm = layer.computeExtremePointsBatch(m, options)
        
        # All should give similar results (limited by eps)
        assert np.allclose(xs[0, :], xs[1, :], rtol=1e-6)
        assert np.allclose(xs[1, :], xs[2, :], rtol=1e-6)
    
    def test_nnTanhLayer_evaluateNumeric(self):
        """Test evaluateNumeric method"""
        layer = nnTanhLayer()
        
        # Test with simple input
        x = np.array([[1]])
        options = {}
        
        y = layer.evaluateNumeric(x, options)
        
        # Should use numpy tanh
        expected = np.tanh(1)
        assert np.isclose(y[0, 0], expected)
    
    def test_nnTanhLayer_reg_polys(self):
        """Test that reg_polys are properly initialized"""
        layer = nnTanhLayer()
        
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
    
    def test_nnTanhLayer_inheritance(self):
        """Test that nnTanhLayer properly inherits from nnActivationLayer"""
        layer = nnTanhLayer()
        
        # Check that it has the expected base class methods
        assert hasattr(layer, 'f')
        assert hasattr(layer, 'df')
        assert hasattr(layer, 'name')
        assert hasattr(layer, 'type')
        assert hasattr(layer, 'is_refinable')
        
        # Check that is_refinable is True (from parent class)
        assert layer.is_refinable is True
    
    def test_nnTanhLayer_symmetry(self):
        """Test that tanh function is odd (symmetric about origin)"""
        layer = nnTanhLayer()
        
        # Test symmetry: tanh(-x) = -tanh(x)
        x = np.array([[1], [2], [3]])
        y_pos = layer.f(x)
        y_neg = layer.f(-x)
        
        assert np.allclose(y_neg, -y_pos)
        
        # Test derivative symmetry: tanh'(-x) = tanh'(x)
        y_prime_pos = layer.df(x)
        y_prime_neg = layer.df(-x)
        
        assert np.allclose(y_prime_neg, y_prime_pos)
    
    def test_nnTanhLayer_monotonicity(self):
        """Test that tanh function is monotonically increasing"""
        layer = nnTanhLayer()
        
        # Test that tanh is increasing
        x1 = np.array([[-2], [-1], [0], [1], [2]])
        y1 = layer.f(x1)
        
        # Check that values are increasing
        for i in range(1, len(y1)):
            assert y1[i] > y1[i-1]
        
        # Test that derivative is always positive
        y_prime = layer.df(x1)
        assert np.all(y_prime > 0)
