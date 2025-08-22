"""
Test for nnHelper.lookupDf function

This test verifies that the lookupDf function works correctly for caching derivatives of layers.
"""

import pytest
import numpy as np
import sympy as sp
from cora_python.nn.nnHelper.lookupDf import lookupDf, clear_lookup_table, get_lookup_table_info


class MockLayer:
    """Mock layer class for testing"""
    
    def __init__(self, name, f=None, has_symbolic=True):
        self.name = name
        self.f = f if f is not None else lambda x: x**2
        self.has_symbolic = has_symbolic
    
    def getSymbolicFunction(self, x):
        """Get symbolic function for the layer"""
        if self.has_symbolic:
            return x**2
        else:
            raise AttributeError("No symbolic function available")
    
    def __class__(self):
        """Return class name for lookup table key"""
        return type('MockLayer', (), {'__name__': self.name})()


class TestLookupDf:
    """Test class for lookupDf function"""
    
    def setup_method(self):
        """Clear lookup table before each test"""
        clear_lookup_table()
    
    def test_lookupDf_basic(self):
        """Test basic lookupDf functionality"""
        # Create a mock layer
        layer = MockLayer("TestLayer")
        
        # Get first derivative
        df1 = lookupDf(layer, 1)
        
        # Check that it's a callable function
        assert callable(df1)
        
        # Test the derivative function
        x_test = 2.0
        result = df1(x_test)
        
        # For f(x) = x^2, df/dx = 2x, so df(2) = 4
        expected = 2 * x_test
        assert np.isclose(result, expected, atol=1e-10)
    
    def test_lookupDf_zero_derivative(self):
        """Test lookupDf with i=0 (original function)"""
        layer = MockLayer("TestLayer")
        
        # Get original function
        f0 = lookupDf(layer, 0)
        
        # Check that it's the original function
        assert callable(f0)
        
        # Test the function
        x_test = 3.0
        result = f0(x_test)
        expected = x_test**2
        assert np.isclose(result, expected, atol=1e-10)
    
    def test_lookupDf_multiple_derivatives(self):
        """Test lookupDf with multiple derivatives"""
        layer = MockLayer("TestLayer")
        
        # Get multiple derivatives
        df1 = lookupDf(layer, 1)
        df2 = lookupDf(layer, 2)
        
        # Check that both are callable
        assert callable(df1)
        assert callable(df2)
        
        # Test the derivatives
        x_test = 2.0
        
        # df/dx = 2x, d²f/dx² = 2
        result1 = df1(x_test)
        result2 = df2(x_test)
        
        expected1 = 2 * x_test
        expected2 = 2
        
        assert np.isclose(result1, expected1, atol=1e-10)
        assert np.isclose(result2, expected2, atol=1e-10)
    
    def test_lookupDf_caching(self):
        """Test that lookupDf caches results"""
        layer = MockLayer("TestLayer")
        
        # Get derivative twice
        df1_first = lookupDf(layer, 1)
        df1_second = lookupDf(layer, 1)
        
        # Should be the same function object (cached)
        assert df1_first is df1_second
        
        # Check lookup table info
        info = get_lookup_table_info()
        assert "TestLayer" in info
        assert info["TestLayer"]["known_derivatives"] == [1]
    
    def test_lookupDf_different_layers(self):
        """Test lookupDf with different layer types"""
        layer1 = MockLayer("Layer1")
        layer2 = MockLayer("Layer2")
        
        # Get derivatives for both layers
        df1_layer1 = lookupDf(layer1, 1)
        df1_layer2 = lookupDf(layer2, 1)
        
        # Should be different functions
        assert df1_layer1 is not df1_layer2
        
        # Check lookup table info
        info = get_lookup_table_info()
        assert "Layer1" in info
        assert "Layer2" in info
        assert len(info) == 2
    
    def test_lookupDf_no_symbolic_function(self):
        """Test lookupDf with layer that has no symbolic function"""
        # Create layer without symbolic function
        layer = MockLayer("NoSymbolicLayer", has_symbolic=False)
        
        # Should fall back to numerical differentiation
        df1 = lookupDf(layer, 1)
        
        # Check that it's a callable function
        assert callable(df1)
        
        # Test the numerical derivative
        x_test = 2.0
        result = df1(x_test)
        
        # Should be approximately correct
        expected = 2 * x_test  # For f(x) = x^2
        assert np.isclose(result, expected, atol=1e-6)
    
    def test_lookupDf_higher_order_derivatives(self):
        """Test lookupDf with higher order derivatives"""
        layer = MockLayer("TestLayer")
        
        # Get higher order derivatives
        df3 = lookupDf(layer, 3)
        df4 = lookupDf(layer, 4)
        
        # Check that they are callable
        assert callable(df3)
        assert callable(df4)
        
        # Test the derivatives
        x_test = 2.0
        
        # For f(x) = x^2:
        # df/dx = 2x, d²f/dx² = 2, d³f/dx³ = 0, d⁴f/dx⁴ = 0
        result3 = df3(x_test)
        result4 = df4(x_test)
        
        expected3 = 0
        expected4 = 0
        
        assert np.isclose(result3, expected3, atol=1e-10)
        assert np.isclose(result4, expected4, atol=1e-10)
    
    def test_lookupDf_custom_function(self):
        """Test lookupDf with custom function"""
        # Create layer with custom function
        def custom_f(x):
            return x**3 + 2*x
        
        layer = MockLayer("CustomLayer", f=custom_f)
        
        # Get derivatives
        df1 = lookupDf(layer, 1)
        df2 = lookupDf(layer, 2)
        
        # Test the derivatives
        x_test = 2.0
        
        # For f(x) = x³ + 2x:
        # df/dx = 3x² + 2, d²f/dx² = 6x
        result1 = df1(x_test)
        result2 = df2(x_test)
        
        expected1 = 3 * x_test**2 + 2
        expected2 = 6 * x_test
        
        assert np.isclose(result1, expected1, atol=1e-10)
        assert np.isclose(result2, expected2, atol=1e-10)
    
    def test_lookupDf_clear_table(self):
        """Test clear_lookup_table function"""
        layer = MockLayer("TestLayer")
        
        # Add some entries to lookup table
        lookupDf(layer, 1)
        lookupDf(layer, 2)
        
        # Check that table has entries
        info_before = get_lookup_table_info()
        assert len(info_before) > 0
        
        # Clear table
        clear_lookup_table()
        
        # Check that table is empty
        info_after = get_lookup_table_info()
        assert len(info_after) == 0
    
    def test_lookupDf_get_table_info(self):
        """Test get_lookup_table_info function"""
        layer1 = MockLayer("Layer1")
        layer2 = MockLayer("Layer2")
        
        # Add entries to lookup table
        lookupDf(layer1, 1)
        lookupDf(layer1, 2)
        lookupDf(layer2, 1)
        
        # Get table info
        info = get_lookup_table_info()
        
        # Check structure
        assert "Layer1" in info
        assert "Layer2" in info
        
        # Check Layer1 info
        layer1_info = info["Layer1"]
        assert "known_derivatives" in layer1_info
        assert "max_derivative" in layer1_info
        assert layer1_info["known_derivatives"] == [1, 2]
        assert layer1_info["max_derivative"] == 2
        
        # Check Layer2 info
        layer2_info = info["Layer2"]
        assert layer2_info["known_derivatives"] == [1]
        assert layer2_info["max_derivative"] == 1
    
    def test_lookupDf_edge_cases(self):
        """Test lookupDf edge cases"""
        # Test with very high order derivative
        layer = MockLayer("TestLayer")
        
        # Get a high order derivative
        df10 = lookupDf(layer, 10)
        
        # Should be callable
        assert callable(df10)
        
        # For polynomial f(x) = x², all derivatives beyond order 2 should be 0
        x_test = 5.0
        result = df10(x_test)
        expected = 0
        
        assert np.isclose(result, expected, atol=1e-10)
    
    def test_lookupDf_numerical_fallback(self):
        """Test numerical differentiation fallback"""
        # Create layer with complex function that might not have symbolic derivative
        def complex_f(x):
            return np.sin(x) + np.exp(x)
        
        layer = MockLayer("ComplexLayer", f=complex_f, has_symbolic=False)
        
        # Should use numerical differentiation
        df1 = lookupDf(layer, 1)
        
        # Check that it's callable
        assert callable(df1)
        
        # Test numerical derivative
        x_test = 1.0
        result = df1(x_test)
        
        # Should be approximately correct
        # df/dx = cos(x) + exp(x)
        expected = np.cos(x_test) + np.exp(x_test)
        assert np.isclose(result, expected, atol=1e-6)
    
    def test_lookupDf_reuse_cached_derivatives(self):
        """Test that cached derivatives are reused"""
        layer = MockLayer("TestLayer")
        
        # Get derivative 1
        df1_first = lookupDf(layer, 1)
        
        # Get derivative 2 (should use cached df1)
        df2 = lookupDf(layer, 2)
        
        # Get derivative 1 again (should be cached)
        df1_second = lookupDf(layer, 1)
        
        # Should be the same objects
        assert df1_first is df1_second
        
        # Check lookup table info
        info = get_lookup_table_info()
        assert "TestLayer" in info
        assert info["TestLayer"]["known_derivatives"] == [1, 2]
