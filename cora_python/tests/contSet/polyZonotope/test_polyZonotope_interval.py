"""
test_polyZonotope_interval - unit test function for interval conversion methods

This is a Python translation of the MATLAB CORA test implementation.

Authors: MATLAB: Niklas Kochdumper, Tobias Ladner
         Python: AI Assistant
"""

import pytest
import numpy as np
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.zonotope.zonotope import Zonotope


class TestPolyZonotopeInterval:
    """Test class for PolyZonotope interval conversion methods"""
    
    def test_interval_basic_interval_method(self):
        """Test basic interval method"""
        # Create polynomial zonotope
        c = np.array([3, 4])
        G = np.array([[2, 0, 1], [0, 2, 1]])
        E = np.array([[1, 0, 1], [0, 1, 1]])
        GI = np.array([[0], [0]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Calculate enclosing interval
        I = pZ.interval('interval')
        
        # Should be equivalent to zonotope interval
        z = Zonotope(c, np.hstack([G, GI]))
        I_ref = z.interval()
        
        # Check that intervals are approximately equal
        assert np.allclose(I.inf, I_ref.inf, atol=1e-10)
        assert np.allclose(I.sup, I_ref.sup, atol=1e-10)
    
    def test_interval_bernstein_method(self):
        """Test Bernstein polynomial method"""
        # Create polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[2, -1, 2], [0, -2, -3]])
        E = np.array([[1, 0, 1], [0, 1, 3]])
        GI = np.array([[0], [0]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Calculate interval using Bernstein method
        I = pZ.interval('bernstein')
        
        # Should return a valid interval
        assert isinstance(I, Interval)
        assert I.inf.shape == (2,)
        assert I.sup.shape == (2,)
        assert np.all(I.inf <= I.sup)
    
    def test_interval_split_method(self):
        """Test split method"""
        # Create polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([[0.5], [0.1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Calculate interval using split method
        I = pZ.interval('split', splits=4)
        
        # Should return a valid interval
        assert isinstance(I, Interval)
        assert I.inf.shape == (2,)
        assert I.sup.shape == (2,)
        assert np.all(I.inf <= I.sup)
    
    def test_interval_bnb_method(self):
        """Test branch and bound method"""
        # Create polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([[0.5], [0.1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Calculate interval using bnb method
        I = pZ.interval('bnb', splits=3)
        
        # Should return a valid interval
        assert isinstance(I, Interval)
        assert I.inf.shape == (2,)
        assert I.sup.shape == (2,)
        assert np.all(I.inf <= I.sup)
    
    def test_interval_bnbAdv_method(self):
        """Test advanced branch and bound method"""
        # Create polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([[0.5], [0.1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Calculate interval using bnbAdv method
        I = pZ.interval('bnbAdv', splits=3)
        
        # Should return a valid interval
        assert isinstance(I, Interval)
        assert I.inf.shape == (2,)
        assert I.sup.shape == (2,)
        assert np.all(I.inf <= I.sup)
    
    def test_interval_globOpt_method(self):
        """Test global optimization method"""
        # Create polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([[0.5], [0.1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Calculate interval using globOpt method
        I = pZ.interval('globOpt', splits=3)
        
        # Should return a valid interval
        assert isinstance(I, Interval)
        assert I.inf.shape == (2,)
        assert I.sup.shape == (2,)
        assert np.all(I.inf <= I.sup)
    
    def test_interval_empty_polyZonotope(self):
        """Test interval with empty polynomial zonotope"""
        pZ = PolyZonotope.empty(2)
        
        # All methods should handle empty sets
        methods = ['interval', 'bernstein', 'split', 'bnb', 'bnbAdv', 'globOpt']
        for method in methods:
            I = pZ.interval(method)
            assert isinstance(I, Interval)
            assert I.inf.shape == (2,)
            assert I.sup.shape == (2,)
    
    def test_interval_single_dimension(self):
        """Test interval with single dimension polynomial zonotope"""
        c = np.array([1])
        G = np.array([[2, 1]])
        E = np.array([[1, 2]])
        GI = np.array([[0.5]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Test all methods
        methods = ['interval', 'bernstein', 'split', 'bnb', 'bnbAdv', 'globOpt']
        for method in methods:
            I = pZ.interval(method)
            assert isinstance(I, Interval)
            assert I.inf.shape == (1,)
            assert I.sup.shape == (1,)
            assert I.inf[0] <= I.sup[0]
    
    def test_interval_high_dimension(self):
        """Test interval with high dimension polynomial zonotope"""
        n = 5
        c = np.random.rand(n)
        G = np.random.rand(n, 3)
        E = np.random.randint(0, 3, (2, 3))
        GI = np.random.rand(n, 2)
        pZ = PolyZonotope(c, G, GI, E)
        
        # Test interval method (most reliable)
        I = pZ.interval('interval')
        assert isinstance(I, Interval)
        assert I.inf.shape == (n,)
        assert I.sup.shape == (n,)
        assert np.all(I.inf <= I.sup)
    
    def test_interval_method_comparison(self):
        """Test that different methods give reasonable results"""
        # Create polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([[0.5], [0.1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Get intervals from different methods
        I_interval = pZ.interval('interval')
        I_bernstein = pZ.interval('bernstein')
        
        # Both should be valid intervals
        assert isinstance(I_interval, Interval)
        assert isinstance(I_bernstein, Interval)
        
        # Both should have same dimensions
        assert I_interval.inf.shape == I_bernstein.inf.shape
        assert I_interval.sup.shape == I_bernstein.sup.shape
        
        # Both should be bounded
        assert np.all(np.isfinite(I_interval.inf))
        assert np.all(np.isfinite(I_interval.sup))
        assert np.all(np.isfinite(I_bernstein.inf))
        assert np.all(np.isfinite(I_bernstein.sup))
    
    def test_interval_invalid_method(self):
        """Test interval with invalid method"""
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([[0.5], [0.1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Should raise ValueError for invalid method
        with pytest.raises(ValueError):
            pZ.interval('invalid_method')
    
    def test_interval_invalid_input(self):
        """Test interval with invalid input"""
        # Should raise ValueError for non-polyZonotope input
        with pytest.raises(ValueError):
            from .interval import interval
            interval("not_a_polyZonotope")
    
    def test_interval_no_dependent_generators(self):
        """Test interval with no dependent generators"""
        c = np.array([1, 2])
        G = np.array([]).reshape(2, 0)  # Empty dependent generators
        E = np.array([]).reshape(0, 0)  # Empty exponent matrix
        GI = np.array([[0.5, 0.1], [0.2, 0.3]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Should work with interval method
        I = pZ.interval('interval')
        assert isinstance(I, Interval)
        assert I.inf.shape == (2,)
        assert I.sup.shape == (2,)
        assert np.all(I.inf <= I.sup)
    
    def test_interval_no_independent_generators(self):
        """Test interval with no independent generators"""
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([]).reshape(2, 0)  # Empty independent generators
        pZ = PolyZonotope(c, G, GI, E)
        
        # Should work with all methods
        methods = ['interval', 'bernstein', 'split', 'bnb', 'bnbAdv', 'globOpt']
        for method in methods:
            I = pZ.interval(method)
            assert isinstance(I, Interval)
            assert I.inf.shape == (2,)
            assert I.sup.shape == (2,)
            assert np.all(I.inf <= I.sup)
    
    def test_interval_splits_parameter(self):
        """Test interval with different splits parameter"""
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([[0.5], [0.1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Test with different number of splits
        for splits in [2, 4, 8, 16]:
            I = pZ.interval('split', splits=splits)
            assert isinstance(I, Interval)
            assert I.inf.shape == (2,)
            assert I.sup.shape == (2,)
            assert np.all(I.inf <= I.sup)
