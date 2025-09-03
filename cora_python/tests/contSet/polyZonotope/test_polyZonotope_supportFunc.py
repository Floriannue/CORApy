"""
test_polyZonotope_supportFunc - unit test function for supportFunc_ method

This is a Python translation of the MATLAB CORA test implementation.

Authors: MATLAB: Niklas Kochdumper, Tobias Ladner
         Python: AI Assistant
"""

import pytest
import numpy as np
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.contSet.interval.interval import Interval


class TestPolyZonotopeSupportFunc:
    """Test class for PolyZonotope supportFunc_ method"""
    
    def test_supportFunc_basic_interval_method(self):
        """Test basic supportFunc with interval method"""
        # Create polynomial zonotope
        c = np.array([3, 4])
        G = np.array([[2, 0, 1], [0, 2, 1]])
        E = np.array([[1, 0, 1], [0, 1, 1]])
        GI = np.array([[0], [0]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Test direction vector
        dir_vec = np.array([1, 1])
        
        # Test different types
        range_result = pZ.supportFunc_(dir_vec, 'range', 'interval')
        lower_result = pZ.supportFunc_(dir_vec, 'lower', 'interval')
        upper_result = pZ.supportFunc_(dir_vec, 'upper', 'interval')
        
        # Check results
        assert isinstance(range_result, Interval)
        assert isinstance(lower_result, (int, float, np.number))
        assert isinstance(upper_result, (int, float, np.number))
        
        # Check consistency
        assert np.isclose(lower_result, range_result.inf)
        assert np.isclose(upper_result, range_result.sup)
    
    def test_supportFunc_bernstein_method(self):
        """Test supportFunc with Bernstein method"""
        # Create polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[2, -1, 2], [0, -2, -3]])
        E = np.array([[1, 0, 1], [0, 1, 3]])
        GI = np.array([[0], [0]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Test direction vector
        dir_vec = np.array([1, 0])
        
        # Test Bernstein method
        range_result = pZ.supportFunc_(dir_vec, 'range', 'bernstein')
        lower_result = pZ.supportFunc_(dir_vec, 'lower', 'bernstein')
        upper_result = pZ.supportFunc_(dir_vec, 'upper', 'bernstein')
        
        # Check results
        assert isinstance(range_result, Interval)
        assert isinstance(lower_result, (int, float, np.number))
        assert isinstance(upper_result, (int, float, np.number))
        
        # Check consistency
        assert np.isclose(lower_result, range_result.inf)
        assert np.isclose(upper_result, range_result.sup)
    
    def test_supportFunc_split_method(self):
        """Test supportFunc with split method"""
        # Create polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([[0.5], [0.1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Test direction vector
        dir_vec = np.array([0, 1])
        
        # Test split method
        range_result = pZ.supportFunc_(dir_vec, 'range', 'split', splits=4)
        lower_result = pZ.supportFunc_(dir_vec, 'lower', 'split', splits=4)
        upper_result = pZ.supportFunc_(dir_vec, 'upper', 'split', splits=4)
        
        # Check results
        assert isinstance(range_result, Interval)
        assert isinstance(lower_result, (int, float, np.number))
        assert isinstance(upper_result, (int, float, np.number))
        
        # Check consistency
        assert np.isclose(lower_result, range_result.inf)
        assert np.isclose(upper_result, range_result.sup)
    
    def test_supportFunc_bnb_method(self):
        """Test supportFunc with branch and bound method"""
        # Create polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([[0.5], [0.1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Test direction vector
        dir_vec = np.array([1, 1])
        
        # Test bnb method
        range_result = pZ.supportFunc_(dir_vec, 'range', 'bnb', maxOrder=3)
        lower_result = pZ.supportFunc_(dir_vec, 'lower', 'bnb', maxOrder=3)
        upper_result = pZ.supportFunc_(dir_vec, 'upper', 'bnb', maxOrder=3)
        
        # Check results
        assert isinstance(range_result, Interval)
        assert isinstance(lower_result, (int, float, np.number))
        assert isinstance(upper_result, (int, float, np.number))
        
        # Check consistency
        assert np.isclose(lower_result, range_result.inf)
        assert np.isclose(upper_result, range_result.sup)
    
    def test_supportFunc_bnbAdv_method(self):
        """Test supportFunc with advanced branch and bound method"""
        # Create polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([[0.5], [0.1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Test direction vector
        dir_vec = np.array([1, 1])
        
        # Test bnbAdv method
        range_result = pZ.supportFunc_(dir_vec, 'range', 'bnbAdv', maxOrder=3)
        lower_result = pZ.supportFunc_(dir_vec, 'lower', 'bnbAdv', maxOrder=3)
        upper_result = pZ.supportFunc_(dir_vec, 'upper', 'bnbAdv', maxOrder=3)
        
        # Check results
        assert isinstance(range_result, Interval)
        assert isinstance(lower_result, (int, float, np.number))
        assert isinstance(upper_result, (int, float, np.number))
        
        # Check consistency
        assert np.isclose(lower_result, range_result.inf)
        assert np.isclose(upper_result, range_result.sup)
    
    def test_supportFunc_globOpt_method(self):
        """Test supportFunc with global optimization method"""
        # Create polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([[0.5], [0.1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Test direction vector
        dir_vec = np.array([1, 1])
        
        # Test globOpt method
        range_result = pZ.supportFunc_(dir_vec, 'range', 'globOpt', maxOrder=3, tol=1e-3)
        lower_result = pZ.supportFunc_(dir_vec, 'lower', 'globOpt', maxOrder=3, tol=1e-3)
        upper_result = pZ.supportFunc_(dir_vec, 'upper', 'globOpt', maxOrder=3, tol=1e-3)
        
        # Check results
        assert isinstance(range_result, Interval)
        assert isinstance(lower_result, (int, float, np.number))
        assert isinstance(upper_result, (int, float, np.number))
        
        # Check consistency
        assert np.isclose(lower_result, range_result.inf)
        assert np.isclose(upper_result, range_result.sup)
    
    def test_supportFunc_quadProg_method(self):
        """Test supportFunc with quadratic programming method"""
        # Create polynomial zonotope
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([[0.5], [0.1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Test direction vector
        dir_vec = np.array([1, 1])
        
        # Test quadProg method
        range_result = pZ.supportFunc_(dir_vec, 'range', 'quadProg')
        lower_result = pZ.supportFunc_(dir_vec, 'lower', 'quadProg')
        upper_result = pZ.supportFunc_(dir_vec, 'upper', 'quadProg')
        
        # Check results
        assert isinstance(range_result, Interval)
        assert isinstance(lower_result, (int, float, np.number))
        assert isinstance(upper_result, (int, float, np.number))
        
        # Check consistency
        assert np.isclose(lower_result, range_result.inf)
        assert np.isclose(upper_result, range_result.sup)
    
    def test_supportFunc_empty_polyZonotope(self):
        """Test supportFunc with empty polynomial zonotope"""
        pZ = PolyZonotope.empty(2)
        dir_vec = np.array([1, 1])
        
        # Test with empty set
        lower_result = pZ.supportFunc_(dir_vec, 'lower', 'interval')
        upper_result = pZ.supportFunc_(dir_vec, 'upper', 'interval')
        
        # Should return infinite values for empty set
        assert lower_result == np.inf
        assert upper_result == -np.inf
    
    def test_supportFunc_single_dimension(self):
        """Test supportFunc with single dimension polynomial zonotope"""
        c = np.array([1])
        G = np.array([[2, 1]])
        E = np.array([[1, 2]])
        GI = np.array([[0.5]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Test direction vector
        dir_vec = np.array([1])
        
        # Test all methods
        methods = ['interval', 'bernstein', 'split', 'bnb', 'bnbAdv', 'globOpt', 'quadProg']
        for method in methods:
            range_result = pZ.supportFunc_(dir_vec, 'range', method)
            assert isinstance(range_result, Interval)
            assert range_result.inf.shape == (1,)
            assert range_result.sup.shape == (1,)
            assert range_result.inf[0] <= range_result.sup[0]
    
    def test_supportFunc_different_directions(self):
        """Test supportFunc with different direction vectors"""
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([[0.5], [0.1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        # Test different direction vectors
        directions = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 1]),
            np.array([-1, 1]),
            np.array([2, -1])
        ]
        
        for dir_vec in directions:
            range_result = pZ.supportFunc_(dir_vec, 'range', 'interval')
            assert isinstance(range_result, Interval)
            assert range_result.inf.shape == (1,)
            assert range_result.sup.shape == (1,)
            assert range_result.inf[0] <= range_result.sup[0]
    
    def test_supportFunc_consistency_between_methods(self):
        """Test consistency between different methods"""
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([[0.5], [0.1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        dir_vec = np.array([1, 1])
        
        # Get results from different methods
        methods = ['interval', 'bernstein', 'split', 'bnb', 'bnbAdv', 'globOpt', 'quadProg']
        results = []
        
        for method in methods:
            range_result = pZ.supportFunc_(dir_vec, 'range', method)
            results.append(range_result)
        
        # All results should be valid intervals
        for result in results:
            assert isinstance(result, Interval)
            assert result.inf.shape == (1,)
            assert result.sup.shape == (1,)
            assert result.inf[0] <= result.sup[0]
    
    def test_supportFunc_invalid_type(self):
        """Test supportFunc with invalid type parameter"""
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([[0.5], [0.1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        dir_vec = np.array([1, 1])
        
        # Should work with valid types
        valid_types = ['lower', 'upper', 'range']
        for type_ in valid_types:
            result = pZ.supportFunc_(dir_vec, type_, 'interval')
            assert result is not None
    
    def test_supportFunc_invalid_method(self):
        """Test supportFunc with invalid method parameter"""
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([[0.5], [0.1]])
        pZ = PolyZonotope(c, G, GI, E)
        
        dir_vec = np.array([1, 1])
        
        # Should handle invalid method gracefully (fallback to interval)
        result = pZ.supportFunc_(dir_vec, 'range', 'invalid_method')
        assert isinstance(result, Interval)
    
    def test_supportFunc_no_dependent_generators(self):
        """Test supportFunc with no dependent generators"""
        c = np.array([1, 2])
        G = np.array([]).reshape(2, 0)  # Empty dependent generators
        E = np.array([]).reshape(0, 0)  # Empty exponent matrix
        GI = np.array([[0.5, 0.1], [0.2, 0.3]])
        pZ = PolyZonotope(c, G, GI, E)
        
        dir_vec = np.array([1, 1])
        
        # Should work with interval method
        range_result = pZ.supportFunc_(dir_vec, 'range', 'interval')
        assert isinstance(range_result, Interval)
        assert range_result.inf.shape == (1,)
        assert range_result.sup.shape == (1,)
        assert range_result.inf[0] <= range_result.sup[0]
    
    def test_supportFunc_no_independent_generators(self):
        """Test supportFunc with no independent generators"""
        c = np.array([0, 0])
        G = np.array([[1, 2], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        GI = np.array([]).reshape(2, 0)  # Empty independent generators
        pZ = PolyZonotope(c, G, GI, E)
        
        dir_vec = np.array([1, 1])
        
        # Should work with all methods
        methods = ['interval', 'bernstein', 'split', 'bnb', 'bnbAdv', 'globOpt', 'quadProg']
        for method in methods:
            range_result = pZ.supportFunc_(dir_vec, 'range', method)
            assert isinstance(range_result, Interval)
            assert range_result.inf.shape == (1,)
            assert range_result.sup.shape == (1,)
            assert range_result.inf[0] <= range_result.sup[0]
