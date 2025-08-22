"""
test_vertices - unit test function for vertices

Syntax:
    pytest test_vertices.py

Inputs:
    -

Outputs:
    test results

Other modules required: none
Subfunctions: none

See also: none

Authors: AI Assistant
Written: 2025
Last update: ---
Last revision: ---
"""

import pytest
import numpy as np
from unittest.mock import patch

# Import actual CORA classes
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contSet.emptySet import EmptySet
from cora_python.contSet.fullspace import Fullspace
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.conZonotope import ConZonotope
from cora_python.contSet.conPolyZono import ConPolyZono

from cora_python.contSet.contSet.vertices import vertices


class TestVertices:
    """Test class for vertices function"""
    
    def test_vertices_basic_functionality(self):
        """Test basic vertices functionality"""
        
        # Create a simple 2D polytope
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        S = Polytope(A, b)
        
        result = vertices(S)
        
        # Should return a 2D array with vertices
        assert result.shape[0] == 2
        assert result.shape[1] > 0
    
    def test_vertices_with_method(self):
        """Test vertices with specific method"""
        
        # Create a simple 2D polytope
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        S = Polytope(A, b)
        
        result = vertices(S, method='lcon2vert')
        
        # Should return a 2D array with vertices
        assert result.shape[0] == 2
        assert result.shape[1] > 0
    
    def test_vertices_empty_set(self):
        """Test vertices with empty set"""
        
        # Create an empty set
        S = EmptySet(2)
        
        result = vertices(S)
        
        # Should return empty array with correct dimensions
        assert result.shape == (2, 0)
    
    def test_vertices_empty_result_handling(self):
        """Test vertices when result is empty but set is not empty"""
        
        # Create a polytope that will return empty vertices
        # Use a polytope with conflicting constraints that results in empty set
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, -2, 1, -2])  # Conflicting constraints: x <= 1 and x >= 2
        S = Polytope(A, b)
        
        result = vertices(S)
        
        # Should return empty array with correct dimensions
        assert result.shape == (2, 0)
    
    def test_vertices_polytope_methods(self):
        """Test vertices with Polytope using different methods"""
        
        # Create a simple 2D polytope
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        S = Polytope(A, b)
        
        valid_methods = ['cdd', 'lcon2vert', 'comb']
        for method in valid_methods:
            result = vertices(S, method=method)
            assert result.shape[0] == 2
        
        # Test validation with invalid method
        with patch('cora_python.g.macros.CHECKS_ENABLED.CHECKS_ENABLED', return_value=True):
            with pytest.raises(Exception):  # Should raise validation error
                vertices(S, method='invalid')
    
    def test_vertices_polytope_method_passing(self):
        """Test that method parameter is correctly passed to polytope.vertices_"""
        
        # Create a simple 2D polytope
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        S = Polytope(A, b)
        
        # Test that method parameter is passed through
        result = vertices(S, method='comb')
        assert result.shape[0] == 2
    
    def test_vertices_conpolyzonotope_method(self):
        """Test vertices with ConPolyZono using numeric method"""
        
        # Create a simple ConPolyZono with valid parameters
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        E = np.array([[1, 0], [0, 1]])  # 2x2 matrix
        A = np.array([[1, 0]])
        b = np.array([1])
        EC = np.array([[1, 0], [0, 1]])  # 2x2 matrix to match E dimensions
        GI = np.array([[0, 0]])  # Add GI parameter (restricted generators)
        S = ConPolyZono(c, G, E, A, b, EC, GI)
        
        # Valid numeric method (number of splits)
        result = vertices(S, method=10)
        assert result.shape[0] == 2
        
        # Test validation with invalid method
        with patch('cora_python.g.macros.CHECKS_ENABLED.CHECKS_ENABLED', return_value=True):
            with pytest.raises(Exception):  # Should raise validation error
                vertices(S, method=0)
            
            with pytest.raises(Exception):  # Should raise validation error
                vertices(S, method=-5)
    
    def test_vertices_conzonotope_methods(self):
        """Test vertices with ConZonotope using different methods"""
        
        # Create a simple ConZonotope
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        A = np.array([[1, 0]])
        b = np.array([1])
        S = ConZonotope(c, G, A, b)
        
        valid_methods = ['default', 'template']
        for method in valid_methods:
            result = vertices(S, method=method)
            assert result.shape[0] == 2
        
        # Test validation with invalid method
        with patch('cora_python.g.macros.CHECKS_ENABLED.CHECKS_ENABLED', return_value=True):
            with pytest.raises(Exception):  # Should raise validation error
                vertices(S, method='invalid')
    
    def test_vertices_general_methods(self):
        """Test vertices with general set types using different methods"""
        
        # Create a simple zonotope
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        S = Zonotope(c, G)
        
        valid_methods = ['convHull', 'iterate', 'polytope']
        for method in valid_methods:
            result = vertices(S, method=method)
            assert result.shape[0] == 2
        
        # Test validation with invalid method
        with patch('cora_python.g.macros.CHECKS_ENABLED.CHECKS_ENABLED', return_value=True):
            with pytest.raises(Exception):  # Should raise validation error
                vertices(S, method='invalid')
    
    def test_vertices_interval(self):
        """Test vertices with Interval"""
        
        # Create a simple 2D interval
        inf = np.array([-1, -1])
        sup = np.array([1, 1])
        S = Interval(inf, sup)
        
        result = vertices(S)
        
        # Should return a 2D array with vertices
        assert result.shape[0] == 2
        assert result.shape[1] > 0
    
    def test_vertices_ellipsoid(self):
        """Test vertices with Ellipsoid"""
        
        # Create a simple 2D ellipsoid with column vector center
        c = np.array([[0], [0]])  # Column vector
        Q = np.array([[1, 0], [0, 1]])  # Square matrix
        S = Ellipsoid(c, Q)
        
        result = vertices(S)
        
        # Should return a 2D array with vertices
        assert result.shape[0] == 2
        assert result.shape[1] > 0
    
    def test_vertices_fullspace(self):
        """Test vertices with Fullspace"""
        
        # Create a 2D fullspace
        S = Fullspace(2)
        
        result = vertices(S)
        
        # Should return a 2D array with vertices
        assert result.shape[0] == 2
        assert result.shape[1] > 0
    
    def test_vertices_emptyset(self):
        """Test vertices with EmptySet"""
        
        # Create an empty set
        S = EmptySet(2)
        
        result = vertices(S)
        
        # Should return empty array with correct dimensions
        assert result.shape == (2, 0)


if __name__ == "__main__":
    pytest.main([__file__]) 