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
from unittest.mock import Mock, patch
from cora_python.contSet.contSet.vertices import vertices


class MockContSet:
    """Mock ContSet for testing vertices method"""
    
    def __new__(cls, class_name="Zonotope", *args, **kwargs):
        # Dynamically create a new class with the desired name that inherits from this one.
        # This ensures that obj.__class__.__name__ gives the mocked name.
        # We pass through any other args to __init__.
        NewCls = type(class_name, (MockContSet,), {})
        instance = object.__new__(NewCls)
        return instance
    
    def __init__(self, class_name="Zonotope", dim_val=2, empty=False):
        self._dim = dim_val
        self._empty = empty
        
    def dim(self):
        return self._dim
    
    def isemptyobject(self):
        return self._empty
    
    def vertices_(self, method, *args):
        """Mock implementation of vertices_"""
        if self._empty:
            return np.array([])
        
        class_name = self.__class__.__name__
        
        # Return mock vertices for different set types
        if class_name == "Interval":
            # Return corner points for 2D interval
            if self._dim == 2:
                return np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
            else:
                return np.ones((self._dim, 2**self._dim))
        else:
            # Return some default vertices
            return np.random.rand(self._dim, 4)
    
    def representsa_(self, setType, tol):
        return self._empty and setType == 'emptySet'


class TestVertices:
    """Test class for vertices function"""
    
    def test_vertices_basic(self):
        """Test basic vertices functionality"""
        
        S = MockContSet("Interval", 2)
        result = vertices(S)
        
        assert result.shape == (2, 4)  # 2D interval has 4 vertices
        assert isinstance(result, np.ndarray)
    
    def test_vertices_polytope_default_method(self):
        """Test vertices with Polytope using default method"""
        
        S = MockContSet("Polytope", 2)
        result = vertices(S)
        
        assert result.shape[0] == 2  # Dimension should match
        assert isinstance(result, np.ndarray)
    
    def test_vertices_polytope_methods(self):
        """Test vertices with Polytope using different methods"""
        
        S = MockContSet("Polytope", 2)
        
        valid_methods = ['cdd', 'lcon2vert']
        for method in valid_methods:
            result = vertices(S, method=method)
            assert result.shape[0] == 2
        
        # Test invalid method
        with pytest.raises(ValueError, match="Invalid method"):
            vertices(S, method='invalid')
    
    def test_vertices_conpolyzonotope_method(self):
        """Test vertices with ConPolyZono using numeric method"""
        
        S = MockContSet("ConPolyZono", 2)
        
        # Valid numeric method (number of splits)
        result = vertices(S, method=10)
        assert result.shape[0] == 2
        
        # Test invalid method - non-positive
        with pytest.raises(ValueError, match="must be a positive number"):
            vertices(S, method=0)
        
        with pytest.raises(ValueError, match="must be a positive number"):
            vertices(S, method=-5)
    
    def test_vertices_conzonotope_methods(self):
        """Test vertices with ConZonotope using different methods"""
        
        S = MockContSet("ConZonotope", 2)
        
        valid_methods = ['default', 'template']
        for method in valid_methods:
            result = vertices(S, method=method)
            assert result.shape[0] == 2
        
        # Test invalid method
        with pytest.raises(ValueError, match="Invalid method"):
            vertices(S, method='invalid')
    
    def test_vertices_general_methods(self):
        """Test vertices with general set types using different methods"""
        
        S = MockContSet("Zonotope", 2)
        
        valid_methods = ['convHull', 'iterate', 'polytope']
        for method in valid_methods:
            result = vertices(S, method=method)
            assert result.shape[0] == 2
        
        # Test invalid method
        with pytest.raises(ValueError, match="Invalid method"):
            vertices(S, method='invalid')
    
    def test_vertices_empty_set(self):
        """Test vertices with empty set"""
        
        S = MockContSet("Interval", 2, empty=True)
        
        result = vertices(S)
        
        # Should return empty array with correct dimensions
        assert result.shape == (2, 0)
    
    def test_vertices_empty_result_handling(self):
        """Test vertices when result is empty but set is not empty"""
        
        class EmptyResultSet:
            def __init__(self):
                self._dim = 3
                self._empty = False
                
            def dim(self):
                return self._dim
                
            def isemptyobject(self):
                return self._empty
                
            def vertices_(self, method, *args):
                return np.array([])
            
            def representsa_(self, setType, tol):
                return False  # Not an empty set
        
        S = EmptyResultSet()
        S.__class__.__name__ = "CustomSet"
        result = vertices(S)
        
        # Should return properly shaped empty array
        assert result.shape == (3, 0)
    
    def test_vertices_exception_handling_empty_set(self):
        """Test vertices exception handling when set represents empty set"""
        
        class ExceptionSet:
            def __init__(self):
                self._dim = 2
                self._empty = False
                
            def dim(self):
                return self._dim
                
            def isemptyobject(self):
                return self._empty
                
            def vertices_(self, method, *args):
                raise RuntimeError("Some error")
            
            def representsa_(self, setType, tol):
                return setType == 'emptySet'
        
        S = ExceptionSet()
        S.__class__.__name__ = "CustomSet"
        result = vertices(S)
        
        # Should handle exception and return empty array
        assert result.shape == (2, 0)
    
    def test_vertices_exception_handling_non_empty_set(self):
        """Test vertices exception handling when set is not empty"""
        
        class ExceptionSet:
            def __init__(self):
                self._dim = 2
                self._empty = False
                
            def dim(self):
                return self._dim
                
            def isemptyobject(self):
                return self._empty
                
            def vertices_(self, method, *args):
                raise RuntimeError("Some error")
            
            def representsa_(self, setType, tol):
                return False
        
        S = ExceptionSet()
        S.__class__.__name__ = "CustomSet"
        
        # Should re-raise the exception
        with pytest.raises(RuntimeError, match="Some error"):
            vertices(S)
    
    def test_vertices_with_additional_args(self):
        """Test vertices with additional arguments"""
        
        class ArgsSet:
            def __init__(self):
                self._dim = 2
                self._empty = False
                
            def dim(self):
                return self._dim
                
            def isemptyobject(self):
                return self._empty
                
            def vertices_(self, method, *args):
                # Verify additional arguments are passed
                assert len(args) == 2
                assert args[0] == 'extra_arg1'
                assert args[1] == 'extra_arg2'
                return np.ones((self._dim, 3))
            
            def representsa_(self, setType, tol):
                return False
        
        S = ArgsSet()
        S.__class__.__name__ = "CustomSet"
        result = vertices(S, 'convHull', 'extra_arg1', 'extra_arg2')
        
        assert result.shape == (2, 3)
    
    def test_vertices_high_dimension(self):
        """Test vertices with high-dimensional sets"""
        
        S = MockContSet("Interval", 5)
        result = vertices(S)
        
        assert result.shape[0] == 5
        assert result.shape[1] > 0


if __name__ == "__main__":
    pytest.main([__file__]) 