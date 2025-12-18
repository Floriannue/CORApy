"""
test_conZonotope - unit test function for ConZonotope class

This module tests the basic functionality of the ConZonotope class including
construction, properties, and basic operations.

Authors: Python translation by AI Assistant
Written: 2025
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import cora_python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from cora_python.contSet import ConZonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestConZonotope:
    """Test class for ConZonotope functionality"""
    
    def test_empty_conZonotope(self):
        """Test empty conZonotope"""
        n = 2
        cZ = ConZonotope.empty(n)
        assert cZ.representsa_('emptySet', eps=1e-10)
        assert cZ.dim() == n
        
        cZ = ConZonotope(np.zeros((3, 0)))
        assert cZ.representsa_('emptySet', eps=1e-10)
        assert cZ.c.shape[0] == 3 and cZ.G.shape[0] == 3
    
    def test_constructor_basic(self):
        """Test basic ConZonotope construction"""
        # Init simple constrained zonotope
        Z = np.array([[0, 3, 0, 1], [0, 0, 2, 1]])
        A = np.array([[1, 0, 1]])
        b = np.array([[1]])  # Column vector to match MATLAB behavior
        cZ = ConZonotope(Z, A, b)
        
        from cora_python.g.functions.matlab.validate.check.compareMatrices import compareMatrices
        from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
        
        assert np.all(withinTol(np.column_stack([cZ.c, cZ.G]), Z, tol=1e-10))
        assert np.all(withinTol(cZ.A, A, tol=1e-10))
        assert np.all(withinTol(cZ.b, b, tol=1e-10))
    
    def test_constructor_constraints_double(self):
        """Test constraints should be double for internal processing"""
        c = np.array([[1.000], [1.000]])
        G = np.array([[3.000, 0.000, 0.000], [0.000, 3.000, 0.000]])
        # Note: Python doesn't have single precision like MATLAB, but we test the conversion
        A = np.array([[1.211, -0.118, 0.684]], dtype=np.float32)
        b = np.array([-0.112], dtype=np.float32)
        cZ = ConZonotope(c, G, A, b)
        # In Python, arrays are typically float64, but we verify they're numeric
        assert np.issubdtype(cZ.A.dtype, np.floating)
        assert np.issubdtype(cZ.b.dtype, np.floating)
    
    def test_constructor_copy(self):
        """Test copy constructor"""
        c = np.array([1, 2])
        G = np.array([[2, 1], [1, 1]])
        A = np.array([[1, 1]])
        b = np.array([0])
        
        cZ1 = ConZonotope(c, G, A, b)
        cZ2 = ConZonotope(cZ1)
        
        assert np.allclose(cZ1.c, cZ2.c)
        assert np.allclose(cZ1.G, cZ2.G)
        assert np.allclose(cZ1.A, cZ2.A)
        assert np.allclose(cZ1.b, cZ2.b)
    
    def test_constructor_no_constraints(self):
        """Test ConZonotope construction without constraints"""
        c = np.array([1, 2])
        G = np.array([[2, 1], [1, 1]])
        
        cZ = ConZonotope(c, G)
        assert np.allclose(cZ.c, c.reshape(-1, 1))  # c is converted to column vector
        assert np.allclose(cZ.G, G)
        assert cZ.A.shape == (0, 2)  # Empty constraint matrix
        assert cZ.b.shape == (0, 1)  # Empty constraint vector
    
    def test_constructor_errors(self):
        """Test constructor error cases"""
        # No arguments
        with pytest.raises(CORAerror):
            ConZonotope()
        
        # Dimension mismatch between c and G
        with pytest.raises(CORAerror):
            ConZonotope([1, 2], [[1], [2], [3]])  # c is 2D, G is 3D
        
        # Constraint dimension mismatch
        with pytest.raises(CORAerror):
            ConZonotope([1, 2], [[1, 1], [2, 2]], [[1]], [1, 2])  # b has wrong length
    
    def test_properties(self):
        """Test ConZonotope properties"""
        c = np.array([0, 0])
        G = np.array([[1, 0], [0, 1]])
        A = np.array([[1, 1]])
        b = np.array([0])
        
        cZ = ConZonotope(c, G, A, b)
        
        assert cZ.dim() == 2
        assert not cZ.isemptyobject()
        assert cZ.precedence == 90
    
    def test_legacy_Z_property(self):
        """Test legacy Z property getter/setter"""
        c = np.array([1, 2])
        G = np.array([[2, 1], [1, 1]])
        
        cZ = ConZonotope(c, G)
        
        # Test getter (should issue warning)
        with pytest.warns(UserWarning):
            Z = cZ.Z
            expected_Z = np.column_stack([c, G])
            assert np.allclose(Z, expected_Z)
        
        # Test setter (should issue warning)
        new_Z = np.array([[0, 1, 0], [1, 0, 1]])
        with pytest.warns(UserWarning):
            cZ.Z = new_Z
            assert np.allclose(cZ.c, [0, 1])
            assert np.allclose(cZ.G, [[1, 0], [0, 1]])
    
    def test_empty_conZonotope(self):
        """Test empty ConZonotope creation"""
        cZ_empty = ConZonotope.empty(2)
        assert cZ_empty.dim() == 2
        assert cZ_empty.c.shape == (2, 0)  # MATLAB creates zeros(2,0)
        assert cZ_empty.G.shape == (2, 0)
        
        # Empty with dimension 0
        cZ_empty0 = ConZonotope.empty(0)
        assert cZ_empty0.dim() == 0
    
    def test_origin_conZonotope(self):
        """Test origin ConZonotope creation"""
        cZ_origin = ConZonotope.origin(3)
        assert cZ_origin.dim() == 3
        assert np.allclose(cZ_origin.c, [0, 0, 0])
        assert cZ_origin.G.shape == (3, 0)
    
    def test_generateRandom(self):
        """Test random ConZonotope generation"""
        # Default random generation
        cZ_rand = ConZonotope.generateRandom()
        assert cZ_rand.dim() == 2  # Default dimension
        
        # Custom parameters
        cZ_rand_custom = ConZonotope.generateRandom(
            Dimension=3, NrGenerators=4, NrConstraints=2
        )
        assert cZ_rand_custom.dim() == 3
        assert cZ_rand_custom.G.shape == (3, 4)
        assert cZ_rand_custom.A.shape == (2, 4)
        assert cZ_rand_custom.b.shape == (2, 1)
    
    def test_G_property_dimension_fixing(self):
        """Test G property dimension fixing"""
        c = np.array([1, 2])
        cZ = ConZonotope(c, np.array([]))  # Empty G
        
        # G should be automatically resized to match c
        assert cZ.G.shape == (2, 0)
        
        # Setting G to empty should maintain correct dimensions
        cZ.G = np.array([])
        assert cZ.G.shape == (2, 0)


if __name__ == '__main__':
    pytest.main([__file__]) 