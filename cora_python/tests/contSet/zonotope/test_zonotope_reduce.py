"""
test_zonotope_reduce - unit test function of reduce

Syntax:
    python -m pytest test_zonotope_reduce.py

Inputs:
    -

Outputs:
    test results

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 26-July-2016 (MATLAB)
Last update: 08-October-2024 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np

from cora_python.contSet.zonotope import Zonotope


class TestZonotopeReduce:
    """Test class for zonotope reduce method"""
    
    def test_basic_reduce_girard(self):
        """Test basic reduction using Girard method"""
        # Create zonotope with many generators
        c = np.array([0, 0])
        G = np.array([[1, 0.5, 0.3, 0.2, 0.1], 
                      [0, 1, 0.4, 0.3, 0.2]])
        Z = Zonotope(c, G)
        
        # Reduce to order 2
        Z_reduced = Z.reduce('girard', 2)
        
        # Should have fewer generators but same dimension
        assert Z_reduced.dim() == Z.dim()
        assert Z_reduced.G.shape[1] <= 2 * Z.dim()  # Order 2 means at most 2*n generators
    
    def test_reduce_combastel(self):
        """Test reduction using Combastel method"""
        # Create test zonotope
        c = np.array([1, 2])
        G = np.array([[1, 0.5, 0.3, 0.2], 
                      [0, 1, 0.4, 0.3]])
        Z = Zonotope(c, G)
        
        # Reduce to order 1
        Z_reduced = Z.reduce('combastel', 1)
        
        # Should be axis-aligned box
        assert Z_reduced.dim() == Z.dim()
        assert Z_reduced.G.shape[1] <= Z.dim()  # At most n generators for order 1
    
    def test_reduce_containment(self):
        """Test that original zonotope is contained in reduced one"""
        c = np.array([0, 0])
        G = np.array([[1, 0.5, 0.3, 0.2, 0.1, 0.05], 
                      [0, 1, 0.4, 0.3, 0.2, 0.15]])
        Z = Zonotope(c, G)
        
        # Reduce using different methods
        Z_reduced_girard = Z.reduce('girard', 2)
        Z_reduced_combastel = Z.reduce('combastel', 2)
        
        # Sample points from original zonotope
        points = Z.randPoint_(50)
        
        # All points should be contained in reduced zonotopes
        for i in range(points.shape[1]):
            point = points[:, i]
            assert Z_reduced_girard.contains_(point)
            assert Z_reduced_combastel.contains_(point)
    
    def test_reduce_identity_high_order(self):
        """Test that reduction with high order preserves zonotope"""
        c = np.array([1, 2])
        G = np.array([[1, 0.5], [0, 1]])
        Z = Zonotope(c, G)
        
        # Reduce with very high order (should not change anything)
        high_order = 10
        Z_reduced = Z.reduce('girard', high_order)
        
        # Should be equal to original
        assert Z.isequal(Z_reduced)
    
    def test_reduce_empty_zonotope(self):
        """Test reduction of empty zonotope"""
        Z_empty = Zonotope.empty(2)
        Z_reduced = Z_empty.reduce('girard', 1)
        
        assert Z_reduced.isemptyobject()
        assert Z_reduced.dim() == 2
    
    def test_reduce_origin(self):
        """Test reduction of origin zonotope"""
        Z_origin = Zonotope.origin(3)
        Z_reduced = Z_origin.reduce('girard', 1)
        
        # Should still be origin
        np.testing.assert_array_equal(Z_reduced.c.flatten(), np.zeros(3))
    
    def test_reduce_single_generator(self):
        """Test reduction when already few generators"""
        c = np.array([1, 2])
        G = np.array([[1], [0]])  # Only one generator
        Z = Zonotope(c, G)
        
        Z_reduced = Z.reduce('girard', 1)
        
        # Should not change much since already minimal
        assert Z_reduced.dim() == Z.dim()
    
    def test_reduce_1d(self):
        """Test reduction of 1D zonotope"""
        c = np.array([5])
        G = np.array([[1, 0.5, 0.3, 0.2, 0.1]])
        Z = Zonotope(c, G)
        
        Z_reduced = Z.reduce('girard', 1)
        
        assert Z_reduced.dim() == 1
        assert Z_reduced.G.shape[1] <= 1  # Order 1 for 1D means at most 1 generator
    
    def test_reduce_different_methods_containment(self):
        """Test that different reduction methods preserve containment"""
        c = np.array([0, 0, 0])
        G = np.random.rand(3, 10) * 0.5  # Random generators
        Z = Zonotope(c, G)
        
        methods = ['girard', 'combastel']
        order = 2
        
        reduced_zonotopes = []
        for method in methods:
            Z_reduced = Z.reduce(method, order)
            reduced_zonotopes.append(Z_reduced)
            
            # Original should be contained
            points = Z.randPoint_(20)
            for i in range(points.shape[1]):
                assert Z_reduced.contains_(points[:, i])
    
    def test_reduce_volume_preservation_order(self):
        """Test that higher order reduction preserves more volume"""
        c = np.array([0, 0])
        G = np.array([[1, 0.5, 0.3, 0.2, 0.1], 
                      [0, 1, 0.4, 0.3, 0.2]])
        Z = Zonotope(c, G)
        
        Z_order1 = Z.reduce('girard', 1)
        Z_order2 = Z.reduce('girard', 2)
        Z_order3 = Z.reduce('girard', 3)
        
        # Higher order should have more generators (less reduction)
        assert Z_order1.G.shape[1] <= Z_order2.G.shape[1]
        assert Z_order2.G.shape[1] <= Z_order3.G.shape[1]


if __name__ == "__main__":
    test_instance = TestZonotopeReduce()
    
    # Run all tests
    test_instance.test_basic_reduce_girard()
    test_instance.test_reduce_combastel()
    test_instance.test_reduce_containment()
    test_instance.test_reduce_identity_high_order()
    test_instance.test_reduce_empty_zonotope()
    test_instance.test_reduce_origin()
    test_instance.test_reduce_single_generator()
    test_instance.test_reduce_1d()
    test_instance.test_reduce_different_methods_containment()
    test_instance.test_reduce_volume_preservation_order()
    
    print("All zonotope reduce tests passed!") 