"""
Test for nnHelper.getOrderIndices functions

This test verifies that the getOrderIndices functions work correctly for polynomial evaluation.
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.getOrderIndicesG import getOrderIndicesG
from cora_python.nn.nnHelper.getOrderIndicesGI import getOrderIndicesGI


class TestGetOrderIndicesG:
    """Test class for getOrderIndicesG function"""
    
    def test_getOrderIndicesG_basic(self):
        """Test basic getOrderIndicesG functionality"""
        # Test with simple case
        G = np.array([[1, 0.5, 0.3], [0.2, 0.8, 0.1]])
        order = 2
        
        start_idx, end_idx = getOrderIndicesG(G, order)
        
        # Check that results are integers
        assert isinstance(start_idx, int)
        assert isinstance(end_idx, int)
        
        # Check that start_idx <= end_idx
        assert start_idx <= end_idx
        
        # Check that indices are non-negative
        assert start_idx >= 0
        assert end_idx >= 0
    
    def test_getOrderIndicesG_different_orders(self):
        """Test getOrderIndicesG with different orders"""
        G = np.array([[1, 0.5], [0.3, 0.8]])
        
        # Test order 1
        start1, end1 = getOrderIndicesG(G, 1)
        assert start1 <= end1
        
        # Test order 2
        start2, end2 = getOrderIndicesG(G, 2)
        assert start2 <= end2
        
        # Test order 3
        start3, end3 = getOrderIndicesG(G, 3)
        assert start3 <= end3
        
        # Higher orders should have larger ranges
        assert (end1 - start1) <= (end2 - start2)
        assert (end2 - start2) <= (end3 - start3)
    
    def test_getOrderIndicesG_edge_cases(self):
        """Test getOrderIndicesG edge cases"""
        # Test with empty matrix
        G = np.array([]).reshape(0, 0)
        order = 1
        
        start_idx, end_idx = getOrderIndicesG(G, order)
        assert isinstance(start_idx, int)
        assert isinstance(end_idx, int)
        
        # Test with single element
        G = np.array([[5]])
        order = 1
        
        start_idx, end_idx = getOrderIndicesG(G, order)
        assert isinstance(start_idx, int)
        assert isinstance(end_idx, int)
        
        # Test with order 0
        G = np.array([[1, 0.5], [0.3, 0.8]])
        order = 0
        
        start_idx, end_idx = getOrderIndicesG(G, order)
        assert isinstance(start_idx, int)
        assert isinstance(end_idx, int)
    
    def test_getOrderIndicesG_large_matrices(self):
        """Test getOrderIndicesG with larger matrices"""
        # Create larger matrix
        np.random.seed(42)
        G = np.random.rand(10, 20)
        
        for order in [1, 2, 3, 5]:
            start_idx, end_idx = getOrderIndicesG(G, order)
            
            assert isinstance(start_idx, int)
            assert isinstance(end_idx, int)
            assert start_idx <= end_idx
            assert start_idx >= 0
            assert end_idx >= 0
    
    def test_getOrderIndicesG_consistency(self):
        """Test that getOrderIndicesG produces consistent results"""
        G = np.array([[1, 0.5, 0.3], [0.2, 0.8, 0.1]])
        order = 2
        
        # Call multiple times
        start1, end1 = getOrderIndicesG(G, order)
        start2, end2 = getOrderIndicesG(G, order)
        
        # Should be consistent
        assert start1 == start2
        assert end1 == end2


class TestGetOrderIndicesGI:
    """Test class for getOrderIndicesGI function"""
    
    def test_getOrderIndicesGI_basic(self):
        """Test basic getOrderIndicesGI functionality"""
        # Test with simple case
        GI = np.array([[0.2, 0.4], [0.1, 0.3]])
        G = np.array([[1, 0.5], [0.3, 0.8]])
        order = 2
        
        start_idx, end_idx = getOrderIndicesGI(GI, G, order)
        
        # Check that results are integers
        assert isinstance(start_idx, int)
        assert isinstance(end_idx, int)
        
        # Check that start_idx <= end_idx
        assert start_idx <= end_idx
        
        # Check that indices are non-negative
        assert start_idx >= 0
        assert end_idx >= 0
    
    def test_getOrderIndicesGI_different_orders(self):
        """Test getOrderIndicesGI with different orders"""
        GI = np.array([[0.2, 0.4], [0.1, 0.3]])
        G = np.array([[1, 0.5], [0.3, 0.8]])
        
        # Test order 1
        start1, end1 = getOrderIndicesGI(GI, G, 1)
        assert start1 <= end1
        
        # Test order 2
        start2, end2 = getOrderIndicesGI(GI, G, 2)
        assert start2 <= end2
        
        # Test order 3
        start3, end3 = getOrderIndicesGI(GI, G, 3)
        assert start3 <= end3
        
        # Higher orders should have larger ranges
        assert (end1 - start1) <= (end2 - start2)
        assert (end2 - start2) <= (end3 - start3)
    
    def test_getOrderIndicesGI_edge_cases(self):
        """Test getOrderIndicesGI edge cases"""
        # Test with empty matrices
        GI = np.array([]).reshape(0, 0)
        G = np.array([]).reshape(0, 0)
        order = 1
        
        start_idx, end_idx = getOrderIndicesGI(GI, G, order)
        assert isinstance(start_idx, int)
        assert isinstance(end_idx, int)
        
        # Test with single element
        GI = np.array([[5]])
        G = np.array([[3]])
        order = 1
        
        start_idx, end_idx = getOrderIndicesGI(GI, G, order)
        assert isinstance(start_idx, int)
        assert isinstance(end_idx, int)
        
        # Test with order 0
        GI = np.array([[0.2, 0.4], [0.1, 0.3]])
        G = np.array([[1, 0.5], [0.3, 0.8]])
        order = 0
        
        start_idx, end_idx = getOrderIndicesGI(GI, G, order)
        assert isinstance(start_idx, int)
        assert isinstance(end_idx, int)
    
    def test_getOrderIndicesGI_different_dimensions(self):
        """Test getOrderIndicesGI with different dimensional matrices"""
        # Test with different dimensions
        GI = np.array([[0.2, 0.4, 0.1], [0.1, 0.3, 0.2]])
        G = np.array([[1, 0.5], [0.3, 0.8]])
        order = 2
        
        start_idx, end_idx = getOrderIndicesGI(GI, G, order)
        assert isinstance(start_idx, int)
        assert isinstance(end_idx, int)
        assert start_idx <= end_idx
        
        # Test with more rows in G than GI
        GI = np.array([[0.2, 0.4], [0.1, 0.3]])
        G = np.array([[1, 0.5, 0.2], [0.3, 0.8, 0.1], [0.4, 0.6, 0.3]])
        order = 2
        
        start_idx, end_idx = getOrderIndicesGI(GI, G, order)
        assert isinstance(start_idx, int)
        assert isinstance(end_idx, int)
        assert start_idx <= end_idx
    
    def test_getOrderIndicesGI_large_matrices(self):
        """Test getOrderIndicesGI with larger matrices"""
        # Create larger matrices
        np.random.seed(42)
        GI = np.random.rand(5, 10)
        G = np.random.rand(5, 15)
        
        for order in [1, 2, 3, 5]:
            start_idx, end_idx = getOrderIndicesGI(GI, G, order)
            
            assert isinstance(start_idx, int)
            assert isinstance(end_idx, int)
            assert start_idx <= end_idx
            assert start_idx >= 0
            assert end_idx >= 0
    
    def test_getOrderIndicesGI_consistency(self):
        """Test that getOrderIndicesGI produces consistent results"""
        GI = np.array([[0.2, 0.4], [0.1, 0.3]])
        G = np.array([[1, 0.5], [0.3, 0.8]])
        order = 2
        
        # Call multiple times
        start1, end1 = getOrderIndicesGI(GI, G, order)
        start2, end2 = getOrderIndicesGI(GI, G, order)
        
        # Should be consistent
        assert start1 == start2
        assert end1 == end2


class TestGetOrderIndicesIntegration:
    """Test class for integration between getOrderIndices functions"""
    
    def test_getOrderIndices_integration(self):
        """Test integration between getOrderIndices functions"""
        # Test data
        GI = np.array([[0.2, 0.4], [0.1, 0.3]])
        G = np.array([[1, 0.5], [0.3, 0.8]])
        
        # Test both functions together
        for order in [1, 2, 3]:
            # Get indices for G
            start_G, end_G = getOrderIndicesG(G, order)
            
            # Get indices for GI
            start_GI, end_GI = getOrderIndicesGI(GI, G, order)
            
            # All should be valid
            assert isinstance(start_G, int)
            assert isinstance(end_G, int)
            assert isinstance(start_GI, int)
            assert isinstance(end_GI, int)
            
            assert start_G <= end_G
            assert start_GI <= end_GI
            assert start_G >= 0
            assert end_G >= 0
            assert start_GI >= 0
            assert end_GI >= 0
    
    def test_getOrderIndices_relationship(self):
        """Test relationship between G and GI indices"""
        GI = np.array([[0.2, 0.4], [0.1, 0.3]])
        G = np.array([[1, 0.5], [0.3, 0.8]])
        
        # For the same order, GI indices should typically be different from G indices
        # since they represent different generator types
        order = 2
        
        start_G, end_G = getOrderIndicesG(G, order)
        start_GI, end_GI = getOrderIndicesGI(GI, G, order)
        
        # Both should be valid
        assert start_G <= end_G
        assert start_GI <= end_GI
        
        # They might be the same or different, but both should be valid
        assert start_G >= 0
        assert end_G >= 0
        assert start_GI >= 0
        assert end_GI >= 0
