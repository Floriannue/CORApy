"""
Test for nnHelper.heap class

This test verifies that the heap class works correctly for neural network operations.
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.heap import Heap


class TestHeap:
    """Test class for Heap implementation"""
    
    def test_heap_creation(self):
        """Test heap creation with various input types"""
        # Test with scalar keys
        heap_entries = [5, 3, 2, 4, 7]
        heap = Heap(heap_entries)
        assert not heap.isempty()
        
        # Test with empty list
        empty_heap = Heap([])
        assert empty_heap.isempty()
        
        # Test with single element
        single_heap = Heap([42])
        assert not single_heap.isempty()
        assert single_heap.min().key == 42
    
    def test_heap_min_heap_property(self):
        """Test that heap maintains min-heap property"""
        heap_entries = [5, 3, 2, 4, 7, 1, 8]
        heap = Heap(heap_entries)
        
        # Extract all elements - they should come out in ascending order
        extracted = []
        while not heap.isempty():
            extracted.append(heap.pop().key)
        
        # Check that elements are in ascending order
        assert extracted == sorted(extracted)
    
    def test_heap_insert(self):
        """Test heap insert operation"""
        heap = Heap([])
        
        # Insert elements one by one
        heap.insert(type('Entry', (), {'key': 5, 'i': 0})())
        heap.insert(type('Entry', (), {'key': 3, 'i': 1})())
        heap.insert(type('Entry', (), {'key': 7, 'i': 2})())
        
        # Check min element
        assert heap.min().key == 3
        
        # Insert smaller element
        heap.insert(type('Entry', (), {'key': 1, 'i': 3})())
        assert heap.min().key == 1
    
    def test_heap_pop(self):
        """Test heap pop operation"""
        heap_entries = [5, 3, 2, 4, 7]
        heap = Heap(heap_entries)
        
        # Pop elements and verify min-heap property
        prev_val = float('-inf')
        while not heap.isempty():
            current_val = heap.pop().key
            assert current_val >= prev_val
            prev_val = current_val
    
    def test_heap_replace_min(self):
        """Test heap replace_min operation"""
        heap_entries = [5, 3, 2, 4, 7]
        heap = Heap(heap_entries)
        
        # Get current min
        old_min = heap.min()
        assert old_min.key == 2
        
        # Replace with new element
        new_entry = type('Entry', (), {'key': 1, 'i': 5})()
        old_min_returned = heap.replace_min(new_entry)
        
        # Check that old min was returned
        assert old_min_returned.key == 2
        
        # Check that new min is correct
        assert heap.min().key == 1
    
    def test_heap_array_keys(self):
        """Test heap with array keys (lexicographic ordering)"""
        # Create entries with array keys
        heap_entries = [
            type('Entry', (), {'key': np.array([2, 1]), 'i': 0})(),
            type('Entry', (), {'key': np.array([1, 2]), 'i': 1})(),
            type('Entry', (), {'key': np.array([1, 1]), 'i': 2})(),
            type('Entry', (), {'key': np.array([2, 2]), 'i': 3})()
        ]
        
        heap = Heap(heap_entries)
        
        # Extract elements - they should come out in lexicographic order
        extracted = []
        while not heap.isempty():
            extracted.append(heap.pop().key.tolist())
        
        # Check lexicographic ordering
        expected = [[1, 1], [1, 2], [2, 1], [2, 2]]
        assert extracted == expected
    
    def test_heap_mixed_key_types(self):
        """Test heap with mixed key types"""
        heap_entries = [
            type('Entry', (), {'key': 5, 'i': 0})(),
            type('Entry', (), {'key': np.array([1, 2]), 'i': 1})(),
            type('Entry', (), {'key': 3, 'i': 2})(),
            type('Entry', (), {'key': np.array([0, 1]), 'i': 3})()
        ]
        
        heap = Heap(heap_entries)
        
        # Should handle mixed types gracefully
        assert not heap.isempty()
    
    def test_heap_edge_cases(self):
        """Test heap edge cases"""
        # Test with negative numbers
        heap_entries = [-5, -3, -2, -4, -7]
        heap = Heap(heap_entries)
        
        extracted = []
        while not heap.isempty():
            extracted.append(heap.pop().key)
        
        assert extracted == sorted(extracted)
        
        # Test with floats
        heap_entries = [5.5, 3.3, 2.2, 4.4, 7.7]
        heap = Heap(heap_entries)
        
        extracted = []
        while not heap.isempty():
            extracted.append(heap.pop().key)
        
        assert extracted == sorted(extracted)
    
    def test_heap_duplicate_keys(self):
        """Test heap with duplicate keys"""
        heap_entries = [3, 3, 3, 1, 1, 2, 2]
        heap = Heap(heap_entries)
        
        extracted = []
        while not heap.isempty():
            extracted.append(heap.pop().key)
        
        # Should maintain stability and order
        assert extracted == sorted(extracted)
    
    def test_heap_large_dataset(self):
        """Test heap with larger dataset"""
        # Generate larger dataset
        np.random.seed(42)  # For reproducibility
        heap_entries = np.random.randint(0, 1000, 100).tolist()
        heap = Heap(heap_entries)
        
        extracted = []
        while not heap.isempty():
            extracted.append(heap.pop().key)
        
        # Check that elements are in ascending order
        assert extracted == sorted(extracted)
    
    def test_heap_empty_operations(self):
        """Test heap operations on empty heap"""
        heap = Heap([])
        
        # Test operations on empty heap
        assert heap.isempty()
        
        # Insert into empty heap
        heap.insert(type('Entry', (), {'key': 42, 'i': 0})())
        assert not heap.isempty()
        assert heap.min().key == 42
        
        # Pop from heap with one element
        entry = heap.pop()
        assert entry.key == 42
        assert heap.isempty()
    
    def test_heap_single_element_operations(self):
        """Test heap operations with single element"""
        heap = Heap([42])
        
        # Test min
        assert heap.min().key == 42
        
        # Test pop
        entry = heap.pop()
        assert entry.key == 42
        assert heap.isempty()
        
        # Test replace_min
        heap = Heap([42])
        new_entry = type('Entry', (), {'key': 10, 'i': 1})()
        old_entry = heap.replace_min(new_entry)
        assert old_entry.key == 42
        assert heap.min().key == 10
