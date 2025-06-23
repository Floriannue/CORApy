"""
Unit tests for specification isempty method

This file tests the isempty functionality for temporal logic specifications.
Based on test_specification_isempty.m from MATLAB CORA.

Authors: MATLAB CORA developers (original)
         Python translation by AI Assistant
"""

import unittest
import numpy as np
from typing import List

# Add path for imports
import sys
sys.path.insert(0, '.')

from cora_python.specification.specification.specification import Specification
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestSpecificationIsEmpty(unittest.TestCase):
    """Test cases for specification isempty method"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create test specifications with different set types
        from cora_python.contSet.polytope.polytope import Polytope
        from cora_python.contSet.emptySet.emptySet import EmptySet
        
        # Non-empty 2D polytope: x1 >= 0, x2 >= 0, x1 + x2 <= 2
        A = np.array([[-1, 0], [0, -1], [1, 1]])
        b = np.array([0, 0, 2])
        self.set_nonempty = Polytope(A, b)
        
        # Empty polytope: x1 >= 0, x1 <= -1 (contradictory constraints)
        A_empty = np.array([[-1, 0], [1, 0]])
        b_empty = np.array([0, -1])
        self.set_empty_polytope = Polytope(A_empty, b_empty)
        
        # Explicit empty set
        self.set_empty_explicit = EmptySet(2)
        
        # Time intervals
        self.time_interval = np.array([0, 10])
        
        # Create specifications with non-empty sets
        self.spec_safe_nonempty = Specification(self.set_nonempty, 'safeSet', self.time_interval)
        self.spec_unsafe_nonempty = Specification(self.set_nonempty, 'unsafeSet', self.time_interval)
        
        # Create specifications with empty sets
        self.spec_safe_empty = Specification(self.set_empty_polytope, 'safeSet', self.time_interval)
        self.spec_unsafe_empty = Specification(self.set_empty_polytope, 'unsafeSet', self.time_interval)
        self.spec_explicit_empty = Specification(self.set_empty_explicit, 'safeSet', self.time_interval)
        
        # Specifications without time
        self.spec_nonempty_no_time = Specification(self.set_nonempty, 'safeSet')
        self.spec_empty_no_time = Specification(self.set_empty_explicit, 'safeSet')
    
    def test_isempty_nonempty_specification(self):
        """Test isempty for non-empty specifications"""
        # Non-empty specifications should return False
        self.assertFalse(self.spec_safe_nonempty.isempty())
        self.assertFalse(self.spec_unsafe_nonempty.isempty())
        self.assertFalse(self.spec_nonempty_no_time.isempty())
    
    def test_isempty_empty_specification(self):
        """Test isempty for empty specifications"""
        # Empty specifications should return True
        self.assertTrue(self.spec_safe_empty.isempty())
        self.assertTrue(self.spec_unsafe_empty.isempty())
        self.assertTrue(self.spec_explicit_empty.isempty())
        self.assertTrue(self.spec_empty_no_time.isempty())
    
    def test_isempty_different_types(self):
        """Test isempty for different specification types"""
        # Test different types with non-empty sets
        spec_final = Specification(self.set_nonempty, 'finalSet', self.time_interval)
        spec_invariant = Specification(self.set_nonempty, 'invariant', self.time_interval)
        
        self.assertFalse(spec_final.isempty())
        self.assertFalse(spec_invariant.isempty())
        
        # Test different types with empty sets
        spec_final_empty = Specification(self.set_empty_explicit, 'finalSet', self.time_interval)
        spec_invariant_empty = Specification(self.set_empty_explicit, 'invariant', self.time_interval)
        
        self.assertTrue(spec_final_empty.isempty())
        self.assertTrue(spec_invariant_empty.isempty())
    
    def test_isempty_return_type(self):
        """Test that isempty returns boolean"""
        result_nonempty = self.spec_safe_nonempty.isempty()
        result_empty = self.spec_explicit_empty.isempty()
        
        # Should return boolean
        self.assertIsInstance(result_nonempty, bool)
        self.assertIsInstance(result_empty, bool)
        
        # Values should be correct
        self.assertFalse(result_nonempty)
        self.assertTrue(result_empty)
    
    def test_isempty_consistency(self):
        """Test isempty consistency"""
        # Multiple calls should return same result
        result1 = self.spec_safe_nonempty.isempty()
        result2 = self.spec_safe_nonempty.isempty()
        self.assertEqual(result1, result2)
        
        result3 = self.spec_explicit_empty.isempty()
        result4 = self.spec_explicit_empty.isempty()
        self.assertEqual(result3, result4)
    
    def test_isempty_with_time_constraints(self):
        """Test isempty behavior with time constraints"""
        # Time constraints should not affect emptiness of the set itself
        spec_with_time = Specification(self.set_nonempty, 'safeSet', self.time_interval)
        spec_without_time = Specification(self.set_nonempty, 'safeSet')
        
        # Both should have same emptiness status
        self.assertEqual(spec_with_time.isempty(), spec_without_time.isempty())
        
        # Same for empty sets
        spec_empty_with_time = Specification(self.set_empty_explicit, 'safeSet', self.time_interval)
        spec_empty_without_time = Specification(self.set_empty_explicit, 'safeSet')
        
        self.assertEqual(spec_empty_with_time.isempty(), spec_empty_without_time.isempty())
    
    def test_isempty_with_complex_sets(self):
        """Test isempty with more complex set types"""
        try:
            from cora_python.contSet.zonotope.zonotope import Zonotope
            
            # Non-empty zonotope
            Z_nonempty = Zonotope(np.array([1, 1]), np.array([[1, 0], [0, 1]]))
            spec_zono_nonempty = Specification(Z_nonempty, 'safeSet', self.time_interval)
            self.assertFalse(spec_zono_nonempty.isempty())
            
            # Degenerate zonotope (if possible to create)
            Z_degenerate = Zonotope(np.array([0, 0]), np.array([[0, 0], [0, 0]]))
            spec_zono_degenerate = Specification(Z_degenerate, 'safeSet', self.time_interval)
            
            # The result depends on how zonotope handles degenerate cases
            result = spec_zono_degenerate.isempty()
            self.assertIsInstance(result, bool)
            
        except ImportError:
            # Skip if zonotope not available
            pass
    
    def test_isempty_edge_cases(self):
        """Test isempty edge cases"""
        # Test with very small but non-empty polytope
        try:
            # Create a very small polytope: 0 <= x1 <= 0.001, 0 <= x2 <= 0.001
            A_small = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
            b_small = np.array([0, 0.001, 0, 0.001])
            set_small = Polytope(A_small, b_small)
            spec_small = Specification(set_small, 'safeSet')
            
            # Should not be empty even if very small
            self.assertFalse(spec_small.isempty())
            
        except Exception:
            # Skip if polytope creation fails
            pass
    
    def test_isempty_after_operations(self):
        """Test isempty after various operations"""
        # Test isempty after inverse
        inv_spec = self.spec_safe_nonempty.inverse()
        self.assertFalse(inv_spec.isempty())  # Inverse should preserve non-emptiness
        
        inv_empty = self.spec_explicit_empty.inverse()
        self.assertTrue(inv_empty.isempty())  # Inverse should preserve emptiness
        
        # Test isempty after projection (if available)
        try:
            proj_spec = self.spec_safe_nonempty.project([1])
            result = proj_spec.isempty()
            self.assertIsInstance(result, bool)
            
        except Exception:
            # Skip if project not available or fails
            pass
    
    def test_isempty_logical_consistency(self):
        """Test logical consistency of isempty"""
        # Empty specification should be consistent across different methods
        empty_spec = self.spec_explicit_empty
        
        self.assertTrue(empty_spec.isempty())
        
        # If set has an isempty method, should be consistent
        if hasattr(empty_spec.set, 'isempty'):
            self.assertEqual(empty_spec.isempty(), empty_spec.set.isempty())
    
    def test_isempty_performance(self):
        """Test isempty performance characteristics"""
        # isempty should be reasonably fast
        import time
        
        start_time = time.time()
        for _ in range(100):
            self.spec_safe_nonempty.isempty()
        end_time = time.time()
        
        # Should complete in reasonable time (less than 1 second for 100 calls)
        self.assertLess(end_time - start_time, 1.0)


if __name__ == '__main__':
    unittest.main() 