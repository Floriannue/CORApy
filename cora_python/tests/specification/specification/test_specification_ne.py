"""
Unit tests for specification ne method (!=)

This file tests the ne (not equal) functionality for temporal logic specifications.
Based on test_specification_ne.m from MATLAB CORA.

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


class TestSpecificationNe(unittest.TestCase):
    """Test cases for specification ne method (!=)"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create test specifications
        from cora_python.contSet.polytope.polytope import Polytope
        
        # 2D polytope: x1 >= 0, x2 >= 0, x1 + x2 <= 2
        A = np.array([[-1, 0], [0, -1], [1, 1]])
        b = np.array([0, 0, 2])
        self.set_2d = Polytope(A, b)
        
        # Different 2D polytope: x1 >= 0, x2 >= 0, x1 + x2 <= 1
        A_diff = np.array([[-1, 0], [0, -1], [1, 1]])
        b_diff = np.array([0, 0, 1])
        self.set_2d_diff = Polytope(A_diff, b_diff)
        
        # Time intervals
        self.time_interval = np.array([0, 10])
        self.time_interval_diff = np.array([0, 5])
        
        # Create identical specifications
        self.spec1 = Specification(self.set_2d, 'safeSet', self.time_interval)
        self.spec1_copy = Specification(self.set_2d, 'safeSet', self.time_interval)
        
        # Create different specifications
        self.spec_diff_set = Specification(self.set_2d_diff, 'safeSet', self.time_interval)
        self.spec_diff_type = Specification(self.set_2d, 'unsafeSet', self.time_interval)
        self.spec_diff_time = Specification(self.set_2d, 'safeSet', self.time_interval_diff)
        
        # Specifications without time
        self.spec_no_time1 = Specification(self.set_2d, 'safeSet')
        self.spec_no_time2 = Specification(self.set_2d, 'safeSet')
        self.spec_no_time_diff = Specification(self.set_2d_diff, 'safeSet')
    
    def test_ne_identical_specifications(self):
        """Test != for identical specifications"""
        # Identical specifications should not be "not equal"
        self.assertFalse(self.spec1 != self.spec1_copy)
        self.assertFalse(self.spec_no_time1 != self.spec_no_time2)
        
        # Self comparison
        self.assertFalse(self.spec1 != self.spec1)
    
    def test_ne_different_sets(self):
        """Test != for specifications with different sets"""
        # Different sets should be "not equal"
        self.assertTrue(self.spec1 != self.spec_diff_set)
        self.assertTrue(self.spec_no_time1 != self.spec_no_time_diff)
    
    def test_ne_different_types(self):
        """Test != for specifications with different types"""
        # Different types should be "not equal"
        self.assertTrue(self.spec1 != self.spec_diff_type)
    
    def test_ne_different_time(self):
        """Test != for specifications with different time intervals"""
        # Different time intervals should be "not equal"
        self.assertTrue(self.spec1 != self.spec_diff_time)
        
        # Specification with time vs without time
        self.assertTrue(self.spec1 != self.spec_no_time1)
    
    def test_ne_operator_consistency(self):
        """Test != operator consistency with == operator"""
        # != should be inverse of ==
        specs = [
            (self.spec1, self.spec1_copy),
            (self.spec1, self.spec_diff_set),
            (self.spec1, self.spec_diff_type),
            (self.spec1, self.spec_diff_time),
            (self.spec_no_time1, self.spec_no_time2),
        ]
        
        for spec_a, spec_b in specs:
            eq_result = spec_a == spec_b
            ne_result = spec_a != spec_b
            self.assertEqual(eq_result, not ne_result)
    
    def test_ne_return_type(self):
        """Test that != returns boolean"""
        result = self.spec1 != self.spec_diff_set
        self.assertIsInstance(result, bool)
        self.assertTrue(result)
        
        result2 = self.spec1 != self.spec1_copy
        self.assertIsInstance(result2, bool)
        self.assertFalse(result2)
    
    def test_ne_symmetry(self):
        """Test != symmetry property"""
        # a != b should equal b != a
        test_pairs = [
            (self.spec1, self.spec_diff_set),
            (self.spec1, self.spec_diff_type),
            (self.spec1, self.spec_diff_time),
            (self.spec_no_time1, self.spec1),
        ]
        
        for spec_a, spec_b in test_pairs:
            result_ab = spec_a != spec_b
            result_ba = spec_b != spec_a
            self.assertEqual(result_ab, result_ba)
    
    def test_ne_transitivity_property(self):
        """Test != transitivity-related properties"""
        # While != is not transitive, we can test consistency
        # If a != b and b != c, then we should have consistent behavior
        
        # All different specifications
        specs = [self.spec1, self.spec_diff_set, self.spec_diff_type]
        
        for i in range(len(specs)):
            for j in range(len(specs)):
                result = specs[i] != specs[j]
                if i == j:
                    self.assertFalse(result)  # Self should be equal
                else:
                    # Different specs should be not equal
                    self.assertTrue(result)
    
    def test_ne_with_invalid_types(self):
        """Test != with invalid comparison types"""
        # Compare with non-specification objects
        self.assertTrue(self.spec1 != "string")
        self.assertTrue(self.spec1 != 42)
        self.assertTrue(self.spec1 != None)
        self.assertTrue(self.spec1 != [1, 2, 3])
        
        # Should not raise errors, just return True
    
    def test_ne_all_specification_types(self):
        """Test != for all specification types"""
        types = ['safeSet', 'unsafeSet', 'finalSet', 'invariant']
        
        # Create specifications of each type
        specs = {}
        for spec_type in types:
            specs[spec_type] = Specification(self.set_2d, spec_type, self.time_interval)
        
        # Test != between different types
        for type1 in types:
            for type2 in types:
                result = specs[type1] != specs[type2]
                if type1 == type2:
                    self.assertFalse(result)  # Same type should be equal
                else:
                    self.assertTrue(result)   # Different types should be not equal
    
    def test_ne_complex_scenarios(self):
        """Test != for complex scenarios"""
        # Test with complex set combinations
        try:
            from cora_python.contSet.zonotope.zonotope import Zonotope
            
            # Zonotope specification
            Z = Zonotope(np.array([1, 1]), np.array([[1, 0], [0, 1]]))
            spec_zono = Specification(Z, 'safeSet', self.time_interval)
            
            # Should be different from polytope specifications
            self.assertTrue(self.spec1 != spec_zono)
            
            # Same zonotope should be equal
            spec_zono2 = Specification(Z, 'safeSet', self.time_interval)
            self.assertFalse(spec_zono != spec_zono2)
            
        except ImportError:
            # Skip if zonotope not available
            pass
    
    def test_ne_edge_cases(self):
        """Test != edge cases"""
        # Test with empty sets
        try:
            from cora_python.contSet.emptySet.emptySet import EmptySet
            
            empty_set1 = EmptySet(2)
            empty_set2 = EmptySet(2)
            
            spec_empty1 = Specification(empty_set1, 'safeSet', self.time_interval)
            spec_empty2 = Specification(empty_set2, 'safeSet', self.time_interval)
            
            # Two empty specifications should be equal (not "not equal")
            self.assertFalse(spec_empty1 != spec_empty2)
            
        except ImportError:
            # Skip if EmptySet not available
            pass
    
    def test_ne_performance(self):
        """Test != performance"""
        # != should be reasonably fast
        import time
        
        start_time = time.time()
        for _ in range(1000):
            _ = self.spec1 != self.spec_diff_set
        end_time = time.time()
        
        # Should complete in reasonable time
        self.assertLess(end_time - start_time, 1.0)
    
    def test_ne_consistency_with_isequal(self):
        """Test != consistency with isequal method"""
        # != should be inverse of isequal (if available)
        if hasattr(self.spec1, 'isequal'):
            test_pairs = [
                (self.spec1, self.spec1_copy),
                (self.spec1, self.spec_diff_set),
                (self.spec1, self.spec_diff_type),
            ]
            
            for spec_a, spec_b in test_pairs:
                isequal_result = spec_a.isequal(spec_b)
                ne_result = spec_a != spec_b
                self.assertEqual(isequal_result, not ne_result)


if __name__ == '__main__':
    unittest.main() 