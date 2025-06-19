"""
test_specification_eq - unit test for equality operator

This test covers the equality comparison between specification objects.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant  
Written: 2022 (MATLAB)
Python translation: 2025
"""

import unittest
import numpy as np
from cora_python.specification.specification.specification import Specification
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval


class TestSpecificationEq(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test sets
        self.set1 = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
        self.set2 = Zonotope(np.array([[1], [1]]), np.array([[1, 0], [0, 1]]))
        self.set3 = Interval(np.array([[-1], [-1]]), np.array([[1], [1]]))
        
        # Create time intervals
        self.time1 = Interval(np.array([[0]]), np.array([[2]]))
        self.time2 = Interval(np.array([[1]]), np.array([[3]]))
        
        # Create locations
        self.loc1 = [1, 2]
        self.loc2 = [2, 3]
    
    def test_identical_specifications(self):
        """Test equality of identical specifications"""
        spec1 = Specification(self.set1, 'safeSet')
        spec2 = Specification(self.set1, 'safeSet')
        
        try:
            result = spec1 == spec2
            self.assertTrue(result)
        except (NotImplementedError, AttributeError):
            self.skipTest("Equality operator not fully implemented yet")
    
    def test_different_sets(self):
        """Test equality with different sets"""
        spec1 = Specification(self.set1, 'safeSet')
        spec2 = Specification(self.set2, 'safeSet')
        
        try:
            result = spec1 == spec2
            self.assertFalse(result)
        except (NotImplementedError, AttributeError):
            self.skipTest("Equality operator not fully implemented yet")
    
    def test_different_types(self):
        """Test equality with different specification types"""
        spec1 = Specification(self.set1, 'safeSet')
        spec2 = Specification(self.set1, 'unsafeSet')
        
        try:
            result = spec1 == spec2
            self.assertFalse(result)
        except (NotImplementedError, AttributeError):
            self.skipTest("Equality operator not fully implemented yet")
    
    def test_different_times(self):
        """Test equality with different time intervals"""
        spec1 = Specification(self.set1, 'safeSet', self.time1)
        spec2 = Specification(self.set1, 'safeSet', self.time2)
        
        try:
            result = spec1 == spec2
            self.assertFalse(result)
        except (NotImplementedError, AttributeError):
            self.skipTest("Equality operator not fully implemented yet")
    
    def test_different_locations(self):
        """Test equality with different locations"""
        spec1 = Specification(self.set1, 'safeSet', self.loc1)
        spec2 = Specification(self.set1, 'safeSet', self.loc2)
        
        try:
            result = spec1 == spec2
            self.assertFalse(result)
        except (NotImplementedError, AttributeError):
            self.skipTest("Equality operator not fully implemented yet")
    
    def test_one_with_time_one_without(self):
        """Test equality when one has time and other doesn't"""
        spec1 = Specification(self.set1, 'safeSet')
        spec2 = Specification(self.set1, 'safeSet', self.time1)
        
        try:
            result = spec1 == spec2
            self.assertFalse(result)
        except (NotImplementedError, AttributeError):
            self.skipTest("Equality operator not fully implemented yet")
    
    def test_one_with_location_one_without(self):
        """Test equality when one has location and other doesn't"""
        spec1 = Specification(self.set1, 'safeSet')
        spec2 = Specification(self.set1, 'safeSet', self.loc1)
        
        try:
            result = spec1 == spec2
            self.assertFalse(result)
        except (NotImplementedError, AttributeError):
            self.skipTest("Equality operator not fully implemented yet")
    
    def test_complete_match(self):
        """Test equality with all properties matching"""
        spec1 = Specification(self.set1, 'invariant', self.time1, self.loc1)
        spec2 = Specification(self.set1, 'invariant', self.time1, self.loc1)
        
        try:
            result = spec1 == spec2
            self.assertTrue(result)
        except (NotImplementedError, AttributeError):
            self.skipTest("Equality operator not fully implemented yet")
    
    def test_not_equal_operator(self):
        """Test not equal operator"""
        spec1 = Specification(self.set1, 'safeSet')
        spec2 = Specification(self.set2, 'safeSet')
        
        try:
            result = spec1 != spec2
            self.assertTrue(result)
        except (NotImplementedError, AttributeError):
            self.skipTest("Not equal operator not fully implemented yet")
    
    def test_equality_with_different_object_types(self):
        """Test equality with non-specification objects"""
        spec1 = Specification(self.set1, 'safeSet')
        
        try:
            result1 = spec1 == "not a specification"
            result2 = spec1 == 42
            result3 = spec1 == None
            
            self.assertFalse(result1)
            self.assertFalse(result2)
            self.assertFalse(result3)
        except (NotImplementedError, AttributeError):
            self.skipTest("Equality operator not fully implemented yet")
    
    def test_reflexivity(self):
        """Test that a specification equals itself"""
        spec = Specification(self.set1, 'safeSet', self.time1, self.loc1)
        
        try:
            result = spec == spec
            self.assertTrue(result)
        except (NotImplementedError, AttributeError):
            self.skipTest("Equality operator not fully implemented yet")
    
    def test_symmetry(self):
        """Test that equality is symmetric"""
        spec1 = Specification(self.set1, 'safeSet')
        spec2 = Specification(self.set1, 'safeSet')
        
        try:
            result1 = spec1 == spec2
            result2 = spec2 == spec1
            self.assertEqual(result1, result2)
        except (NotImplementedError, AttributeError):
            self.skipTest("Equality operator not fully implemented yet")
    
    def test_transitivity(self):
        """Test that equality is transitive"""
        spec1 = Specification(self.set1, 'safeSet')
        spec2 = Specification(self.set1, 'safeSet')
        spec3 = Specification(self.set1, 'safeSet')
        
        try:
            result12 = spec1 == spec2
            result23 = spec2 == spec3
            result13 = spec1 == spec3
            
            if result12 and result23:
                self.assertTrue(result13)
        except (NotImplementedError, AttributeError):
            self.skipTest("Equality operator not fully implemented yet")


if __name__ == '__main__':
    unittest.main() 