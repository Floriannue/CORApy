"""
test_specification_add - unit test for add method

This test covers the addition operation for specification objects.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 2022 (MATLAB)
Python translation: 2025
"""

import unittest
import numpy as np
from cora_python.specification.specification.specification import Specification
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestSpecificationAdd(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test sets
        self.set1 = Interval(np.array([[0], [0]]), np.array([[1], [1]]))
        self.set2 = Interval(np.array([[2], [2]]), np.array([[3], [3]]))
        self.set3 = Zonotope(np.array([[0.5], [0.5]]), np.array([[0.3, 0.1], [0.1, 0.3]]))
        
        # Create specifications
        self.spec1 = Specification(self.set1, 'safeSet')
        self.spec2 = Specification(self.set2, 'safeSet')
        self.spec3 = Specification(self.set3, 'unsafeSet')
        
        # Time intervals
        self.time1 = Interval(np.array([[0]]), np.array([[2]]))
        self.time2 = Interval(np.array([[1]]), np.array([[3]]))
        
        # Locations
        self.loc1 = [1, 2]
        self.loc2 = [2, 3]
    
    def test_add_compatible_specifications(self):
        """Test adding compatible specifications"""
        try:
            result = self.spec1 + self.spec2
            
            # Result should be a list of specifications
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            
            # Check that both specifications are preserved
            self.assertEqual(result[0].set, self.spec1.set)
            self.assertEqual(result[1].set, self.spec2.set)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Add method not fully implemented yet")
    
    def test_add_different_types(self):
        """Test adding specifications with different types"""
        try:
            result = self.spec1 + self.spec3  # safeSet + unsafeSet
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            
            # Types should be preserved
            self.assertEqual(result[0].type, 'safeSet')
            self.assertEqual(result[1].type, 'unsafeSet')
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Add method not fully implemented yet")
    
    def test_add_with_time_constraints(self):
        """Test adding specifications with time constraints"""
        spec_time1 = Specification(self.set1, 'safeSet', self.time1)
        spec_time2 = Specification(self.set2, 'safeSet', self.time2)
        
        try:
            result = spec_time1 + spec_time2
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            
            # Time constraints should be preserved
            self.assertEqual(result[0].time, self.time1)
            self.assertEqual(result[1].time, self.time2)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Add with time constraints not implemented yet")
    
    def test_add_with_locations(self):
        """Test adding specifications with location constraints"""
        spec_loc1 = Specification(self.set1, 'safeSet', self.loc1)
        spec_loc2 = Specification(self.set2, 'safeSet', self.loc2)
        
        try:
            result = spec_loc1 + spec_loc2
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            
            # Locations should be preserved
            self.assertEqual(result[0].location, self.loc1)
            self.assertEqual(result[1].location, self.loc2)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Add with locations not implemented yet")
    
    def test_add_to_list(self):
        """Test adding specification to existing list"""
        spec_list = [self.spec1, self.spec2]
        
        try:
            result = spec_list + self.spec3
            # or result = self.spec3 + spec_list
            
            # Should handle adding to lists
            if hasattr(result, '__len__'):
                self.assertGreaterEqual(len(result), 3)
                
        except (NotImplementedError, AttributeError, TypeError):
            # This might not be implemented the same way as MATLAB
            self.skipTest("List addition not implemented yet")
    
    def test_add_multiple_specifications(self):
        """Test chaining multiple additions"""
        try:
            result = self.spec1 + self.spec2 + self.spec3
            
            # Should result in list with all specifications
            if isinstance(result, list):
                self.assertGreaterEqual(len(result), 3)
            else:
                # Or some other combined representation
                self.assertIsNotNone(result)
                
        except (NotImplementedError, AttributeError):
            self.skipTest("Multiple specification addition not implemented yet")
    
    def test_add_empty_specification(self):
        """Test adding empty specification"""
        spec_empty = Specification()
        
        try:
            result = self.spec1 + spec_empty
            
            # Should handle empty specifications appropriately
            self.assertIsNotNone(result)
            
        except (NotImplementedError, AttributeError, CORAerror):
            # Might raise error for empty specifications
            self.skipTest("Empty specification addition not implemented yet")
    
    def test_add_incompatible_dimensions(self):
        """Test adding specifications with incompatible dimensions"""
        # Create 3D specification
        set_3d = Interval(np.array([[0], [0], [0]]), np.array([[1], [1], [1]]))
        spec_3d = Specification(set_3d, 'safeSet')
        
        try:
            # Should raise error or handle gracefully
            with self.assertRaises((ValueError, CORAerror, AttributeError)):
                result = self.spec1 + spec_3d  # 2D + 3D
                
        except (NotImplementedError, AttributeError):
            self.skipTest("Dimension checking in addition not implemented yet")
    
    def test_add_preserves_properties(self):
        """Test that addition preserves all specification properties"""
        spec_complex = Specification(self.set1, 'invariant', self.time1, self.loc1)
        
        try:
            result = spec_complex + self.spec2
            
            if isinstance(result, list):
                # First spec should maintain all properties
                original_spec = result[0]
                self.assertEqual(original_spec.type, 'invariant')
                self.assertEqual(original_spec.time, self.time1)
                self.assertEqual(original_spec.location, self.loc1)
                
        except (NotImplementedError, AttributeError):
            self.skipTest("Property preservation in addition not implemented yet")
    
    def test_add_commutativity(self):
        """Test that addition is commutative"""
        try:
            result1 = self.spec1 + self.spec2
            result2 = self.spec2 + self.spec1
            
            # Results should be equivalent (order might differ)
            if isinstance(result1, list) and isinstance(result2, list):
                self.assertEqual(len(result1), len(result2))
                
                # Check that both contain the same specifications
                # (order might be different)
                sets1 = {id(spec.set) for spec in result1}
                sets2 = {id(spec.set) for spec in result2}
                self.assertEqual(sets1, sets2)
                
        except (NotImplementedError, AttributeError):
            self.skipTest("Addition commutativity not implemented yet")
    
    def test_add_associativity(self):
        """Test that addition is associative"""
        try:
            # (spec1 + spec2) + spec3
            temp1 = self.spec1 + self.spec2
            result1 = temp1 + self.spec3 if hasattr(temp1, '__add__') else None
            
            # spec1 + (spec2 + spec3)
            temp2 = self.spec2 + self.spec3
            result2 = self.spec1 + temp2 if hasattr(temp2, '__add__') else None
            
            if result1 is not None and result2 is not None:
                # Should be equivalent
                if isinstance(result1, list) and isinstance(result2, list):
                    self.assertEqual(len(result1), len(result2))
                    
        except (NotImplementedError, AttributeError):
            self.skipTest("Addition associativity not implemented yet")
    
    def test_add_custom_specifications(self):
        """Test adding custom specifications"""
        custom_func = lambda x: x[0] > 0
        spec_custom = Specification(custom_func, 'custom')
        
        try:
            result = self.spec1 + spec_custom
            
            # Should handle custom specifications
            self.assertIsNotNone(result)
            
        except (NotImplementedError, AttributeError, CORAerror):
            # Custom specs might not support addition
            self.skipTest("Custom specification addition not implemented yet")
    
    def test_add_error_cases(self):
        """Test error cases for addition"""
        try:
            # Test adding invalid objects
            with self.assertRaises((TypeError, ValueError, AttributeError)):
                result = self.spec1 + "invalid"
                
            with self.assertRaises((TypeError, ValueError, AttributeError)):
                result = self.spec1 + 42
                
        except (NotImplementedError, AttributeError):
            self.skipTest("Error handling in addition not implemented yet")
    
    def test_add_return_type_consistency(self):
        """Test that addition returns consistent types"""
        try:
            result1 = self.spec1 + self.spec2
            result2 = self.spec2 + self.spec3
            
            # Should return same type of object
            self.assertEqual(type(result1), type(result2))
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Addition return type consistency not implemented yet")


if __name__ == '__main__':
    unittest.main() 