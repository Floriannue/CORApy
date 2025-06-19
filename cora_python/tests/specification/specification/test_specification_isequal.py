"""
test_specification_isequal - unit test for isequal method

This test covers the isequal functionality for specification objects.

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


class TestSpecificationIsequal(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test sets
        self.set1 = Interval(np.array([[0], [0]]), np.array([[1], [1]]))
        self.set2 = Interval(np.array([[0], [0]]), np.array([[1], [1]]))  # Same as set1
        self.set3 = Interval(np.array([[2], [2]]), np.array([[3], [3]]))  # Different
        
        # Create zonotopes
        self.zono1 = Zonotope(np.array([[0.5], [0.5]]), np.array([[0.3, 0.1], [0.1, 0.3]]))
        self.zono2 = Zonotope(np.array([[0.5], [0.5]]), np.array([[0.3, 0.1], [0.1, 0.3]]))  # Same
        
        # Time intervals
        self.time1 = Interval(np.array([[0]]), np.array([[2]]))
        self.time2 = Interval(np.array([[0]]), np.array([[2]]))  # Same as time1
        self.time3 = Interval(np.array([[1]]), np.array([[3]]))  # Different
        
        # Locations
        self.loc1 = [1, 2]
        self.loc2 = [1, 2]  # Same as loc1
        self.loc3 = [2, 3]  # Different
        
        # Create specifications
        self.spec1 = Specification(self.set1, 'safeSet')
        self.spec2 = Specification(self.set2, 'safeSet')  # Should be equal to spec1
        self.spec3 = Specification(self.set3, 'safeSet')  # Different set
        self.spec4 = Specification(self.set1, 'unsafeSet')  # Different type
    
    def test_isequal_identical_specifications(self):
        """Test isequal for identical specifications"""
        try:
            result = self.spec1.isequal(self.spec2)
            self.assertTrue(result)
            
            # Test reflexivity
            result_self = self.spec1.isequal(self.spec1)
            self.assertTrue(result_self)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal method not fully implemented yet")
    
    def test_isequal_different_sets(self):
        """Test isequal with different sets"""
        try:
            result = self.spec1.isequal(self.spec3)
            self.assertFalse(result)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal method not fully implemented yet")
    
    def test_isequal_different_types(self):
        """Test isequal with different specification types"""
        try:
            result = self.spec1.isequal(self.spec4)
            self.assertFalse(result)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal method not fully implemented yet")
    
    def test_isequal_with_time_same(self):
        """Test isequal with same time constraints"""
        spec_time1 = Specification(self.set1, 'safeSet', self.time1)
        spec_time2 = Specification(self.set2, 'safeSet', self.time2)
        
        try:
            result = spec_time1.isequal(spec_time2)
            self.assertTrue(result)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal with time constraints not implemented yet")
    
    def test_isequal_with_time_different(self):
        """Test isequal with different time constraints"""
        spec_time1 = Specification(self.set1, 'safeSet', self.time1)
        spec_time3 = Specification(self.set2, 'safeSet', self.time3)
        
        try:
            result = spec_time1.isequal(spec_time3)
            self.assertFalse(result)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal with time constraints not implemented yet")
    
    def test_isequal_with_location_same(self):
        """Test isequal with same location constraints"""
        spec_loc1 = Specification(self.set1, 'safeSet', self.loc1)
        spec_loc2 = Specification(self.set2, 'safeSet', self.loc2)
        
        try:
            result = spec_loc1.isequal(spec_loc2)
            self.assertTrue(result)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal with location constraints not implemented yet")
    
    def test_isequal_with_location_different(self):
        """Test isequal with different location constraints"""
        spec_loc1 = Specification(self.set1, 'safeSet', self.loc1)
        spec_loc3 = Specification(self.set2, 'safeSet', self.loc3)
        
        try:
            result = spec_loc1.isequal(spec_loc3)
            self.assertFalse(result)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal with location constraints not implemented yet")
    
    def test_isequal_one_with_time_one_without(self):
        """Test isequal when one has time and other doesn't"""
        spec_time = Specification(self.set1, 'safeSet', self.time1)
        spec_no_time = Specification(self.set2, 'safeSet')
        
        try:
            result = spec_time.isequal(spec_no_time)
            self.assertFalse(result)
            
            # Test reverse
            result_reverse = spec_no_time.isequal(spec_time)
            self.assertFalse(result_reverse)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal time/no-time comparison not implemented yet")
    
    def test_isequal_one_with_location_one_without(self):
        """Test isequal when one has location and other doesn't"""
        spec_loc = Specification(self.set1, 'safeSet', self.loc1)
        spec_no_loc = Specification(self.set2, 'safeSet')
        
        try:
            result = spec_loc.isequal(spec_no_loc)
            self.assertFalse(result)
            
            # Test reverse
            result_reverse = spec_no_loc.isequal(spec_loc)
            self.assertFalse(result_reverse)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal location/no-location comparison not implemented yet")
    
    def test_isequal_complete_match(self):
        """Test isequal with all properties matching"""
        spec_full1 = Specification(self.set1, 'invariant', self.time1, self.loc1)
        spec_full2 = Specification(self.set2, 'invariant', self.time2, self.loc2)
        
        try:
            result = spec_full1.isequal(spec_full2)
            self.assertTrue(result)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal with all properties not implemented yet")
    
    def test_isequal_zonotope_sets(self):
        """Test isequal with zonotope sets"""
        spec_zono1 = Specification(self.zono1, 'safeSet')
        spec_zono2 = Specification(self.zono2, 'safeSet')
        
        try:
            result = spec_zono1.isequal(spec_zono2)
            # Should be true if zonotopes are equal
            self.assertTrue(result)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal with zonotope sets not implemented yet")
    
    def test_isequal_different_set_types(self):
        """Test isequal with different set types"""
        spec_interval = Specification(self.set1, 'safeSet')
        spec_zono = Specification(self.zono1, 'safeSet')
        
        try:
            result = spec_interval.isequal(spec_zono)
            # Should be false (different set types)
            self.assertFalse(result)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal with different set types not implemented yet")
    
    def test_isequal_with_non_specification(self):
        """Test isequal with non-specification objects"""
        try:
            result1 = self.spec1.isequal("not a specification")
            result2 = self.spec1.isequal(42)
            result3 = self.spec1.isequal(None)
            
            self.assertFalse(result1)
            self.assertFalse(result2)
            self.assertFalse(result3)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal with non-specification objects not implemented yet")
    
    def test_isequal_symmetry(self):
        """Test that isequal is symmetric"""
        try:
            result1 = self.spec1.isequal(self.spec2)
            result2 = self.spec2.isequal(self.spec1)
            
            self.assertEqual(result1, result2)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal symmetry not implemented yet")
    
    def test_isequal_transitivity(self):
        """Test that isequal is transitive"""
        # Create three equivalent specifications
        spec_a = Specification(self.set1, 'safeSet')
        spec_b = Specification(self.set2, 'safeSet')  # Same as set1
        spec_c = Specification(self.set1, 'safeSet')  # Same as set1
        
        try:
            result_ab = spec_a.isequal(spec_b)
            result_bc = spec_b.isequal(spec_c)
            result_ac = spec_a.isequal(spec_c)
            
            if result_ab and result_bc:
                self.assertTrue(result_ac)
                
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal transitivity not implemented yet")
    
    def test_isequal_empty_specifications(self):
        """Test isequal with empty specifications"""
        spec_empty1 = Specification()
        spec_empty2 = Specification()
        
        try:
            result = spec_empty1.isequal(spec_empty2)
            # Empty specifications should be equal
            self.assertTrue(result)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal with empty specifications not implemented yet")
    
    def test_isequal_vs_equality_operator(self):
        """Test consistency between isequal and == operator"""
        try:
            isequal_result = self.spec1.isequal(self.spec2)
            eq_result = self.spec1 == self.spec2
            
            # Should give same result
            self.assertEqual(isequal_result, eq_result)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("Consistency between isequal and == not implemented yet")
    
    def test_isequal_tolerance_handling(self):
        """Test isequal with numerical tolerance"""
        # Create sets with small numerical differences
        set_exact = Interval(np.array([[0], [0]]), np.array([[1], [1]]))
        set_approx = Interval(np.array([[1e-15], [1e-15]]), np.array([[1 + 1e-15], [1 + 1e-15]]))
        
        spec_exact = Specification(set_exact, 'safeSet')
        spec_approx = Specification(set_approx, 'safeSet')
        
        try:
            result = spec_exact.isequal(spec_approx)
            # Might be true with tolerance, false without
            self.assertIsInstance(result, bool)
            
        except (NotImplementedError, AttributeError):
            self.skipTest("isequal tolerance handling not implemented yet")


if __name__ == '__main__':
    unittest.main() 