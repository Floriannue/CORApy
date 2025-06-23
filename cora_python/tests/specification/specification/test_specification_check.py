"""
Test specification check functionality

Tests the check method for specifications according to MATLAB behavior.
Tests all specification types with proper geometric verification.

Authors: Python test by AI Assistant
Written: 2025
"""

import unittest
import numpy as np
from cora_python.specification.specification.specification import Specification
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval


class TestSpecificationCheck(unittest.TestCase):
    """Test class for specification check functionality"""

    def setUp(self):
        """Set up test fixtures with MATLAB test data"""
        # Test data from MATLAB: set = polytope([1 1]/sqrt(2),1);
        A = np.array([[1, 1]]) / np.sqrt(2)  # [0.707, 0.707]
        b = np.array([1])
        self.set_spec = Polytope(A, b)
        
        # Test data from MATLAB: Z = zonotope([-10;-8],[1 0 -2; 2 -1 1]);
        self.Z = Zonotope(np.array([[-10], [-8]]), np.array([[1, 0, -2], [2, -1, 1]]))
        
        # Verify the geometric truth: All zonotope vertices should satisfy constraint
        vertices = self.Z.vertices()
        constraint_values = np.dot(A[0], vertices)  # [0.707, 0.707] * vertices
        self.assertTrue(np.all(constraint_values <= 1.0 + 1e-10), 
                       f"All vertices should satisfy constraint, got: {constraint_values}")

    def test_check_safeSet_contained(self):
        """Test safeSet specification with contained zonotope"""
        # Create safeSet specification
        spec = Specification(self.set_spec, 'safeSet')
        
        # Check the zonotope - should be True since zonotope is contained in polytope
        res, indSpec, indObj = spec.check(self.Z)
        
        # The zonotope is contained in the polytope, so it should be safe
        self.assertTrue(res, "Zonotope should be contained in polytope (safe)")
        self.assertEqual(indSpec, 0, "No specification should be violated")
        self.assertEqual(indObj, 1, "No object should be violated")

    def test_check_safeSet_not_contained(self):
        """Test safeSet specification with non-contained set"""
        # Create a zonotope that violates the constraint
        # Center at [2, 2] which gives constraint value 2.828 > 1
        Z_outside = Zonotope(np.array([[2], [2]]), np.array([[0.1, 0], [0, 0.1]]))
        
        # Verify it's actually outside
        vertices = Z_outside.vertices()
        constraint_values = np.dot(np.array([1, 1]) / np.sqrt(2), vertices)
        self.assertTrue(np.all(constraint_values > 1.0), 
                       f"All vertices should violate constraint, got: {constraint_values}")
        
        # Create safeSet specification  
        spec = Specification(self.set_spec, 'safeSet')
        
        # Check the zonotope - should be False since zonotope is not contained
        res, indSpec, indObj = spec.check(Z_outside)
        
        self.assertFalse(res, "Zonotope should not be contained in polytope (unsafe)")
        self.assertEqual(indSpec, 1, "First specification should be violated")
        self.assertEqual(indObj, 1, "First object should be violated")

    def test_check_unsafeSet_no_intersection(self):
        """Test unsafeSet specification with non-intersecting zonotope"""
        # Create unsafeSet specification
        spec = Specification(self.set_spec, 'unsafeSet')
        
        # The zonotope is contained in the polytope, so it DOES intersect
        # According to MATLAB: res = ~isIntersecting_(set,S,'exact',1e-8)
        # Since there IS intersection, result should be False (unsafe)
        res, indSpec, indObj = spec.check(self.Z)
        
        self.assertFalse(res, "Contained zonotope intersects with unsafe set (unsafe)")
        self.assertEqual(indSpec, 1, "First specification should be violated")
        self.assertEqual(indObj, 1, "First object should be violated")

    def test_check_unsafeSet_no_intersection_disjoint(self):
        """Test unsafeSet specification with truly disjoint sets"""
        # Create a zonotope far away that doesn't intersect
        Z_far = Zonotope(np.array([[10], [10]]), np.array([[0.1, 0], [0, 0.1]]))
        
        # Verify it's actually outside and doesn't intersect
        vertices = Z_far.vertices()
        constraint_values = np.dot(np.array([1, 1]) / np.sqrt(2), vertices)
        self.assertTrue(np.all(constraint_values > 1.0), 
                       f"All vertices should be outside constraint, got: {constraint_values}")
        
        # Create unsafeSet specification
        spec = Specification(self.set_spec, 'unsafeSet')
        
        # Check the zonotope - should be True since no intersection
        res, indSpec, indObj = spec.check(Z_far)
        
        self.assertTrue(res, "Disjoint zonotope should not intersect unsafe set (safe)")
        self.assertEqual(indSpec, 0, "No specification should be violated")
        self.assertEqual(indObj, 1, "No object should be violated")

    def test_check_invariant_intersecting(self):
        """Test invariant specification with intersecting zonotope"""
        # Create invariant specification
        spec = Specification(self.set_spec, 'invariant')
        
        # The zonotope intersects (is contained in) the polytope
        # According to MATLAB: res = isIntersecting_(set,S,'approx',1e-8)
        # Since there IS intersection, result should be True
        res, indSpec, indObj = spec.check(self.Z)
        
        self.assertTrue(res, "Zonotope should intersect with invariant set")
        self.assertEqual(indSpec, 0, "No specification should be violated")
        self.assertEqual(indObj, 1, "No object should be violated")

    def test_check_invariant_not_intersecting(self):
        """Test invariant specification with non-intersecting zonotope"""
        # Create a zonotope far away that doesn't intersect
        Z_far = Zonotope(np.array([[10], [10]]), np.array([[0.1, 0], [0, 0.1]]))
        
        # Create invariant specification
        spec = Specification(self.set_spec, 'invariant')
        
        # Check the zonotope - should be False since no intersection
        res, indSpec, indObj = spec.check(Z_far)
        
        self.assertFalse(res, "Disjoint zonotope should not intersect invariant set")
        self.assertEqual(indSpec, 1, "First specification should be violated")
        self.assertEqual(indObj, 1, "First object should be violated")

    def test_check_multiple_specifications(self):
        """Test checking multiple specifications"""
        # Create multiple specifications
        spec1 = Specification(self.set_spec, 'safeSet')  # Should pass
        
        # Create a second polytope that the zonotope doesn't intersect
        A2 = np.array([[1, 0]])  # x <= 5
        b2 = np.array([5])
        set_spec2 = Polytope(A2, b2)
        spec2 = Specification(set_spec2, 'invariant')  # Should pass (zonotope intersects)
        
        specs = [spec1, spec2]
        
        # Check all specifications
        res, indSpec, indObj = specs[0].check(self.Z)  # Check first spec
        self.assertTrue(res, "First specification should pass")
        
        res, indSpec, indObj = specs[1].check(self.Z)  # Check second spec  
        self.assertTrue(res, "Second specification should pass")

    def test_check_point_input(self):
        """Test check with point input"""
        # Create safeSet specification
        spec = Specification(self.set_spec, 'safeSet')
        
        # Test point inside the set
        point_inside = np.array([[-1], [-1]])  # Constraint: 0.707*(-1) + 0.707*(-1) = -1.414 <= 1 ✓
        constraint_val = np.dot(np.array([1, 1]) / np.sqrt(2), point_inside.flatten())
        self.assertLessEqual(constraint_val, 1.0, f"Point should be inside, constraint value: {constraint_val}")
        
        res, indSpec, indObj = spec.check(point_inside)
        self.assertTrue(res, "Point inside polytope should satisfy safeSet")
        
        # Test point outside the set
        point_outside = np.array([[2], [2]])  # Constraint: 0.707*2 + 0.707*2 = 2.828 > 1 ✗
        constraint_val = np.dot(np.array([1, 1]) / np.sqrt(2), point_outside.flatten())
        self.assertGreater(constraint_val, 1.0, f"Point should be outside, constraint value: {constraint_val}")
        
        res, indSpec, indObj = spec.check(point_outside)
        self.assertFalse(res, "Point outside polytope should violate safeSet")

    def test_check_custom_specification(self):
        """Test check with custom function specification"""
        # Custom function that checks if center is in third quadrant
        def custom_func(S):
            if hasattr(S, 'center'):
                center = S.center()
                return center[0] < 0 and center[1] < 0
            elif isinstance(S, np.ndarray):
                return S[0] < 0 and S[1] < 0
            return False
        
        spec = Specification(custom_func, 'custom')
        
        # Test with zonotope (center [-10, -8] in third quadrant)
        res, indSpec, indObj = spec.check(self.Z)
        self.assertTrue(res, "Zonotope center should be in third quadrant")
        
        # Test with point in first quadrant  
        point_first = np.array([[1], [1]])
        res, indSpec, indObj = spec.check(point_first)
        self.assertFalse(res, "Point in first quadrant should violate custom spec")


if __name__ == '__main__':
    unittest.main() 