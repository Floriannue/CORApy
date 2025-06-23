"""
test_interval_generateRandom - unit tests for interval generateRandom method

Syntax:
    python -m pytest test_interval_generateRandom.py

Authors: Mark Wetzlinger, Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 27-September-2019 (MATLAB)
Python translation: 2025
"""

import unittest
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.check import withinTol


class TestIntervalGenerateRandom(unittest.TestCase):
    """Test cases for interval generateRandom method"""
    
    def test_generateRandom_empty_call(self):
        """Test generateRandom with no arguments"""
        I = Interval.generateRandom()
        
        # Should create a valid interval
        self.assertIsInstance(I, Interval)
        self.assertGreater(I.dim(), 0)
        self.assertFalse(I.isemptyobject())
    
    def test_generateRandom_dimension_only(self):
        """Test generateRandom with dimension only"""
        n = 3
        I = Interval.generateRandom('Dimension', n)
        
        self.assertEqual(I.dim(), n)
        self.assertFalse(I.isemptyobject())
    
    def test_generateRandom_center_only(self):
        """Test generateRandom with center only"""
        c = np.array([[3], [2], [1]])
        I = Interval.generateRandom('Center', c)
        
        self.assertTrue(np.all(withinTol(I.center(), c, 1e-12)))
    
    def test_generateRandom_max_radius_only(self):
        """Test generateRandom with max radius only"""
        r = 2
        I = Interval.generateRandom('MaxRadius', r)
        
        self.assertLessEqual(np.max(I.rad()), r)
    
    def test_generateRandom_max_radius_per_dimension(self):
        """Test generateRandom with different max radius per dimension"""
        r_nd = np.array([[2], [3], [1]])
        I = Interval.generateRandom('MaxRadius', r_nd)
        
        self.assertTrue(np.all(r_nd >= I.rad()))
    
    def test_generateRandom_dimension_and_center(self):
        """Test generateRandom with dimension and center"""
        n = 3
        c = np.array([[3], [2], [1]])
        I = Interval.generateRandom('Dimension', n, 'Center', c)
        
        self.assertEqual(I.dim(), n)
        self.assertTrue(np.all(withinTol(I.center(), c, 1e-12)))
    
    def test_generateRandom_all_parameters(self):
        """Test generateRandom with dimension, center, and max radius"""
        n = 3
        c = np.array([[3], [2], [1]])
        r = 2
        I = Interval.generateRandom('Dimension', n, 'Center', c, 'MaxRadius', r)
        
        self.assertEqual(I.dim(), n)
        self.assertTrue(np.all(withinTol(I.center(), c, 1e-12)))
        self.assertLessEqual(np.max(I.rad()), r)
    
    def test_generateRandom_dimension_center_mismatch(self):
        """Test generateRandom with mismatched dimension and center"""
        # dimension and center don't match
        with self.assertRaises(Exception):  # Should throw error
            Interval.generateRandom('Dimension', 2, 'Center', np.ones((3, 1)))
    
    def test_generateRandom_center_wrong_shape(self):
        """Test generateRandom with wrong center shape"""
        # dimension and center don't match (wrong shape)
        with self.assertRaises(Exception):  # Should throw error
            Interval.generateRandom('Dimension', 3, 'Center', np.ones((3, 2)))
    
    def test_generateRandom_matrix_dimension(self):
        """Test generateRandom with matrix dimensions"""
        n = [2, 3]
        I = Interval.generateRandom('Dimension', n)
        
        # Check that dimensions match
        expected_shape = tuple(n)
        self.assertEqual(I.inf.shape, expected_shape)
        self.assertEqual(I.sup.shape, expected_shape)
    
    def test_generateRandom_matrix_center(self):
        """Test generateRandom with matrix center"""
        c = np.array([[2, 5, 4], [1, -1, 2]])
        I = Interval.generateRandom('Center', c)
        
        self.assertTrue(np.all(withinTol(I.center(), c, 1e-12)))
    
    def test_generateRandom_matrix_max_radius(self):
        """Test generateRandom with matrix max radius"""
        r = 4
        I = Interval.generateRandom('MaxRadius', r)
        
        self.assertLessEqual(np.max(I.rad()), r)
    
    def test_generateRandom_matrix_max_radius_per_element(self):
        """Test generateRandom with different max radius per matrix element"""
        r_nd = np.array([[1, 0.5, 2], [2, 5, 2]])
        I = Interval.generateRandom('MaxRadius', r_nd)
        
        self.assertTrue(np.all(r_nd >= I.rad()))
    
    def test_generateRandom_matrix_all_parameters(self):
        """Test generateRandom with all matrix parameters"""
        n = [2, 3]
        c = np.array([[2, 5, 4], [1, -1, 2]])
        r = 4
        I = Interval.generateRandom('Dimension', n, 'Center', c, 'MaxRadius', r)
        
        expected_shape = tuple(n)
        self.assertEqual(I.inf.shape, expected_shape)
        self.assertEqual(I.sup.shape, expected_shape)
        self.assertTrue(np.all(withinTol(I.center(), c, 1e-12)))
        self.assertLessEqual(np.max(I.rad()), r)
    
    def test_generateRandom_nd_arrays(self):
        """Test generateRandom with n-dimensional arrays"""
        # 3D array
        dims = [2, 2, 3]
        I = Interval.generateRandom('Dimension', dims)
        
        expected_shape = tuple(dims)
        self.assertEqual(I.inf.shape, expected_shape)
        self.assertEqual(I.sup.shape, expected_shape)
        
        # 4D array
        dims = [2, 1, 3, 4]
        I = Interval.generateRandom('Dimension', dims)
        
        expected_shape = tuple(dims)
        self.assertEqual(I.inf.shape, expected_shape)
        self.assertEqual(I.sup.shape, expected_shape)
    
    def test_generateRandom_reproducibility(self):
        """Test that generateRandom produces different results on repeated calls"""
        I1 = Interval.generateRandom('Dimension', 3)
        I2 = Interval.generateRandom('Dimension', 3)
        
        # Should be different intervals (with very high probability)
        self.assertFalse(I1.isequal(I2))


if __name__ == '__main__':
    unittest.main() 