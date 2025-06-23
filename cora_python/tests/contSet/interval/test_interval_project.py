"""
test_interval_project - unit tests for interval project method

Syntax:
    python -m pytest test_interval_project.py

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 17-September-2019 (MATLAB)
Python translation: 2025
"""

import unittest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalProject(unittest.TestCase):
    """Test cases for interval project method"""
    
    def test_project_basic(self):
        """Test basic projection functionality"""
        # create interval
        lower = np.array([[-3], [-9], [-4], [-7], [-1]])
        upper = np.array([[4], [2], [6], [3], [8]])
        Int = Interval(lower, upper)
        
        # project interval using dimension indices
        dimensions = [0, 2, 3]  # Python uses 0-based indexing
        I_proj1 = Int.project(dimensions)
        
        # true solution
        lower_true = np.array([[-3], [-4], [-7]])
        upper_true = np.array([[4], [6], [3]])
        I_true = Interval(lower_true, upper_true)
        
        # check if projected intervals are equal
        self.assertTrue(I_proj1.isequal(I_true))
    
    def test_project_logical_indexing(self):
        """Test projection using logical indexing"""
        # create interval
        lower = np.array([[-3], [-9], [-4], [-7], [-1]])
        upper = np.array([[4], [2], [6], [3], [8]])
        Int = Interval(lower, upper)
        
        # project interval using logical indexing
        dimensions = [True, False, True, True, False]
        I_proj2 = Int.project(dimensions)
        
        # true solution
        lower_true = np.array([[-3], [-4], [-7]])
        upper_true = np.array([[4], [6], [3]])
        I_true = Interval(lower_true, upper_true)
        
        # check if projected intervals are equal
        self.assertTrue(I_proj2.isequal(I_true))
    
    def test_project_single_dimension(self):
        """Test projection to single dimension"""
        # create interval
        lower = np.array([[-3], [-9], [-4]])
        upper = np.array([[4], [2], [6]])
        Int = Interval(lower, upper)
        
        # project to single dimension
        I_proj = Int.project([1])  # Second dimension (0-based)
        
        # true solution
        I_true = Interval(np.array([[-9]]), np.array([[2]]))
        
        # check if projected intervals are equal
        self.assertTrue(I_proj.isequal(I_true))
    
    def test_project_all_dimensions(self):
        """Test projection to all dimensions (should return same interval)"""
        # create interval
        lower = np.array([[-3], [-9]])
        upper = np.array([[4], [2]])
        Int = Interval(lower, upper)
        
        # project to all dimensions
        I_proj = Int.project([0, 1])
        
        # should be equal to original interval
        self.assertTrue(I_proj.isequal(Int))
    
    def test_project_empty_interval(self):
        """Test projection of empty interval"""
        # create empty interval
        Int = Interval.empty(5)
        
        # project empty interval
        I_proj = Int.project([0, 2, 3])
        
        # should still be empty with correct dimension
        self.assertTrue(I_proj.isemptyobject())
        self.assertEqual(I_proj.dim(), 3)
    
    def test_project_numpy_array_dimensions(self):
        """Test projection using numpy array for dimensions"""
        # create interval
        lower = np.array([[-3], [-9], [-4], [-7]])
        upper = np.array([[4], [2], [6], [3]])
        Int = Interval(lower, upper)
        
        # project using numpy array
        dimensions = np.array([0, 2])
        I_proj = Int.project(dimensions)
        
        # true solution
        lower_true = np.array([[-3], [-4]])
        upper_true = np.array([[4], [6]])
        I_true = Interval(lower_true, upper_true)
        
        # check if projected intervals are equal
        self.assertTrue(I_proj.isequal(I_true))


if __name__ == '__main__':
    unittest.main() 