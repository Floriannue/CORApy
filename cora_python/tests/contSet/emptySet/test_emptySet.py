import unittest
import numpy as np
from cora_python.contSet.emptySet.emptySet import EmptySet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestEmptySet(unittest.TestCase):
    """Test cases for EmptySet class constructor and basic methods."""

    def test_constructor_basic(self):
        """Test basic emptySet constructor with dimension."""
        n = 2
        E = EmptySet(n)
        
        self.assertEqual(E.dimension, n)

    def test_constructor_0d(self):
        """Test emptySet constructor with 0 dimension."""
        E = EmptySet(0)
        
        self.assertEqual(E.dimension, 0)

    def test_constructor_3d(self):
        """Test emptySet constructor for 3D."""
        n = 3
        E = EmptySet(n)
        
        self.assertEqual(E.dimension, n)

    def test_constructor_1d(self):
        """Test emptySet constructor for 1D."""
        n = 1
        E = EmptySet(n)
        
        self.assertEqual(E.dimension, n)

    def test_constructor_high_dimension(self):
        """Test emptySet constructor for high dimension."""
        n = 10
        E = EmptySet(n)
        
        self.assertEqual(E.dimension, n)

    def test_isemptyobject(self):
        """Test isemptyobject method."""
        E = EmptySet(2)
        
        # According to MATLAB implementation, isemptyobject always returns false for emptySet
        # The emptySet object itself is not "empty" - it represents the empty set
        self.assertFalse(E.isemptyobject())

    def test_mtimes(self):
        """Test mtimes method - should return self."""
        E = EmptySet(2)
        
        # Test with scalar
        result = E.mtimes(2.0)
        self.assertIs(result, E)
        
        # Test with matrix
        matrix = np.array([[1, 0], [0, 1]])
        result = E.mtimes(matrix)
        self.assertIs(result, E)

    def test_plus(self):
        """Test plus method - should return self."""
        E = EmptySet(2)
        
        # Test with scalar
        result = E.plus(2.0)
        self.assertIs(result, E)
        
        # Test with vector
        vector = np.array([[1], [2]])
        result = E.plus(vector)
        self.assertIs(result, E)

    def test_and_(self):
        """Test and_ method - should return self."""
        E1 = EmptySet(2)
        E2 = EmptySet(2)
        
        result = E1.and_(E2)
        self.assertIs(result, E1)

    def test_dim(self):
        """Test dim method."""
        dimensions = [0, 1, 2, 3, 5, 10]
        
        for n in dimensions:
            E = EmptySet(n)
            self.assertEqual(E.dim(), n)



    def test_center(self):
        """Test center method - should return empty array."""
        E = EmptySet(2)
        
        center = E.center()
        self.assertEqual(center.shape, (2, 0))
        self.assertEqual(center.size, 0)

    def test_center_different_dimensions(self):
        """Test center method for different dimensions."""
        dimensions = [0, 1, 3, 5]
        
        for n in dimensions:
            E = EmptySet(n)
            center = E.center()
            self.assertEqual(center.shape, (n, 0))
            self.assertEqual(center.size, 0)


if __name__ == '__main__':
    unittest.main() 