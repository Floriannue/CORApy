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

    def test_constructor_default(self):
        """Test emptySet constructor with default dimension."""
        E = EmptySet()
        
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
        
        self.assertTrue(E.isemptyobject())

    def test_ismember(self):
        """Test ismember method - should always return False."""
        E = EmptySet(2)
        point = np.array([[1], [2]])
        
        self.assertFalse(E.ismember(point))

    def test_ismember_different_points(self):
        """Test ismember method with different points."""
        E = EmptySet(3)
        
        # Test various points
        points = [
            np.array([[0], [0], [0]]),
            np.array([[1], [2], [3]]),
            np.array([[-1], [-2], [-3]]),
            np.array([[0.5], [1.5], [2.5]])
        ]
        
        for point in points:
            self.assertFalse(E.ismember(point))

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

    def test_minus(self):
        """Test minus method - should return self."""
        E = EmptySet(2)
        
        # Test with scalar
        result = E.minus(2.0)
        self.assertIs(result, E)
        
        # Test with vector
        vector = np.array([[1], [2]])
        result = E.minus(vector)
        self.assertIs(result, E)

    def test_or_(self):
        """Test or_ method - should return self."""
        E1 = EmptySet(2)
        E2 = EmptySet(2)
        
        result = E1.or_(E2)
        self.assertIs(result, E1)

    def test_and_(self):
        """Test and_ method - should return self."""
        E1 = EmptySet(2)
        E2 = EmptySet(2)
        
        result = E1.and_(E2)
        self.assertIs(result, E1)

    def test_sup(self):
        """Test sup method - should return negative infinity."""
        E = EmptySet(2)
        direction = np.array([[1], [0]])
        
        result = E.sup(direction)
        self.assertEqual(result, -np.inf)

    def test_inf(self):
        """Test inf method - should return positive infinity."""
        E = EmptySet(2)
        direction = np.array([[1], [0]])
        
        result = E.inf(direction)
        self.assertEqual(result, np.inf)

    def test_dim(self):
        """Test dim method."""
        dimensions = [0, 1, 2, 3, 5, 10]
        
        for n in dimensions:
            E = EmptySet(n)
            self.assertEqual(E.dim(), n)

    def test_isempty(self):
        """Test isempty method - should always return True."""
        dimensions = [0, 1, 2, 3, 5]
        
        for n in dimensions:
            E = EmptySet(n)
            self.assertTrue(E.isempty())

    def test_center(self):
        """Test center method - should return empty array."""
        E = EmptySet(2)
        
        center = E.center()
        self.assertEqual(center.shape, (0, 1))
        self.assertEqual(center.size, 0)

    def test_center_different_dimensions(self):
        """Test center method for different dimensions."""
        dimensions = [0, 1, 3, 5]
        
        for n in dimensions:
            E = EmptySet(n)
            center = E.center()
            self.assertEqual(center.shape, (0, 1))
            self.assertEqual(center.size, 0)


if __name__ == '__main__':
    unittest.main() 