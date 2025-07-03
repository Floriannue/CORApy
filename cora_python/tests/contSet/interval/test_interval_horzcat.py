import pytest
import numpy as np
from cora_python.contSet.interval import Interval

class TestIntervalHorzcat:

    def test_horzcat_basic_row_vectors(self):
        """Test concatenation of two 1D-intervals (row vectors)."""
        i1 = Interval([-1], [1])
        i2 = Interval([2], [3])
        res = i1.horzcat(i2)
        
        expected_inf = np.array([[-1, 2]])
        expected_sup = np.array([[1, 3]])
        
        np.testing.assert_array_equal(res.inf, expected_inf)
        np.testing.assert_array_equal(res.sup, expected_sup)

    def test_horzcat_multiple_row_vectors(self):
        """Test concatenation of multiple 1D-intervals."""
        i1 = Interval([-1], [1])
        i2 = Interval([0], [2])
        i3 = Interval([1], [3])
        res = i1.horzcat(i2, i3)

        expected_inf = np.array([[-1, 0, 1]])
        expected_sup = np.array([[1, 2, 3]])
        
        np.testing.assert_array_equal(res.inf, expected_inf)
        np.testing.assert_array_equal(res.sup, expected_sup)

    def test_horzcat_column_vectors(self):
        """Test concatenation of column-vector intervals."""
        i1 = Interval([[-1], [0]], [[1], [2]])
        i2 = Interval([[2], [3]], [[4], [5]])
        res = i1.horzcat(i2)

        expected_inf = np.array([[-1, 2], [0, 3]])
        expected_sup = np.array([[1, 4], [2, 5]])
        
        np.testing.assert_array_equal(res.inf, expected_inf)
        np.testing.assert_array_equal(res.sup, expected_sup)
        
    def test_horzcat_matrix_intervals(self):
        """Test concatenation of matrix intervals."""
        i1 = Interval(np.ones((2, 2)), 2 * np.ones((2, 2)))
        i2 = Interval(3 * np.ones((2, 3)), 4 * np.ones((2, 3)))
        res = i1.horzcat(i2)

        np.testing.assert_array_equal(res.inf, np.hstack([i1.inf, i2.inf]))
        np.testing.assert_array_equal(res.sup, np.hstack([i1.sup, i2.sup]))
        assert res.inf.shape == (2, 5)

    def test_horzcat_with_numeric(self):
        """Test concatenation with numeric values."""
        i1 = Interval([-1], [1])
        num = 2
        res = i1.horzcat(num)

        expected_inf = np.array([[-1, 2]])
        expected_sup = np.array([[1, 2]])
        
        np.testing.assert_array_equal(res.inf, expected_inf)
        np.testing.assert_array_equal(res.sup, expected_sup)

    def test_horzcat_with_numpy_array(self):
        """Test concatenation with a numpy array."""
        i1 = Interval([[-1],[0]], [[1],[2]])
        arr = np.array([[2],[3]])
        res = i1.horzcat(arr)
        
        expected_inf = np.array([[-1, 2], [0, 3]])
        expected_sup = np.array([[1, 2], [2, 3]])

        np.testing.assert_array_equal(res.inf, expected_inf)
        np.testing.assert_array_equal(res.sup, expected_sup)

    def test_horzcat_single_argument(self):
        """Test horzcat with a single interval argument."""
        i1 = Interval([-1, 0], [1, 2])
        res = i1.horzcat() # This is equivalent to horzcat(I1)
        
        # Should return a 2D version of the original
        np.testing.assert_array_equal(res.inf, np.atleast_2d(i1.inf))
        np.testing.assert_array_equal(res.sup, np.atleast_2d(i1.sup))

    def test_horzcat_with_empty(self):
        """Test concatenation with an empty interval."""
        i1 = Interval([1, 2], [3, 4])
        i_empty = Interval.empty()
        
        # Concatenating with empty should not change the interval
        res1 = i1.horzcat(i_empty)
        np.testing.assert_array_equal(res1.inf, np.atleast_2d(i1.inf))
        
        res2 = i_empty.horzcat(i1)
        np.testing.assert_array_equal(res2.inf, np.atleast_2d(i1.inf))

    def test_dimension_mismatch(self):
        """Test that concatenation with mismatched dimensions raises an error."""
        i1 = Interval(np.zeros((2, 2)))
        i2 = Interval(np.zeros((3, 2)))
        with pytest.raises(ValueError):
            i1.horzcat(i2) 