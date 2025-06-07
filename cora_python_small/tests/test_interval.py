import unittest
import numpy as np
from cora_python_small.interval.interval import Interval

class TestInterval(unittest.TestCase):
    """Test suite for the Interval class."""

    def test_instantiation(self):
        """Tests Interval instantiation with various bound types."""
        # Scalar bounds
        i1 = Interval(1, 10)
        self.assertEqual(i1.inf, 1)
        self.assertEqual(i1.sup, 10)

        # List bounds
        i2 = Interval([-1, 0], [5, 6])
        # Assuming internal conversion to numpy arrays or list-like behavior
        if isinstance(i2.inf, np.ndarray):
            np.testing.assert_array_equal(i2.inf, np.array([-1, 0]))
            np.testing.assert_array_equal(i2.sup, np.array([5, 6]))
        else:
            self.assertEqual(i2.inf, [-1, 0])
            self.assertEqual(i2.sup, [5, 6])

        # NumPy array bounds
        i3_inf = np.array([1.0, 2.0])
        i3_sup = np.array([3.0, 4.0])
        i3 = Interval(i3_inf, i3_sup)
        np.testing.assert_array_equal(i3.inf, i3_inf)
        np.testing.assert_array_equal(i3.sup, i3_sup)

        # Test that input arrays are not modified if they are numpy arrays
        i3_inf_copy = np.copy(i3_inf)
        i3_sup_copy = np.copy(i3_sup)
        i4 = Interval(i3_inf, i3_sup)
        np.testing.assert_array_equal(i3_inf, i3_inf_copy, "Input inf array modified")
        np.testing.assert_array_equal(i3_sup, i3_sup_copy, "Input sup array modified")


    def test_repr(self):
        """Tests the __repr__ method."""
        i1 = Interval(1, 2)
        self.assertEqual(repr(i1), "Interval(inf=1, sup=2)")

        i2_inf = np.array([-1, 0])
        i2_sup = np.array([5, 6])
        i2 = Interval(i2_inf, i2_sup)
        # repr on numpy arrays includes "array(...)"
        expected_repr = f"Interval(inf={i2_inf!r}, sup={i2_sup!r})"
        # We need to be careful with how numpy array's repr is formatted.
        # For simplicity, let's check if essential parts are there.
        self.assertTrue(repr(i2).startswith("Interval(inf=array"))
        self.assertTrue("sup=array" in repr(i2))


    def test_add_intervals(self):
        """Tests addition of Interval objects."""
        # Scalar intervals
        i1 = Interval(1, 2)
        i2 = Interval(3, 4)
        result = i1 + i2
        self.assertEqual(result.inf, 4)
        self.assertEqual(result.sup, 6)

        # NumPy array intervals
        i3_inf = np.array([1, 5])
        i3_sup = np.array([2, 6])
        i3 = Interval(i3_inf, i3_sup)

        i4_inf = np.array([3, 0])
        i4_sup = np.array([4, 1])
        i4 = Interval(i4_inf, i4_sup)

        result_np = i3 + i4
        np.testing.assert_array_equal(result_np.inf, np.array([4, 5]))
        np.testing.assert_array_equal(result_np.sup, np.array([6, 7]))

        # Type error for adding non-Interval
        with self.assertRaisesRegex(TypeError, "Operand must be an Interval object."):
            _ = Interval(1, 2) + "not_an_interval"

        # Dimension mismatch
        i_dim1 = Interval(np.array([1]), np.array([2]))
        i_dim2 = Interval(np.array([1,0]), np.array([2,1]))
        with self.assertRaisesRegex(ValueError, "Interval dimensions must match for addition."):
            _ = i_dim1 + i_dim2

    def test_mul_scalar(self):
        """Tests multiplication of an Interval by a scalar."""
        # Positive scalar
        i1 = Interval(1, 2)
        res1 = i1 * 3
        self.assertEqual(res1.inf, 3)
        self.assertEqual(res1.sup, 6)

        # Negative scalar
        res2 = i1 * -3
        self.assertEqual(res2.inf, -6)
        self.assertEqual(res2.sup, -3)

        # NumPy array with positive scalar
        i_np = Interval(np.array([1, 10]), np.array([2, 12]))
        res_np1 = i_np * 2
        np.testing.assert_array_equal(res_np1.inf, np.array([2, 20]))
        np.testing.assert_array_equal(res_np1.sup, np.array([4, 24]))

        # NumPy array with negative scalar
        res_np2 = i_np * -2
        np.testing.assert_array_equal(res_np2.inf, np.array([-4, -24]))
        np.testing.assert_array_equal(res_np2.sup, np.array([-2, -20]))

    def test_rmul_scalar(self):
        """Tests right multiplication of an Interval by a scalar."""
        i1 = Interval(1, 2)
        res1 = 3 * i1
        self.assertEqual(res1.inf, 3)
        self.assertEqual(res1.sup, 6)

        res2 = -3 * i1
        self.assertEqual(res2.inf, -6)
        self.assertEqual(res2.sup, -3)

        # Numpy array
        i_np = Interval(np.array([1, 10]), np.array([2, 12]))
        res_np1 = 2 * i_np
        np.testing.assert_array_equal(res_np1.inf, np.array([2, 20]))
        np.testing.assert_array_equal(res_np1.sup, np.array([4, 24]))


    def test_mul_intervals(self):
        """Tests multiplication of two Interval objects."""
        # Case 1: All positive
        i1 = Interval(1, 2)
        i2 = Interval(3, 4)
        res1 = i1 * i2
        self.assertEqual(res1.inf, 3) # min(1*3, 1*4, 2*3, 2*4) = min(3,4,6,8)=3
        self.assertEqual(res1.sup, 8) # max(3,4,6,8)=8

        # Case 2: Mixed signs
        i3 = Interval(-1, 2)
        i4 = Interval(3, 4) # [-1,2]*[3,4] -> min(-3,-4,6,8)=-4, max(-3,-4,6,8)=8
        res2 = i3 * i4
        self.assertEqual(res2.inf, -4)
        self.assertEqual(res2.sup, 8)

        # Case 3: Both mixed signs
        i5 = Interval(-1, 2)
        i6 = Interval(-3, 4) # [-1,2]*[-3,4] -> min(3,-4,-6,8)=-6, max(3,-4,-6,8)=8
        res3 = i5 * i6
        self.assertEqual(res3.inf, -6)
        self.assertEqual(res3.sup, 8)

        # NumPy array intervals
        i7 = Interval(np.array([-1, 2]), np.array([2, 3]))
        i8 = Interval(np.array([3, -4]), np.array([4, -2]))
        # El1: [-1,2]*[3,4] -> p=(-3,-4,6,8) -> inf=-4, sup=8
        # El2: [2,3]*[-4,-2] -> p=(-8,-4,-12,-6) -> inf=-12, sup=-4
        res_np = i7 * i8
        np.testing.assert_array_equal(res_np.inf, np.array([-4, -12]))
        np.testing.assert_array_equal(res_np.sup, np.array([8, -4]))

        # Type error for multiplying non-Interval/non-scalar
        with self.assertRaisesRegex(TypeError, "Operand must be a scalar or an Interval object."):
            _ = Interval(1, 2) * "not_an_interval"

        # Dimension mismatch
        i_dim1 = Interval(np.array([1]), np.array([2]))
        i_dim2 = Interval(np.array([1,0]), np.array([2,1]))
        with self.assertRaisesRegex(ValueError, "Interval dimensions must match for element-wise multiplication."):
            _ = i_dim1 * i_dim2


if __name__ == '__main__':
    unittest.main()
