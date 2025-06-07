import numpy as np
from ..contSet.ContSet import ContSet

class Interval(ContSet):
    """
    Represents an n-dimensional interval.

    An interval is defined by a lower bound (inf) and an upper bound (sup)
    for each dimension.
    """

    def __init__(self, inf, sup):
        """
        Initializes an Interval object.

        Args:
            inf: The lower bound of the interval (e.g., a list or NumPy array).
            sup: The upper bound of the interval (e.g., a list or NumPy array).
        """
        self.inf = inf
        self.sup = sup

    def __repr__(self):
        """
        Returns a string representation of the Interval object.
        """
        return f"Interval(inf={self.inf}, sup={self.sup})"

    def __add__(self, other):
        """
        Computes the Minkowski sum of this interval with another interval.

        Args:
            other: Another Interval object.

        Returns:
            A new Interval object representing the Minkowski sum.

        Raises:
            TypeError: If other is not an Interval object.
            ValueError: If the dimensions of the intervals do not match.
        """
        if not isinstance(other, Interval):
            raise TypeError("Operand must be an Interval object.")

        # Assuming self.inf, self.sup, other.inf, other.sup are NumPy arrays
        # or types that support element-wise addition and have a 'shape' attribute.
        if np.shape(self.inf) != np.shape(other.inf) or \
           np.shape(self.sup) != np.shape(other.sup):
            raise ValueError("Interval dimensions must match for addition.")

        new_inf = self.inf + other.inf
        new_sup = self.sup + other.sup
        return Interval(new_inf, new_sup)

    def __mul__(self, other):
        """
        Computes the multiplication of this interval by a scalar or another interval.

        Args:
            other: A scalar (int or float) or another Interval object.

        Returns:
            A new Interval object representing the product.

        Raises:
            TypeError: If other is not a scalar or an Interval object.
            ValueError: If multiplying by another Interval and dimensions do not match.
        """
        if isinstance(other, (int, float)):
            # Scalar multiplication
            scalar = other
            if scalar >= 0:
                new_inf = self.inf * scalar
                new_sup = self.sup * scalar
            else:
                # When multiplying by a negative scalar, bounds flip
                new_inf = self.sup * scalar
                new_sup = self.inf * scalar
            return Interval(new_inf, new_sup)
        elif isinstance(other, Interval):
            # Interval multiplication
            # Ensure dimensions match for element-wise operation
            if np.shape(self.inf) != np.shape(other.inf) or \
               np.shape(self.sup) != np.shape(other.sup):
                raise ValueError("Interval dimensions must match for element-wise multiplication.")

            # Calculate all four products
            # self.inf = a, self.sup = b
            # other.inf = c, other.sup = d
            # products: ac, ad, bc, bd
            p1 = self.inf * other.inf  # ac
            p2 = self.inf * other.sup  # ad
            p3 = self.sup * other.inf  # bc
            p4 = self.sup * other.sup  # bd

            new_inf = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
            new_sup = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
            return Interval(new_inf, new_sup)
        else:
            raise TypeError("Operand must be a scalar or an Interval object.")

    def __rmul__(self, other):
        """
        Computes the right multiplication of this interval by a scalar.
        This handles cases like `scalar * interval_object`.

        Args:
            other: A scalar (int or float).

        Returns:
            A new Interval object representing the product.

        Raises:
            TypeError: If other is not a scalar.
        """
        # This method is called when the left operand does not support multiplication
        # with the interval type, but the right operand (self) does.
        # We expect 'other' to be a scalar here.
        if not isinstance(other, (int, float)):
            raise TypeError("Operand must be a scalar when multiplying from the left.")
        # The logic is the same as scalar multiplication in __mul__
        return self.__mul__(other)
