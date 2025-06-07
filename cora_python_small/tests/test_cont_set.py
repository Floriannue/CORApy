import unittest
from cora_python_small.contSet.ContSet import ContSet

class TestContSet(unittest.TestCase):
    """Test suite for the ContSet base class."""

    def test_instantiation(self):
        """Tests if ContSet can be instantiated."""
        try:
            cs = ContSet()
            self.assertIsInstance(cs, ContSet, "ContSet should be instantiable.")
        except TypeError:
            # This might be raised if ContSet is truly abstract (e.g. uses ABC)
            # For now, we assume it's not strictly abstract or can be instantiated.
            # If instantiation is not allowed, this test needs to change
            # to assert that TypeError is raised.
            self.fail("ContSet raised TypeError on instantiation. If abstract, test needs adjustment.")

if __name__ == '__main__':
    unittest.main()
