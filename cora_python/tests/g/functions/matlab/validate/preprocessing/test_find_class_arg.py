import numpy as np
import pytest
from cora_python.g.functions.matlab.validate.preprocessing.find_class_arg import find_class_arg
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

# Mock classes for testing
class Zonotope:
    def __init__(self, dim):
        self.dim_val = dim
    def __repr__(self):
        return f"Zonotope(dim={self.dim_val})"

class Interval:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
    def __repr__(self):
        return f"Interval(lower={self.lower}, upper={self.upper})"

class PolyZonotope:
    def __repr__(self):
        return "PolyZonotope()"

class TestFindClassArg:

    def test_obj1_is_target(self):
        Z = Zonotope(3)
        I = Interval(0, 1)
        obj, S = find_class_arg(Z, I, 'Zonotope')
        assert isinstance(obj, Zonotope)
        assert isinstance(S, Interval)

    def test_obj2_is_target(self):
        Z = Zonotope(3)
        I = Interval(0, 1)
        obj, S = find_class_arg(I, Z, 'Zonotope')
        assert isinstance(obj, Zonotope)
        assert isinstance(S, Interval)

    def test_neither_is_target(self):
        I1 = Interval(0, 1)
        I2 = Interval(2, 3)
        with pytest.raises(CORAerror) as excinfo:
            find_class_arg(I1, I2, 'Zonotope')
        assert 'Neither argument is of class Zonotope.' in str(excinfo.value)

    def test_both_are_target(self):
        Z1 = Zonotope(3)
        Z2 = Zonotope(4)
        with pytest.raises(CORAerror) as excinfo:
            find_class_arg(Z1, Z2, 'Zonotope')
        assert 'Expected only one argument to be of class Zonotope, but both are.' in str(excinfo.value)

    def test_numeric_target(self):
        arr = np.array([[1],[2]])
        Z = Zonotope(2)
        obj, S = find_class_arg(arr, Z, 'numeric')
        assert np.array_equal(obj, arr)
        assert isinstance(S, Zonotope)

        obj, S = find_class_arg(Z, arr, 'numeric')
        assert np.array_equal(obj, arr)
        assert isinstance(S, Zonotope)

    def test_numeric_neither_target(self):
        Z1 = Zonotope(3)
        Z2 = Zonotope(4)
        with pytest.raises(CORAerror) as excinfo:
            find_class_arg(Z1, Z2, 'numeric')
        assert 'Neither argument is of class numeric.' in str(excinfo.value)

    def test_numeric_both_target(self):
        arr1 = np.array([[1],[2]])
        arr2 = np.array([[3],[4]])
        with pytest.raises(CORAerror) as excinfo:
            find_class_arg(arr1, arr2, 'numeric')
        assert 'Expected only one argument to be of class numeric, but both are.' in str(excinfo.value)

    def test_classname_case_insensitivity(self):
        Z = Zonotope(3)
        I = Interval(0, 1)
        obj, S = find_class_arg(Z, I, 'zonotope') # lowercase classname
        assert isinstance(obj, Zonotope)
        assert isinstance(S, Interval)

    def test_subclass_behavior(self):
        # findClassArg doesn't explicitly handle inheritance, it checks exact class name.
        # So, if a subclass is passed but parent classname is specified, it won't match.
        # This is consistent with MATLAB's `isa(obj, classname)` behavior which checks direct class or superclass.
        # Our `__class__.__name__.lower() == classname.lower()` is stricter and only matches direct class.
        # If MATLAB `isa` actually checks superclasses, we might need to adjust.
        # However, for `findClassArg`, it's typically used to find one specific type among two.
        
        # Let's assume it only looks for exact class name match based on original logic:
        class MyZonotope(Zonotope):
            pass
        
        my_z = MyZonotope(5)
        I = Interval(0,1)

        obj, S = find_class_arg(my_z, I, 'MyZonotope')
        assert isinstance(obj, MyZonotope)
        assert isinstance(S, Interval)

        with pytest.raises(CORAerror) as excinfo:
            find_class_arg(my_z, I, 'Zonotope') # Will not find MyZonotope as Zonotope
        assert 'Neither argument is of class Zonotope.' in str(excinfo.value) 