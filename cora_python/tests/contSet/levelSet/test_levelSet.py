"""
Unit tests for the LevelSet class constructor.
"""

import pytest
import sympy as sp
from cora_python.contSet.levelSet import LevelSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_single_equation():
    """Tests the constructor with a single symbolic equation."""
    x, y = sp.symbols('x y')
    eq = x**2 + y**2 - 4
    
    # Test with '=='
    ls1 = LevelSet(eq, [x, y], '==')
    assert isinstance(ls1, LevelSet)
    assert str(ls1.eq) == str(eq)
    
    # Test with '<'
    ls2 = LevelSet(eq, [x, y], '<')
    assert isinstance(ls2, LevelSet)

    # Test with '<='
    ls3 = LevelSet(eq, [x, y], '<=')
    assert isinstance(ls3, LevelSet)

def test_multiple_equations():
    """Tests the constructor with multiple symbolic equations."""
    a, b = sp.symbols('a b')
    eq1 = sp.sin(a) + sp.log(b)
    eq2 = abs(a) * b

    # Test with ['<=', '<=']
    ls1 = LevelSet([eq1, eq2], [a, b], ['<=', '<='])
    assert isinstance(ls1, LevelSet)
    
    # Test with ['<=', '<']
    ls2 = LevelSet([eq1, eq2], [a, b], ['<=', '<'])
    assert isinstance(ls2, LevelSet)

def test_independent_variables():
    """Tests the constructor with independent variables."""
    a, b, x, y = sp.symbols('a b x y')
    eqs = [a, b, x, y]
    ls = LevelSet(eqs, [a, b, x, y], ['<=', '<=', '<=', '<='])
    assert isinstance(ls, LevelSet)

def test_unused_variable_in_vars():
    """Tests the constructor with an unused variable in the variable list."""
    a, b, y = sp.symbols('a b y')
    eq = a + y
    ls = LevelSet(eq, [a, b, y], '==')
    assert isinstance(ls, LevelSet)

def test_errors():
    """Tests error handling for invalid inputs."""
    a, b = sp.symbols('a b')
    eq1 = sp.sin(a) + sp.log(b)
    eq2 = abs(a) * b
    
    # Test for wrong number of comparison operators
    with pytest.raises(CORAerror):
         LevelSet([eq1, eq2], [a, b], '==')

if __name__ == '__main__':
    pytest.main() 