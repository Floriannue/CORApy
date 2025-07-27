"""
Test cases for aux_minkDiffZono function
"""

import numpy as np
import pytest
from cora_python.g.functions.helper.sets.contSet.zonotope.aux_minkDiffZono import aux_minkDiffZono
from cora_python.contSet.zonotope.zonotope import Zonotope


def test_aux_minkDiffZono_basic():
    """Test basic functionality with simple zonotopes"""
    # Create simple zonotopes
    minuend = Zonotope(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
    subtrahend = Zonotope(np.array([0.5, 0.5]), np.array([[0.5, 0], [0, 0.5]]))
    
    result = aux_minkDiffZono(minuend, subtrahend, 'approx')
    
    assert isinstance(result, Zonotope)
    assert result.dim() == 2


def test_aux_minkDiffZono_inner_method():
    """Test with inner method"""
    minuend = Zonotope(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
    subtrahend = Zonotope(np.array([0.5, 0.5]), np.array([[0.5, 0], [0, 0.5]]))
    
    result = aux_minkDiffZono(minuend, subtrahend, 'inner')
    
    assert isinstance(result, Zonotope)
    assert result.dim() == 2


def test_aux_minkDiffZono_outer_method():
    """Test with outer method"""
    minuend = Zonotope(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
    subtrahend = Zonotope(np.array([0.5, 0.5]), np.array([[0.5, 0], [0, 0.5]]))
    
    result = aux_minkDiffZono(minuend, subtrahend, 'outer')
    
    assert isinstance(result, Zonotope)
    assert result.dim() == 2


def test_aux_minkDiffZono_outer_coarse_method():
    """Test with outer:coarse method"""
    minuend = Zonotope(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
    subtrahend = Zonotope(np.array([0.5, 0.5]), np.array([[0.5, 0], [0, 0.5]]))
    
    result = aux_minkDiffZono(minuend, subtrahend, 'outer:coarse')
    
    assert isinstance(result, Zonotope)
    assert result.dim() == 2


def test_aux_minkDiffZono_exact_2d():
    """Test exact method for 2D zonotopes"""
    minuend = Zonotope(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
    subtrahend = Zonotope(np.array([0.5, 0.5]), np.array([[0.5, 0], [0, 0.5]]))
    
    result = aux_minkDiffZono(minuend, subtrahend, 'exact')
    
    assert isinstance(result, Zonotope)
    assert result.dim() == 2


def test_aux_minkDiffZono_unknown_method():
    """Test with unknown method"""
    minuend = Zonotope(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
    subtrahend = Zonotope(np.array([0.5, 0.5]), np.array([[0.5, 0], [0, 0.5]]))
    
    with pytest.raises(Exception):
        aux_minkDiffZono(minuend, subtrahend, 'unknown_method') 