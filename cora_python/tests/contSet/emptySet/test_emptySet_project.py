"""
Test cases for project method of EmptySet class
"""

import pytest
from cora_python.contSet.emptySet import EmptySet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_project_basic():
    """Test basic projection functionality"""
    # init empty set
    n = 4
    O = EmptySet(n)
    
    # project to subspace
    projDims = [1, 3]
    O_ = O.project(projDims)
    
    # true solution
    O_true = EmptySet(len(projDims))
    
    # compare solutions
    assert O_.dimension == O_true.dimension


def test_project_single_dimension():
    """Test projection to single dimension"""
    O = EmptySet(3)
    O_ = O.project([2])
    
    assert O_.dimension == 1


def test_project_multiple_dimensions():
    """Test projection to multiple dimensions"""
    O = EmptySet(5)
    O_ = O.project([1, 2, 4])
    
    assert O_.dimension == 3


def test_project_all_dimensions():
    """Test projection to all dimensions"""
    O = EmptySet(3)
    O_ = O.project([1, 2, 3])
    
    assert O_.dimension == 3


def test_project_invalid_dimensions_negative():
    """Test projection with negative dimensions (should raise error)"""
    O = EmptySet(4)
    projDims = [-1, 2]
    
    with pytest.raises(CORAerror, match='interval is not inside the valid domain'):
        O.project(projDims)


def test_project_invalid_dimensions_out_of_range():
    """Test projection with dimensions out of range (should raise error)"""
    O = EmptySet(4)
    projDims = [3, 5]
    
    with pytest.raises(CORAerror, match='interval is not inside the valid domain'):
        O.project(projDims)


def test_project_dimension_zero():
    """Test projection with dimension 0 (should raise error)"""
    O = EmptySet(3)
    projDims = [0, 1]
    
    with pytest.raises(CORAerror, match='interval is not inside the valid domain'):
        O.project(projDims)


def test_project_dimension_too_large():
    """Test projection with dimension larger than original (should raise error)"""
    O = EmptySet(2)
    projDims = [1, 3]
    
    with pytest.raises(CORAerror, match='interval is not inside the valid domain'):
        O.project(projDims) 