import numpy as np
import pytest

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.polygon import Polygon
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_contSet_polygon():
    # Test that error is thrown for non-2D sets
    z_3d = Zonotope(np.zeros((3, 1)), np.eye(3))
    with pytest.raises(CORAerror) as e:
        z_3d.polygon()
    assert "Given set must be 2-dimensional" in str(e.value)

    # Test conversion of a 2D set
    z_2d = Zonotope(np.array([[1], [1]]), np.array([[1, 0], [0, 1]]))
    p = z_2d.polygon()

    # Check if the result is a Polygon
    assert isinstance(p, Polygon)

    # Check if vertices are correct (for a simple zonotope, vertices are easy to calculate)
    expected_v = np.array([
        [2., 2., 0., 0.],
        [2., 0., 0., 2.]
    ])
    
    # The vertices might be in a different order
    assert p.V.shape == expected_v.shape
    
    # Sort both sets of vertices to compare them
    sorted_p_v = p.V[:, np.lexsort(p.V)]
    sorted_expected_v = expected_v[:, np.lexsort(expected_v)]
    
    np.testing.assert_array_almost_equal(sorted_p_v, sorted_expected_v) 