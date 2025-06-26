import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3D

from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

@pytest.fixture
def setup_plots():
    # Setup for tests: close all plots before and after
    plt.close('all')
    yield
    plt.close('all')

def test_plotRandPoint_dimensions(setup_plots):
    # 1D plot
    z_1d = Zonotope(np.array([0]), np.array([1]))
    handle_1d = z_1d.plotRandPoint([0], 10)
    assert isinstance(handle_1d, Line2D)

    # 2D plot
    z_2d = Zonotope(np.zeros((2, 1)), np.eye(2))
    handle_2d = z_2d.plotRandPoint([0, 1], 10)
    assert isinstance(handle_2d, Line2D)

    # 3D plot
    z_3d = Zonotope(np.zeros((3, 1)), np.eye(3))
    handle_3d = z_3d.plotRandPoint([0, 1, 2], 10)
    assert isinstance(handle_3d, Line3D)

def test_plotRandPoint_error_handling(setup_plots):
    # Test error for > 3 dimensions
    z_4d = Zonotope(np.zeros((4, 1)), np.eye(4))
    with pytest.raises(CORAerror) as e:
        z_4d.plotRandPoint([0, 1, 2, 3], 10)
    assert "Number of dimensions to plot has to be <= 3" in str(e.value) 