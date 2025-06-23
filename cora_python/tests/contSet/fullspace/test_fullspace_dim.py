"""
test_fullspace_dim - unit test function of dim

Syntax:
    res = test_fullspace_dim()

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Mark Wetzlinger
Written:       05-April-2023
Last update:   ---
Last revision: ---
"""

from cora_python.contSet.fullspace import Fullspace


def test_fullspace_dim():
    """Test fullspace dim method"""
    
    # n-dimensional fullspace
    n = 2
    fs = Fullspace(n)
    assert fs.dim() == n


if __name__ == "__main__":
    test_fullspace_dim() 