"""
test_fullspace_display - unit test function of display

Syntax:
    res = test_fullspace_display()

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


def test_fullspace_display():
    """Test fullspace display method"""
    
    # n-dimensional fullspace
    n = 2
    fs = Fullspace(n)
    
    # only has to run through without errors...
    display_result = fs.display()
    assert isinstance(display_result, str)
    assert "fullspace:" in display_result
    assert str(n) in display_result


if __name__ == "__main__":
    test_fullspace_display() 