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
    
    # Get display string using display_()
    display_result = fs.display_()
    assert isinstance(display_result, str)
    assert "fullspace:" in display_result or "R^" in display_result
    assert str(n) in display_result
    
    # Test that display() prints the same
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        fs.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_result
    finally:
        sys.stdout = old_stdout
    
    # Test that __str__ returns the same
    assert str(fs) == display_result


if __name__ == "__main__":
    test_fullspace_display() 