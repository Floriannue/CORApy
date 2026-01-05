"""
display - displays the properties of a fullspace object (dimension) on
   the command window

Syntax:
   display(fs)

Inputs:
   fs - fullspace object

Outputs:
   -

Example: 
   fs = fullspace(2);
   display(fs);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       22-March-2023
Last update:   ---
Last revision: ---
"""


def display_(fs):
    """
    Displays the properties of a fullspace object (internal function that returns string)
    
    Args:
        fs: fullspace object
        
    Returns:
        str: Display string for the fullspace object
    """
    
    # Build display string similar to MATLAB
    result = f"fullspace:\n"
    result += f"- dimension: {fs.dimension}"
    
    return result


def display(fs):
    """
    Displays the properties of a fullspace object (prints to stdout)
    
    Args:
        fs: fullspace object
    """
    print(display_(fs), end='') 