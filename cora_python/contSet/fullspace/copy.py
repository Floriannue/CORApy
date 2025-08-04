"""
copy - copies the fullspace object (used for dynamic dispatch)

Syntax:
   fs_out = copy(fs)

Inputs:
   fs - fullspace object

Outputs:
   fs_out - copied fullspace object

Example: 
   fs = fullspace(2);
   fs_out = copy(fs);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       30-September-2024
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

def copy(fs):
    """
    Copies the fullspace object (used for dynamic dispatch)
    
    Args:
        fs: fullspace object
        
    Returns:
        fs_out: copied fullspace object
    """
    # call copy constructor
    fs_out = type(fs)(fs)
    
    return fs_out

# ------------------------------ END OF CODE ------------------------------ 