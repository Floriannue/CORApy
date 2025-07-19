"""
enclosePoints - enclose a point cloud with a fullspace object

Syntax:
   fs = enclosePoints(points)

Inputs:
   points - point cloud (nxm matrix)

Outputs:
   fs - fullspace object

Example: 
   points = [2 4 -2; 1 0 5];
   fs = fullspace.enclosePoints(points);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       25-April-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

from cora_python.contSet.fullspace import Fullspace

def enclosePoints(points):
    """
    Enclose a point cloud with a fullspace object
    
    Args:
        points: point cloud (nxm matrix)
        
    Returns:
        fs: fullspace object
    """

    
    fs = Fullspace(points.shape[0])
    
    return fs

# ------------------------------ END OF CODE ------------------------------ 