"""
vertices_ - returns the vertices of a full-dimensional space

Syntax:
   V = vertices_(fs)

Inputs:
   fs - fullspace object

Outputs:
   V - vertices

Example: 
   fs = fullspace(2);
   V = vertices(fs);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/vertices

Authors:       Mark Wetzlinger
Written:       25-April-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def vertices_(fs, method: str = 'convHull', *args, **kwargs):
    """
    Returns the vertices of a full-dimensional space
    
    Args:
        fs: fullspace object
        method: Method for vertex computation (unused, kept for interface compatibility)
        *args: additional arguments (unused)
        **kwargs: Additional keyword arguments (unused)
        
    Returns:
        V: vertices
    """
    if fs.dimension == 0:
        raise CORAerror('CORA:notSupported',
                       'Vertices computation of R^0 not supported.')
    
    # convert to interval and compute vertices
    I = fs.interval()
    V = I.vertices_()
    
    return V

# ------------------------------ END OF CODE ------------------------------ 