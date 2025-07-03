"""
contains_ - check if polygon object contains another polygon or points

Syntax:
    [res,cert,scaling] = contains_(pgon1, pgon2, method,tol,maxEval,certToggle,scalingToggle,varargin)

Inputs:
    pgon1 - polygon
    pgon2 - polygon or numeric array of points
    method - 'exact'
    tol - tolerance for the containment check
    maxEval - Currently has no effect
    certToggle - if set to True, cert will be computed
    scalingToggle - if set to True, scaling will be computed

Outputs:
    res - true/false
    cert - certificate
    scaling - scaling factor

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       13-March-2020 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Union, Tuple

if TYPE_CHECKING:
    from .polygon import Polygon
    from ..contSet import ContSet

try:
    from shapely.geometry import Point
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def contains_(pgon1: 'Polygon', pgon2: Union['ContSet', np.ndarray], method: str = 'exact',
              tol: float = 1e-12, maxEval: int = 200, certToggle: bool = False, 
              scalingToggle: bool = False, *varargin) -> Tuple[bool, float, float]:
    """
    Check if polygon object pgon1 contains polygon pgon2 or points
    
    Args:
        pgon1: polygon object
        pgon2: polygon object or numeric array of points
        method: method for computation (currently only 'exact')
        tol: tolerance for containment check
        maxEval: maximum evaluations (currently no effect)
        certToggle: whether to compute certification
        scalingToggle: whether to compute scaling
        
    Returns:
        tuple: (res, cert, scaling)
    """
    
    # The code is not yet ready to deal with scaling or cert
    cert = np.nan
    scaling = np.inf
    
    if scalingToggle or certToggle:
        raise CORAerror('CORA:notSupported',
            "The computation of the scaling factor or cert " +
            "for polygon containment is not yet implemented.")
    
    if not SHAPELY_AVAILABLE:
        raise CORAerror('CORA:missingDependency',
            'Shapely is required for polygon operations')
    
    # Enlarge pgon1 by tolerance (buffer operation)
    pgon1_buffered = pgon1.set.buffer(tol) if hasattr(pgon1.set, 'buffer') else pgon1.set
    
    if isinstance(pgon2, np.ndarray):
        # Check point containment
        if pgon2.ndim == 1:
            # Single point
            point = Point(pgon2[0], pgon2[1])
            res = pgon1_buffered.contains(point) or pgon1_buffered.touches(point)
        else:
            # Multiple points
            if pgon2.shape[0] == 2:
                # Points are in columns: pgon2 is 2×n
                points = [Point(pgon2[0, i], pgon2[1, i]) for i in range(pgon2.shape[1])]
            else:
                # Points are in rows: pgon2 is n×2  
                points = [Point(pgon2[i, 0], pgon2[i, 1]) for i in range(pgon2.shape[0])]
            
            res = np.array([pgon1_buffered.contains(p) or pgon1_buffered.touches(p) 
                           for p in points])
            
    elif hasattr(pgon2, '__class__') and hasattr(pgon2, '__bases__') and any('ContSet' in str(base) for base in type(pgon2).__bases__):
        # Convert to polygon
        from .polygon import Polygon
        if not isinstance(pgon2, Polygon):
            # Try to convert other contSet to polygon
            try:
                vertices = pgon2.vertices()
                pgon2 = Polygon(vertices)
            except:
                raise CORAerror('CORA:notSupported',
                    'This set representation cannot be converted to polygon!')
        
        # Compute union
        union_poly = unary_union([pgon1.set, pgon2.set])
        
        # Check if area of pgon1 is identical to area of union
        A1 = pgon1.set.area
        A2 = union_poly.area
        
        # Use tolerance for comparison
        res = abs(A1 - A2) <= tol
        
    else:
        # Try to convert to numpy array first
        try:
            pgon2 = np.asarray(pgon2)
            # Recursive call with numpy array
            return contains_(pgon1, pgon2, method, tol, maxEval, certToggle, scalingToggle, *varargin)
        except:
            raise CORAerror('CORA:notSupported',
                'This set representation is not supported!')
    
    return res, cert, scaling 