"""
reduce - reduces the order of a zonotope, the resulting zonotope is an
    over-approximation of the original zonotope

Syntax:
    Z = reduce(Z, method)
    Z = reduce(Z, method, order)
    Z = reduce(Z, method, order, filterLength)
    Z = reduce(Z, method, order, filterLength, option)

Inputs:
    Z - zonotope object
    method - string specifying the reduction method:
                   - 'adaptive'        Thm. 3.2. in [6]
                   - 'cluster'         Sec. III.B in [3]
                   - 'combastel'       Sec. 3.2 in [4]
                   - 'constOpt'        Sec. III.D in [3]
                   - 'girard'          Sec. 4 in [2]
                   - 'methA'           Sec. 2.5.5 in [1]
                   - 'methB'           Sec. 2.5.5 in [1]
                   - 'methC'           Sec. 2.5.5 in [1]
                   - 'pca'             Sec. III.A in [3]
                   - 'scott'           Appendix of [5]
                   - 'sadraddini'      Proposition 6 in [8]
                   - 'scale'
                   - 'scaleHausdorff'  
                   - 'redistribute'
                   - 'valero'          
    order - order of reduced zonotope
    filterLength - ???
    options - ???
    alg - ???

Outputs:
    Z - zonotope object
    dHerror - (optional, only 'adaptive' and 'scaleHausdorff') 
              over-approximation of the Hausdorff distance between the 
              original and reduced zonotope
    gredIdx - index of reduced generators

References:
    [1] M. Althoff. "Reachability analysis and its application to the 
        safety assessment of autonomous cars", 2010
    [2] A. Girard. "Reachability of uncertain linear systems using
        zonotopes". 2005
    [3] A. Kopetzki et al. "Methods for order reduction of zonotopes", 
        CDC 2017
    [4] C. Combastel. "A state bounding observer based on zonotopes",
        ECC 2003
    [5] J. Scott et al. "Constrained zonotopes: A new tool for set-based 
        estimation and fault detection", Automatica 2016
    [6] M. Wetzlinger et al. "Adaptive parameter tuning for reachability
        analysis of nonlinear systems", HSCC 2021.
    [7] C.E. Valero et al. "On minimal volume zonotope order reduction",
        Automatica 2021 (in revision)
    [8] S. Sadraddini and R. Tedrake. "Linear Encodings for Polytope
        Containment Problems", CDC 2019 (ArXiV version)

Other m-files required: none
Subfunctions: see below
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       24-January-2007 (MATLAB)
Last update:   06-October-2024 (MW, refactor including priv_) (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Union, Optional, Tuple
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def reduce(Z: 'Zonotope', method: str, order: Optional[int] = 1, 
          filterLength: Optional[int] = None, option: Optional[str] = None) -> Union['Zonotope', Tuple['Zonotope', ...]]:
    """
    Reduces the order of a zonotope
    
    Args:
        Z: Zonotope object
        method: Reduction method string
        order: Order of reduced zonotope (default: 1)
        filterLength: Optional filter length for some methods
        option: Optional additional option
        
    Returns:
        Zonotope or tuple: Reduced zonotope object, and optionally additional outputs
        
    Raises:
        CORAerror: If method is not supported
    """
    from .zonotope import Zonotope
    from .private.priv_reduceMethods import (
        priv_reduceGirard, priv_reduceIdx, priv_reduceAdaptive, priv_reduceCombastel,
        priv_reducePCA, priv_reduceMethA, priv_reduceMethB, priv_reduceMethC,
        priv_reduceMethE, priv_reduceMethF, priv_reduceRedistribute, priv_reduceCluster,
        priv_reduceScott, priv_reduceValero, priv_reduceSadraddini, priv_reduceScale,
        priv_reduceScaleHausdorff, priv_reduceConstOpt
    )
    
    # Handle default values
    if order is None:
        order = 1
    if filterLength is None:
        filterLength = []
    if option is None:
        option = []
    
    # Remove substring necessary for special reduction for polyZonotopes
    if method.startswith('approxdep_'):
        method = method.replace('approxdep_', '')
    
    # Select method
    if method == 'girard':
        return priv_reduceGirard(Z, order)
    
    elif method == 'idx':
        # note: var 'order' is not an order here
        return priv_reduceIdx(Z, order)
    
    elif method == 'adaptive':
        # note: var 'order' is not an order here!
        # Check if we should track details (from Z attribute if available)
        track_details = False
        if hasattr(Z, '_track_reduction_details'):
            track_details = Z._track_reduction_details
            delattr(Z, '_track_reduction_details')  # Clean up
        # Pass tracking flag - priv_reduceAdaptive accepts type as 3rd arg and track_details as keyword
        # But we need to pass track_details, so use the dict approach for backward compatibility
        if track_details:
            option_dict = {'type': 'girard', 'track_details': True}
            Z_reduced, dHerror, gredIdx = priv_reduceAdaptive(Z, order, option_dict)
        else:
            Z_reduced, dHerror, gredIdx = priv_reduceAdaptive(Z, order, 'girard')
        return Z_reduced, dHerror, gredIdx
    
    elif method == 'adaptive-penven':
        # note: var 'order' is not an order here!
        Z_reduced, dHerror, gredIdx = priv_reduceAdaptive(Z, order, 'penven')
        return Z_reduced, dHerror, gredIdx
    
    elif method == 'combastel':
        return priv_reduceCombastel(Z, order)
    
    elif method == 'pca':
        return priv_reducePCA(Z, order)
    
    elif method == 'methA':
        return priv_reduceMethA(Z, order)
    
    elif method == 'methB':
        return priv_reduceMethB(Z, order, filterLength)
    
    elif method == 'methC':
        return priv_reduceMethC(Z, order, filterLength)
    
    elif method == 'methE':
        return priv_reduceMethE(Z, order)
    
    elif method == 'methF':
        return priv_reduceMethF(Z)
    
    elif method == 'redistribute':
        return priv_reduceRedistribute(Z, order)
    
    elif method == 'cluster':
        return priv_reduceCluster(Z, order, option)
    
    elif method == 'scott':
        return priv_reduceScott(Z, order)
    
    elif method == 'valero':
        return priv_reduceValero(Z, order)
    
    elif method == 'sadraddini':
        return priv_reduceSadraddini(Z, order)
    
    elif method == 'scale':
        return priv_reduceScale(Z, order)
    
    elif method == 'scaleHausdorff':
        Z_reduced, dHerror = priv_reduceScaleHausdorff(Z, order)
        return Z_reduced, dHerror
    
    elif method == 'constOpt':
        option = 'svd'
        alg = 'interior-point'
        return priv_reduceConstOpt(Z, order, option, alg)
    
    else:
        # wrong method
        raise CORAerror('CORA:wrongValue', 'second',
                       "'adaptive', 'adaptive-penven', 'cluster', 'combastel', 'constOpt', 'girard'" +
                       "'methA', 'methB', 'methC', 'pca', 'scott', 'redistribute', 'sadraddini'" +
                       "'scale', 'scaleHausdorff', or 'valero'")


 