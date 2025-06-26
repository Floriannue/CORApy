"""
cartProd_ method for zonotope class
"""

import numpy as np
from typing import Union
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cora_python.contSet.interval import Interval
    from cora_python.contSet.conZonotope import ConZonotope
    from cora_python.contSet.zonoBundle import ZonoBundle
    from cora_python.contSet.polytope import Polytope
    from cora_python.contSet.polyZonotope import PolyZonotope
    from cora_python.contSet.conPolyZono import ConPolyZono
    
def cartProd_(Z: Union[Zonotope, np.ndarray], S: Union[Zonotope, np.ndarray, 'Interval', 'ConZonotope', 'ZonoBundle', 'Polytope', 'PolyZonotope', 'ConPolyZono']) -> Zonotope:
    """
    Computes the Cartesian product of two zonotopes
    
    Args:
        Z: zonotope object or numeric array
        S: contSet object or numeric array
        
    Returns:
        Zonotope object representing the Cartesian product
    """
    from ..interval import Interval
    from ..conZonotope import ConZonotope
    from ..zonoBundle import ZonoBundle
    from ..polytope import Polytope
    from ..polyZonotope import PolyZonotope
    from ..conPolyZono import ConPolyZono
    # First or second set is zonotope
    if isinstance(Z, Zonotope):
        
        # Different cases for different set representations
        if isinstance(S, Zonotope):
            # Both are zonotopes - direct Cartesian product
            new_c = np.vstack([Z.c, S.c])
            new_G = np.block([[Z.G, np.zeros((Z.G.shape[0], S.G.shape[1]))],
                              [np.zeros((S.G.shape[0], Z.G.shape[1])), S.G]])
            return Zonotope(new_c, new_G)
            
        elif isinstance(S, np.ndarray):
            # S is a numeric vector
            new_c = np.vstack([Z.c, S.reshape(-1, 1)])
            new_G = np.vstack([Z.G, np.zeros((S.size, Z.G.shape[1]))])
            return Zonotope(new_c, new_G)
            
        elif isinstance(S, Interval):
            # Convert interval to zonotope and compute Cartesian product
            S_zono = S.zonotope()
            return Z.cartProd_(S_zono)
            
        elif isinstance(S, ConZonotope):
            # Convert to conZonotope and compute
            Z_con = Z.conZonotope()
            return Z_con.cartProd_(S)
            
        elif isinstance(S, ZonoBundle):
            # Convert to zonoBundle and compute
            Z_bundle = Z.zonoBundle()
            return Z_bundle.cartProd_(S)
            
        elif isinstance(S, Polytope):
            # Convert to polytope and compute
            Z_poly = Z.polytope()
            return Z_poly.cartProd_(S)
            
        elif isinstance(S, PolyZonotope):
            # Convert to polyZonotope and compute
            Z_pz = Z.polyZonotope()
            return Z_pz.cartProd_(S)
            
        elif isinstance(S, ConPolyZono):
            # Convert to conPolyZono and compute
            Z_cpz = Z.conPolyZono()
            return Z_cpz.cartProd_(S)
            
        else:
            # Throw error for unsupported arguments
            raise CORAerror('CORA:noops', f"Cartesian product not supported between {type(Z)} and {type(S)}")
    
    elif isinstance(S, Zonotope):
        
        # First argument is a vector
        if isinstance(Z, np.ndarray):
            new_c = np.vstack([Z.reshape(-1, 1), S.c])
            new_G = np.vstack([np.zeros((Z.size, S.G.shape[1])), S.G])
            return Zonotope(new_c, new_G)
        else:
            # Throw error for unsupported arguments
            raise CORAerror('CORA:noops', f"Cartesian product not supported between {type(Z)} and {type(S)}")
    
    else:
        # Throw error for unsupported arguments
        raise CORAerror('CORA:noops', f"Cartesian product not supported between {type(Z)} and {type(S)}") 