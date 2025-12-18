"""
test_polyZonotope - unit test function for constructor

Syntax:
    res = test_polyZonotope

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       28-April-2023
Last update:   04-October-2024 (MW, check default properties)
Last revision: ---
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope


class TestPolyZonotope:
    """Test class for polyZonotope constructor"""
    
    def test_empty_polyZonotope(self):
        """Test empty polyZonotope"""
        n = 2
        pZ = PolyZonotope.empty(n)
        assert pZ.c.shape == (n, 0)
        assert pZ.G.shape == (n, 0)
        assert pZ.GI.shape == (n, 0)
        assert pZ.E.shape == (0, 0)
        assert pZ.id.shape == (0, 1)
        assert pZ.representsa_('emptySet', 1e-10)
    
    def test_all_syntaxes(self):
        """Test all different syntaxes from constructor"""
        # Create polynomial zonotope
        c = np.array([[0], [0]])
        G = np.array([[2, 0, 1], [0, 2, 1]])
        GI = np.array([[0], [0.5]])
        E = np.array([[1, 0, 3], [0, 1, 1]])
        E_def = np.eye(3)
        id = np.array([[5], [6]])
        id_def2 = np.array([[1], [2]])
        id_def3 = np.array([[1], [2], [3]])
        
        # Only center
        pZ = PolyZonotope(c)
        assert pZ.G.shape == (2, 0)
        
        # Only center and dependent generator matrix
        pZ = PolyZonotope(c, G)
        assert pZ.GI.shape == (2, 0)
        assert np.allclose(pZ.E, E_def)
        assert np.allclose(pZ.id.flatten(), id_def3.flatten())
        
        # Center and both generator matrices
        pZ = PolyZonotope(c, G, GI)
        assert np.allclose(pZ.E, E_def)
        assert np.allclose(pZ.id.flatten(), id_def3.flatten())
        
        # Only independent generator matrix
        pZ = PolyZonotope(c, np.array([]).reshape(0, 0), GI)
        assert pZ.G.shape == (2, 0)
        assert pZ.E.shape == (0, 0)
        assert pZ.id.shape == (0, 1)
        
        # Both generator matrices and exponent matrix
        pZ = PolyZonotope(c, G, GI, E)
        assert np.allclose(pZ.id.flatten(), id_def2.flatten())
        
        # No independent generator matrix
        pZ = PolyZonotope(c, G, np.array([]).reshape(0, 0), E)
        assert pZ.GI.shape == (2, 0)
        assert np.allclose(pZ.id.flatten(), id_def2.flatten())
        
        # No independent generator matrix, with identifiers
        pZ = PolyZonotope(c, G, np.array([]).reshape(0, 0), E, id)
        assert pZ.GI.shape == (2, 0)
        
        # All input arguments
        pZ = PolyZonotope(c, G, GI, E, id)
        
        # Copy constructor
        pZ = PolyZonotope(pZ)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

