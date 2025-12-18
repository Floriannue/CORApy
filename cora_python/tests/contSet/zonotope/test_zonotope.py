"""
test_zonotope - unit test function of zonotope (constructor)

Syntax:
    res = test_zonotope

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       27-July-2021
Last update:   ---
Last revision: ---
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.compareMatrices import compareMatrices


class TestZonotope:
    """Test class for zonotope constructor"""
    
    def test_empty_zonotopes(self):
        """Test empty zonotopes"""
        # Empty zonotope using empty method
        Z = Zonotope.empty(2)
        assert Z.representsa_('emptySet')
        assert Z.dim() == 2
        
        # Empty zonotope using zeros(3,0)
        Z = Zonotope(np.zeros((3, 0)))
        assert Z.representsa_('emptySet')
        assert Z.c.shape[0] == 3
        assert Z.G.shape[0] == 3
        
        # Empty zonotope using zeros(3,0), []
        Z = Zonotope(np.zeros((3, 0)), np.array([]).reshape(0, 0))
        assert Z.representsa_('emptySet')
        assert Z.c.shape[0] == 3
        assert Z.G.shape[0] == 3
        
        # Empty zonotope using zeros(3,0), zeros(3,0)
        Z = Zonotope(np.zeros((3, 0)), np.zeros((3, 0)))
        assert Z.representsa_('emptySet')
        assert Z.c.shape[0] == 3
        assert Z.G.shape[0] == 3
    
    def test_admissible_initializations(self):
        """Test admissible initializations"""
        # Random center, random generator matrix
        c = np.array([[3], [3], [2]])
        G = np.array([[2, -4, -6, 3, 5], [1, -7, 3, -5, 2], [0, 4, -7, 3, 2]])
        Zmat = np.column_stack([c, G])
        
        # Center and generator matrix
        Z = Zonotope(c, G)
        assert compareMatrices(Z.c, c)
        assert compareMatrices(Z.G, G)
        
        # Center only
        Z = Zonotope(c)
        assert compareMatrices(Z.c, c)
        assert Z.G.size == 0 and Z.G.shape[0] == 3
        
        # Combined matrix [c, G]
        Z = Zonotope(Zmat)
        assert compareMatrices(Z.c, Zmat[:, 0:1])
        assert compareMatrices(Z.G, Zmat[:, 1:])
    
    def test_wrong_instantiations(self):
        """Test wrong instantiations"""
        c = np.array([[3], [3], [2]])
        G = np.array([[2, -4, -6, 3, 5], [1, -7, 3, -5, 2], [0, 4, -7, 3, 2]])
        
        # Center and generator matrix do not match
        c_plus1 = np.array([[4], [6], [-2], [3]])
        G_plus1 = np.array([[2, -4, -6, 3, 5], [1, -7, 3, -5, 2], [0, 4, -7, 3, 2], [2, 0, 5, -4, 2]])
        np.random.seed(42)  # For reproducibility
        randLogicals = np.random.randn(*G.shape) > 0
        c_NaN = c.copy().astype(float)  # Convert to float to allow NaN
        c_NaN[1] = np.nan
        G_NaN = G.copy().astype(float)  # Convert to float to allow NaN
        G_NaN[randLogicals] = np.nan
        
        # Center and generator matrix do not match
        with pytest.raises(CORAerror) as exc_info:
            Zonotope(c_plus1, G)
        assert exc_info.value.identifier == 'CORA:wrongInputInConstructor'
        
        with pytest.raises(CORAerror) as exc_info:
            Zonotope(c, G_plus1)
        assert exc_info.value.identifier == 'CORA:wrongInputInConstructor'
        
        # Center is empty
        with pytest.raises(CORAerror) as exc_info:
            Zonotope(np.array([]).reshape(0, 0), G)
        assert exc_info.value.identifier == 'CORA:wrongInputInConstructor'
        
        # Center has NaN entry
        with pytest.raises(CORAerror) as exc_info:
            Zonotope(c_NaN, G)
        assert exc_info.value.identifier == 'CORA:wrongValue'
        
        # Generator matrix has NaN entries
        with pytest.raises(CORAerror) as exc_info:
            Zonotope(c, G_NaN)
        assert exc_info.value.identifier == 'CORA:wrongValue'
        
        # Too many input arguments
        with pytest.raises(CORAerror) as exc_info:
            Zonotope(c, G, G)
        assert exc_info.value.identifier == 'CORA:numInputArgsConstructor'


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 