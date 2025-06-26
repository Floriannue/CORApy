"""
test_generateRandom - unit test function for polyZonotope generateRandom

Tests the random polyZonotope generation functionality.

Authors: MATLAB: Mark Wetzlinger
         Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope


class TestPolyZonotopeGenerateRandom:
    """Test class for polyZonotope generateRandom method"""
    
    def test_generateRandom_empty_call(self):
        """Test generateRandom with empty call"""
        pZ = PolyZonotope.generateRandom()
        assert isinstance(pZ, PolyZonotope)
        assert pZ.dim() > 0
    
    def test_generateRandom_dimension_only(self):
        """Test generateRandom with only dimension specified"""
        n = 3
        pZ = PolyZonotope.generateRandom(Dimension=n)
        assert pZ.dim() == n
    
    def test_generateRandom_nr_generators_only(self):
        """Test generateRandom with only number of generators specified"""
        nrGens = 10
        pZ = PolyZonotope.generateRandom(NrGenerators=nrGens)
        assert isinstance(pZ, PolyZonotope)
        # Number of generators check would depend on implementation
    
    def test_generateRandom_nr_factors_only(self):
        """Test generateRandom with only number of factors specified"""
        nrFac = 3
        pZ = PolyZonotope.generateRandom(NrFactors=nrFac)
        assert isinstance(pZ, PolyZonotope)
        assert pZ.E.shape[0] == nrFac
    
    def test_generateRandom_nr_ind_generators_only(self):
        """Test generateRandom with only number of independent generators specified"""
        nrIndGens = 5
        pZ = PolyZonotope.generateRandom(NrIndGenerators=nrIndGens)
        assert isinstance(pZ, PolyZonotope)
        assert pZ.GI.shape[1] == nrIndGens
    
    def test_generateRandom_dimension_and_generators(self):
        """Test generateRandom with dimension and number of generators"""
        n = 3
        nrGens = 10
        pZ = PolyZonotope.generateRandom(Dimension=n, NrGenerators=nrGens)
        assert pZ.dim() == n
    
    def test_generateRandom_dimension_and_factors(self):
        """Test generateRandom with dimension and number of factors"""
        n = 3
        nrFac = 3
        pZ = PolyZonotope.generateRandom(Dimension=n, NrFactors=nrFac)
        assert pZ.dim() == n and pZ.E.shape[0] == nrFac
    
    def test_generateRandom_dimension_generators_factors(self):
        """Test generateRandom with dimension, generators, and factors"""
        n = 3
        nrGens = 10
        nrFac = 3
        pZ = PolyZonotope.generateRandom(Dimension=n, NrGenerators=nrGens, NrFactors=nrFac)
        assert pZ.dim() == n and pZ.E.shape[0] == nrFac
    
    def test_generateRandom_dimension_factors_ind_generators(self):
        """Test generateRandom with dimension, factors, and independent generators"""
        n = 3
        nrFac = 3
        nrIndGens = 5
        pZ = PolyZonotope.generateRandom(Dimension=n, NrFactors=nrFac, NrIndGenerators=nrIndGens)
        assert pZ.dim() == n and pZ.E.shape[0] == nrFac
    
    def test_generateRandom_dimension_generators_ind_generators(self):
        """Test generateRandom with dimension, generators, and independent generators"""
        n = 3
        nrGens = 10
        nrIndGens = 5
        pZ = PolyZonotope.generateRandom(Dimension=n, NrGenerators=nrGens, NrIndGenerators=nrIndGens)
        assert pZ.dim() == n and pZ.GI.shape[1] == nrIndGens
    
    def test_generateRandom_all_parameters(self):
        """Test generateRandom with all parameters specified"""
        n = 3
        nrGens = 10
        nrFac = 3
        nrIndGens = 5
        pZ = PolyZonotope.generateRandom(
            Dimension=n, 
            NrGenerators=nrGens, 
            NrFactors=nrFac, 
            NrIndGenerators=nrIndGens
        )
        assert (pZ.dim() == n and 
                pZ.E.shape[0] == nrFac and 
                pZ.GI.shape[1] == nrIndGens)
    
    def test_generateRandom_various_dimensions(self):
        """Test generateRandom with various dimensions"""
        for n in [1, 2, 5, 10]:
            pZ = PolyZonotope.generateRandom(Dimension=n)
            assert pZ.dim() == n
            assert not pZ.isemptyobject()
    
    def test_generateRandom_properties(self):
        """Test properties of randomly generated polyZonotope"""
        pZ = PolyZonotope.generateRandom(Dimension=4, NrFactors=2, NrIndGenerators=3)
        
        # Should have correct dimension
        assert pZ.dim() == 4
        
        # Should not be empty
        assert not pZ.isemptyobject()
        
        # Should have correct matrix dimensions
        assert pZ.c.shape[0] == 4  # center dimension
        assert pZ.E.shape[0] == 2  # number of factors
        assert pZ.GI.shape[0] == 4  # independent generators dimension
        assert pZ.GI.shape[1] == 3  # number of independent generators
    
    def test_generateRandom_edge_cases(self):
        """Test edge cases for generateRandom"""
        # Minimum dimension
        pZ = PolyZonotope.generateRandom(Dimension=1)
        assert pZ.dim() == 1
        
        # No factors
        pZ = PolyZonotope.generateRandom(Dimension=2, NrFactors=0)
        assert pZ.dim() == 2
        assert pZ.E.shape[0] == 0
        
        # No independent generators
        pZ = PolyZonotope.generateRandom(Dimension=2, NrIndGenerators=0)
        assert pZ.dim() == 2
        assert pZ.GI.shape[1] == 0 