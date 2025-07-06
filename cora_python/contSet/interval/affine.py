from cora_python.contSet.affine import Affine

def affine(I, *varargin):
    """
    affine - conversion to affine objects
    
    Syntax:
        I = affine(I)
        
    Inputs:
        I - interval object
        name - a cell containing a name of a variable
        opt_method - method used to calculate interval over-approximations of
                     taylor models 
                      'int': standard interval arithmetic (default)
                      'bnb': branch and bound method is used to find min/max
                      'bnbAdv': branch and bound with re-expansion of Taylor models
        eps - precision for the selected optimization method (opt_method = 'bnb', 
              opt_method = 'bnbAdv', and opt_method = 'linQuad')
        tolerance - monomials with coefficients smaller than this value are
                    moved to the remainder
    
    Outputs:
        aff - affine object
    """
    
    return Affine(I.infimum(), I.supremum(), *varargin) 