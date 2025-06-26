from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def conZonotope(self, *varargin):
    """
    conversion to conZonotope objects

    Syntax:
        cZ = conZonotope(S)

    Inputs:
        S - contSet object

    Outputs:
        cZ - conZonotope object
    """

    # is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:noops', self) 