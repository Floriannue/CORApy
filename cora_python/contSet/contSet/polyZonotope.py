from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def polyZonotope(self, *varargin):
    """
    conversion to polyZonotope objects

    Syntax:
        pZ = polyZonotope(S)

    Inputs:
        S - contSet object

    Outputs:
        pZ - polyZonotope object
    """

    # is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:noops', self) 