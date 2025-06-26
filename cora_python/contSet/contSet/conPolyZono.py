from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def conPolyZono(self, *varargin):
    """
    conversion to conPolyZono objects

    Syntax:
        cPZ = conPolyZono(S)

    Inputs:
        S - contSet object

    Outputs:
        cPZ - conPolyZono object
    """

    # is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:noops', self) 