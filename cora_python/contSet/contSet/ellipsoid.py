from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def ellipsoid(self, *varargin):
    """
    conversion to ellipsoid objects

    Syntax:
        E = ellipsoid(S)

    Inputs:
        S - contSet object

    Outputs:
        E - ellipsoid object
    """

    # is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:noops', self) 