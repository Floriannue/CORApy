from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def interval(self, *varargin):
    """
    conversion to interval objects

    Syntax:
        I = interval(S)

    Inputs:
        S - contSet object

    Outputs:
        I - interval object
    """

    # is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:noops', self) 